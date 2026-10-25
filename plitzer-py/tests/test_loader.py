import gc
import os
import signal
import threading
import time
import warnings
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

import plitzer
from conftest import SETTINGS, MakeLoader

Tokens = npt.NDArray[np.int32]


def global_batch(batches: Sequence[plitzer.Batch]) -> Tokens:
    """The tokens of one step of all ranks, in rank order."""
    return np.concatenate([np.from_dlpack(b.tokens) for b in batches])


def epoch(loaders: Sequence[plitzer.Loader]) -> list[Tokens]:
    """The rest of the epoch: each step's global batch."""
    return [global_batch(batches) for batches in zip(*loaders, strict=True)]


def same(a: Sequence[Tokens], b: Sequence[Tokens]) -> bool:
    return len(a) == len(b) and all(
        np.array_equal(x, y) for x, y in zip(a, b, strict=True)
    )


def test_ranks_split_the_global_batch(make: MakeLoader) -> None:
    one = epoch([make()])
    two = epoch([make(rank=0, world=2), make(rank=1, world=2)])
    assert len(one) > 0
    assert same(one, two)


def test_iteration_ends_each_epoch(make: MakeLoader) -> None:
    loader = make()
    first = epoch([loader])
    assert loader.epoch == 1
    second = epoch([loader])
    assert loader.epoch == 2
    assert not same(first, second)


def test_iterator_protocol(make: MakeLoader) -> None:
    loader = make()
    first = epoch([loader])
    with pytest.raises(StopIteration):
        next(loader)  # an ended iterator stays ended
    loader.set_epoch(0)  # changes nothing, so the iterator stays ended
    with pytest.raises(StopIteration):
        next(loader)
    assert loader.epoch == 1
    second = epoch([loader])
    assert not same(first, second)


def test_set_epoch_after_stop_iteration(make: MakeLoader) -> None:
    """A trainer that steps with next() and calls set_epoch each epoch."""
    want = make()
    first, second = epoch([want]), epoch([want])
    loader = make()
    got: list[list[Tokens]] = []
    for e in range(2):
        loader.set_epoch(e)
        got.append([])
        try:
            while True:
                tokens = np.from_dlpack(next(loader).tokens)
                got[e].append(tokens.astype(np.int32, copy=False))
        except StopIteration:
            pass
    assert same(got[0], first) and same(got[1], second)


def test_epoch_is_the_next_step(make: MakeLoader) -> None:
    steps = len(epoch([make()]))
    loader = make()
    for _ in range(steps):
        next(loader)
    # The last step is taken: the next step, and a checkpoint, are epoch 1.
    assert loader.epoch == 1
    with pytest.raises(StopIteration):
        next(loader)
    assert loader.epoch == 1


def test_checkpoint_during_a_step(make: MakeLoader) -> None:
    """A checkpoint from another thread while steps wait without the GIL."""
    loader = make()
    stop = threading.Event()
    saved: list[bytes] = []
    errors: list[BaseException] = []

    def watch() -> None:
        while not stop.is_set():
            try:
                saved.append(loader.checkpoint())
                assert loader.epoch in (0, 1)
            except BaseException as e:  # reported below, in the test thread
                errors.append(e)
                return

    watcher = threading.Thread(target=watch)
    watcher.start()
    epoch([loader])
    stop.set()
    watcher.join()
    assert not errors, errors
    assert saved
    for part in saved[:: max(1, len(saved) // 50)]:
        plitzer.merge([part])  # world 1: every part is a whole checkpoint


def test_non_strict_zip_keeps_ranks_together(make: MakeLoader) -> None:
    one = make()
    want = [epoch([one]), epoch([one])]
    two = [make(rank=r, world=2) for r in range(2)]
    for w in want:
        # zip stops at the first StopIteration and never asks rank 1 for its
        # end; `iter` moves rank 1 on to the next epoch too.
        got = [global_batch(pair) for pair in zip(*two, strict=False)]
        assert same(got, w)


def test_no_default_cache_without_a_user_folder(
    dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ["PLITZER_CACHE_DIR", "LOCALAPPDATA", "HOME"]:
        monkeypatch.delenv(name, raising=False)
    # A relative folder would differ for each working directory: unset.
    monkeypatch.setenv("XDG_CACHE_HOME", ".cache")
    with pytest.raises(plitzer.ConfigError, match="cache"):
        plitzer.Loader(str(dataset), **SETTINGS)


def test_default_cache_is_made_and_shared(
    dataset: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "new" / "cache"
    monkeypatch.setenv("PLITZER_CACHE_DIR", str(cache))
    for location in [str(dataset), str(dataset) + os.sep, dataset.as_uri()]:
        next(plitzer.Loader(location, **SETTINGS))
    assert len(list(cache.iterdir())) == 1, "one index for each dataset"


def test_resume_on_another_world(make: MakeLoader) -> None:
    want = epoch([make()])
    for k in [0, 1, len(want) // 2]:
        two = [make(rank=r, world=2) for r in range(2)]
        head = [global_batch([next(rank) for rank in two]) for _ in range(k)]
        parts = [bytearray(rank.checkpoint()) for rank in reversed(two)]
        checkpoint = bytearray(plitzer.merge(parts))  # a buffer works too
        four = [make(rank=r, world=4) for r in range(4)]
        for rank in four:
            rank.resume(checkpoint)
        assert same(head + epoch(four), want), f"resume at {k}"


def test_resume_refuses_other_runs(make: MakeLoader) -> None:
    saved = make().checkpoint()
    other = make(seq_len=16)
    with pytest.raises(plitzer.CheckpointError, match="seq_len"):
        other.resume(saved)
    with pytest.raises(plitzer.CheckpointError):
        other.resume(b"not a checkpoint")
    with pytest.raises(TypeError):
        plitzer.merge([list(saved)])  # type: ignore[list-item]
    with pytest.raises(plitzer.CheckpointError, match="missing"):
        plitzer.merge([make(rank=1, world=2).checkpoint()])


def fnv1a(data: bytes) -> int:
    h = 0xCBF29CE484222325
    for b in data:
        h = ((h ^ b) * 0x100000001B3) % 2**64
    return h


def test_resume_refuses_a_hand_made_checkpoint(make: MakeLoader) -> None:
    """Checksummed bytes with lane states that no run makes are refused at
    resume, with nothing changed, never a panic."""
    loader = make()
    for _ in range(3):
        next(loader)
    good = loader.checkpoint()
    # The last lane state is 20 bytes (step u64, doc u64, offset u32) before
    # the 8-byte checksum; move its leftover start back to doc 0.
    at = len(good) - 8 - 20
    doc = int.from_bytes(good[at + 8 : at + 16], "little")
    refused = 0
    for start in range(doc):
        body = bytearray(good[:-8])
        body[at + 8 : at + 20] = start.to_bytes(8, "little") + bytes(4)
        bad = bytes(body) + fnv1a(bytes(body)).to_bytes(8, "little")
        try:
            loader.resume(bad)
        except plitzer.CheckpointError:
            refused += 1
            assert loader.checkpoint() == good, "a refused resume changes nothing"
        loader.resume(good)
    assert refused > 0


def test_bad_settings(make: MakeLoader, tmp_path: Path) -> None:
    with pytest.raises(plitzer.ConfigError, match="divide"):
        make(world=3)
    with pytest.raises(plitzer.ConfigError, match="world"):
        make(world=-1)
    with pytest.raises(plitzer.ConfigError, match="seed"):
        make(seed=2**200)
    with pytest.raises(plitzer.ConfigError, match="epoch"):
        make().set_epoch(-1)
    with pytest.raises(plitzer.ConfigError, match="lanes must be an int, not float"):
        make(lanes=4.0)  # type: ignore[arg-type]
    with pytest.raises(plitzer.ConfigError, match="docs must be an int, not bool"):
        make(docs=True)
    make().set_epoch(np.int64(1))  # has __index__

    class Broken:
        def __index__(self) -> int:
            raise ZeroDivisionError("from __index__")

    with pytest.raises(ZeroDivisionError):  # not hidden as "out of range"
        make().set_epoch(Broken())

    with pytest.raises(plitzer.ConfigError, match="#"):
        plitzer.Loader(str(tmp_path.as_uri()) + "/data#2", **SETTINGS)
    blocker = tmp_path / "file"
    blocker.write_bytes(b"")
    with pytest.raises(plitzer.ConfigError, match="cache"):
        make(cache=blocker / "index")
    with pytest.raises(plitzer.ConfigError):
        make(storage_options={"region": "x"})
    with pytest.raises(plitzer.ConfigError, match="budget"):
        next(make(budget=1))
    with pytest.raises(plitzer.DataError):
        plitzer.Loader(tmp_path / "missing", **SETTINGS)
    assert issubclass(plitzer.ConfigError, plitzer.PlitzerError)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
def test_forked_child_makes_its_own_loader(make: MakeLoader) -> None:
    loader = make()
    first = np.from_dlpack(next(loader).tokens).copy()
    with warnings.catch_warnings():
        # Python warns that fork in a process with threads can deadlock.
        warnings.simplefilter("ignore", DeprecationWarning)
        pid = os.fork()
    if pid == 0:
        try:
            next(loader)
            code = 1
        except RuntimeError:
            # The copied loader is refused, also its checkpoint, whose lock a
            # parent thread could hold. A new loader works.
            try:
                loader.checkpoint()
                code = 4
            except RuntimeError:
                fresh = np.from_dlpack(next(make()).tokens)
                code = 0 if np.array_equal(fresh, first) else 3
        except BaseException:
            code = 2
        # Freeing the copied loader must not wait on the parent's runtime.
        del loader
        gc.collect()
        os._exit(code)
    # Without the guard the child would wait forever on threads it lacks.
    deadline = time.monotonic() + 20
    while (done := os.waitpid(pid, os.WNOHANG))[0] == 0:
        if time.monotonic() > deadline:
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
            pytest.fail("the forked child hung")
        time.sleep(0.05)
    assert os.waitstatus_to_exitcode(done[1]) == 0
