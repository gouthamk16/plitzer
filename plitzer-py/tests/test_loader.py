import gc
import os
import signal
import sys
import threading
import time
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if sys.platform != "win32":
    import resource
import numpy.typing as npt
import pyarrow as pa
import pytest

import plitzer
from conftest import SETTINGS, MakeLoader

if TYPE_CHECKING:
    from typing_extensions import CapsuleType

Tokens = npt.NDArray[np.int32]


def epoch(loaders: Sequence[plitzer.Loader]) -> list[Tokens]:
    """The rest of the epoch: each step's global batch, all ranks in order."""
    return [
        np.concatenate([np.from_dlpack(b.tokens) for b in batches])
        for batches in zip(*loaders, strict=True)
    ]


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
        got = [
            np.concatenate([np.from_dlpack(b.tokens) for b in pair])
            for pair in zip(*two, strict=False)
        ]
        assert same(got, w)


def test_batch_shapes_and_padding(make: MakeLoader) -> None:
    batch = next(make())
    assert batch.tokens is batch.tokens
    assert batch.tokens.shape == (4, 8)
    assert batch.lengths.shape == (4,)
    tokens = np.from_dlpack(batch.tokens)
    lengths = np.from_dlpack(batch.lengths)
    assert tokens.dtype == np.int32 and lengths.dtype == np.int32
    for row, n in zip(tokens, lengths, strict=True):
        assert (row[n:] == -1).all(), "pad 2**32 - 1 reads as -1"
        assert (row[:n] >= 0).all()


class Legacy:
    """A producer as old consumers call it: `__dlpack__()` with no arguments."""

    def __init__(self, array: plitzer.Array) -> None:
        self.array = array

    def __dlpack__(self, *, stream: None = None, **_: object) -> "CapsuleType":
        return self.array.__dlpack__()

    def __dlpack_device__(self) -> tuple[int, int]:
        return self.array.__dlpack_device__()


def test_dlpack_versioned_and_legacy(make: MakeLoader) -> None:
    batch = next(make())
    assert '"dltensor_versioned"' in repr(batch.tokens.__dlpack__(max_version=(1, 0)))
    assert '"dltensor"' in repr(batch.tokens.__dlpack__())
    versioned = np.from_dlpack(batch.tokens)
    legacy = np.from_dlpack(Legacy(batch.tokens))
    assert np.array_equal(versioned, legacy)
    assert batch.tokens.__dlpack_device__() == (1, 0)


def max_rss_kib() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


@pytest.mark.skipif(sys.platform != "linux", reason="ru_maxrss is in KiB on Linux")
def test_unused_capsules_are_freed(make: MakeLoader) -> None:
    batch = next(make())
    for _ in range(10_000):  # reach the peak RSS before measuring
        batch.tokens.__dlpack__()
    before = max_rss_kib()
    # Each leaked capsule would keep about 200 bytes: about 60 MiB in all.
    for _ in range(150_000):
        batch.tokens.__dlpack__()
        batch.tokens.__dlpack__(max_version=(1, 0))
    gc.collect()
    assert max_rss_kib() - before < 16 * 1024


def test_no_default_cache_without_a_user_folder(
    dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ["PLITZER_CACHE_DIR", "XDG_CACHE_HOME", "LOCALAPPDATA", "HOME"]:
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(plitzer.ConfigError, match="cache"):
        plitzer.Loader(str(dataset), **SETTINGS)  # type: ignore[arg-type]


def test_default_cache_is_made_and_shared(
    dataset: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "new" / "cache"
    monkeypatch.setenv("PLITZER_CACHE_DIR", str(cache))
    for location in [str(dataset), str(dataset) + os.sep, dataset.as_uri()]:
        next(plitzer.Loader(location, **SETTINGS))  # type: ignore[arg-type]
    assert len(list(cache.iterdir())) == 1, "one index for each dataset"


def test_data_outlives_the_batch_and_the_loader(make: MakeLoader) -> None:
    loader = make()
    batch = next(loader)
    tokens = np.from_dlpack(batch.tokens)
    table = pa.array(batch.tokens)
    want = tokens.copy()
    del batch, loader
    gc.collect()
    # Fill the heap, so a freed buffer would likely be reused and differ.
    noise = [np.full(32, 7, np.int32) for _ in range(10_000)]
    assert np.array_equal(tokens, want)
    assert np.array_equal(table.to_numpy_ndarray(), want)
    del noise


def test_copy_and_device(make: MakeLoader) -> None:
    batch = next(make())
    shared = np.from_dlpack(batch.tokens)
    copied = np.from_dlpack(batch.tokens, copy=True)
    copied[0, 0] = 12345
    assert shared[0, 0] != 12345
    with pytest.raises(BufferError):
        batch.tokens.__dlpack__(dl_device=(2, 0))


def test_arrow_export(make: MakeLoader) -> None:
    batch = next(make())
    tokens = pa.array(batch.tokens)
    assert isinstance(tokens.type, pa.FixedShapeTensorType)
    assert tokens.type.shape == [8]
    assert np.array_equal(tokens.to_numpy_ndarray(), np.from_dlpack(batch.tokens))
    lengths = pa.array(batch.lengths)
    assert lengths.type == pa.int32()
    assert np.array_equal(lengths.to_numpy(), np.from_dlpack(batch.lengths))


def test_resume_on_another_world(make: MakeLoader) -> None:
    want = epoch([make()])
    for k in [0, 1, len(want) // 2]:
        two = [make(rank=r, world=2) for r in range(2)]
        head = [
            np.concatenate([np.from_dlpack(next(rank).tokens) for rank in two])
            for _ in range(k)
        ]
        checkpoint = plitzer.merge([rank.checkpoint() for rank in reversed(two)])
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


def test_bad_settings(make: MakeLoader, tmp_path: Path) -> None:
    with pytest.raises(plitzer.ConfigError, match="divide"):
        make(world=3)
    with pytest.raises(plitzer.ConfigError, match="world"):
        make(world=-1)
    with pytest.raises(plitzer.ConfigError, match="#"):
        plitzer.Loader(str(tmp_path.as_uri()) + "/data#2", **SETTINGS)  # type: ignore[arg-type]
    blocker = tmp_path / "file"
    blocker.write_bytes(b"")
    with pytest.raises(plitzer.ConfigError, match="cache"):
        make(cache=blocker / "index")
    with pytest.raises(plitzer.ConfigError):
        make(storage_options={"region": "x"})
    with pytest.raises(plitzer.ConfigError, match="budget"):
        next(make(budget=1))
    missing = str(tmp_path / "missing")
    with pytest.raises(plitzer.DataError):
        plitzer.Loader(missing, **SETTINGS)  # type: ignore[arg-type]
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
            # The copied loader is refused, but a new one works.
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
