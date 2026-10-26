import multiprocessing
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt
import pytest
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

import plitzer
import plitzer.torch as pt
from conftest import SETTINGS

Steps = list[npt.NDArray[np.generic]]


def settings(tmp_path: Path) -> pt.LoaderSettings:
    """SETTINGS with a cache in `tmp_path`, as `Dataset` keywords."""
    s = SETTINGS
    return pt.LoaderSettings(
        seed=s["seed"],
        lanes=s["lanes"],
        window=s["window"],
        seq_len=s["seq_len"],
        docs=s["docs"],
        cap=s["cap"],
        pad=s["pad"],
        budget=s["budget"],
        cache=str(tmp_path / "index"),
    )


def reference(dataset: Path, tmp_path: Path) -> list[Steps]:
    """Epochs 0 and 1 of one GPU, from `plitzer.Loader`."""
    loader = plitzer.Loader(dataset, **settings(tmp_path))
    return [[np.from_dlpack(b.tokens).copy() for b in loader] for _ in range(2)]


def same(a: Steps, b: Steps) -> bool:
    return len(a) == len(b) and all(
        np.array_equal(x, y) for x, y in zip(a, b, strict=True)
    )


def tokens(batches: list[pt.TokenBatch]) -> Steps:
    return [b.tokens.numpy().copy() for b in batches]


def test_matches_the_loader(dataset: Path, tmp_path: Path) -> None:
    want = reference(dataset, tmp_path)
    ds = pt.Dataset(dataset, **settings(tmp_path))
    for epoch in want:
        got = list(ds)
        assert same(tokens(got), epoch)
        b = got[0]
        assert b.tokens.dtype == torch.int32 and b.lengths.dtype == torch.int32
        assert b.tokens.shape == (SETTINGS["lanes"], SETTINGS["seq_len"])
        assert b.lengths.shape == (SETTINGS["lanes"],)
    assert ds.epoch == 2


def test_dataloader_passes_batches_on(dataset: Path, tmp_path: Path) -> None:
    want = reference(dataset, tmp_path)[0]
    ds = pt.Dataset(dataset, **settings(tmp_path))
    got = list(DataLoader(ds, batch_size=None, num_workers=0))
    assert all(isinstance(b, pt.TokenBatch) for b in got)
    assert same(tokens(got), want)
    with pytest.raises(RuntimeError, match="num_workers=0"):
        next(iter(DataLoader(ds, batch_size=None, num_workers=1)))


def test_state_dict_resumes(dataset: Path, tmp_path: Path) -> None:
    want = reference(dataset, tmp_path)
    flat = want[0] + want[1]
    for k in [0, len(want[0]) // 2, len(want[0])]:
        ds = pt.Dataset(dataset, **settings(tmp_path))
        it = iter(ds)
        head = [next(it).tokens.numpy().copy() for _ in range(k)]
        path = tmp_path / f"state-{k}.pt"
        torch.save(ds.state_dict(), path)
        state = torch.load(path, weights_only=True)
        again = pt.Dataset(dataset, **settings(tmp_path))
        again.load_state_dict(state)
        # A state after the last step is at the start of the next epoch.
        tail = tokens(list(again))
        assert same(head + tail, flat[: len(head) + len(tail)]), k


def test_load_state_dict_refuses_bad_states(dataset: Path, tmp_path: Path) -> None:
    ds = pt.Dataset(dataset, **settings(tmp_path))
    bad_states: list[dict[str, object]] = [{}, {"plitzer": "text"}, {"other": b""}]
    for bad in bad_states:
        with pytest.raises(plitzer.CheckpointError, match="plitzer"):
            ds.load_state_dict(bad)
    with pytest.raises(plitzer.CheckpointError):
        ds.load_state_dict({"plitzer": b"not a checkpoint"})
    # A group without torch.distributed initialized is refused.
    group = cast(dist.ProcessGroup, object())
    with pytest.raises(plitzer.ConfigError, match="group"):
        pt.Dataset(dataset, group=group, **settings(tmp_path))


def run_rank(rank: int, world: int, store: str, job: tuple[str, str, str]) -> None:
    """One rank of a gloo job: resumes from `<out>/start` if it exists, takes
    the rest of the epoch, or `steps` steps, and saves its tokens and state."""
    location, out, steps = job
    dist.init_process_group(
        "gloo", init_method=Path(store).as_uri(), rank=rank, world_size=world
    )
    ds = pt.Dataset(location, **settings(Path(out)))
    start = Path(out, "start")
    if start.exists():
        ds.load_state_dict({"plitzer": start.read_bytes()})
    it = iter(ds)
    taken = [
        b.tokens.numpy().copy() for _, b in zip(range(int(steps)), it, strict=False)
    ]
    np.save(Path(out, f"tokens-{rank}.npy"), np.stack(taken))
    Path(out, f"state-{rank}").write_bytes(ds.state_dict()["plitzer"])
    dist.destroy_process_group()


def spawn(world: int, location: Path, out: Path, steps: int) -> list[bytes]:
    """Runs `run_rank` on `world` processes, and gives the state of each."""
    store = str(out / "store")
    job = (str(location), str(out), str(steps))
    # Spawn, not fork: a forked child cannot use the parent's Rust threads.
    context = multiprocessing.get_context("spawn")
    ranks = [
        context.Process(target=run_rank, args=(r, world, store, job), daemon=True)
        for r in range(world)
    ]
    for p in ranks:
        p.start()
    for p in ranks:
        p.join(timeout=120)
        if p.exitcode is None:
            p.kill()
        assert p.exitcode == 0, f"a rank failed with {p.exitcode}"
    return [(out / f"state-{r}").read_bytes() for r in range(world)]


def global_steps(world: int, out: Path) -> Steps:
    ranks = [np.load(out / f"tokens-{r}.npy") for r in range(world)]
    return list(np.concatenate(ranks, axis=1))


@pytest.mark.skipif(
    not (dist.is_available() and dist.is_gloo_available()), reason="needs gloo"
)
def test_state_dict_gathers_all_ranks(dataset: Path, tmp_path: Path) -> None:
    """Two ranks save the same global state; four ranks resume from it and
    give the tail of a one-GPU run."""
    want = reference(dataset, tmp_path)[0]
    k = len(want) // 2
    two = tmp_path / "two"
    two.mkdir()
    states = spawn(2, dataset, two, k)
    assert states[0] == states[1], "every rank holds the same state"
    assert same(global_steps(2, two), want[:k])
    four = tmp_path / "four"
    four.mkdir()
    (four / "start").write_bytes(states[0])
    spawn(4, dataset, four, len(want) - k)
    assert same(global_steps(4, four), want[k:])
    # One GPU resumes from it too.
    one = pt.Dataset(dataset, **settings(tmp_path))
    one.load_state_dict({"plitzer": states[0]})
    assert same(tokens(list(one)), want[k:])
