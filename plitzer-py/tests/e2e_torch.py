"""The end-to-end trainer with `plitzer.torch`: a gloo job whose ranks save
one global state through the collective `state_dict()`."""

import os
from collections.abc import Iterable
from pathlib import Path

import torch
import torch.distributed as dist

import e2e
import plitzer.torch as pt
from e2e import Arrays, Job


def settings(job: Job) -> pt.LoaderSettings:
    s = e2e.SETTINGS
    return pt.LoaderSettings(
        seed=s["seed"],
        lanes=s["lanes"],
        window=s["window"],
        seq_len=s["seq_len"],
        docs=s["docs"],
        cap=s["cap"],
        pad=s["pad"],
        budget=s["budget"],
        cache=job.cache,
    )


def torch_rank(rank: int, job: Job) -> None:
    """One rank: every rank calls `state_dict()`, and rank 0 saves it."""
    store = Path(job.out, "store").as_uri()
    dist.init_process_group("gloo", init_method=store, rank=rank, world_size=job.world)
    ds = pt.Dataset(job.location, **settings(job))
    if job.resume:
        ds.load_state_dict(torch.load(job.resume, weights_only=True))

    def epoch() -> Iterable[Arrays]:
        for b in ds:
            yield b.tokens.numpy(), b.lengths.numpy()

    def save(step: int) -> None:
        state = ds.state_dict()  # a collective: every rank calls it
        if rank == 0:
            path = Path(job.out, f"state-{step}.pt")
            temp = path.with_name(path.name + ".tmp")
            torch.save({"step": step, "plitzer": state["plitzer"]}, temp)
            os.replace(temp, path)

    e2e.train(job, rank, ds.lanes, epoch, save)
    dist.destroy_process_group()


def last_torch_checkpoint(job: Job) -> tuple[int, str | None]:
    """The newest state that rank 0 saved (none: step 0)."""
    for step in range(job.steps - job.steps % e2e.EVERY, 0, -e2e.EVERY):
        path = Path(job.out, f"state-{step}.pt")
        if path.exists():
            return step, str(path)
    return 0, None
