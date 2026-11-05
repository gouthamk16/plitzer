"""The end-to-end trainer with `plitzer.torch`: a gloo job whose ranks save
one global state through the collective `state_dict()`."""

import io
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
        try:
            state = ds.state_dict()  # a collective: every rank calls it
        except RuntimeError:
            # A peer was killed in this gather: see e2e.PEER_DIED.
            os._exit(e2e.PEER_DIED)
        path = Path(job.out, f"state-{step}")
        if rank == 0 and path.exists():  # the epoch end: see e2e.train
            saved = torch.load(path, weights_only=True)["plitzer"]
            assert saved == state["plitzer"], "the state changed after the epoch end"
        elif rank == 0:
            data = io.BytesIO()
            torch.save(state, data)
            e2e.atomic_write(path, data.getvalue())

    e2e.train(job, rank, ds.lanes, epoch, save)
    dist.destroy_process_group()


def torch_checkpoint(job: Job, step: int | None) -> tuple[int, str]:
    """The state saved at `step`, else the newest state that rank 0 saved,
    as a real trainer finds it."""
    if step is None:
        step = max(e2e.saved(Path(job.out), "state-"))
    return step, str(Path(job.out, f"state-{step}"))
