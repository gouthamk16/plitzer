"""The phases of the baseline, each in a fresh process: the conversion,
the timed epoch, the tail and the resume. A process times its own work, so
imports and the start of the process do not count, and no cache in
memory, worker or shared-memory segment passes from one phase to the next.
"""

import hashlib
import importlib
import multiprocessing
import os
import sys
import time
from collections.abc import Iterator
from multiprocessing.connection import Connection
from pathlib import Path
from typing import NamedTuple, Protocol

import numpy as np
import numpy.typing as npt

Doc = npt.NDArray[np.int32]
# Each name is a module of plitzer.bench and its class. The child imports it
# after it starts, so that this module needs none of their dependencies.
LOADERS = {
    "plitzer": ("plitzer_adapter", "Plitzer"),
    "mosaic": ("mosaic", "Mosaic"),
    "hf": ("hf", "HF"),
    "hf-stream": ("hf", "HFStream"),
}
# Loaders that pack documents into sequences: their batches hold sequences,
# not documents, so documents per second does not apply to them.
PACKED = {"plitzer"}


class Run(NamedTuple):
    seed: int
    batch: int  # documents in each batch, or sequences for a packed loader
    workers: int  # loader worker processes, where the loader has them
    seq_len: int = 2048  # tokens in each sequence of a packed loader


class Opened(Protocol):
    """A loader at the start of an epoch, or at a saved state."""

    def __iter__(self) -> Iterator[list[Doc]]: ...
    def state(self) -> object:
        """A state after the batches given so far, to restore with ``open``."""
        ...


class Loader(Protocol):
    def convert(self, data: Path, work: Path) -> Path:
        """The data in the loader's own form, under ``work``, or ``data``."""
        ...

    def open(self, data: Path, prepared: Path, run: Run, state: object) -> Opened:
        """An epoch of ``prepared``, from ``state``, or the start if None."""
        ...


class Job(NamedTuple):
    phase: str  # "convert", "epoch", "tail" or "resume"
    loader: str
    data: Path
    work: Path
    run: Run
    resume_at: int  # the batches before the resume, 1 or more
    prepared: Path | None = None  # from the conversion
    state: object = None  # the state to resume from


class Phase(NamedTuple):
    seconds: float  # to the end; for a resume, to its first batch
    cpu: float | None  # CPU seconds of the process and its workers, if known
    docs: int = 0
    tokens: int = 0
    tail: str = ""  # digest of the batches from `resume_at`, or of a resume
    tail_docs: int = 0
    tail_tokens: int = 0
    state: object = None  # after `resume_at` batches
    prepared: Path | None = None


def parquet_files(data: Path) -> list[Path]:
    return sorted(data.rglob("*.parquet"))


class Tail:
    """A digest of a run of batches, in order, and their documents."""

    def __init__(self) -> None:
        self._hash = hashlib.sha256()
        self.docs = self.tokens = 0

    def add(self, batch: list[Doc]) -> None:
        self.docs += len(batch)
        self.tokens += sum(len(doc) for doc in batch)
        self._hash.update(len(batch).to_bytes(8, "little"))
        for doc in batch:
            self._hash.update(len(doc).to_bytes(8, "little"))
            self._hash.update(np.ascontiguousarray(doc, np.int32).tobytes())

    def digest(self) -> str:
        return self._hash.hexdigest()

    def phase(self, seconds: float) -> "Phase":
        return Phase(
            seconds,
            None,
            tail=self.digest(),
            tail_docs=self.docs,
            tail_tokens=self.tokens,
        )


def cpu() -> float:
    t = os.times()
    return t.user + t.system + t.children_user + t.children_system


def loader(name: str) -> Loader:
    module, cls = LOADERS[name]
    made: Loader = getattr(importlib.import_module(f"plitzer.bench.{module}"), cls)()
    return made


def workers_counted() -> bool:
    """Whether ``os.times()`` counts the CPU time of the workers: only for
    joined children of this process. Windows gives no child times, and a
    forkserver (Linux's default from Python 3.14) is the workers' parent."""
    return (
        sys.platform != "win32" and multiprocessing.get_start_method() != "forkserver"
    )


def epoch(opened: Opened, resume_at: int) -> Phase:
    """One timed epoch: counts, and the state after ``resume_at`` batches.
    It digests nothing, so that only the loader's work is timed."""
    t0, c0 = time.perf_counter(), cpu()
    docs = tokens = batches = 0
    state = None
    for batch in opened:
        docs += len(batch)
        tokens += sum(len(d) for d in batch)
        batches += 1
        if batches == resume_at:
            state = opened.state()
    if state is None or batches == resume_at:
        raise ValueError(f"resume_at {resume_at} is not inside {batches} batches")
    seconds, used = time.perf_counter() - t0, cpu() - c0
    return Phase(
        seconds, used if workers_counted() else None, docs, tokens, state=state
    )


def tail(opened: Opened, resume_at: int) -> Phase:
    """The digest of the batches after the first ``resume_at``, untimed."""
    digest = Tail()
    for i, batch in enumerate(opened):
        if i >= resume_at:
            digest.add(batch)
    return digest.phase(0.0)


def resume(opened: Opened, t0: float) -> Phase:
    """The rest of an epoch from a state, timed to its first batch."""
    first: float | None = None
    digest = Tail()
    for batch in opened:
        first = first or time.perf_counter() - t0
        digest.add(batch)
    seconds = first or time.perf_counter() - t0
    return digest.phase(seconds)


def child(conn: Connection, job: Job) -> None:
    load = loader(job.loader)
    t0, c0 = time.perf_counter(), cpu()
    if job.phase == "convert":
        out = load.convert(job.data, job.work)
        conn.send(Phase(time.perf_counter() - t0, cpu() - c0, prepared=out))
        return
    assert job.prepared is not None, "the conversion runs first"
    # The epoch's clock starts after `open`, the resume's before it: a
    # resume waits for the loader to start, and an epoch measures its pace.
    opened = load.open(job.data, job.prepared, job.run, job.state)
    if job.phase == "epoch":
        conn.send(epoch(opened, job.resume_at))
    elif job.phase == "tail":
        conn.send(tail(opened, job.resume_at))
    else:
        conn.send(resume(opened, t0))


def phase(job: Job) -> Phase:
    """``job`` in a new process."""
    ctx = multiprocessing.get_context("spawn")
    receive, send = ctx.Pipe(duplex=False)
    process = ctx.Process(target=child, args=(send, job))
    process.start()
    send.close()
    try:
        got = receive.recv()
    except EOFError:
        got = None
    process.join()
    if not isinstance(got, Phase) or process.exitcode != 0:
        raise RuntimeError(f"{job.loader} {job.phase} failed (exit {process.exitcode})")
    return got
