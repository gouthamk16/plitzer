"""The baseline: how existing loaders read the Parquet files that plitzer
reads. For each loader it measures the conversion, one shuffled epoch
(documents and tokens per second, CPU cores used) and a resume at
mid-epoch: the wait for the first batch, whether the rest of the epoch is
the one an uninterrupted run gives, and how many documents it loses.

    python -m plitzer.bench.baseline DATA --loaders mosaic,hf,hf-stream \\
        --batch 64 --workers 4

Each phase runs in a fresh process, so no cache in memory, worker or
shared-memory segment passes from one phase to the next. The adapters
(plitzer.bench.mosaic and plitzer.bench.hf) need the ``baseline`` extra.
"""

import argparse
import hashlib
import importlib
import multiprocessing
import os
import sys
import tempfile
import time
from collections.abc import Iterator, Sequence
from multiprocessing.connection import Connection
from pathlib import Path
from typing import NamedTuple, Protocol

import numpy as np
import numpy.typing as npt
import pyarrow.parquet as pq

Doc = npt.NDArray[np.int32]
# Each name is a module of plitzer.bench and its class. The child imports it
# after it starts, so that this module needs none of their dependencies.
LOADERS = {
    "mosaic": ("mosaic", "Mosaic"),
    "hf": ("hf", "HF"),
    "hf-stream": ("hf", "HFStream"),
}


class Run(NamedTuple):
    seed: int
    batch: int  # documents in each batch
    workers: int  # loader worker processes, where the loader has them


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
    phase: str  # "convert", "epoch" or "resume"
    loader: str
    data: Path
    work: Path
    run: Run
    resume_at: int  # the batch at which the resume starts
    state: object  # the state to resume from


class Phase(NamedTuple):
    seconds: float  # to the end; for a resume, to its first batch
    cpu: float  # CPU seconds of the process and its joined workers
    docs: int = 0
    tokens: int = 0
    tail: str = ""  # digest of the batches from `resume_at`, or of a resume
    tail_docs: int = 0
    state: object = None  # after `resume_at` batches
    prepared: Path | None = None


class Result(NamedTuple):
    loader: str
    convert_s: float
    convert_bytes: int
    docs_per_s: float
    tokens_per_s: float
    cores: float | None  # None where the OS gives no CPU time of workers
    resume_s: float  # to the first batch
    exact: bool  # the rest of the epoch is the one of an uninterrupted run
    lost: int  # documents of the rest of the epoch that the resume misses


def parquet_files(data: Path) -> list[Path]:
    return sorted(data.rglob("*.parquet"))


class Tail:
    """A digest of a run of batches, in order, and their documents."""

    def __init__(self) -> None:
        self._hash = hashlib.sha256()
        self.docs = 0

    def add(self, batch: list[Doc]) -> None:
        self.docs += len(batch)
        self._hash.update(len(batch).to_bytes(8, "little"))
        for doc in batch:
            self._hash.update(len(doc).to_bytes(8, "little"))
            self._hash.update(np.ascontiguousarray(doc, np.int32).tobytes())

    def digest(self) -> str:
        return self._hash.hexdigest()


def cpu() -> float:
    t = os.times()
    return t.user + t.system + t.children_user + t.children_system


def loader(name: str) -> Loader:
    module, cls = LOADERS[name]
    made: Loader = getattr(importlib.import_module(f"plitzer.bench.{module}"), cls)()
    return made


def epoch(opened: Opened, resume_at: int) -> Phase:
    """One epoch: counts, the state after ``resume_at`` batches, and the
    digest of the batches after it."""
    t0, c0 = time.perf_counter(), cpu()
    docs = tokens = 0
    state = None
    tail = Tail()
    for i, batch in enumerate(opened):
        docs += len(batch)
        tokens += sum(len(d) for d in batch)
        if i >= resume_at:
            tail.add(batch)
        if i == resume_at - 1:
            state = opened.state()
    seconds, used = time.perf_counter() - t0, cpu() - c0
    return Phase(seconds, used, docs, tokens, tail.digest(), tail.docs, state)


def resume(opened: Opened, t0: float, c0: float) -> Phase:
    """The rest of an epoch from a state, timed to its first batch."""
    first: tuple[float, float] | None = None
    tail = Tail()
    for batch in opened:
        first = first or (time.perf_counter() - t0, cpu() - c0)
        tail.add(batch)
    seconds, used = first or (time.perf_counter() - t0, cpu() - c0)
    return Phase(seconds, used, tail=tail.digest(), tail_docs=tail.docs)


def child(conn: Connection, job: Job) -> None:
    load = loader(job.loader)
    prepared = job.work / "prepared"
    t0, c0 = time.perf_counter(), cpu()
    if job.phase == "convert":
        out = load.convert(job.data, job.work)
        conn.send(Phase(time.perf_counter() - t0, cpu() - c0, prepared=out))
        return
    where = Path(prepared.read_text())
    if job.phase == "epoch":
        conn.send(epoch(load.open(job.data, where, job.run, None), job.resume_at))
        return
    conn.send(resume(load.open(job.data, where, job.run, job.state), t0, c0))


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


def dir_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def measure(name: str, data: Path, work: Path, run: Run, resume_at: int) -> Result:
    """The three phases of loader ``name`` over ``data``, using ``work``."""
    job = Job("convert", name, data, work, run, resume_at, None)
    convert = phase(job)
    assert convert.prepared is not None
    (work / "prepared").write_text(str(convert.prepared))
    size = 0 if convert.prepared == data else dir_bytes(convert.prepared)
    run_ = phase(job._replace(phase="epoch"))
    back = phase(job._replace(phase="resume", state=run_.state))
    s = run_.seconds
    # Windows gives no CPU time of child processes.
    cores = None if sys.platform == "win32" else run_.cpu / s
    exact = back.tail == run_.tail
    lost = run_.tail_docs - back.tail_docs
    return Result(
        name,
        convert.seconds,
        size,
        run_.docs / s,
        run_.tokens / s,
        cores,
        back.seconds,
        exact,
        lost,
    )


def table(results: Sequence[Result]) -> str:
    rows = [
        "| loader | convert (s) | converted (MiB) | docs/s | Mtokens/s | cores "
        "| resume (s) | exact resume | docs lost |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        cores = "—" if r.cores is None else f"{r.cores:.2f}"
        rows.append(
            f"| {r.loader} | {r.convert_s:.1f} | {r.convert_bytes / 2**20:.0f} "
            f"| {r.docs_per_s:,.0f} | {r.tokens_per_s / 1e6:.1f} | {cores} "
            f"| {r.resume_s:.2f} | {'yes' if r.exact else 'no'} | {r.lost:,} |"
        )
    return "\n".join(rows)


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    p.add_argument("data", type=Path)
    p.add_argument("--loaders", default=",".join(LOADERS))
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume-at", type=int, help="batch; default mid-epoch")
    p.add_argument(
        "--work", type=Path, help="for converted data; default a temp folder"
    )
    a = p.parse_args(argv)
    run = Run(a.seed, a.batch, a.workers)
    resume_at = a.resume_at or max(1, count_docs(a.data) // a.batch // 2)
    results = []
    with tempfile.TemporaryDirectory(dir=a.work) as tmp:
        for name in a.loaders.split(","):
            work = Path(tmp, name)
            work.mkdir()
            results.append(measure(name, a.data, work, run, resume_at))
    print(table(results))


def count_docs(data: Path) -> int:
    return sum(pq.read_metadata(p).num_rows for p in parquet_files(data))


if __name__ == "__main__":
    main()
