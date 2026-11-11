"""The baseline: how existing loaders read the Parquet files that plitzer
reads. For each loader it measures the conversion, one shuffled epoch
(documents and tokens per second, CPU cores used) and a resume at
mid-epoch: the wait for the first batch, whether the rest of the epoch is
the one an uninterrupted run gives, and how many documents it loses.

    python -m plitzer.bench.baseline DATA --loaders plitzer,mosaic,hf,hf-stream \\
        --batch 64 --workers 4

Each phase runs in a fresh process, so no cache in memory, worker or
shared-memory segment passes from one phase to the next. The adapters
(plitzer.bench.mosaic and plitzer.bench.hf) need the ``baseline`` extra;
plitzer's (plitzer.bench.plitzer_adapter) needs only plitzer.
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

from plitzer import _plitzer

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
    lost_tokens: int  # their tokens, which also compare packed loaders
    packed: bool = False

    @property
    def ns_per_token(self) -> float | None:
        """CPU nanoseconds for each token: the cores per token per second."""
        return None if self.cores is None else self.cores / self.tokens_per_s * 1e9


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
    return Phase(
        0.0,
        None,
        tail=digest.digest(),
        tail_docs=digest.docs,
        tail_tokens=digest.tokens,
    )


def resume(opened: Opened, t0: float) -> Phase:
    """The rest of an epoch from a state, timed to its first batch."""
    first: float | None = None
    digest = Tail()
    for batch in opened:
        first = first or time.perf_counter() - t0
        digest.add(batch)
    seconds = first or time.perf_counter() - t0
    return Phase(
        seconds,
        None,
        tail=digest.digest(),
        tail_docs=digest.docs,
        tail_tokens=digest.tokens,
    )


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


def warm(data: Path) -> None:
    """Reads the source files once, so that each loader starts with them in
    the page cache. A conversion reads them and writes a copy that is still
    in memory for its epoch; with no conversion, a loader would otherwise
    read them from the disk, and its row would measure the disk."""
    for path in parquet_files(data):
        with path.open("rb") as f:
            while f.read(16 << 20):
                pass


def dir_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def measure(name: str, data: Path, work: Path, run: Run, resume_at: int) -> Result:
    """The phases of loader ``name`` over ``data``, using ``work``."""
    if name not in LOADERS:
        raise ValueError(f"no loader {name!r}; the loaders are {', '.join(LOADERS)}")
    if resume_at < 1:
        raise ValueError("resume_at is a count of batches, 1 or more")
    warm(data)
    convert = phase(Job("convert", name, data, work, run, resume_at))
    job = Job("epoch", name, data, work, run, resume_at, convert.prepared)
    size = 0 if convert.prepared == data else dir_bytes(work)
    timed = phase(job)
    want = phase(job._replace(phase="tail"))
    back = phase(job._replace(phase="resume", state=timed.state))
    s = timed.seconds
    return Result(
        loader=name,
        convert_s=convert.seconds,
        convert_bytes=size,
        docs_per_s=timed.docs / s,
        tokens_per_s=timed.tokens / s,
        cores=None if timed.cpu is None else timed.cpu / s,
        resume_s=back.seconds,
        exact=back.tail == want.tail,
        lost=want.tail_docs - back.tail_docs,
        lost_tokens=want.tail_tokens - back.tail_tokens,
        packed=name in PACKED,
    )


def table(results: Sequence[Result]) -> str:
    rows = [
        "| loader | convert (s) | converted (MiB) | docs/s | Mtokens/s | cores "
        "| CPU ns/token | resume (s) | exact resume | tokens lost |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        cores = "—" if r.cores is None else f"{r.cores:.2f}"
        ns = "—" if r.ns_per_token is None else f"{r.ns_per_token:.1f}"
        docs = "—" if r.packed else f"{r.docs_per_s:,.0f}"
        rows.append(
            f"| {r.loader} | {r.convert_s:.1f} | {r.convert_bytes / 2**20:.1f} "
            f"| {docs} | {r.tokens_per_s / 1e6:.1f} | {cores} | {ns} "
            f"| {r.resume_s:.2f} | {'yes' if r.exact else 'no'} | {r.lost_tokens:,} |"
        )
    return "\n".join(rows)


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    p.add_argument("data", type=Path)
    p.add_argument("--loaders", type=loader_names, default=list(LOADERS))
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seq-len", type=int, default=2048, help="for packed loaders")
    p.add_argument("--resume-at", type=int, help="batches before; default half")
    p.add_argument(
        "--work", type=Path, help="for converted data; default a temp folder"
    )
    a = p.parse_args(argv)
    if "plitzer" in a.loaders and plitzer_debug():
        p.error("plitzer is a debug build; build it with maturin develop --release")
    run = Run(a.seed, a.batch, a.workers, a.seq_len)
    half = max(1, count_docs(a.data) // a.batch // 2)
    resume_at = half if a.resume_at is None else a.resume_at
    results = []
    with tempfile.TemporaryDirectory(dir=a.work) as tmp:
        for name in a.loaders:
            work = Path(tmp, name)
            work.mkdir()
            results.append(measure(name, a.data, work, run, resume_at))
    print(table(results))


def loader_names(value: str) -> list[str]:
    names = value.split(",")
    unknown = [n for n in names if n not in LOADERS]
    if unknown or len(set(names)) != len(names):
        raise argparse.ArgumentTypeError(
            f"names from {', '.join(LOADERS)}, each once; not {value!r}"
        )
    return names


def plitzer_debug() -> bool:
    """Whether the plitzer extension is a debug build, whose speed is not
    plitzer's: it runs several times slower than a release build."""
    return _plitzer.DEBUG


def count_docs(data: Path) -> int:
    return sum(pq.read_metadata(p).num_rows for p in parquet_files(data))


if __name__ == "__main__":
    main()
