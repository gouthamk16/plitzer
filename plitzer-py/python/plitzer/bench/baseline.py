"""The baseline: how existing loaders read the Parquet files that plitzer
reads. For each loader it measures the conversion, one shuffled epoch
(documents and tokens per second, CPU cores used) and a resume at
mid-epoch: the wait for the first batch, whether the rest of the epoch is
the one an uninterrupted run gives, and how many documents it loses.

    python -m plitzer.bench.baseline DATA --loaders plitzer,mosaic,hf,hf-stream \\
        --batch 64 --workers 4

``--batch`` is in documents. A packed loader (plitzer) gets the same tokens
in each batch, in sequences of ``--seq-len``, and each loader resumes at
half of its own epoch. Each phase runs in a fresh process
(plitzer.bench.phases). The adapters
(plitzer.bench.mosaic and plitzer.bench.hf) need the ``baseline`` extra;
plitzer's (plitzer.bench.plitzer_adapter) needs only plitzer.
"""

import argparse
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

import pyarrow.parquet as pq

from plitzer import _plitzer
from plitzer.bench.phases import LOADERS, PACKED, Job, Run, parquet_files, phase


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

    @property
    def ns_per_token(self) -> float | None:
        """CPU nanoseconds for each token: the cores per token per second."""
        return None if self.cores is None else self.cores / self.tokens_per_s * 1e9


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


class Shape(NamedTuple):
    docs: int
    tokens: int  # leaf values of the token column: tokens, and empty rows


def shape(data: Path, column: str = "tokens") -> Shape:
    """The documents and tokens of ``data``, from the footers."""
    docs = tokens = 0
    for path in parquet_files(data):
        meta = pq.read_metadata(path)
        leaves = [
            i
            for i in range(meta.num_columns)
            if meta.schema.column(i).path.split(".")[0] == column
        ]
        docs += meta.num_rows
        for g in range(meta.num_row_groups):
            tokens += sum(meta.row_group(g).column(i).num_values for i in leaves)
    return Shape(docs, tokens)


def for_loader(name: str, run: Run, data: Shape) -> tuple[Run, int]:
    """The run of loader ``name``, and the batches of its epoch. A packed
    loader gets the tokens of ``run.batch`` documents in each batch, in
    sequences of ``run.seq_len``, so that each loader moves the same tokens
    for each batch."""
    if name not in PACKED:
        return run, -(-data.docs // run.batch)
    per_doc = data.tokens / max(1, data.docs)
    lanes = max(1, round(run.batch * per_doc / run.seq_len))
    return run._replace(batch=lanes), -(-data.tokens // (lanes * run.seq_len))


def measure(
    name: str, data: Path, work: Path, run: Run, resume_at: int | None = None
) -> Result:
    """The phases of loader ``name`` over ``data``, using ``work``. The
    resume is after ``resume_at`` batches; by default, half of its epoch."""
    if name not in LOADERS:
        raise ValueError(f"no loader {name!r}; the loaders are {', '.join(LOADERS)}")
    run, batches = for_loader(name, run, shape(data))
    resume_at = max(1, batches // 2) if resume_at is None else resume_at
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
        docs = "—" if r.loader in PACKED else f"{r.docs_per_s:,.0f}"
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
    results = []
    with tempfile.TemporaryDirectory(dir=a.work) as tmp:
        for name in a.loaders:
            work = Path(tmp, name)
            work.mkdir()
            results.append(measure(name, a.data, work, run, a.resume_at))
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


if __name__ == "__main__":
    main()
