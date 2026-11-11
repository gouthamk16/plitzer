"""Read amplification: the bytes that all GPUs of a job fetch in one epoch,
divided by the token-chunk bytes of the dataset. One GPU reads each chunk
once, so its amplification is 1. The GPUs' lanes split each window of W
row groups into stretches in storage order, and two neighbours share at
most the one row group at their edge. So the GPUs read at most
world - 1 more row groups than each window holds: over an epoch of n row
groups in ceil(n/W) windows, at most 1 + ceil(n/W)(world - 1)/n, which is
about 1 + (world - 1)/W.

    python -m plitzer.bench.amplification DATA --lanes 512 --window 64 \\
        --worlds 1,8,64 --seq-len 2048

DATA is a local folder of Parquet files. Each rank runs in turn in this
process, for epochs 0 and 1, and counts the bytes of epoch 1.
"""

import argparse
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

import pyarrow.parquet as pq

import plitzer


class Settings(NamedTuple):
    """The settings of each rank's ``plitzer.Loader``."""

    lanes: int
    window: int
    seq_len: int
    budget: int
    seed: int = 0
    column: str = "tokens"


class Dataset(NamedTuple):
    row_groups: int
    bytes: int  # of the token chunks that one GPU fetches in an epoch


class World(NamedTuple):
    """One epoch of a job of ``len(per_gpu)`` GPUs."""

    window: int
    per_gpu: list[int]  # the bytes that each rank fetched
    dataset: Dataset

    @property
    def amplification(self) -> float:
        return sum(self.per_gpu) / self.dataset.bytes

    @property
    def bound(self) -> float:
        """In row groups, not bytes: an edge row group can be larger than
        the mean."""
        n = self.dataset.row_groups
        windows = -(-n // self.window)
        return 1 + windows * (len(self.per_gpu) - 1) / n


def scan(location: Path, column: str) -> Dataset:
    """The row groups under ``location``, and the bytes of the ``column``
    chunks of those with rows, which the loader fetches."""
    total = row_groups = 0
    for path in location.rglob("*.parquet"):
        meta = pq.read_metadata(path)
        top = [
            meta.schema.column(i).path.split(".")[0] for i in range(meta.num_columns)
        ]
        leaf = top.index(column)
        row_groups += meta.num_row_groups
        for g in range(meta.num_row_groups):
            group = meta.row_group(g)
            if group.num_rows > 0:
                total += group.column(leaf).total_compressed_size
    return Dataset(row_groups, total)


def rank_bytes(
    location: Path, settings: Settings, rank: int, world: int, cache: Path
) -> int:
    """The bytes that rank ``rank`` of ``world`` fetches in one epoch: its
    second, so that the windows that the prefetch reads into the next epoch
    count once at each end."""
    s = settings
    loader = plitzer.Loader(
        location,
        seed=s.seed,
        lanes=s.lanes,
        window=s.window,
        seq_len=s.seq_len,
        budget=s.budget,
        column=s.column,
        cache=cache,
        rank=rank,
        world=world,
    )
    for _ in loader:
        pass
    first = settled(loader)
    for _ in loader:
        pass
    return settled(loader) - first


QUIET = 0.2
"""Seconds with no new bytes after which the prefetch counts as stopped: far
longer than one read of a local row group, so a slow read is not missed."""


def settled(loader: plitzer.Loader) -> int:
    """`bytes_read` once the prefetch stops: it reads a fixed number of
    windows past the slowest lane, so the count it stops at depends only on
    where the lanes stand, not on how fast the reads were."""
    last = -1
    while loader.bytes_read != last:
        last = loader.bytes_read
        time.sleep(QUIET)
    return last


def measure(
    location: Path, settings: Settings, worlds: Sequence[int], cache: Path
) -> list[World]:
    """One epoch of every rank of each of ``worlds``, with the index cached
    at ``cache``."""
    dataset = scan(location, settings.column)
    return [
        World(
            settings.window,
            [rank_bytes(location, settings, r, w, cache) for r in range(w)],
            dataset,
        )
        for w in worlds
    ]


def table(results: Sequence[World]) -> str:
    """The results as a Markdown table."""
    mib = 1 << 20
    rows = [
        "| W | world | read (MiB) | per GPU, max (MiB) | amplification | bound |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        rows.append(
            f"| {r.window} | {len(r.per_gpu)} | {sum(r.per_gpu) / mib:.0f} "
            f"| {max(r.per_gpu) / mib:.1f} | {r.amplification:.3f} | {r.bound:.3f} |"
        )
    return "\n".join(rows)


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    p.add_argument("data", type=Path)
    p.add_argument("--lanes", type=int, required=True)
    p.add_argument("--window", type=int, nargs="+", required=True)
    p.add_argument(
        "--worlds", type=lambda v: [int(w) for w in v.split(",")], required=True
    )
    p.add_argument("--seq-len", type=int, required=True)
    p.add_argument("--budget", type=int, default=8 << 30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--column", default="tokens")
    a = p.parse_args(argv)
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        for window in a.window:
            s = Settings(a.lanes, window, a.seq_len, a.budget, a.seed, a.column)
            results.extend(measure(a.data, s, a.worlds, Path(tmp, "index")))
    print(table(results))


if __name__ == "__main__":
    main()
