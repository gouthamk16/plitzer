"""End-to-end harness: train on several processes, kill them at a random
step, resume on another number of processes, and compare the stream of
batches with a run that never stopped.

The trainer logs one line per lane and step, `step lane digest`, so the
stream does not depend on the number of processes that made it. The
ranks start with `spawn`: a forked child cannot use the parent's Rust
threads.
"""

import hashlib
import multiprocessing
import os
import random
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq

import plitzer
from conftest import Settings

# 12 lanes: the world can be 1, 2, 3, 4 or 6.
WORLDS = [1, 2, 3, 4, 6]
SETTINGS: Settings = {
    "seed": 5,
    "lanes": 12,
    "window": 4,
    "seq_len": 16,
    "docs": 2,
    "cap": 24,
    "pad": 2**32 - 1,
    "budget": 1 << 30,
}
EVERY = 3  # steps between checkpoints
PAUSE = 0.02  # seconds of "training" per step, so a kill lands mid-run
DEADLINE = 180.0  # seconds for one run of all its ranks

Stream = dict[tuple[int, int], str]
Arrays = tuple[npt.NDArray[np.generic], npt.NDArray[np.generic]]


@dataclass(frozen=True)
class Job:
    """One run of `world` ranks, from step `start` to step `steps`."""

    location: str
    out: str  # this run's logs, checkpoints and barrier files
    cache: str
    world: int
    start: int
    steps: int
    resume: str | None  # the checkpoint to resume from


def write_dataset(folder: Path) -> None:
    """Eight files of row groups of 5 rows, every token unique."""
    folder.mkdir()
    for f in range(8):
        rows = [
            [f * 100_000 + r * 100 + t for t in range(1 + 5 * ((3 * f + r) % 7))]
            for r in range(11 + 2 * f)
        ]
        table = pa.table({"tokens": pa.array(rows, pa.list_(pa.int32()))})
        pq.write_table(table, folder / f"{f}.parquet", row_group_size=5)


def epoch_steps(location: Path, cache: Path) -> int:
    """The steps in one epoch on one GPU."""
    return sum(1 for _ in plitzer.Loader(location, **SETTINGS, cache=str(cache)))


def train(
    job: Job,
    rank: int,
    lanes: range,
    epoch: Callable[[], Iterable[Arrays]],
    save: Callable[[int], None],
) -> None:
    """Steps from `job.start` to `job.steps`, an epoch at a time. Logs each
    lane's batch and calls `save` every `EVERY` steps."""
    log = os.open(Path(job.out, f"log-{rank}"), os.O_WRONLY | os.O_CREAT | os.O_APPEND)
    step = job.start
    while step < job.steps:
        taken = 0
        for tokens, lengths in epoch():
            lines = "".join(
                f"{step} {lane} {digest(tokens[i], int(lengths[i]))}\n"
                for i, lane in enumerate(lanes)
            )
            os.write(log, lines.encode())  # one write: a kill leaves no half step
            time.sleep(PAUSE)
            step, taken = step + 1, taken + 1
            if step % EVERY == 0:
                save(step)
            if step == job.steps:
                break
        assert taken > 0, "an epoch gave no batch"
    os.close(log)


def digest(tokens: npt.NDArray[np.generic], length: int) -> str:
    data = tokens.tobytes() + length.to_bytes(4, "little")
    return hashlib.sha256(data).hexdigest()[:16]


def atomic_write(path: Path, data: bytes) -> None:
    """Writes `data` so that a kill leaves either the old file or the new."""
    temp = path.with_name(path.name + ".tmp")
    temp.write_bytes(data)
    os.replace(temp, path)


def loader_rank(rank: int, job: Job) -> None:
    """One rank with `plitzer.Loader`: it saves its own part, and the parts
    are merged at the resume."""
    loader = plitzer.Loader(
        job.location, **SETTINGS, cache=job.cache, rank=rank, world=job.world
    )
    if job.resume:
        loader.resume(Path(job.resume).read_bytes())
    barrier(job, rank)

    def epoch() -> Iterable[Arrays]:
        for b in loader:
            yield np.from_dlpack(b.tokens), np.from_dlpack(b.lengths)

    def save(step: int) -> None:
        atomic_write(Path(job.out, f"part-{step}-{rank}"), loader.checkpoint())

    train(job, rank, loader.lanes, epoch, save)


def barrier(job: Job, rank: int) -> None:
    """Waits for every rank to be ready, so that no rank finishes before the
    slowest one starts. torch's ranks need none: their gather keeps them in
    step."""
    Path(job.out, f"ready-{rank}").touch()
    while not all(Path(job.out, f"ready-{r}").exists() for r in range(job.world)):
        time.sleep(0.005)


def run(
    target: Callable[[int, Job], None], job: Job, kill_at: int | None
) -> list[int | None]:
    """Runs every rank of `job`. With `kill_at`, kills them all once each
    has logged that step. Gives the exit code of each rank."""
    Path(job.out).mkdir()
    spawn = multiprocessing.get_context("spawn")
    ranks = [
        spawn.Process(target=target, args=(r, job), daemon=True)
        for r in range(job.world)
    ]
    for p in ranks:
        p.start()
    deadline = time.monotonic() + DEADLINE
    while any(p.is_alive() for p in ranks) and time.monotonic() < deadline:
        if kill_at is not None and reached(job, kill_at):
            break
        time.sleep(0.005)
    for p in ranks:
        if p.is_alive():
            p.kill()
        p.join()
    return [p.exitcode for p in ranks]


def reached(job: Job, step: int) -> bool:
    """Whether every rank of `job` has logged `step`."""
    for r in range(job.world):
        steps = {s for s, _ in read(Path(job.out, f"log-{r}"))}
        if step not in steps:
            return False
    return True


def read(path: Path) -> Stream:
    """The complete lines of a log; a kill can leave the last one cut."""
    if not path.exists():
        return {}
    text = path.read_text()
    stream: Stream = {}
    for line in text[: text.rfind("\n") + 1].splitlines():
        step, lane, digest = line.split()
        stream[int(step), int(lane)] = digest
    return stream


def stream(out: Path, world: int) -> Stream:
    joined: Stream = {}
    for r in range(world):
        joined.update(read(out / f"log-{r}"))
    return joined


def last_loader_checkpoint(job: Job) -> tuple[int, str | None]:
    """The newest step that every rank saved a part of, and the file of
    those parts merged into one global checkpoint (none: step 0)."""
    out = Path(job.out)
    for step in range(job.steps - job.steps % EVERY, 0, -EVERY):
        parts = [out / f"part-{step}-{r}" for r in range(job.world)]
        if all(p.exists() for p in parts):
            merged = out / f"merged-{step}"
            merged.write_bytes(plitzer.merge([p.read_bytes() for p in parts]))
            return step, str(merged)
    return 0, None


@dataclass(frozen=True)
class Scenario:
    seed: int
    kill_at: int
    old: int  # world before the kill
    new: int  # world after it

    def __str__(self) -> str:
        return (
            f"seed {self.seed}: kill at step {self.kill_at}, "
            f"world {self.old} then {self.new}"
        )


def scenario(seed: int, steps: int) -> Scenario:
    """A random kill step and two different worlds, from `seed`. The kill
    leaves a few steps, so the ranks are still running when it comes."""
    rng = random.Random(seed)
    old, new = rng.sample(WORLDS, 2)
    return Scenario(seed, rng.randint(1, steps - 4), old, new)


def check(want: Stream, got: Stream, steps: int, case: str) -> None:
    """The streams hold the same batch for each (step, lane) of the run."""
    keys = [(s, lane) for s in range(steps) for lane in range(SETTINGS["lanes"])]
    for key in keys:
        assert got.get(key) == want.get(key), f"{case}: first difference at {key}"
    assert set(got) == set(keys), f"{case}: steps or lanes outside the run"


def reference(data: Path, base: Path, steps: int) -> Stream:
    """One GPU, no stop."""
    out = base / "reference"
    job = Job(str(data), str(out), str(base / "cache"), 1, 0, steps, None)
    assert run(loader_rank, job, None) == [0]
    return stream(out, 1)


def kill_and_resume(
    target: Callable[[int, Job], None],
    last: Callable[[Job], tuple[int, str | None]],
    data: Path,
    base: Path,
    case: Scenario,
    steps: int,
) -> Stream:
    """Runs world `case.old` of `target`, kills it at `case.kill_at`, and
    runs world `case.new` from the newest checkpoint that `last` finds.
    Gives the old run's stream before that step and the new run's after."""
    cache, base = str(base / "cache"), base / f"seed-{case.seed}"
    base.mkdir()
    old = Job(str(data), str(base / "old"), cache, case.old, 0, steps, None)
    codes = run(target, old, case.kill_at)
    assert all(c != 0 for c in codes), f"{case}: a rank ended before the kill"
    start, resume = last(old)
    new = Job(str(data), str(base / "new"), cache, case.new, start, steps, resume)
    assert run(target, new, None) == [0] * case.new, f"{case}: the resumed run"
    before = stream(Path(old.out), case.old)
    return {
        **{k: v for k, v in before.items() if k[0] < start},
        **stream(Path(new.out), case.new),
    }
