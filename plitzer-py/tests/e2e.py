"""End-to-end harness: train on several processes, kill them at a random
step, resume on another number of processes, and compare the stream of
batches with a run that never stopped.

The trainer logs one line per lane and step, `step lane digest`, so the
stream does not depend on the number of processes that made it. It saves
a checkpoint at every step, so the resume step is the kill step: once
every rank has logged step k, every rank has saved step k. The ranks
start with `spawn`, since a forked child cannot use the parent's Rust
threads. This module imports only numpy and plitzer, so that each rank
starts fast.
"""

import hashlib
import multiprocessing
import os
import random
import signal
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import numpy.typing as npt

import plitzer

if TYPE_CHECKING:
    from conftest import Settings

# 12 lanes: the world can be 1, 2, 3, 4 or 6.
WORLDS = [1, 2, 3, 4, 6]
SETTINGS: "Settings" = {
    "seed": 5,
    "lanes": 12,
    "window": 4,
    "seq_len": 16,
    "docs": 2,
    "cap": 24,
    "pad": 2**32 - 1,
    "budget": 1 << 30,
}
PAUSE = 0.02  # seconds of "training" per step, so a kill lands mid-run
DEADLINE = 180.0  # seconds for one run of all its ranks
FOREVER = 10**9  # the steps of a run that only a kill ends
# The exit code of a killed rank: SIGKILL on POSIX; on Windows,
# multiprocessing reports its TerminateProcess as SIGTERM.
KILLED = -signal.SIGTERM if sys.platform == "win32" else -signal.SIGKILL

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
    resume: str | None  # the global checkpoint to resume from


def digest(tokens: npt.NDArray[np.generic], length: int) -> str:
    data = tokens.tobytes() + length.to_bytes(4, "little")
    return hashlib.sha256(data).hexdigest()[:16]


def train(
    job: Job,
    rank: int,
    lanes: range,
    epoch: Callable[[], Iterable[Arrays]],
    save: Callable[[int], None],
) -> None:
    """Steps from `job.start` to `job.steps`, an epoch at a time. Logs each
    lane's batch and calls `save` with the step count after each step."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_BINARY", 0)
    log = os.open(Path(job.out, f"log-{rank}"), flags)
    step = job.start
    while step < job.steps:
        taken = 0
        for tokens, lengths in epoch():
            line = "".join(
                f"{step} {lane} {digest(tokens[i], int(lengths[i]))}\n"
                for i, lane in enumerate(lanes)
            ).encode()
            # One write: a kill leaves no half-logged step behind.
            assert os.write(log, line) == len(line), "a short write to the log"
            time.sleep(PAUSE)
            step, taken = step + 1, taken + 1
            save(step)
            if step == job.steps:
                break
        else:
            # After StopIteration too: the state must be the one saved
            # after the last batch, the start of the next epoch.
            save(step)
        assert taken > 0, "an epoch gave no batch"
    os.close(log)


def atomic_write(path: Path, data: bytes) -> None:
    """Writes `data` so that a kill leaves either no file or the whole. A
    second write of one step must give the same bytes: see `train`."""
    if path.exists():
        assert path.read_bytes() == data, f"{path.name} changed after the epoch end"
        return
    temp = path.with_name(path.name + ".tmp")
    temp.write_bytes(data)
    # Windows refuses a rename while a scanner holds the new file open.
    for attempt in range(20):
        try:
            os.replace(temp, path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.01)


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
    """Waits for every rank to be ready, so that the kill finds them all
    running. torch's ranks need none: their gather keeps them in step."""
    Path(job.out, f"ready-{rank}").touch()
    while not all(Path(job.out, f"ready-{r}").exists() for r in range(job.world)):
        time.sleep(0.005)


def loader_checkpoint(job: Job, step: int) -> str:
    """The parts that every rank saved at `step`, merged into one file."""
    parts = [Path(job.out, f"part-{step}-{r}") for r in range(job.world)]
    merged = Path(job.out, f"merged-{step}")
    merged.write_bytes(plitzer.merge([p.read_bytes() for p in parts]))
    return str(merged)


class Ran(NamedTuple):
    killed: bool  # the kill came, at `kill_at`
    codes: list[int | None]  # the exit code of each rank


def run(target: Callable[[int, Job], None], job: Job, kill_at: int | None) -> Ran:
    """Runs every rank of `job`. With `kill_at`, kills them all once each
    has logged that step."""
    Path(job.out).mkdir()
    spawn = multiprocessing.get_context("spawn")
    ranks = [
        spawn.Process(target=target, args=(r, job), daemon=True)
        for r in range(job.world)
    ]
    for p in ranks:
        p.start()
    deadline, killed = time.monotonic() + DEADLINE, False
    while any(p.is_alive() for p in ranks) and time.monotonic() < deadline:
        if kill_at is not None and reached(job, kill_at):
            killed = True
            break
        time.sleep(0.005)
    # Kill them all before a join: a gloo rank that sees a dead peer could
    # end by itself first.
    for p in ranks:
        if p.is_alive():
            p.kill()
    for p in ranks:
        p.join()
    return Ran(killed, [p.exitcode for p in ranks])


def reached(job: Job, step: int) -> bool:
    """Whether every rank of `job` has logged `step`. A rank logs its steps
    in order, so its last complete line tells; only the end is read."""
    for r in range(job.world):
        path = Path(job.out, f"log-{r}")
        if not path.exists():
            return False
        with path.open("rb") as log:
            log.seek(max(0, path.stat().st_size - 4096))
            tail = log.read()
        lines = tail[: tail.rfind(b"\n")].splitlines()
        if not lines or int(lines[-1].split()[0]) < step:
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


class Reference(NamedTuple):
    """One GPU, no stop."""

    ends: list[int]  # the step count at the end of each epoch
    steps: int  # the steps of the run
    want: Stream


def reference(location: Path, cache: Path, epochs: float) -> Reference:
    """One GPU, no stop, in this process, for `epochs` epochs."""
    loader = plitzer.Loader(location, **SETTINGS, cache=str(cache))
    want: list[dict[int, str]] = []
    ends = []
    for _ in range(int(epochs) + 1):
        for b in loader:
            tokens, lengths = np.from_dlpack(b.tokens), np.from_dlpack(b.lengths)
            lanes = range(SETTINGS["lanes"])
            want.append({i: digest(tokens[i], int(lengths[i])) for i in lanes})
        ends.append(len(want))
    whole = int(epochs)
    last = ends[whole] - (ends[whole - 1] if whole else 0)
    steps = (ends[whole - 1] if whole else 0) + int(last * (epochs - whole))
    stream = {(s, i): d for s in range(steps) for i, d in want[s].items()}
    return Reference(ends, steps, stream)


@dataclass(frozen=True)
class Scenario:
    seed: int
    kill_at: int  # also the resume step
    old: int  # world before the kill
    new: int  # world after it

    def __str__(self) -> str:
        return (
            f"seed {self.seed}: kill and resume at step {self.kill_at}, "
            f"world {self.old} then {self.new}"
        )


def scenario(seed: int, ends: list[int], steps: int) -> Scenario:
    """A kill step and two different worlds, from `seed`. One time in
    three the kill step is an epoch boundary, where the checkpoint stands
    at the start of the next epoch."""
    rng = random.Random(seed)
    old, new = rng.sample(WORLDS, 2)
    boundaries = [e for e in ends if e < steps]
    if boundaries and rng.random() < 1 / 3:
        return Scenario(seed, rng.choice(boundaries), old, new)
    return Scenario(seed, rng.randint(1, steps - 1), old, new)


def kill_and_resume(
    target: Callable[[int, Job], None],
    checkpoint: Callable[[Job, int], str],
    location: Path,
    cache: Path,
    base: Path,
    case: Scenario,
    steps: int,
) -> Stream:
    """Runs world `case.old` of `target` until it is killed at
    `case.kill_at`, then world `case.new` from the `checkpoint` of that
    step to `steps`. Gives the old run's stream before that step and the
    new run's after it."""
    base = base / f"seed-{case.seed}"
    base.mkdir()
    at = (str(location), str(cache))
    old = Job(at[0], str(base / "old"), at[1], case.old, 0, FOREVER, None)
    ran = run(target, old, case.kill_at)
    assert ran.killed and ran.codes == [KILLED] * case.old, f"{case}: {ran}"
    start, resume = case.kill_at, checkpoint(old, case.kill_at)
    new = Job(at[0], str(base / "new"), at[1], case.new, start, steps, resume)
    codes = run(target, new, None).codes
    assert codes == [0] * case.new, f"{case}: the resumed run: {codes}"
    before = stream(Path(old.out), case.old)
    after = stream(Path(new.out), case.new)
    return {**{k: v for k, v in before.items() if k[0] < start}, **after}


def check(want: Stream, got: Stream, steps: int, case: str) -> None:
    """The streams hold the same batch for each (step, lane) of the run."""
    keys = [(s, lane) for s in range(steps) for lane in range(SETTINGS["lanes"])]
    for key in keys:
        assert got.get(key) == want.get(key), f"{case}: first difference at {key}"
    assert set(got) == set(keys), f"{case}: steps or lanes outside the run"
