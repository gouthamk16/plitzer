"""End-to-end harness: train on several processes, kill them at a random
step, resume on another number of processes, and compare the stream of
batches with a run that never stopped (see e2e_stream).

The trainer logs one line per lane and step, `step lane digest`, so the
stream does not depend on the number of processes that made it. It saves
a checkpoint at every step; once every rank has logged step k, every rank
has saved step k, so the newest complete checkpoint after the kill is at
step k or later. The ranks start with `spawn`, since a forked child cannot
use the parent's Rust threads. This module imports only numpy and plitzer,
so that each rank starts fast.
"""

import multiprocessing
import os
import random
import signal
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from multiprocessing.synchronize import Barrier
from pathlib import Path
from typing import NamedTuple

import numpy as np
import numpy.typing as npt

import plitzer
from e2e_stream import SETTINGS, Stream, digest, stream

WORLDS = [1, 2, 3, 4, 6]  # the divisors of SETTINGS["lanes"] that are worlds
PAUSE = 0.02  # seconds of "training" per step, so a kill lands mid-run
DEADLINE = 180.0  # seconds for one run of all its ranks
# The exit code of a killed rank: SIGKILL on POSIX; on Windows,
# multiprocessing reports its TerminateProcess as SIGTERM.
KILLED = -signal.SIGTERM if sys.platform == "win32" else -signal.SIGKILL
# The exit code of a gloo rank whose gather failed because a peer died.
PEER_DIED = 3
# The run to kill stops at step kill_at + MARGIN - 1 and waits there for
# the kill. So the resume step is below the end of the run also when the
# kill comes late, and a kill on time still lands mid-run.
MARGIN = 6

Arrays = tuple[npt.NDArray[np.generic], npt.NDArray[np.generic]]


@dataclass(frozen=True)
class Job:
    """One run of `world` ranks, from step `start` to step `steps`."""

    location: str
    out: str  # this run's logs and checkpoints
    cache: str
    world: int
    start: int
    steps: int
    resume: str | None  # the global checkpoint to resume from
    hold: bool  # wait for the kill at `steps`, not end


Rank = Callable[[int, Job, Barrier], None]


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
    while job.hold:
        time.sleep(1)
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
    for attempt in range(9):  # about 2.5 s in all
        try:
            os.replace(temp, path)
            return
        except PermissionError:
            if attempt == 8:
                raise
            time.sleep(0.01 * 2**attempt)


def loader_rank(rank: int, job: Job, ready: Barrier) -> None:
    """One rank with `plitzer.Loader`: it saves its own part, and the parts
    are merged at the resume."""
    loader = plitzer.Loader(
        job.location, **SETTINGS, cache=job.cache, rank=rank, world=job.world
    )
    if job.resume:
        loader.resume(Path(job.resume).read_bytes())
    ready.wait()

    def epoch() -> Iterable[Arrays]:
        for b in loader:
            yield np.from_dlpack(b.tokens), np.from_dlpack(b.lengths)

    def save(step: int) -> None:
        atomic_write(Path(job.out, f"part-{rank}-{step}"), loader.checkpoint())

    train(job, rank, loader.lanes, epoch, save)


def saved(out: Path, prefix: str) -> set[int]:
    """The steps of the files `<prefix><step>` in `out`; a kill can leave a
    `.tmp` file, which does not count."""
    names = (p.name for p in out.glob(f"{prefix}*") if not p.name.endswith(".tmp"))
    return {int(n[len(prefix) :]) for n in names}


def loader_checkpoint(job: Job, step: int | None) -> tuple[int, str]:
    """The parts saved at `step`, else at the newest step that every rank
    saved, as a real trainer finds it after a crash, merged into one file."""
    out = Path(job.out)
    if step is None:
        ranks = range(job.world)
        step = max(set.intersection(*(saved(out, f"part-{r}-") for r in ranks)))
    parts = [out / f"part-{r}-{step}" for r in range(job.world)]
    merged = out / f"merged-{step}"
    merged.write_bytes(plitzer.merge([p.read_bytes() for p in parts]))
    return step, str(merged)


class Ran(NamedTuple):
    killed: bool  # the kill came, at `kill_at`
    codes: list[int | None]  # the exit code of each rank


def run(target: Rank, job: Job, kill_at: int | None) -> Ran:
    """Runs every rank of `job`. With `kill_at`, kills them all once each
    has logged that step."""
    Path(job.out).mkdir()
    spawn = multiprocessing.get_context("spawn")
    # Every rank runs before the kill can come.
    ready = spawn.Barrier(job.world)
    ranks = [
        spawn.Process(target=target, args=(r, job, ready), daemon=True)
        for r in range(job.world)
    ]
    killed = False
    try:
        for p in ranks:
            p.start()
        deadline = time.monotonic() + DEADLINE
        while any(p.is_alive() for p in ranks) and time.monotonic() < deadline:
            if kill_at is not None and reached(job, kill_at):
                killed = True
                break
            # A rank that fails leaves the others waiting for it; one that
            # ends by itself before a kill never reaches it.
            ended = [p.exitcode for p in ranks if p.exitcode is not None]
            if any(ended) or (kill_at is not None and ended):
                break
            time.sleep(0.005)
    finally:
        # Kill them all before a join, and also when this loop fails: an
        # old run never ends by itself.
        started = [p for p in ranks if p.pid is not None]
        for p in started:
            if p.is_alive():
                p.kill()
        for p in started:
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


@dataclass(frozen=True)
class Scenario:
    seed: int
    kill_at: int  # every rank has logged this step before the kill
    old: int  # world before the kill
    new: int  # world after it
    boundary: bool  # kill_at ends an epoch: resume from exactly there

    def __str__(self) -> str:
        return (
            f"seed {self.seed}: kill after step {self.kill_at}, "
            f"world {self.old} then {self.new}"
        )


def scenario(seed: int, ends: list[int], steps: int) -> Scenario:
    """A kill step and two different worlds, from `seed`. One time in
    three the kill step is an epoch boundary, where the checkpoint stands
    at the start of the next epoch."""
    rng = random.Random(seed)
    old, new = rng.sample(WORLDS, 2)
    # The resume must leave steps to compare: see MARGIN.
    last = steps - MARGIN
    boundaries = [e for e in ends if e <= last]
    if boundaries and rng.random() < 1 / 3:
        return Scenario(seed, rng.choice(boundaries), old, new, True)
    return Scenario(seed, rng.randint(1, last), old, new, False)


def kill_and_resume(
    target: Rank,
    checkpoint: Callable[[Job, int | None], tuple[int, str]],
    location: Path,
    cache: Path,
    base: Path,
    case: Scenario,
    steps: int,
    gloo: bool = False,
) -> Stream:
    """Runs world `case.old` of `target` until it is killed once every rank
    has logged `case.kill_at`, then world `case.new` from the newest
    checkpoint that `checkpoint` finds after the kill, to `steps`. Gives
    the old run's stream before that step and the new run's after it."""
    print(case)  # pytest shows it with any failure, also one without `case`
    base = base / f"seed-{case.seed}"
    base.mkdir()
    at = (str(location), str(cache))
    hold = case.kill_at + MARGIN - 1
    old = Job(at[0], str(base / "old"), at[1], case.old, 0, hold, None, True)
    ran = run(target, old, case.kill_at)
    if gloo:
        # A gloo rank that sees a dead peer can end just before its own
        # kill, with PEER_DIED; any other end is a failure of its own.
        ends = {KILLED, PEER_DIED}
        killed = ran.killed and KILLED in ran.codes and set(ran.codes) <= ends
    else:
        killed = ran.killed and ran.codes == [KILLED] * case.old
    assert killed, f"{case}: not killed: {ran}"
    start, resume = checkpoint(old, case.kill_at if case.boundary else None)
    assert case.kill_at <= start < steps, f"{case}: resume at {start}"
    print(f"resume at step {start}")
    new = Job(at[0], str(base / "new"), at[1], case.new, start, steps, resume, False)
    codes = run(target, new, None).codes
    assert codes == [0] * case.new, f"{case}, resumed at {start}: {codes}"
    before = stream(Path(old.out), case.old)
    after = stream(Path(new.out), case.new)
    return {**{k: v for k, v in before.items() if k[0] < start}, **after}
