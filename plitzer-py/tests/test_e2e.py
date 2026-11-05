import os
import random
from pathlib import Path

import pytest

import e2e


@pytest.fixture
def data(tmp_path: Path) -> Path:
    e2e.write_dataset(tmp_path / "data")
    return tmp_path / "data"


def seeds(count: int) -> list[int]:
    """`PLITZER_E2E_SEED` to repeat a failure, else new seeds each run."""
    given = os.environ.get("PLITZER_E2E_SEED")
    if given:
        return [int(given)]
    return [random.randrange(2**32) for _ in range(count)]


def test_kill_and_resume_on_another_world(data: Path, tmp_path: Path) -> None:
    """Training is killed at a random step on one number of processes and
    resumed on another, from the parts that each rank saved; the stream
    equals a run that never stopped."""
    steps = e2e.epoch_steps(data, tmp_path / "cache") * 5 // 2
    want = e2e.reference(data, tmp_path, steps)
    for seed in seeds(3):
        case = e2e.scenario(seed, steps)
        got = e2e.kill_and_resume(
            e2e.loader_rank, e2e.last_loader_checkpoint, data, tmp_path, case, steps
        )
        e2e.check(want, got, steps, str(case))
