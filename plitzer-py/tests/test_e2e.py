from pathlib import Path

import e2e
from conftest import E2E, e2e_seeds


def test_kill_and_resume_on_another_world(e2e_data: E2E, tmp_path: Path) -> None:
    """Training is killed at a random step on one number of processes and
    resumed on another, from the parts that each rank saved; the stream
    equals a run that never stopped."""
    d = e2e_data
    for seed in e2e_seeds(3):
        case = e2e.scenario(seed, d.epoch, d.steps)
        got = e2e.kill_and_resume(
            e2e.loader_rank, e2e.loader_checkpoint, d.data, tmp_path, case, d.steps
        )
        e2e.check(d.want, got, d.steps, str(case))
