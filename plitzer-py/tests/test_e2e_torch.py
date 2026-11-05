from pathlib import Path

import pytest
import torch.distributed as dist

import e2e
import e2e_torch
from test_e2e import data, seeds

__all__ = ["data"]  # the fixture


@pytest.mark.skipif(
    not (dist.is_available() and dist.is_gloo_available()), reason="needs gloo"
)
def test_kill_and_resume_with_torch(data: Path, tmp_path: Path) -> None:
    """The same kill and resume with plitzer.torch over gloo, from the one
    global state that state_dict() gathers and rank 0 saves with torch."""
    steps = e2e.epoch_steps(data, tmp_path / "cache") * 5 // 2
    want = e2e.reference(data, tmp_path, steps)
    case = e2e.scenario(seeds(1)[0], steps)
    got = e2e.kill_and_resume(
        e2e_torch.torch_rank,
        e2e_torch.last_torch_checkpoint,
        data,
        tmp_path,
        case,
        steps,
    )
    e2e.check(want, got, steps, str(case))
