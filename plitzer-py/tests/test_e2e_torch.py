from pathlib import Path

import pytest
import torch.distributed as dist

import e2e
import e2e_torch
from conftest import E2E, e2e_seeds


@pytest.mark.skipif(
    not (dist.is_available() and dist.is_gloo_available()), reason="needs gloo"
)
def test_kill_and_resume_with_torch(e2e_data: E2E, tmp_path: Path) -> None:
    """The same kill and resume with plitzer.torch over gloo, from the one
    global state that state_dict() gathers and rank 0 saves with torch."""
    d = e2e_data
    case = e2e.scenario(e2e_seeds(1)[0], d.ref.ends, d.ref.steps)
    got = e2e.kill_and_resume(
        e2e_torch.torch_rank,
        e2e_torch.torch_checkpoint,
        d.data,
        d.cache,
        tmp_path,
        case,
        d.ref.steps,
    )
    e2e.check(d.ref.want, got, d.ref.steps, str(case))
