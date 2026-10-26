import numpy as np
import numpy.typing as npt
import pytest

import plitzer
from conftest import MakeLoader

Tokens = npt.NDArray[np.int32]


def steps(loaders: list[plitzer.Loader]) -> list[Tokens]:
    """The rest of the epoch: each step's global batch, in rank order."""
    return [
        np.concatenate([np.from_dlpack(b.tokens) for b in batches])
        for batches in zip(*loaders, strict=True)
    ]


def ranks(make: MakeLoader, world: int, budget: int) -> list[plitzer.Loader]:
    return [make(rank=r, world=world, budget=budget) for r in range(world)]


def smallest_budget(make: MakeLoader, world: int) -> int:
    """The fewest KiB at which a fresh epoch runs, in bytes."""
    lo, hi = 1, 1 << 20
    while lo < hi:
        mid = (lo + hi) // 2
        try:
            steps(ranks(make, world, mid * 1024))
            hi = mid
        except plitzer.ConfigError:
            lo = mid + 1
    return lo * 1024


@pytest.mark.parametrize("world", [1, 2])
def test_resume_with_the_fresh_budget(make: MakeLoader, world: int) -> None:
    """A resume at any step runs with the smallest budget of a fresh run."""
    budget = smallest_budget(make, world)
    with pytest.raises(plitzer.ConfigError, match="budget"):
        steps(ranks(make, world, budget - 1024))
    want = steps(ranks(make, world, budget))
    for k in range(len(want)):
        head = ranks(make, world, budget)
        for _ in range(k):
            for rank in head:
                next(rank)
        saved = plitzer.merge([rank.checkpoint() for rank in head])
        tail = ranks(make, world, budget)
        for rank in tail:
            rank.resume(saved)
        got = want[:k] + steps(tail)
        assert all(np.array_equal(a, b) for a, b in zip(got, want, strict=True)), k
