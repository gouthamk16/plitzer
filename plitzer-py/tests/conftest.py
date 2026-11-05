import importlib.util
import os
import random
from collections.abc import Mapping
from pathlib import Path
from typing import NamedTuple, Protocol, TypedDict, Unpack

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import e2e
import plitzer

# torch is an optional extra. Find it without importing it, which is slow.
collect_ignore = (
    [] if importlib.util.find_spec("torch") else ["test_torch.py", "test_e2e_torch.py"]
)


class Extras(TypedDict, total=False):
    """The keywords of `plitzer.Loader` that have defaults."""

    rank: int
    world: int
    column: str
    cache: str | os.PathLike[str] | None
    storage_options: Mapping[str, str | bool] | None


class Settings(TypedDict):
    """The keywords that `plitzer.Loader` needs."""

    seed: int
    lanes: int
    window: int
    seq_len: int
    docs: int
    cap: int
    pad: int
    budget: int


class Args(Settings, Extras):
    """The keywords of a whole `plitzer.Loader` call."""


class Overrides(Extras, total=False):
    """Any keyword of `plitzer.Loader`, to replace one of SETTINGS."""

    seed: int
    lanes: int
    window: int
    seq_len: int
    docs: int
    cap: int
    pad: int
    budget: int


SETTINGS: Settings = {
    "seed": 11,
    "lanes": 4,
    "window": 3,
    "seq_len": 8,
    "docs": 2,
    "cap": 10,
    "pad": 2**32 - 1,
    "budget": 1 << 30,
}


class MakeLoader(Protocol):
    def __call__(self, **overrides: Unpack[Overrides]) -> plitzer.Loader: ...


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    """Four files of row groups of 5 rows, every token unique."""
    data = tmp_path / "data"
    data.mkdir()
    for f in range(4):
        rows = [
            [f * 10_000 + r * 100 + t for t in range(3 * ((f + r) % 7))]
            for r in range(6 + 3 * f)
        ]
        table = pa.table({"tokens": pa.array(rows, pa.list_(pa.int32()))})
        pq.write_table(table, data / f"{f}.parquet", row_group_size=5)
    return data


@pytest.fixture
def make(dataset: Path, tmp_path: Path) -> MakeLoader:
    """A Loader over `dataset` with SETTINGS, which keywords override."""

    def make(**overrides: Unpack[Overrides]) -> plitzer.Loader:
        args: Args = {**SETTINGS, "cache": str(tmp_path / "index"), **overrides}
        return plitzer.Loader(str(dataset), **args)

    return make


class E2E(NamedTuple):
    """The end-to-end tests' dataset, its warm index cache, and the run
    that never stops, over 2.5 epochs."""

    data: Path
    cache: Path
    ref: e2e.Reference


@pytest.fixture(scope="session")
def e2e_data(tmp_path_factory: pytest.TempPathFactory) -> E2E:
    """Eight files of row groups of 5 rows, every token unique, and the
    stream of one GPU over 2.5 epochs."""
    base = tmp_path_factory.mktemp("e2e")
    data = base / "data"
    data.mkdir()
    for f in range(8):
        rows = [
            [f * 100_000 + r * 100 + t for t in range(1 + 5 * ((3 * f + r) % 7))]
            for r in range(11 + 2 * f)
        ]
        table = pa.table({"tokens": pa.array(rows, pa.list_(pa.int32()))})
        pq.write_table(table, data / f"{f}.parquet", row_group_size=5)
    return E2E(data, base / "cache", e2e.reference(data, base / "cache", 2.5))


def e2e_seeds(count: int) -> list[int]:
    """`PLITZER_E2E_SEED` to repeat a failure, else new seeds each run."""
    given = os.environ.get("PLITZER_E2E_SEED")
    if given:
        return [int(given)]
    return [random.randrange(2**32) for _ in range(count)]
