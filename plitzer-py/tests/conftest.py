from collections.abc import Callable
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import plitzer

SETTINGS = {
    "seed": 11,
    "lanes": 4,
    "window": 3,
    "seq_len": 8,
    "docs": 2,
    "cap": 10,
    "pad": 2**32 - 1,
    "budget": 1 << 30,
}

MakeLoader = Callable[..., plitzer.Loader]


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

    def make(**overrides: object) -> plitzer.Loader:
        settings: dict[str, object] = {**SETTINGS, "cache": str(tmp_path / "index")}
        settings.update(overrides)
        return plitzer.Loader(str(dataset), **settings)  # type: ignore[arg-type]

    return make
