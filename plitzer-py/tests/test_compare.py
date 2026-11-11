"""plitzer in the baseline harness. These need no other loader, so they run
in every Python job; test_baseline.py tests the others."""

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from plitzer.bench import baseline

RUN = baseline.Run(seed=3, batch=4, workers=0, seq_len=8)
TOKENS = 900


@pytest.fixture(scope="module")
def data(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Rows of 0 to 12 unique tokens, 900 in all, in row groups of 10 rows."""
    out = tmp_path_factory.mktemp("compare") / "data"
    out.mkdir()
    rows: list[list[int]] = []
    start = 0
    while start < TOKENS:
        n = min(len(rows) * 7 % 13, TOKENS - start)
        rows.append(list(range(start, start + n)))
        start += n
    table = pa.table({"tokens": pa.array(rows, pa.list_(pa.int32()))})
    pq.write_table(table, out / "0.parquet", row_group_size=10)
    return out


def test_an_epoch_gives_each_token(data: Path, tmp_path: Path) -> None:
    load = baseline.loader("plitzer")
    prepared = load.convert(data, tmp_path)
    got = np.concatenate(
        [np.concatenate(b) for b in load.open(data, prepared, RUN, None)]
    )
    # Every sequence is full, so the epoch ends a little into the next one:
    # by less than one step, and only with tokens it already gave.
    extra = len(got) - TOKENS
    assert 0 <= extra < RUN.batch * RUN.seq_len
    assert set(got.tolist()) == set(range(TOKENS))


def test_resume_is_exact(data: Path, tmp_path: Path) -> None:
    r = baseline.measure("plitzer", data, tmp_path, RUN, resume_at=10)
    assert r.packed and r.exact and r.lost == 0 and r.lost_tokens == 0
    # The conversion is the index sidecar: a few KiB, not a copy.
    assert 0 < r.convert_bytes < 64 << 10
    assert r.tokens_per_s > 0 and r.resume_s > 0
    row = baseline.table([r]).splitlines()[2]
    assert row.startswith("| plitzer |") and "| — |" in row


def test_main_runs_plitzer_and_refuses_a_debug_build(
    data: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = [str(data), "--loaders", "plitzer", "--batch", "4", "--seq-len", "8"]
    args += ["--resume-at", "10", "--work", str(tmp_path)]
    monkeypatch.setattr(baseline, "plitzer_debug", lambda: True)
    with pytest.raises(SystemExit):
        baseline.main(args)
    assert "debug build" in capsys.readouterr().err
    monkeypatch.setattr(baseline, "plitzer_debug", lambda: False)
    baseline.main(args)
    rows = capsys.readouterr().out.strip().splitlines()
    assert len(rows) == 3 and rows[2].startswith("| plitzer |")
