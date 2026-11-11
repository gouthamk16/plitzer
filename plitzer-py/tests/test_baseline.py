import hashlib
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from plitzer.bench import baseline, fineweb

# The adapters need the `baseline` extra: Mosaic, `datasets` and torch.
pytest.importorskip("streaming")
pytest.importorskip("datasets")

RUN = baseline.Run(seed=3, batch=8, workers=2)
LENGTHS = [np.array([(7 * i) % 13 for i in range(g, g + 40)]) for g in (0, 40, 80)]


@pytest.fixture(scope="module")
def data(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """120 documents in three row groups, of 0 to 12 tokens."""
    out = tmp_path_factory.mktemp("baseline") / "data"
    out.mkdir()
    fineweb.write(iter(LENGTHS), out / "0.parquet", seed=1)
    return out


def doc_key(doc: baseline.Doc) -> str:
    return hashlib.sha256(np.asarray(doc, np.int32).tobytes()).hexdigest()


def source(data: Path) -> list[str]:
    column = pq.read_table(data / "0.parquet")["tokens"].to_pylist()
    return sorted(doc_key(np.array(d, np.int32)) for d in column)


@pytest.mark.parametrize("name", list(baseline.LOADERS))
def test_an_epoch_gives_each_document_once(
    name: str, data: Path, tmp_path: Path
) -> None:
    load = baseline.loader(name)
    prepared = load.convert(data, tmp_path)
    one = baseline.Run(seed=3, batch=8, workers=0)
    got = [doc_key(d) for batch in load.open(data, prepared, one, None) for d in batch]
    assert sorted(got) == source(data)


@pytest.mark.parametrize("name", ["mosaic", "hf"])
def test_resume_is_exact(name: str, data: Path, tmp_path: Path) -> None:
    r = baseline.measure(name, data, tmp_path, RUN, resume_at=5)
    assert r.loader == name
    assert r.exact and r.lost == 0
    assert r.convert_bytes > 0 and r.docs_per_s > 0 and r.tokens_per_s > 0
    assert r.resume_s > 0
    assert (r.cores is None) == (not baseline.workers_counted())


def test_cpu_counts_this_process() -> None:
    # Until 0.1 s of CPU, not of wall time: a busy runner can deschedule us.
    start, deadline = baseline.cpu(), time.perf_counter() + 30
    while baseline.cpu() - start < 0.1:
        assert time.perf_counter() < deadline, "cpu() never grew"


def test_resume_at_must_be_inside_the_epoch(data: Path, tmp_path: Path) -> None:
    # 120 documents in batches of 8: 15 batches, the last with none after it.
    with pytest.raises(RuntimeError, match="hf epoch failed"):
        baseline.measure("hf", data, tmp_path / "a", RUN, resume_at=15)
    with pytest.raises(ValueError, match="1 or more"):
        baseline.measure("hf", data, tmp_path / "b", RUN, resume_at=0)


def test_main_refuses_unknown_or_repeated_loaders(data: Path) -> None:
    for loaders in ("hf,hf_stream", "hf,hf"):
        with pytest.raises(SystemExit):
            baseline.main([str(data), "--loaders", loaders])


def test_a_streaming_resume_loses_the_shuffle_buffer(
    data: Path, tmp_path: Path
) -> None:
    # The buffer holds all 120 documents, and a resume does not save it.
    r = baseline.measure("hf-stream", data, tmp_path, RUN, resume_at=5)
    assert r.convert_bytes == 0
    assert not r.exact
    assert r.lost == 120 - 5 * RUN.batch


def test_main_prints_a_row_for_each_loader(
    data: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = [
        "--loaders",
        "hf",
        "--batch",
        "8",
        "--workers",
        "1",
        "--work",
        str(tmp_path),
    ]
    baseline.main([str(data), *args])
    rows = capsys.readouterr().out.strip().splitlines()
    assert len(rows) == 3 and rows[2].startswith("| hf |")
