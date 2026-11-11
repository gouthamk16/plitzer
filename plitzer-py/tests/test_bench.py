import threading
import time
import urllib.error
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

from conftest import SETTINGS, MakeLoader
from plitzer.bench import amplification, fineweb


def test_bytes_read_counts_each_epoch(make: MakeLoader, dataset: Path) -> None:
    loader = make()
    assert loader.bytes_read == 0, "nothing is read before the first step"
    whole = amplification.scan(dataset, "tokens").bytes
    # Each epoch reads the whole dataset, and the prefetch reads at most a
    # few windows into the next one, so a second read of the dataset fails.
    for epoch in (1, 2, 3):
        for _ in loader:
            pass
        assert epoch * whole <= loader.bytes_read < (epoch + 1) * whole


def settings(window: int) -> amplification.Settings:
    s = SETTINGS
    return amplification.Settings(s["lanes"], window, s["seq_len"], s["budget"])


def test_measure_runs_every_rank(dataset: Path, tmp_path: Path) -> None:
    # GPU edges fall inside row groups, so these amplifications are above 1
    # but not exact. `test_measure_even_row_groups` checks exact values.
    cache = tmp_path / "index"
    one, two, four = amplification.measure(dataset, settings(3), [1, 2, 4], cache)
    assert [len(w.per_gpu) for w in (one, two, four)] == [1, 2, 4]
    assert all(w.amplification >= 1.0 for w in (one, two, four))
    assert "| 3 | 4 |" in amplification.table([one, two, four])


@pytest.fixture
def even(tmp_path: Path) -> Path:
    """Eight row groups of 20 tokens each: lane slices are cut by tokens, so
    with equal row groups, GPU edges fall on row-group edges."""
    data = tmp_path / "even"
    data.mkdir()
    rows = [[r * 10 + t for t in range(5)] for r in range(32)]
    table = pa.table({"tokens": pa.array(rows, pa.list_(pa.int32()))})
    pq.write_table(table, data / "0.parquet", row_group_size=4)
    return data


def test_measure_even_row_groups(even: Path, tmp_path: Path) -> None:
    cache = tmp_path / "index"
    # 4 lanes and windows of 4 row groups: each lane has one row group's
    # tokens, so GPUs share none, and the amplification is exactly 1.
    for w in amplification.measure(even, settings(4), [1, 2, 4], cache):
        assert w.amplification == 1.0, len(w.per_gpu)
    # Windows of 2: each lane has half a row group's tokens, and 4 GPUs read each
    # row group twice.
    (w,) = amplification.measure(even, settings(2), [4], cache)
    assert w.amplification == 2.0 == len(w.per_gpu) / w.window


def test_scan_and_bound(dataset: Path) -> None:
    # Files of 6, 9, 12 and 15 rows, in row groups of 5.
    assert amplification.scan(dataset, "tokens").row_groups == 2 + 2 + 3 + 3
    # 512 row groups in windows of 100: five full windows and one of 12,
    # each read with at most 63 more row groups.
    world = amplification.World(100, [0] * 64, amplification.Dataset(512, 1))
    assert world.bound == 1 + 6 * 63 / 512
    assert amplification.World(4, [0], amplification.Dataset(10, 1)).bound == 1


def test_main_prints_a_row_for_each_run(
    dataset: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = "--lanes 4 --window 1 3 --worlds 1,2 --seq-len 8"
    amplification.main([str(dataset), *args.split()])
    rows = capsys.readouterr().out.strip().splitlines()
    assert len(rows) == 2 + 4
    assert rows[2].startswith("| 1 | 1 |") and rows[-1].startswith("| 3 | 2 |")


@pytest.fixture
def source(tmp_path: Path) -> Path:
    """A file shaped like FineWeb-Edu: text and token counts, in row groups
    of 4 rows. The text is large and random, so its chunks are far larger
    than the 64 KiB that pyarrow reads at the end to find the footer."""
    counts = [3, 0, 7, 1, 4, 4, 9, 2, 5, 6]
    rng = np.random.default_rng(0)
    table = pa.table(
        {
            "text": [rng.bytes(30_000).hex() for _ in counts],
            "token_count": pa.array(counts, pa.int64()),
        }
    )
    path = tmp_path / "source.parquet"
    pq.write_table(table, path, row_group_size=4)
    return path


def lengths_of(path: Path) -> list[list[int]]:
    """The document lengths of each row group of a built file."""
    parquet = pq.ParquetFile(path)
    groups = (parquet.read_row_group(g) for g in range(parquet.num_row_groups))
    return [pc.list_value_length(t["tokens"]).to_pylist() for t in groups]


def test_build_keeps_lengths_and_row_groups(source: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    (built,) = fineweb.build([source], out, row_groups=2)
    assert lengths_of(built) == [[3, 0, 7, 1], [4, 4, 9, 2]]
    tokens = pq.read_table(built)["tokens"].combine_chunks().flatten()
    values = tokens.to_numpy()
    assert values.dtype == np.int32 and values.min() >= 0
    assert values.max() < fineweb.VOCAB
    again = fineweb.build([source], tmp_path / "again", row_groups=2)
    assert pq.read_table(again[0]).equals(pq.read_table(built)), "seeded"


class Ranges(BaseHTTPRequestHandler):
    """Serves one file, only in ranges, as Hugging Face's CDN does; records
    the ranges it sends."""

    body = b""
    sent: list[range] = []
    failures = 0  # the next requests that get 503

    def do_GET(self) -> None:  # noqa: N802 (the name http.server calls)
        if self.failures:
            type(self).failures -= 1
            self.send_error(503)
            return
        self.send_range()

    def send_range(self) -> None:
        first, last = self.headers["Range"].removeprefix("bytes=").split("-")
        part = self.body[int(first) : int(last) + 1]
        self.send_response(206)
        size = len(self.body)
        self.send_header("Content-Range", f"bytes {first}-{last}/{size}")
        self.send_header("Content-Length", str(len(part)))
        self.end_headers()
        self.wfile.write(part)
        self.sent.append(range(int(first), int(last) + 1))

    def log_message(self, format: str, *args: object) -> None:
        pass


@pytest.fixture
def served(source: Path) -> Iterator[str]:
    Ranges.body, Ranges.sent, Ranges.failures = source.read_bytes(), [], 0
    server = ThreadingHTTPServer(("127.0.0.1", 0), Ranges)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}/source.parquet"
    server.shutdown()
    server.server_close()


def test_build_from_a_url_reads_ranges(
    served: str, source: Path, tmp_path: Path
) -> None:
    (built,) = fineweb.build([served], tmp_path / "out", row_groups=3)
    assert lengths_of(built) == [[3, 0, 7, 1], [4, 4, 9, 2], [5, 6]]
    # The first row group's text is far from the footer, and never fetched.
    text = pq.read_metadata(source).row_group(0).column(0)
    start = text.dictionary_page_offset or text.data_page_offset
    chunk = range(start, start + text.total_compressed_size)
    assert not [r for r in Ranges.sent if r.start < chunk.stop and chunk.start < r.stop]


def test_range_file_refuses_a_file_url(source: Path) -> None:
    with pytest.raises(OSError, match="not an HTTP URL"):
        fineweb.RangeFile(source.as_uri())


def test_build_retries_a_server_error(
    served: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(time, "sleep", lambda _: None)
    Ranges.failures = fineweb.TRIES - 1
    (built,) = fineweb.build([served], tmp_path / "out", row_groups=1)
    assert lengths_of(built) == [[3, 0, 7, 1]]
    Ranges.failures = fineweb.TRIES
    with pytest.raises(urllib.error.HTTPError, match="503"):
        fineweb.RangeFile(served)


def test_write_refuses_a_row_group_past_int32(tmp_path: Path) -> None:
    path = tmp_path / "big.parquet"
    with pytest.raises(ValueError, match="2\\*\\*31"):
        fineweb.write(iter([np.array([2**30, 2**30])]), path, seed=0)
    assert not path.exists()
