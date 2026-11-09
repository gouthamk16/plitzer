import threading
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
    for epoch in (1, 2):
        for _ in loader:
            pass
        assert loader.bytes_read == epoch * whole


def test_measure_one_gpu_reads_the_dataset(dataset: Path) -> None:
    s = SETTINGS
    settings = amplification.Settings(
        s["lanes"], s["window"], s["seq_len"], s["docs"], s["cap"], s["budget"]
    )
    one, two, four = amplification.measure(dataset, settings, [1, 2, 4])
    assert one.amplification == 1.0
    for world in (two, four):
        assert 1.0 <= world.amplification <= world.bound * 1.5
        assert len(world.per_gpu) in (2, 4)
    assert "| 3 | 4 |" in amplification.table([one, two, four])


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
    args = "--lanes 4 --window 1 3 --worlds 1,2 --seq-len 8 --docs 2 --cap 10"
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

    def do_GET(self) -> None:  # noqa: N802 (the name http.server calls)
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
    Ranges.body, Ranges.sent = source.read_bytes(), []
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
