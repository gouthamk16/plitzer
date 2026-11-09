"""A dataset with the shape of FineWeb-Edu, for benchmarks that depend on
the size of documents and row groups but not on their tokens.

FineWeb-Edu has text and a ``token_count`` column (GPT-2 tokens per
document), but no token column. This reads only the footer and the
``token_count`` chunks of each source file, through HTTP range requests,
and writes a ``tokens`` column of documents with those lengths and the
same row groups. The token values are random: they compress worse than
real tokens, about equally in each row group.

    python -m plitzer.bench.fineweb OUT --files 4 --row-groups 128
"""

import argparse
import http.client
import io
import os
import time
import urllib.error
import urllib.request
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq

if TYPE_CHECKING:
    from typing_extensions import Buffer  # collections.abc.Buffer from 3.12

SOURCE = (
    "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/"
    "sample/10BT/{:03d}_00000.parquet"
)
VOCAB = 50257  # GPT-2's vocabulary, the tokenizer of `token_count`
TRIES = 4  # a build makes hundreds of requests; retry a dropped one


class RangeFile(io.RawIOBase):
    """A read-only file over HTTP: each read is one range request, so a
    Parquet reader fetches only the footer and the chunks it reads."""

    def __init__(self, url: str) -> None:
        super().__init__()
        self._url = url
        self._pos = 0
        # The total size is after the slash of `Content-Range: bytes 0-0/N`.
        with self._get(0, 0) as r:
            total = (r.headers["Content-Range"] or "").rpartition("/")[2]
        if not total.isdigit():
            raise OSError(f"{url}: no size in the Content-Range of a range")
        self._size = int(total)

    def _get(self, first: int, last: int) -> http.client.HTTPResponse:
        """The response to one range request. A connection error or a
        server error is tried again; other errors are not."""
        headers = {"Range": f"bytes={first}-{last}"}
        request = urllib.request.Request(self._url, headers=headers)
        for attempt in range(TRIES):
            try:
                response = urllib.request.urlopen(request, timeout=60)
                break
            except urllib.error.HTTPError as e:
                if e.code < 500 or attempt == TRIES - 1:
                    raise
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                if attempt == TRIES - 1:
                    raise
            time.sleep(2**attempt)
        if not isinstance(response, http.client.HTTPResponse):
            raise OSError(f"{self._url}: not an HTTP URL")
        if response.status != 206:
            response.close()
            raise OSError(f"{self._url}: no range support (HTTP {response.status})")
        return response

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self._pos, io.SEEK_END: self._size}
        if whence not in base or base[whence] + offset < 0:
            raise ValueError(f"cannot seek to {offset} from {whence}")
        self._pos = base[whence] + offset
        return self._pos

    def tell(self) -> int:
        return self._pos

    def readinto(self, buffer: "Buffer") -> int:
        view = memoryview(buffer).cast("B")
        n = min(len(view), self._size - self._pos)
        if n <= 0:
            return 0
        with self._get(self._pos, self._pos + n - 1) as r:
            data = r.read()
        if len(data) != n:
            raise OSError(f"{self._url}: {len(data)} bytes for a range of {n}")
        view[:n] = data
        self._pos += n
        return n


def lengths(source: str | Path, row_groups: int) -> Iterator[npt.NDArray[np.int64]]:
    """The ``token_count`` of each document of the first ``row_groups`` row
    groups of ``source``, a URL or a local file, one row group at a time."""
    is_url = isinstance(source, str) and source.startswith(("http://", "https://"))
    with RangeFile(str(source)) if is_url else open(source, "rb") as f:
        parquet = pq.ParquetFile(f)
        for g in range(min(row_groups, parquet.num_row_groups)):
            column = parquet.read_row_group(g, columns=["token_count"]).column(0)
            yield column.to_numpy().astype(np.int64)


def write(groups: Iterator[npt.NDArray[np.int64]], path: Path, seed: int) -> None:
    """A ``tokens`` column with one row group for each of ``groups``, whose
    documents have those lengths and random tokens. A failed build leaves
    no file at ``path``."""
    rng = np.random.default_rng(seed)
    schema = pa.schema([("tokens", pa.list_(pa.int32()))])
    temp = path.with_name(path.name + ".tmp")
    with pq.ParquetWriter(temp, schema) as writer:
        for counts in groups:
            offsets = np.concatenate([[0], np.cumsum(counts)])
            if offsets[-1] >= 2**31:
                raise ValueError(f"{path.name}: a row group of 2**31 tokens or more")
            values = rng.integers(0, VOCAB, size=int(offsets[-1]), dtype=np.int32)
            array = pa.array(offsets.astype(np.int32))
            tokens = pa.ListArray.from_arrays(array, pa.array(values))
            writer.write_table(pa.table({"tokens": tokens}, schema=schema))
    os.replace(temp, path)


def build(
    sources: Sequence[str | Path], out: Path, row_groups: int, seed: int = 0
) -> list[Path]:
    """One file in ``out`` for each of ``sources``, from its first
    ``row_groups`` row groups."""
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, source in enumerate(sources):
        path = out / f"{i:03d}.parquet"
        write(lengths(source, row_groups), path, seed + i)
        paths.append(path)
    return paths


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    p.add_argument("out", type=Path)
    p.add_argument("--files", type=int, default=4, help="sample/10BT files, from 000")
    p.add_argument("--row-groups", type=int, default=128, help="of each file")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    sources = [SOURCE.format(i) for i in range(a.files)]
    for path in build(sources, a.out, a.row_groups, a.seed):
        print(path)


if __name__ == "__main__":
    main()
