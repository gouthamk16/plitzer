"""Hugging Face ``datasets``, two ways:

- ``HF``: ``load_dataset`` converts the files to an Arrow cache, and a
  shuffled map-style dataset gives each batch by random access. A resume
  starts at the next index of the shuffled order, as
  ``accelerate.skip_first_batches`` does for a map-style sampler.
- ``HFStream``: ``streaming=True`` reads the files with no conversion,
  through a shuffle buffer. ``state_dict()`` saves the position but not the
  buffer, so a resume refills it from later documents.

Both iterate in this process: ``state_dict()`` of an ``IterableDataset``
counts only what this process takes, not what DataLoader workers take.
"""

from collections.abc import Iterator
from pathlib import Path

import datasets
import numpy as np

from plitzer.bench.baseline import Doc, Run, parquet_files

BUFFER = 10_000  # documents in the shuffle buffer of HFStream

datasets.disable_progress_bars()


def files(data: Path) -> list[str]:
    return [str(p) for p in parquet_files(data)]


class MapOpened:
    def __init__(self, dataset: datasets.Dataset, batch: int, start: int) -> None:
        self._dataset, self._batch, self._next = dataset, batch, start

    def __iter__(self) -> Iterator[list[Doc]]:
        while self._next < len(self._dataset):
            rows = self._dataset[self._next : self._next + self._batch]["tokens"]
            self._next += self._batch
            yield [np.asarray(r, np.int32) for r in rows]

    def state(self) -> object:
        return self._next


class HF:
    def convert(self, data: Path, work: Path) -> Path:
        cache = work / "hf"
        datasets.load_dataset(
            "parquet", data_files=files(data), split="train", cache_dir=str(cache)
        )
        return cache

    def open(self, data: Path, prepared: Path, run: Run, state: object) -> MapOpened:
        dataset = datasets.load_dataset(
            "parquet", data_files=files(data), split="train", cache_dir=str(prepared)
        )
        shuffled = dataset.shuffle(seed=run.seed).with_format("numpy")
        start = 0 if state is None else state
        if not isinstance(start, int):
            raise TypeError("an HF state is the index of the next document")
        return MapOpened(shuffled, run.batch, start)


class StreamOpened:
    def __init__(self, dataset: datasets.IterableDataset, batch: int) -> None:
        self._dataset, self._batch = dataset, batch

    def __iter__(self) -> Iterator[list[Doc]]:
        batch: list[Doc] = []
        for row in self._dataset:
            batch.append(np.asarray(row["tokens"], np.int32))
            if len(batch) == self._batch:
                yield batch
                batch = []
        if batch:
            yield batch

    def state(self) -> object:
        return self._dataset.state_dict()


class HFStream:
    def convert(self, data: Path, work: Path) -> Path:
        return data

    def open(self, data: Path, prepared: Path, run: Run, state: object) -> StreamOpened:
        dataset = datasets.load_dataset(
            "parquet", data_files=files(data), split="train", streaming=True
        )
        shuffled = dataset.shuffle(seed=run.seed, buffer_size=BUFFER)
        formatted = shuffled.with_format("numpy")
        if state is not None:
            if not isinstance(state, dict):
                raise TypeError("an HFStream state is the dict of state_dict()")
            formatted.load_state_dict(state)
        return StreamOpened(formatted, run.batch)
