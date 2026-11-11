"""plitzer in the baseline harness. Its "conversion" is the index: it reads
the Parquet footers and keeps them in a sidecar, with no copy of the data.
Each batch is one step of one GPU, ``batch`` sequences of ``seq_len``
tokens, so the harness counts each sequence as a document. A resume goes
through the checkpoint bytes, as a trainer's does.
"""

from collections.abc import Iterator
from pathlib import Path

import numpy as np

import plitzer
from plitzer.bench.baseline import Doc, Run

WINDOW = 64  # row groups in each window
BUDGET = 8 << 30  # bytes of row groups that the loader may read ahead


def loader(data: Path, index: Path, run: Run) -> plitzer.Loader:
    return plitzer.Loader(
        data,
        seed=run.seed,
        lanes=run.batch,
        window=WINDOW,
        seq_len=run.seq_len,
        budget=BUDGET,
        cache=index,
    )


class Opened:
    def __init__(self, loader: plitzer.Loader) -> None:
        self._loader = loader

    def __iter__(self) -> Iterator[list[Doc]]:
        for batch in self._loader:
            yield list(np.from_dlpack(batch.tokens))

    def state(self) -> object:
        return self._loader.checkpoint()


class Plitzer:
    def convert(self, data: Path, work: Path) -> Path:
        index = work / "index"
        loader(data, index, Run(0, 1, 0))
        return index

    def open(self, data: Path, prepared: Path, run: Run, state: object) -> Opened:
        opened = loader(data, prepared, run)
        if state is not None:
            if not isinstance(state, bytes):
                raise TypeError("a plitzer state is checkpoint bytes")
            opened.resume(state)
        return Opened(opened)
