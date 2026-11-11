"""MosaicML Streaming: the Parquet files converted to MDS shards, read by a
shuffled ``StreamingDataset`` through a ``StreamingDataLoader``."""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from streaming import MDSWriter, StreamingDataLoader, StreamingDataset
from streaming.base.util import clean_stale_shared_memory

from plitzer.bench.baseline import Doc, Run, parquet_files


def docs(batch: list[dict[str, bytes]]) -> dict[str, list[Doc]]:
    # A dict, not a list: StreamingDataLoader counts the samples of a batch
    # from its first value, so a list of samples would save the wrong
    # position, and a resume would start at another sample.
    return {"tokens": [np.frombuffer(s["tokens"], np.int32) for s in batch]}


class Opened:
    def __init__(self, loader: StreamingDataLoader) -> None:
        self._loader = loader

    def __iter__(self) -> Iterator[list[Doc]]:
        for batch in self._loader:
            yield batch["tokens"]

    def state(self) -> object:
        return self._loader.state_dict()


class Mosaic:
    def convert(self, data: Path, work: Path) -> Path:
        out = work / "mds"
        # Bytes, as llm-foundry writes tokens: the ndarray encoding refuses
        # an empty document.
        columns = {"tokens": "bytes"}
        with MDSWriter(out=str(out), columns=columns, compression="zstd") as writer:
            for path in parquet_files(data):
                for batch in pq.ParquetFile(path).iter_batches(columns=["tokens"]):
                    column = batch.column(0)
                    offsets = column.offsets.to_numpy()
                    values = column.values.to_numpy().astype(np.int32, copy=False)
                    for i in range(len(column)):
                        doc = values[offsets[i] : offsets[i + 1]]
                        writer.write({"tokens": doc.tobytes()})
        return out

    def open(self, data: Path, prepared: Path, run: Run, state: object) -> Opened:
        # A killed run leaves its shared memory, which a new dataset refuses.
        clean_stale_shared_memory()
        dataset = StreamingDataset(
            local=str(prepared),
            shuffle=True,
            shuffle_seed=run.seed,
            batch_size=run.batch,
        )
        loader = StreamingDataLoader(
            dataset, batch_size=run.batch, num_workers=run.workers, collate_fn=docs
        )
        if state is not None:
            if not isinstance(state, dict):
                raise TypeError("a Mosaic state is the dict of state_dict()")
            loader.load_state_dict(state)
        return Opened(loader)
