# plitzer

A deterministic, resumable dataloader for LLM training that reads pre-tokenized Parquet directly from local disk or S3. The core is written in Rust, with no Python in the hot path.

- **No conversion, no copy:** it reads your Parquet files as they are. A sidecar index of the footers is all it writes.
- **Lossless packing:** every sequence is exactly `seq_len` real tokens. No token is dropped and nothing is padded.
- **Deterministic and exactly resumable:** the order is a pure function of `(seed, epoch)` and the dataset, identical on Linux, macOS and Windows. A checkpoint resumes mid-epoch with the same batches, also on a different number of GPUs.
- **Low read amplification:** each GPU reads one contiguous stretch of each shuffled window, about `1 + (GPUs − 1)/W` of its data.
- **Zero-copy batches:** int32 tokens through DLPack (NumPy, PyTorch) and the Arrow PyCapsule interface.

## Install

```sh
pip install plitzer            # wheels for Linux (x86_64, aarch64), macOS and Windows, CPython 3.11+
pip install "plitzer[torch]"   # with the PyTorch IterableDataset
```

## Quick start

The input is a Parquet dataset with a pre-tokenized `list<int32>` (or `uint32`) column.

```python
import numpy as np
import plitzer

loader = plitzer.Loader(
    "s3://bucket/fineweb-edu",          # or a local folder
    seed=0, lanes=512, window=64,        # B lanes, W row groups per window
    seq_len=2048,                        # every row is 2048 real tokens: none dropped, no padding
    budget=8 << 30,                      # bytes of decoded row groups kept in memory
                                         # (4 per token + 8 per document)
    rank=0, world=8,                     # this GPU, of 8
)
while loader.epoch < 3:                  # also right after loader.resume(state)
    for batch in loader:                 # the rest of this epoch; then epoch + 1
        tokens = np.from_dlpack(batch.tokens)    # or torch.from_dlpack, or pyarrow.array
        ...
    state = loader.checkpoint()          # this GPU's part; plitzer.merge(parts) joins all GPUs
```

With PyTorch, `plitzer.torch.Dataset` takes the same settings, reads the rank and world from `torch.distributed`, and its `state_dict()` is one global checkpoint that resumes on any number of GPUs:

```python
from torch.utils.data import DataLoader
import plitzer.torch

dataset = plitzer.torch.Dataset("s3://bucket/fineweb-edu", seed=0, lanes=512, window=64,
                                seq_len=2048, budget=8 << 30)
loader = DataLoader(dataset, batch_size=None, num_workers=0)  # each item is one step
while dataset.epoch < 3:                 # also right after dataset.load_state_dict(state)
    for batch in loader:                 # batch.tokens: (lanes per GPU, seq_len) int32
        ...
    state = dataset.state_dict()         # call it on every rank: it gathers all GPUs
# later, on any number of GPUs: dataset.load_state_dict(state)
```

S3 credentials come from the usual `AWS_*` variables, or from `storage_options={"endpoint": ..., "region": ...}`.

## Benchmarks

plitzer, MosaicML Streaming and Hugging Face `datasets` reading the same FineWeb-Edu-shaped Parquet files (526 M tokens) on 4 CPU cores:

| loader | conversion | extra copy | Mtokens/s | CPU ns/token | resume (s) | exact resume | tokens lost |
|---|---|---|---|---|---|---|---|
| plitzer | none | none | 391.5 | 10.0 | 0.24 | yes | 0 |
| MosaicML Streaming | 26.5 s | 1.2 GiB | 23.9 | 102.8 | 3.32 | yes | 0 |
| HF `datasets` | 4.2 s | 2.0 GiB | 91.9 | 10.8 | 0.27 | yes | 0 |
| HF `datasets`, streaming | none | none | 2.8 | 388.2 | 3.18 | no | 10.4 M |

The method and the full table are in the [repository README](https://github.com/gouthamk16/plitzer#benchmarks).

## Status and limits

Alpha. The API and the checkpoint format can change between minor versions; a checkpoint from another version is refused with a clear error, never misread.

- Use one loader per GPU process, with `num_workers=0` in PyTorch: the parallelism is in Rust.
- Set the window W to many times your GPU count.
- Input is pre-tokenized text; tokenization, images and dataset mixing are out of scope for 0.1.

## Links

- Source, issues and contributing: <https://github.com/gouthamk16/plitzer>
- Design and decision log: <https://github.com/gouthamk16/plitzer/blob/main/docs/design.md>
- Changelog: <https://github.com/gouthamk16/plitzer/blob/main/CHANGELOG.md>

Licensed under either MIT or Apache-2.0, at your option.
