# plitzer

A deterministic, resumable dataloader for ML training that reads Parquet directly from local disk or S3. The core is written in Rust and exposed to Python.

The order is a pure function of `(seed, epoch)` and the dataset. Documents are packed into fixed-length sequences by *lanes*, which depend on the batch size and never on the GPU count: every sequence is full, with no token dropped and no padding. Resuming mid-epoch is exact, and the sequences stay identical when the GPU count changes.

> **Status:** alpha, 0.1.0. The API and the checkpoint format can still change between minor versions; a checkpoint from another version is refused with a clear error, never misread.

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

With PyTorch (`pip install plitzer[torch]`), `plitzer.torch.Dataset` takes the same settings. It reads the rank and world from `torch.distributed`, and its `state_dict()` is one global checkpoint that resumes on any number of GPUs:

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

The state is the position of the next batch the dataset gives, so a loop that prefetches batches before training them skips those on a resume ([#22](https://github.com/gouthamk16/plitzer/issues/22)).

S3 credentials come from the usual `AWS_*` variables, or from `storage_options={"endpoint": ..., "region": ...}`.

The design, its rationale and the decision log are in [docs/design.md](https://github.com/gouthamk16/plitzer/blob/main/docs/design.md).

## Benchmarks

plitzer against MosaicML Streaming and Hugging Face `datasets`, all reading the same FineWeb-Edu-shaped Parquet files (512 row groups, 526 M tokens) through one harness, `plitzer.bench.baseline`, on 4 CPU cores with the files in the page cache (converted sizes in MiB, tokens lost on a resume at mid-epoch):

| loader | convert (s) | converted (MiB) | docs/s | Mtokens/s | cores | CPU ns/token | resume (s) | exact resume | tokens lost |
|---|---|---|---|---|---|---|---|---|---|
| plitzer | 0.0 | 0.0 | — | 298.5 | 3.95 | 13.2 | 0.34 | yes | 0 |
| mosaic | 27.5 | 1221.8 | 23,475 | 24.1 | 2.66 | 110.5 | 2.79 | yes | 0 |
| hf | 4.8 | 2008.7 | 82,952 | 85.2 | 0.99 | 11.6 | 0.16 | yes | 0 |
| hf-stream | 0.0 | 0.0 | 2,329 | 2.4 | 1.12 | 467.0 | 3.54 | no | 10,373,481 |

plitzer reads the Parquet as it is, with no conversion or copy, delivers 3.5× the tokens of `hf` and 12× those of Mosaic, packs every sequence full with no padding, and resumes exactly. `hf` still spends less CPU per token (11.6 vs 13.2 ns) and resumes faster (0.16 vs 0.34 s), from the Arrow copy it makes first; both are the next work on speed. `--batch` is in documents, and plitzer gets the same tokens in each batch (32 sequences of 2048). Method and caveats: [design §5.14](https://github.com/gouthamk16/plitzer/blob/main/docs/design.md#514-baseline-harness).

```sh
python -m plitzer.bench.baseline DATA --loaders plitzer,mosaic,hf,hf-stream --batch 64 --workers 4
```

## TODO

- [x] **Benchmark baseline:** measure MosaicML Streaming and HF `datasets` on a FineWeb-Edu Parquet sample (conversion time, samples/s, CPU cores used, resume time) (`plitzer.bench.baseline`, [design §5.14](https://github.com/gouthamk16/plitzer/blob/main/docs/design.md#514-baseline-harness))
- [x] **Ordering core:** shuffle row groups per epoch, split each window into lane slices, and shuffle each slice (`rand_chacha`). Property tests:
  - [x] each epoch yields every sample exactly once
  - [x] each lane's slice is one contiguous stretch of storage
  - [x] a saved reference output matches on every OS
- [x] **Lanes:** pack each lane's documents into fixed-length sequences, and checkpoint each lane's position. Property tests:
  - [x] resuming at step k matches the tail of an uninterrupted run
  - [x] global batches are identical across GPU counts
- [x] **Lossless packing:** cut lane slices by tokens and fill each sequence to exactly L, with lanes running on across epochs, so no token is dropped or padded (was 19.3% lost on FineWeb-Edu lengths)
- [x] **Index:** build a row-group index from Parquet footers, with caching
- [x] **I/O:** async reads with `object_store` (local disk and S3, tested against versitygw), plus a prefetch queue capped by bytes
  - [x] read one row group's token column without its footer (local disk and S3)
  - [x] prefetch each GPU's row groups for a window, capped by bytes
- [x] **Python:** PyO3 and maturin bindings, a PyTorch `IterableDataset` with `state_dict()`, and Arrow PyCapsule export
  - [x] typed checkpoint bytes that merge across GPUs, and a per-GPU session that runs epochs and refuses a mismatched resume
  - [x] bindings: a `plitzer` wheel exporting batches through DLPack and Arrow
  - [x] `plitzer.torch` `IterableDataset` with `state_dict()` and `load_state_dict()`
- [x] **End-to-end test:** kill training at a random step, resume on a different GPU count, and check the sample-stream hash matches
- [x] **Read amplification check:** measure bytes read per GPU on real data. Lane slices simulate at 1.01–1.25×. Add a per-node cache only if it's needed. Measured on FineWeb-Edu lengths: at the bound 1 + (world − 1)/W (1.013× at W = 512 on 8 GPUs), so no cache ([design §6](https://github.com/gouthamk16/plitzer/blob/main/docs/design.md#6-read-amplification))
- [x] **Benchmarks:** publish a comparison table against Mosaic and HF ([Benchmarks](#benchmarks))
- [x] **Release:** build wheels in CI and publish to PyPI
  - [x] wheels for Linux, macOS and Windows and an sdist, each tested on its own OS, published by a `v*` tag (`.github/workflows/release.yml`)
  - [x] publish 0.1.0 to PyPI ([pypi.org/project/plitzer](https://pypi.org/project/plitzer/))
- [ ] **Stretch:** in-loader tokenization

## Known limits

- One `Loader` per GPU process, with `num_workers=0` in PyTorch: the parallelism is in Rust, and forking a process with a running runtime is unsafe.
- Read amplification is about 1 + (GPUs − 1)/W: set the window W to many times your GPU count (for example W = 512 costs 1.4 % on 8 GPUs).
- CPU per token and the time to the first batch after a resume are still behind Hugging Face `datasets`' memory-mapped Arrow copy; both are the next work on speed.
- Input is pre-tokenized text only; tokenization, images and dataset mixing are out of scope for 0.1.

## Contributing

Contributions are welcome: bug reports, benchmarks on your own data, and pull requests.

- **Issues:** report a bug with the plitzer version, the OS, a minimal Parquet file or the code that makes one, and the full error. For a wrong order or a wrong resume, give the seed and the settings.
- **Branches:** one branch per change, from `main`, named `feat/<name>` or `fix/<name>`. Every change lands through a pull request with green CI; nobody commits to `main` directly.
- **CI must pass on all three OSes.** The order must be bit-identical on Linux, macOS and Windows, so CI runs fmt, clippy (`-D warnings`), ruff, `mypy --strict`, and the Rust and Python tests on each OS, plus the S3 and baseline jobs. Run the same checks locally first (see Development; [CLAUDE.md](https://github.com/gouthamk16/plitzer/blob/main/CLAUDE.md) lists them all).
- **Do not break the invariants.** Every shuffle goes through `shuffle::shuffle` with `rand_chacha`, never `StdRng` or `rand`'s range sampling. Each epoch gives every token once, a resume reproduces the tail of an uninterrupted run, and the golden outputs match on every OS. A change to the order or the packer keeps the golden and property tests passing; an intended change to a golden value says why in the PR.
- **Tests come with the change.** A bug fix adds the test that fails without it. Prefer a property test where the input space is real.
- **Docs come with the change.** A PR that changes a feature, a design or an interface updates [docs/design.md](https://github.com/gouthamk16/plitzer/blob/main/docs/design.md) in the same PR (status table, component section, decision log, change log), written in ASD-STE100; section 10 of that document gives the rules. Tick the TODO item below in the PR that completes it, and add user-facing changes to [CHANGELOG.md](https://github.com/gouthamk16/plitzer/blob/main/CHANGELOG.md).
- **Code style:** the simplest correct code; files under about 300 lines and functions under about 30; typed structs across module and Rust↔Python boundaries (no `Any` in Python); comments only for what the code cannot say.
- **Licence:** a contribution is licensed under MIT OR Apache-2.0, as the project is, unless you say otherwise.

`main` is protected by the ruleset in [.github/rulesets/main.json](https://github.com/gouthamk16/plitzer/blob/main/.github/rulesets/main.json): pull requests only, with CI green and up to date with `main`, review threads resolved, merge commits, and no force-push or deletion.

## Development

```sh
pip install maturin
maturin develop --release        # build the extension into the active virtualenv
cargo test && python -m pytest plitzer-py/tests
```

[CLAUDE.md](https://github.com/gouthamk16/plitzer/blob/main/CLAUDE.md) lists every check that CI runs. A release is a `v*` tag: CI builds the wheels and the sdist, tests each wheel on its own OS, and publishes them to PyPI ([CHANGELOG.md](https://github.com/gouthamk16/plitzer/blob/main/CHANGELOG.md)).

## License

Licensed under either [MIT](https://github.com/gouthamk16/plitzer/blob/main/LICENSE-MIT) or [Apache-2.0](https://github.com/gouthamk16/plitzer/blob/main/LICENSE-APACHE), at your option.
