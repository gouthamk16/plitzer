# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Status

Plitzer is a Rust-core ML dataloader. It feeds Parquet data from local disk or S3 to PyTorch with an order that is deterministic and resumable. So far it is a library crate with ten modules plus Python bindings: `src/shuffle.rs`, a deterministic Fisher-Yates shuffle; `src/seed.rs`, which derives each shuffle's 32-byte key; `src/order.rs`, the ordering core (row-group windows, lane slices, GPU lane blocks); `src/pack.rs` (with `src/pack/lane.rs` and `src/pack/pieces.rs`), the lossless per-lane packer, whose lanes run on across epochs and resume from a position; `src/index.rs`, the row-group index built from Parquet footers through `object_store`, cached in a sidecar, recording where each row group's token column chunk is; `src/read.rs`, which fetches and decodes one row group's token column from that record alone, never a footer; `src/load.rs`, the per-GPU loader that prefetches its lanes' row groups under a byte budget and drives the packer step by step; `src/checkpoint.rs`, checkpoint bytes (versioned, checksummed) whose per-GPU parts merge into a global checkpoint; and `src/session.rs`, one GPU's lanes across epochs, which validates the config, rolls epochs and resumes from a checkpoint. `src/store.rs` turns a local path or `s3://` URL into a store (S3 reads `AWS_*` from the environment as object_store's `from_env` does, but an empty variable counts as unset; settings that object_store would panic on are refused). `src/codec.rs` holds the binary helpers shared by the sidecar and the checkpoint. `plitzer-py/` is a workspace crate (PyO3 0.30, maturin, abi3 wheels for CPython 3.11+) exposing `plitzer.Loader` (one GPU's `Session`), `Batch.tokens` as an int32 array through DLPack (versioned and legacy) and the Arrow PyCapsule interface, `merge`, and typed exceptions, plus `plitzer.torch.Dataset` (`python/plitzer/torch.py`, the optional `torch` extra), an `IterableDataset` whose `state_dict()` gathers every rank's part into one global checkpoint. `Loader.bytes_read` counts the token-chunk bytes a loader fetched, live. `plitzer.bench` (the optional `bench` extra) holds the benchmark harnesses: `fineweb` writes a dataset with FineWeb-Edu's document lengths and row groups (random tokens), `amplification` measures the bytes every GPU reads in an epoch, and `baseline` (the `baseline` extra: torch, `mosaicml-streaming`, `datasets`) measures MosaicML Streaming and HF `datasets` on the same files: conversion, throughput, CPU cores and an exact-resume check. The rest of the design below is the plan, not existing code. Update this file as pieces land.

## Commands

```sh
cargo build
cargo test
cargo test <name>          # single test, substring match
cargo clippy --all-targets --all-features
cargo fmt
cargo test --features s3   # tests/s3.rs needs PLITZER_S3_ENDPOINT, e.g. a local versitygw

# Python bindings (plitzer-py/), in a virtualenv with maturin, pytest, numpy, pyarrow, and for
# plitzer.torch (mypy and test_torch.py) the CPU torch: pip install torch --index-url https://download.pytorch.org/whl/cpu
maturin develop -m plitzer-py/Cargo.toml
python -m pytest plitzer-py/tests     # test_s3.py runs when PLITZER_S3_ENDPOINT is set; test_torch.py and
                                      # test_e2e_torch.py when torch is installed. The e2e tests kill training
                                      # at a random step; PLITZER_E2E_SEED=<seed> repeats a failing seed.
                                      # test_baseline.py runs when mosaicml-streaming and datasets are installed.
ruff check plitzer-py/python plitzer-py/tests && ruff format --check plitzer-py/python plitzer-py/tests && mypy plitzer-py/python plitzer-py/tests

# Read amplification on FineWeb-Edu shapes (needs huggingface.co and its CDN us.aws.cdn.hf.co)
python -m plitzer.bench.fineweb OUT --files 8 --row-groups 64
python -m plitzer.bench.amplification OUT --lanes 512 --window 16 64 256 512 --worlds 1,8,64,256 \
    --seq-len 2048
# The baseline loaders (CPU torch and torchvision first, from download.pytorch.org/whl/cpu)
python -m plitzer.bench.baseline OUT --loaders mosaic,hf,hf-stream --batch 64 --workers 4
```

`pyproject.toml` is at the repository root and points maturin at `plitzer-py/Cargo.toml`, so `maturin develop`, `maturin sdist` and `pip install .` work from the root. A release is a `v*` tag matching the Cargo version: `.github/workflows/release.yml` builds abi3 wheels (Linux x86_64 and aarch64, macOS universal2, Windows x64) and an sdist, tests each wheel on its own OS, and publishes to PyPI with trusted publishing. Add the version's entry to `CHANGELOG.md` first.

`cargo test` at the root tests only the core crate; the bindings are tested through pytest. Lint with `cargo clippy --workspace --all-targets --all-features` and `cargo fmt --all`.

MSRV is Rust 1.88 (required by `parquet` 60).

CI (`.github/workflows/ci.yml`) runs fmt, clippy with `-D warnings`, ruff and `mypy --strict`, and Rust and Python (3.11, the abi3 floor) tests on Linux, macOS and Windows. The cross-OS matrix is required because ordering output must match on every platform. A separate Linux `s3` job runs `tests/s3.rs` and `plitzer-py/tests/test_s3.py` against versitygw (MinIO no longer publishes images); S3 support is behind the `s3` cargo feature.

## Workflow

- One branch per feature, branched from `main` (`feat/<name>`, `fix/<name>`). Don't use git worktrees.
- Each change lands on `main` through a PR with green CI. Never commit directly to `main`.
- Keep the TODO list in `README.md` current: tick items off in the PR that completes them.
- Keep `docs/design.md` current: every PR that adds or changes a feature, design or interface updates it in the same PR (status table, component section, requirements, decision log, change log). Write it in ASD-STE100; section 10 of the doc lists the steps and rules.

## Planned design

The pitch is MosaicML Streaming's guarantees, read directly from existing Parquet with no conversion step and no Python in the hot path. Initial scope is LLM text: it loads documents and packs them into fixed-length sequences. Images, WebDataset, format conversion and dataset mixing are out of scope for v1.

Pipeline: **index → order → read → pack → Python**

- **Index:** read only Parquet footers (parquet's `ParquetMetaDataPushDecoder` over `object_store`) to build a list of `(file, row_group, num_rows)` plus each row group's token-column chunk `(start, len, codec)`, sorted by path, and cache it as a compact sidecar file. The cache is reused while every file's path, size and ETag match the listing. Footers are large: about 3.1 MiB per FineWeb-Edu file, about 9.4 GB across the dataset. Rereading them on every start isn't an option.
- **Order:** a pure function of `(seed, epoch)`. Nothing about the order is stored.
  - Each epoch, shuffle the row groups (`Purpose::RowGroups`) and group them into windows of W row groups.
  - Within a window, row groups are in storage order. The global batch has B **lanes** (sequence slots), and each lane owns one contiguous slice of the window, cut by **slots** (one per token plus one per empty row: the leaf value count the footer records), so every lane gets the same share of the tokens to within one. A row crossing a slice edge is split between the two lanes. Each lane cuts its slice into pieces (one per row, from the row offsets once its row groups are in memory) and shuffles them with `Purpose::Lane { window, lane }`. Slicing by document counts made lanes drift apart by whole windows (±25% tokens per lane on FineWeb-Edu), so don't go back to it.
- **Pack:** lossless. A lane is a permanent packer over its **stream**: its slices of epoch e's windows, then epoch e+1's, and so on. Each step it copies the rest of its current piece, then whole pieces, cut at exactly L tokens, straight from the row group into the output (one copy per piece, no leftover buffer). Nothing is dropped and nothing is padded; there is no D, cap or pad token. A lane's position is `(epoch, window, token)`: tokens taken from that window's slice, so it is checked and located from the index alone. An epoch is S = ⌈T/(B·L)⌉ steps for the dataset's T tokens, so lanes run on into the next epoch within a few tokens of each other. `Lane::need` finds the missing row groups before `fill` moves anything, which keeps the loader cancel safe. Tokens come in through the `Tokens` trait; the packer adds no EOS.
- **Checkpoint:** version 2: `seed`, `epoch`, `step`, every lane's position `(epoch, window, token)`, and B, W, L and an index fingerprint so a resume with different settings is refused (version 1 is refused). It is bytes (magic, version, FNV-1a checksum), never a dict. Each GPU's `Session::checkpoint()` is a part; `Checkpoint::merge` joins all parts into a global checkpoint that resumes on any GPU count. At the end of an epoch the session stands at `(epoch + 1, step 0)`, and the lanes keep their positions.
- **Splitting across GPUs:** each GPU owns a contiguous block of `B/world` lanes. Lanes depend only on B, never on `world`. That's why resuming on a different GPU count gives identical sequences: each GPU rebuilds its lanes from the checkpoint. B must be divisible by `world`.
- **I/O:** `object_store` plus parquet's sans-I/O `ParquetPushDecoder`, fed with one-column metadata rebuilt from the index record, so no footer is ever re-read (row groups are shuffled across files, so footer re-reads would cost ~1.5× the data). Chunks are fetched in ≤16 MiB pieces. Input is a pre-tokenized list-of-integers column; `int32`/`uint32` tokens are zero-copy. Prefetching is capped by bytes, not by number of batches: each GPU's loader reads its lanes' row groups in window order across epochs (keys `(epoch, window, position)`), at most 2 windows past its slowest lane, taking semaphore permits (KiB, cost `values·4 + (rows+1)·8`) one row group at a time. A step that waits for a read adds that read's permits, so the loader never waits on itself; a budget below one window is refused. A row group is kept while some `(lane, epoch, window)` still uses it, and released only after every lane filled the step. At most 16 reads are in flight. Prefetched data is not part of a checkpoint; a resume reads each lane's current slice again, with no replay. Use a multi-thread tokio runtime so reads continue while Python runs the GPU step. After a failed read the loader returns `Stopped`; resume from `positions()`.
- **Python:** PyO3 and maturin. Batches are exported through DLPack (int32) and the Arrow PyCapsule interface (`arrow.fixed_shape_tensor`). A PyTorch `IterableDataset` provides `state_dict()` and `load_state_dict()`; `state_dict()` is a collective over the data-parallel group, so every rank must call it. Users set `num_workers=0`, since parallelism lives in Rust and forking a process with a running tokio runtime is unsafe.

Still open:
- Not yet measured on real FineWeb-Edu lengths: how well documents mix within a sequence on sorted datasets, per-GPU batch statistics, and the window-size/memory trade-off.

### Invariants that must not break

- Use `rand_chacha` for every shuffle, never `StdRng`. `StdRng`'s algorithm can change between `rand` versions, which would silently break reproducibility. For the same reason, don't use `rand`'s range sampling (`random_range`, formerly `gen_range`, and `SliceRandom`) or `seed_from_u64`. Use `shuffle::shuffle`. It takes a `seed::Key`, which only `seed::key` can build, so every shuffle starts from fresh generator state and distinct `(seed, epoch, purpose)` inputs never share a stream.
- Each epoch of the order yields every sample exactly once, and the packer emits every token of each lane's stream in order: no token is dropped and no sequence is padded.
- Lanes and their slices depend only on `(seed, epoch, B, W)` and the index's slots, never on `world`.
- Resuming at step k reproduces the tail of an uninterrupted run exactly, including when the GPU count changes.
- A saved reference output of the order must match on every OS.

These are enforced with tests: golden outputs pinned in CI on all 3 OSes, and `proptest` property tests in `src/order.rs` and `src/pack/tests.rs`. Any change to the ordering code must keep them passing.

### Known hard problem

Read amplification. If a shuffled batch is split across GPUs naively, every GPU downloads almost every row group in the window, about `world`× the data. Lane slices fix this. A GPU's lanes cover one contiguous stretch of storage, so it reads that stretch and roughly one shared row group at each edge. The cost is at most 1 + ⌈n/W⌉(world − 1)/n in row groups (about 1 + (world − 1)/W). Measured on FineWeb-Edu lengths (`plitzer.bench`, design §6), it sits at that bound: slices are cut by tokens, so each GPU edge falls inside a row group (1.013 at W = 512 on 8 GPUs, 1.122 on 64), and about world/W when world > W. So there is no per-node cache; pick W many times the GPU count. Treat any change to slicing as a trade-off between shuffle quality and I/O.

## Code standards

- Prefer the simplest correct solution over the most extensible one. A function earns its place by being reused or by making the code clearer, not by anticipating future needs.
- Don't write comments that restate what the code does. Comment only what isn't obvious: why a threshold has its value, a workaround for a specific library bug, an invariant you can't see locally.
- Keep files under about 300 lines. A module growing past that is usually doing two things, so split it.
- Keep functions under about 30 lines. A longer function is probably doing more than one job.
- Anything that crosses a module or FFI boundary (Rust↔Python) is a typed struct or class, not a dict or loose object. In Python, don't use `Any`. If a shape really is unknown, take `object` and narrow it.
- Every async call that can fail (S3 and network reads, file I/O, subprocesses) gets explicit error handling where the failure can be acted on, not a blanket catch three layers up.
- Extract logic duplicated three or more times. Two copies usually don't need extracting yet.
- Put imports at the top of the module. Exception: heavy ML imports (`torch`, `jax`) may be deferred into the function that uses them if that measurably speeds up import time. Note why inline when you do.
- Don't commit one-off or throwaway scripts. The benchmark harness is part of the product: it lives in the real module structure and has tests.
