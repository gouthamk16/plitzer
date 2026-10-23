# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Status

Plitzer is a Rust-core ML dataloader. It feeds Parquet data from local disk or S3 to PyTorch with an order that is deterministic and resumable. So far it is a library crate with nine modules: `src/shuffle.rs`, a deterministic Fisher-Yates shuffle; `src/seed.rs`, which derives each shuffle's 32-byte key; `src/order.rs`, the ordering core (row-group windows, lane slices, GPU lane blocks); `src/pack.rs`, the per-lane packer with resumable lane state; `src/index.rs`, the row-group index built from Parquet footers through `object_store`, cached in a sidecar, recording where each row group's token column chunk is; `src/read.rs`, which fetches and decodes one row group's token column from that record alone, never a footer; `src/load.rs`, the per-GPU loader that prefetches its lanes' row groups under a byte budget and drives the packer step by step; `src/checkpoint.rs`, checkpoint bytes (versioned, checksummed) whose per-GPU parts merge into a global checkpoint; and `src/session.rs`, one GPU's lanes across epochs, which validates the config, rolls epochs and resumes from a checkpoint. `src/codec.rs` holds the binary helpers shared by the sidecar and the checkpoint. The rest of the design below is the plan, not existing code. Update this file as pieces land.

## Commands

```sh
cargo build
cargo test
cargo test <name>          # single test, substring match
cargo clippy --all-targets --all-features
cargo fmt
cargo test --features s3 --test s3   # needs PLITZER_S3_ENDPOINT, e.g. a local versitygw
```

MSRV is Rust 1.88 (required by `parquet` 60).

CI (`.github/workflows/ci.yml`) runs fmt, clippy with `-D warnings`, and tests on Linux, macOS and Windows. The cross-OS matrix is required because ordering output must match on every platform. A separate Linux `s3` job runs `tests/s3.rs` against versitygw (MinIO no longer publishes images); S3 support is behind the `s3` cargo feature.

## Workflow

- One branch per feature, branched from `main` (`feat/<name>`, `fix/<name>`). Don't use git worktrees.
- Each change lands on `main` through a PR with green CI. Never commit directly to `main`.
- Keep the TODO list in `README.md` current: tick items off in the PR that completes them.
- Keep `docs/design.md` current: every PR that adds or changes a feature, design or interface updates it in the same PR (status table, component section, requirements, decision log, change log). Write it in ASD-STE100; section 10 of the doc lists the steps and rules.

## Planned design

The pitch is MosaicML Streaming's guarantees, read directly from existing Parquet with no conversion step and no Python in the hot path. Initial scope is LLM text: it loads documents and packs them into fixed-length sequences. Images, WebDataset, format conversion and dataset mixing are out of scope for v1.

Pipeline: **index → order → read → pack → Python**

- **Index:** read only Parquet footers (parquet's `ParquetMetaDataPushDecoder` over `object_store`) to build a list of `(file, row_group, num_rows)` plus each row group's token-column chunk `(start, len, codec)`, sorted by path, and cache it as a compact sidecar file. The cache is reused while every file's path, size and ETag match the listing. Footers are large: about 3.1 MiB per FineWeb-Edu file, about 9.4 GB across the dataset. Rereading them on every start isn't an option.
- **Order:** a pure function of `(seed, epoch, step)`. Nothing about the order is stored.
  - Each epoch, shuffle the row groups (`Purpose::RowGroups`) and group them into windows of W row groups.
  - Within a window, row groups are in storage order. The global batch has B **lanes** (sequence slots), and each lane owns one contiguous slice of the window. Each lane shuffles its slice with `Purpose::Lane { window, lane }`.
- **Pack:** a lane is a permanent packer. Each step it takes D documents from its slice, appends them to its leftover tokens, and emits exactly one sequence of L tokens. Leftover is capped at C tokens; past the cap the *oldest* tokens are dropped, so the leftover is always one run ending at the cursor (`min(step·D, lane_docs)`). A lane that runs short is padded. A lane's state is its next step plus the `(doc, offset)` where its leftover starts. Each epoch ends with `ceil(C/L)` flush steps, so nothing is dropped at epoch end. Tokens come in through the `Tokens` trait; the packer adds no EOS.
- **Checkpoint:** `seed`, `epoch`, plus every lane's state `(step, doc, offset)`, and B, W, L, D, C and an index fingerprint so a resume with different settings is refused. It is bytes (magic, version, FNV-1a checksum), never a dict. Each GPU's `Session::checkpoint()` is a part; `Checkpoint::merge` joins all parts into a global checkpoint that resumes on any GPU count. At the end of an epoch the session stands at `(epoch + 1, step 0)`.
- **Splitting across GPUs:** each GPU owns a contiguous block of `B/world` lanes. Lanes depend only on B, never on `world`. That's why resuming on a different GPU count gives identical sequences: each GPU rebuilds its lanes from the checkpoint. B must be divisible by `world`.
- **I/O:** `object_store` plus parquet's sans-I/O `ParquetPushDecoder`, fed with one-column metadata rebuilt from the index record, so no footer is ever re-read (row groups are shuffled across files, so footer re-reads would cost ~1.5× the data). Chunks are fetched in ≤16 MiB pieces. Input is a pre-tokenized list-of-integers column; `int32`/`uint32` tokens are zero-copy. Prefetching is capped by bytes, not by number of batches: each GPU's loader plans every row group its lanes need with a `[first, last]` step range (from lane slice ranges, not per document), refuses a budget below the plan's peak, takes semaphore permits (KiB, cost `values·4 + (rows+1)·8`) one row group at a time in plan order, and evicts after the last step. At most 16 reads are in flight. Prefetched data is not part of a checkpoint. Use a multi-thread tokio runtime so reads continue while Python runs the GPU step. After a failed read the loader returns `Stopped`; resume from `states()`.
- **Python:** PyO3 and maturin. Batches are exported through the Arrow PyCapsule interface. A PyTorch `IterableDataset` provides `state_dict()` and `load_state_dict()`. Users set `num_workers=0`, since parallelism lives in Rust and forking a process with a running tokio runtime is unsafe.

Still open:
- Not yet measured on real FineWeb-Edu lengths: packing waste (padding plus dropped leftover), how well documents mix within a sequence on sorted datasets, per-GPU batch statistics, and the window-size/memory trade-off.

### Invariants that must not break

- Use `rand_chacha` for every shuffle, never `StdRng`. `StdRng`'s algorithm can change between `rand` versions, which would silently break reproducibility. For the same reason, don't use `rand`'s range sampling (`random_range`, formerly `gen_range`, and `SliceRandom`) or `seed_from_u64`. Use `shuffle::shuffle`. It takes a `seed::Key`, which only `seed::key` can build, so every shuffle starts from fresh generator state and distinct `(seed, epoch, purpose)` inputs never share a stream.
- Each epoch yields every sample exactly once, apart from leftover tokens dropped at the lane cap.
- Lanes and their slices depend only on `(seed, epoch, B, W)`, never on `world`.
- Resuming at step k reproduces the tail of an uninterrupted run exactly, including when the GPU count changes.
- A saved reference output of the order must match on every OS.

These are enforced with tests: golden outputs pinned in CI on all 3 OSes, and `proptest` property tests in `src/order.rs` and `src/pack/tests.rs`. Any change to the ordering code must keep them passing.

### Known hard problem

Read amplification. If a shuffled batch is split across GPUs naively, every GPU downloads almost every row group in the window, about `world`× the data. Lane slices fix this. A GPU's lanes cover one contiguous stretch of storage, so it reads that stretch and roughly one shared row group at each edge. Simulation puts the cost at about 1 + world/W, or 1.01–1.25× at FineWeb-Edu-like sizes. A per-node cache is only a fallback if benchmarks show it's needed. Treat any change to slicing as a trade-off between shuffle quality and I/O.

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
