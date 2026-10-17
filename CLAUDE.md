# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Status

Plitzer is a Rust-core ML dataloader. It feeds Parquet data from local disk or S3 to PyTorch with an order that is deterministic and resumable. So far it is a library crate with one module, `src/shuffle.rs`: a deterministic Fisher-Yates shuffle. The rest of the design below is the plan, not existing code. Update this file as pieces land.

## Commands

```sh
cargo build
cargo test
cargo test <name>          # single test, substring match
cargo clippy --all-targets
cargo fmt
```

CI (`.github/workflows/ci.yml`) runs fmt, clippy with `-D warnings`, and tests on Linux, macOS and Windows. The cross-OS matrix is required because ordering output must match on every platform.

## Workflow

- One branch per feature, branched from `main` (`feat/<name>`, `fix/<name>`). Don't use git worktrees.
- Each change lands on `main` through a PR with green CI. Never commit directly to `main`.
- Keep the TODO list in `README.md` current: tick items off in the PR that completes them.

## Planned design

The pitch is MosaicML Streaming's guarantees, read directly from existing Parquet with no conversion step and no Python in the hot path. Initial scope is text and tabular data, mainly pre-tokenized LLM data. Images, WebDataset, format conversion and dataset mixing are out of scope for v1.

Pipeline: **index → order → read → batch → Python**

- **Index:** read only Parquet footers to build a list of `(file, row_group, num_rows)`. The result is cacheable.
- **Order:** a pure function of `(seed, epoch, position)`. Shuffle the row groups, group them into windows of W row groups, then shuffle rows within each window, seeded by `(seed, epoch, window)`. Nothing about the order is stored. A checkpoint is just those integers.
- **Splitting across GPUs:** the global batch size B is fixed. Step `s` covers positions `[sB, (s+1)B)`, and each GPU takes a contiguous `B/world` slice. That's why global batches stay the same when the GPU count changes. B must be divisible by `world`.
- **I/O:** `object_store` plus the `parquet` crate's async reader, on tokio. Prefetching is capped by bytes, not by number of batches.
- **Python:** PyO3 and maturin. Batches are exported through the Arrow PyCapsule interface. A PyTorch `IterableDataset` provides `state_dict()` and `load_state_dict()`. Users set `num_workers=0`, since parallelism lives in Rust and forking a process with a running tokio runtime is unsafe.

### Invariants that must not break

- Use `rand_chacha` for every shuffle, never `StdRng`. `StdRng`'s algorithm can change between `rand` versions, which would silently break reproducibility. For the same reason, don't use `rand`'s range sampling (`random_range`, formerly `gen_range`, and `SliceRandom`) or `seed_from_u64`. Use `shuffle::shuffle`, which takes a 32-byte seed so every shuffle starts from fresh generator state.
- Each epoch yields every sample exactly once.
- Resuming at step k reproduces the tail of an uninterrupted run exactly, including when the GPU count changes.
- A saved reference output of the order must match on every OS.

These are enforced with property tests. Any change to the ordering code must keep them passing.

### Known hard problem

Read amplification: every GPU's slice draws from the same window of row groups, so naively each GPU downloads every row group in the window. The planned v1 fix is a cache on each machine shared by its GPUs. Treat any change here as a trade-off between shuffle quality and I/O.

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
