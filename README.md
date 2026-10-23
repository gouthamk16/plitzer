# plitzer

A deterministic, resumable dataloader for ML training that reads Parquet directly from local disk or S3. The core is written in Rust and exposed to Python.

Batch order is a pure function of `(seed, epoch, step)`. Documents are packed into fixed-length sequences by *lanes*, which depend on the batch size and never on the GPU count. Resuming mid-epoch is exact, and the sequences stay identical when the GPU count changes.

> **Status:** pre-alpha. Nothing usable yet.

The design, its rationale and the decision log are in [docs/design.md](docs/design.md).

## TODO

- [ ] **Benchmark baseline:** measure MosaicML Streaming and HF `datasets` on a FineWeb-Edu Parquet sample (conversion time, samples/s, CPU cores used, resume time)
- [x] **Ordering core:** shuffle row groups per epoch, split each window into lane slices, and shuffle each slice (`rand_chacha`). Property tests:
  - [x] each epoch yields every sample exactly once
  - [x] each lane's slice is one contiguous stretch of storage
  - [x] a saved reference output matches on every OS
- [x] **Lanes:** pack each lane's documents into fixed-length sequences, carry leftover tokens, and checkpoint each lane's state. Property tests:
  - [x] resuming at step k matches the tail of an uninterrupted run
  - [x] global batches are identical across GPU counts
- [x] **Index:** build a row-group index from Parquet footers, with caching
- [ ] **I/O:** async reads with `object_store` (local disk and S3, tested against versitygw), plus a prefetch queue capped by bytes
  - [x] read one row group's token column without its footer (local disk and S3)
  - [ ] prefetch each GPU's row groups for a window, capped by bytes
- [ ] **Python:** PyO3 and maturin bindings, a PyTorch `IterableDataset` with `state_dict()`, and Arrow PyCapsule export
- [ ] **End-to-end test:** kill training at a random step, resume on a different GPU count, and check the sample-stream hash matches
- [ ] **Read amplification check:** measure bytes read per GPU on real data. Lane slices simulate at 1.01–1.25×. Add a per-node cache only if it's needed
- [ ] **Benchmarks:** publish a comparison table against Mosaic and HF
- [ ] **Release:** build wheels in CI and publish to PyPI
- [ ] **Stretch:** in-loader tokenization

## License

Licensed under either [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.
