# plitzer

A deterministic, resumable dataloader for ML training that reads Parquet directly from local disk or S3. The core is written in Rust and exposed to Python.

Batch order is a pure function of `(seed, epoch, step)`. A checkpoint is three integers, resuming mid-epoch is exact, and global batches stay identical when the GPU count changes.

> **Status:** pre-alpha. Nothing usable yet.

## TODO

- [ ] **Benchmark baseline:** measure MosaicML Streaming and HF `datasets` on a FineWeb-Edu Parquet sample (conversion time, samples/s, CPU cores used, resume time)
- [ ] **Ordering core:** deterministic shuffle of row groups and windows with `rand_chacha`, with property tests:
  - [ ] each epoch yields every sample exactly once
  - [ ] resuming at step k matches the tail of an uninterrupted run
  - [ ] global batches are identical across GPU counts
  - [ ] a saved reference output matches on every OS
- [ ] **Index:** build a row-group index from Parquet footers, with caching
- [ ] **I/O:** async reads with `object_store` (local disk and S3, tested against MinIO), plus a prefetch queue capped by bytes
- [ ] **Python:** PyO3 and maturin bindings, a PyTorch `IterableDataset` with `state_dict()`, and Arrow PyCapsule export
- [ ] **End-to-end test:** kill training at a random step, resume on a different GPU count, and check the sample-stream hash matches
- [ ] **Node-local cache:** reduce read amplification from one copy per GPU to one copy per node
- [ ] **Benchmarks:** publish a comparison table against Mosaic and HF
- [ ] **Release:** build wheels in CI and publish to PyPI
- [ ] **Stretch:** in-loader tokenization and sequence packing

## License

Licensed under either [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.
