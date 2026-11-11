# Changelog

All notable changes to plitzer. The project follows [Semantic Versioning](https://semver.org/); until 1.0, a minor version can change the API or the checkpoint format. A checkpoint from another format version is refused with `CheckpointError`, never misread.

## 0.1.0 (unreleased)

The first release.

- **Loader:** `plitzer.Loader` reads a pre-tokenized `list<int32>` or `list<uint32>` column of Parquet files from a local folder or an `s3://` URL, with no conversion step and no copy of the data. Each step gives one GPU's sequences of exactly `seq_len` tokens as an int32 array, through DLPack and the Arrow PyCapsule interface, with no copy.
- **Lossless packing:** lane slices are cut by tokens, every sequence is full, no token is dropped and nothing is padded. Lanes run on across epoch ends.
- **Deterministic, resumable order:** the order is a pure function of `(seed, epoch)` and the dataset, identical on Linux, macOS and Windows. `checkpoint()` gives versioned, checksummed bytes; `plitzer.merge` joins every GPU's part into a global checkpoint that resumes exactly on any GPU count.
- **PyTorch:** `plitzer.torch.Dataset` (the `torch` extra), an `IterableDataset` whose `state_dict()` gathers all ranks into one global checkpoint.
- **I/O:** a row-group index from the Parquet footers, cached in a sidecar; reads of one token column chunk with no footer; a prefetch capped by bytes, 16 reads in flight.
- **Benchmarks:** `plitzer.bench` (the `bench` extra): FineWeb-Edu-shaped data, read amplification, and a harness against MosaicML Streaming and Hugging Face `datasets` (the `baseline` extra).
