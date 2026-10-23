# Plitzer engineering design

<!-- ste-allow: engineering, training, streaming, sampling -->

| Field | Value |
|---|---|
| Status | Active. It changes with each feature PR. The code is pre-alpha. |
| Last change | 2026-10-23, PR #14 |
| Language standard | ASD-STE100 (Simplified Technical English) |
| Code | `src/` at the commit of this file |

This document gives the design of Plitzer. It tells you what each component does, why it has its current design, and how the tests verify it. Each PR that adds or changes a feature also changes this document (refer to [section 10](#10-maintenance)).

Read sections 1 to 4 for an overview. Read section 5 for the design of one component. Sections 8 and 9 give the history of decisions and changes.

## Contents

1. [Purpose and scope](#1-purpose-and-scope)
2. [Terms](#2-terms)
3. [Requirements](#3-requirements)
4. [System overview](#4-system-overview)
5. [Component design](#5-component-design)
6. [Read amplification](#6-read-amplification)
7. [Open questions](#7-open-questions)
8. [Decision log](#8-decision-log)
9. [Change log](#9-change-log)
10. [Maintenance](#10-maintenance)

## 1. Purpose and scope

Plitzer is a dataloader for ML training. It reads Parquet data from local disk or S3 and supplies batches to PyTorch. The core is in Rust. Python code does not run in the hot path.

The order of the data is deterministic and resumable. The order is a pure function of a small set of integers. A checkpoint stores these integers and the state of each lane. It does not store the order.

Plitzer reads the Parquet files directly. A conversion step is not necessary. This is the primary difference from MosaicML Streaming, which has similar guarantees but needs its own file format.

**In scope for v1:**

- LLM text: documents that Plitzer packs into fixed-length sequences.
- Parquet on local disk and on S3.
- PyTorch through a Python `IterableDataset`.

**Out of scope for v1:**

- Images and WebDataset.
- Format conversion.
- Mixes of datasets.

## 2. Terms

Each term has only one definition in this document.

| Term | Definition |
|---|---|
| Document | One row of a Parquet file. Plitzer identifies it as `(row_group, row)`. |
| Row group | A Parquet row group. It is the smallest unit that Plitzer reads. |
| Index | The list of row groups, their row counts and the place of their token column chunks. The id of a row group is its position in the index. |
| Epoch | One full pass over the dataset. |
| Window | A set of W row groups. The documents of a window mix with each other. |
| W | The number of row groups in a window. |
| Lane | One sequence slot of the global batch. A lane reads documents and packs them into sequences. |
| B | The number of lanes. B is also the global batch size in sequences. |
| L | The length of one sequence, in tokens. |
| world | The number of GPUs. |
| rank | The position of one GPU, from 0 to world − 1. |
| Step | One global batch. Each lane supplies one sequence per step. |
| D | The number of documents that a lane takes at each step. |
| Leftover | The tokens that a lane took but did not put into a sequence yet. |
| C | The cap: the maximum number of leftover tokens that a lane keeps after a step. |
| Cursor | The number of documents that a lane took in the epoch before a step. |
| Lane state | The checkpoint of one lane: the next step and the position where its leftover starts, as `(step, doc, offset)`. |
| Pad token | The token that fills the end of a sequence that has too few tokens. |
| Key | The 32-byte seed of one shuffle. Only `seed::key` makes a key. |
| Read amplification | The bytes that one GPU reads, divided by the bytes of its own data. |
| Prefix | The folder or key prefix in the store below which the dataset files are. |
| Sidecar | The file that keeps the index between starts. |
| ETag | A version tag that the store gives to each file. It changes when the file changes. |
| Fingerprint | A 64-bit hash of the index without the ETags. |
| Budget | The most bytes of decoded tokens that the loader of one GPU keeps in memory. |
| Need plan | The list of row groups that one GPU needs, each with the first and last step that needs it. |
| Token column | The Parquet column that holds the tokens of each document as a list of integers. |
| Column chunk | The bytes of one column in one row group. |
| Codec | The compression method of a column chunk, such as Snappy or Zstd. |

## 3. Requirements

| Id | Requirement | Verification | Status |
|---|---|---|---|
| REQ-1 | The order is a pure function of `(seed, epoch, step)` and the index. | Golden tests in `shuffle`, `seed`, `order` and `pack`. | Met for the order and the packer |
| REQ-2 | Each epoch supplies each document exactly once. The only exception is leftover tokens that a lane drops at its cap. | `order::exactly_once` and `pack::exactly_once_tokens` (property tests). | Met for the order and the packer |
| REQ-3 | Lanes and their slices depend only on `(seed, epoch, B, W)`. They do not depend on world. | `Order` has no world input. `order::gpu_lanes_partition` (property test). | Met |
| REQ-4 | A resume at step k gives the same tail as a run without a stop. This is also true when world changes. | `pack::resume_matches_tail` and `pack::world_invariant` (property tests). Planned: end-to-end test. | Met in the packer |
| REQ-5 | The output is the same on Linux, macOS and Windows. | CI runs the golden tests on the three systems. | Met |
| REQ-6 | Read amplification is approximately 1 + world/W or less. | The reader reads no footer (`read::reads_without_footer`). Each GPU reads only its own slices (`load::reads_only_own_row_groups`). Planned: measurement on FineWeb-Edu. | Partly met |
| REQ-7 | A dependency upgrade does not change the order without a test failure. | Golden tests pin the shuffle output and the key bytes. | Met |
| REQ-8 | A start with an unchanged dataset reads no Parquet footers. | `index::open_uses_cache_without_footers`. | Met |
| REQ-9 | The decoded tokens in memory stay at or below the budget, and a budget that the loader accepts never stops it. | `load::budget_is_checked_and_kept`. | Met |

## 4. System overview

The pipeline has five stages. Each stage gives its output to the next stage.

```mermaid
flowchart LR
    index["Index<br/>Parquet footers"] --> order["Order<br/>windows and lanes"]
    order --> read["Read<br/>object_store"]
    read --> pack["Pack<br/>L-token sequences"]
    pack --> python["Python<br/>IterableDataset"]
    classDef done fill:#d4edda,stroke:#2e7d32,color:#1b5e20
    class index,order,read,pack done
```

| Component | Module | Status | Section |
|---|---|---|---|
| Shuffle | `src/shuffle.rs` | Implemented | [5.1](#51-shuffle) |
| Seed key | `src/seed.rs` | Implemented | [5.2](#52-seed-key) |
| Order | `src/order.rs` | Implemented | [5.3](#53-order) |
| Pack | `src/pack.rs` | Implemented | [5.4](#54-pack) |
| Index | `src/index.rs` | Implemented | [5.5](#55-index) |
| Checkpoint | — | Planned | [5.8](#58-planned-components) |
| Read | `src/read.rs` | Implemented | [5.6](#56-read) |
| Loader | `src/load.rs` | Implemented | [5.7](#57-loader) |
| Python bindings | — | Planned | [5.8](#58-planned-components) |

## 5. Component design

Each subsection has the same parts: function, interface, design, rationale, limits and verification.

### 5.1 Shuffle

**Function:** `shuffle::shuffle` puts a slice into a random order. The order depends only on the key.

**Interface:**

```rust
pub fn shuffle<T>(items: &mut [T], key: Key);
```

**Design:**

- The function makes a new `ChaCha8Rng` from the key for each call.
- It does a Fisher-Yates shuffle.
- The private function `below(rng, n)` gives a uniform integer in `[0, n)`. It uses rejection sampling. It rejects the lowest `2^64 mod n` values, so that each result has the same probability.

**Rationale:**

- `ChaCha8Rng` has a fixed algorithm. `StdRng` can change between `rand` versions (D-1).
- The range functions of `rand` can also change, so Plitzer has its own `below` (D-2).
- A new generator for each call makes each shuffle independent. With a shared generator, the order of one window would depend on all earlier windows (D-3).

**Limits:** `rand_chacha` promises a stable stream only within one minor version. The golden test finds a change of the stream.

**Verification:**

| Test | What it shows |
|---|---|
| `golden` | The output for a fixed key is the same on each OS. It also pins the rejection path of `below`. |
| `is_permutation` | Each item occurs exactly once after the shuffle. |
| `below_is_unbiased` | `below` has no bias toward small values. |
| `all_orders_equally_likely` | Each of the 6 orders of 3 items has the same frequency. This finds the off-by-one error that gives Sattolo's algorithm. |

### 5.2 Seed key

**Function:** `seed::key` makes the key for one shuffle. Different inputs always give different keys.

**Interface:**

```rust
pub enum Purpose { RowGroups, Lane { window: u64, lane: u32 } }
pub struct Key(/* private */ [u8; 32]);
pub fn key(seed: u64, epoch: u64, purpose: Purpose) -> Key;
```

**Design:** The function writes each input into fixed bytes, in little-endian order.

| Bytes | Field | Value for `RowGroups` |
|---|---|---|
| 0–7 | `seed` | `seed` |
| 8–15 | `epoch` | `epoch` |
| 16–19 | tag | 0 (`Lane` has tag 1) |
| 20–23 | `lane` | 0 |
| 24–31 | `window` | 0 |

**Rationale:**

- No two fields share a byte, so two different inputs cannot give the same key.
- The tag keeps `RowGroups` and `Lane { window: 0, lane: 0 }` apart.
- The field of `Key` is private, so other modules cannot make a key by hand (D-4).
- Clippy forbids `SeedableRng::from_seed` outside `shuffle`. Thus no code can make a generator from raw bytes (D-5).

**Limits:** The tag values are part of each saved order. Do not change them.

**Verification:**

| Test | What it shows |
|---|---|
| `golden` | All 32 bytes for inputs with different, nonzero bytes. A field that moves or overlaps changes the result. |
| `purposes_never_collide` | `RowGroups` and `Lane { 0, 0 }` give different keys. |

### 5.3 Order

**Function:** The order module tells each lane which documents to read in an epoch, and in which sequence. It also tells each GPU which lanes it owns.

**Interface:**

```rust
pub struct Doc { pub row_group: u32, pub row: u32 }

pub struct Order<'a> {
    pub seed: u64,
    pub epoch: u64,
    pub lanes: u32,      // B
    pub window: u32,     // W
    pub rows: &'a [u32], // row count of each row group
}

pub struct Window { /* private: index, groups, docs */ }

impl Order<'_> {
    pub fn windows(&self) -> Vec<Window>;
    pub fn lane(&self, window: &Window, lane: u32) -> Vec<Doc>;
    pub fn lane_len(&self, window: &Window, lane: u32) -> u64;
    pub fn lane_groups(&self, window: &Window, lanes: Range<u32>) -> Vec<Range<usize>>;
}
impl Window {
    pub fn groups(&self) -> &[u32];
}

pub fn gpu_lanes(lanes: u32, world: u32, rank: u32) -> Range<u32>;
```

**Design:** The module does these steps for each epoch:

1. Shuffle the ids of all row groups with `Purpose::RowGroups`.
2. Cut the result into windows of W row groups. The last window can have fewer.
3. Sort the row groups of each window into storage order.
4. Put the documents of the window in a line, in storage order. Call the count N.
5. Give lane `l` the positions `[l·N/B, (l+1)·N/B)`.
6. Shuffle the documents of each lane with `Purpose::Lane { window, lane }`.

GPU `rank` owns the lanes `[rank·B/world, (rank+1)·B/world)`.

`lane_len` gives the size of a lane slice without the shuffle of step 6. Each `Window` keeps its document count N, so `lane_len` does not add the row counts again.

**Example:** The index has 8 row groups with the row counts `[3, 1, 4, 1, 5, 9, 2, 6]`. W is 3 and B is 2. Seed 7, epoch 0 gives the windows `[1, 5, 6]`, `[2, 4, 7]` and `[0, 3]`. Window 0 has 1 + 9 + 2 = 12 documents. Each lane gets 6 documents:

```text
position   0    1    2    3    4    5  |  6    7    8    9    10   11
document  1/0  5/0  5/1  5/2  5/3  5/4 | 5/5  5/6  5/7  5/8  6/0  6/1
          ---------- lane 0 ---------- | ---------- lane 1 ----------
```

A document shows as `row_group/row`. Step 6 then shuffles each lane. Lane 0 reads `5/1, 1/0, 5/0, 5/4, 5/2, 5/3`.

**Rationale:**

- Step 3 is the fix for read amplification. A lane slice covers one stretch of storage. Thus a GPU reads only its own row groups and the row groups at its edges (section 6).
- The lane slices depend on B and W, but not on world. A change of world changes only which GPU runs a lane (D-7).
- Only `windows` makes a `Window`, and a `Window` holds its own index. Thus a caller cannot give the row groups of one window with the number of a different window. Two windows never share a lane key (D-8).
- The bounds of the slices use document counts, not token counts. Token balance is a problem for the Pack stage (D-10).
- The bounds use `u128` arithmetic, because `l·N` can be larger than `u64`.

**Limits:**

- `Order::windows` panics if `window` is 0. `Order::lane` panics if `lane` is not less than B. `gpu_lanes` panics if world does not divide B, or if `rank` is not less than world (D-11).
- A lane is empty if N is less than B.
- A `Window` keeps its document count. Use it only with the `Order` that made it.
- Row group ids and row numbers are `u32`. FineWeb-Edu has approximately 3 × 10⁶ row groups: approximately 2,900 files of approximately 1,000 row groups.
- The documents of a window mix only inside each lane. The mix quality on sorted datasets is not measured yet (section 7).

**Verification:**

| Test | Type | What it shows |
|---|---|---|
| `exactly_once` | Property | All lanes of all windows supply each document exactly once. |
| `windows_partition` | Property | Each row group is in one window. Each window is sorted. All windows have W row groups, except the last. |
| `slices_contiguous` | Property | The sorted lane slices, in lane order, make the window in storage order. The lane sizes differ by one or less. |
| `lane_len_matches` | Property | `lane_len` is equal to the length of `lane` for each window and lane. |
| `gpu_lanes_partition` | Property | The lanes of all GPUs cover `0..B` in order, with no overlap. |
| `gpu_lanes_rejects_uneven_world` | Unit | `gpu_lanes` panics if world does not divide B. |
| `golden` | Unit | The output above, and a hash over 500 row groups, is the same on each OS. |
| `epochs_differ` | Unit | The row group shuffle and the lane shuffle both change with the epoch. |

The property tests use small inputs. Thus empty row groups, empty lanes and short last windows occur frequently.

### 5.4 Pack

**Function:** The pack module makes one sequence of L tokens for each lane at each step. It also gives the state of each lane for a checkpoint, and makes a lane again from that state.

**Interface:**

```rust
pub trait Tokens { fn tokens(&self, doc: Doc) -> &[u32]; }

pub struct Packing {
    pub seq_len: u32, // L
    pub docs: u32,    // D
    pub cap: u32,     // C
    pub pad: u32,     // pad token
}

pub struct LaneState { pub step: u64, pub doc: u64, pub offset: u32 }

pub struct Epoch { /* private: a copy of the order inputs, packing, windows, steps */ }
pub struct Lane { /* private: Arc<Epoch>, ... */ }

impl Epoch {
    pub fn new(order: Order<'_>, packing: Packing) -> Self;
    pub fn order(&self) -> Order<'_>;
    pub fn windows(&self) -> &[Window];
    pub fn packing(&self) -> Packing;
    pub fn steps(&self) -> u64;
    pub fn lane_docs(&self, lane: u32) -> u64;
    pub fn lane(self: &Arc<Self>, lane: u32, state: LaneState, tokens: &impl Tokens) -> Lane;
}
impl Lane {
    pub fn fill(&mut self, tokens: &impl Tokens, out: &mut [u32]) -> u32;
    pub fn state(&self) -> LaneState;
}
```

**Design:** The documents of a lane in an epoch are its slices from `Order::lane`, window after window. `Lane::fill` does these steps:

1. Take the next D documents and add their tokens to the leftover.
2. Copy the first L tokens of the leftover into `out`. Fill the remainder of `out` with the pad token.
3. If the leftover has more than C tokens, remove the oldest tokens until C tokens remain.
4. Return the number of real tokens in `out`.

The cursor before step s is `min(s·D, lane_docs)`. It is a function of the step, so a checkpoint does not store it (D-14). The leftover is always the tokens from `(doc, offset)` to the cursor. Thus the lane state is sufficient to make the lane again.

`Epoch::lane` makes each lane, also at the start of an epoch (the default state, at step 0). It reads the documents from the lane state to the cursor again, and removes the first `offset` tokens.

The epoch has `ceil(max lane_docs / D) + ceil(C / L)` steps, or 0 steps if it has no documents. The last `ceil(C / L)` steps empty all leftovers. A lane with no tokens supplies a sequence of pad tokens.

**Example:** L is 4, D is 2 and C is 3. Lane 1 takes these documents, with these tokens:

```text
document   0        1           2           3          4
tokens     19 20    21 ... 27   13 ... 18   3 ... 7    (none)
```

| Step | Sequence | Dropped | Lane state after |
|---|---|---|---|
| 0 | 19 20 21 22 | 23 24 | (1, 1, 4) |
| 1 | 25 26 27 13 | 14 ... 18, 3, 4 | (2, 3, 2) |
| 2 | 5 6 7 pad | — | (3, 5, 0) |
| 3 | pad pad pad pad | — | (4, 5, 0) |

At step 0, the leftover has 9 tokens. The lane copies 4 tokens. Then it removes 23 and 24, so that C = 3 tokens remain.

**Rationale:**

- The cap removes the oldest tokens. Thus the leftover is one continuous run of tokens that ends at the cursor, and `(doc, offset)` can describe it (D-12).
- The last `ceil(C / L)` steps empty the leftovers. Thus the cap is the only cause of dropped tokens, and the epoch length does not depend on token counts (D-13).
- The pack module reads tokens through the `Tokens` trait. The loader implements it with the row groups in memory, and the tests use a corpus in memory (D-16).
- The lane state holds its step. Thus a caller cannot give the state of one step with a different step (D-17).
- `fill` writes into a buffer from the caller. The Python stage can then give one buffer of B/world × L tokens to each GPU, with no copy (D-15).
- Each `Lane` reads the windows of its `Epoch`. Thus the row group shuffle occurs one time for each epoch, not one time for each lane.

**Limits:**

- If D documents have many more than L + C tokens, the sequences are mostly the ends of documents. The cap removes the starts (section 7).
- The packer adds no tokens between documents. The `Tokens` source must add an EOS token if the model needs one.
- A document can have a maximum of `u32::MAX` tokens.
- A checkpoint is correct only with the same `Order` and `Packing` values. The checkpoint struct must store them (section 5.8).
- `Epoch::new` panics if L or D is 0. `fill` panics if `out` does not have L tokens, or after the last step. `Epoch::lane` panics if the step or the state is not possible (D-11).

**Verification:**

| Test | Type | What it shows |
|---|---|---|
| `resume_matches_tail` | Property | For each lane and step k, a lane made from the state at step k supplies the same sequences as the run without a stop. |
| `world_invariant` | Property | B is 12. A run with world a, a stop at step k and a resume with world b gives the global batches of world 1. |
| `exactly_once_tokens` | Property | The real tokens of each sequence are one continuous run of the lane stream, and the runs go forward. With a large C, all lanes together supply each token exactly once. |
| `cap_and_padding` | Property | Pad tokens occur only at the end. The leftover is never more than C. A short sequence occurs only with an empty leftover. At the end of the epoch, each lane state is `(lane_docs, 0)`. |
| `golden` | Unit | The example above, and a hash over 60 row groups, is the same on each OS. |
| `*_panics` | Unit | Each invalid input in the limits causes a panic. |

The property tests use small inputs. Thus empty documents, empty lanes, C = 0, C < L and L = 1 occur frequently.

### 5.5 Index

**Function:** The index module makes the list of all row groups of a dataset, their row counts and the place of their token column chunks. It reads only the Parquet footers. It keeps the list in a sidecar, so that a start with an unchanged dataset reads no footers.

**Interface:**

```rust
pub struct File {
    pub path: String,          // below the prefix, with "/" separators
    pub size: u64,
    pub e_tag: Option<String>,
    pub row_groups: Vec<RowGroup>,
}

pub struct RowGroup { pub rows: u32, pub start: u64, pub len: u32, pub values: u32, pub codec: Codec }

pub struct Index { /* private: column, schema, files, starts */ }

pub enum Error { Store(..), Footer { path, source }, TooLarge { path }, Column { path, reason }, NoFiles { prefix }, Cache(..) }

impl Index {
    pub async fn build(store: &dyn ObjectStore, prefix: &Path, column: &str) -> Result<Index, Error>;
    pub async fn open(store: &dyn ObjectStore, prefix: &Path, column: &str, cache: &std::path::Path) -> Result<Index, Error>;
    pub fn files(&self) -> &[File];
    pub fn column(&self) -> &str;
    pub fn schema(&self) -> &str;                              // the token column's schema, as text
    pub fn row_group(&self, row_group: u32) -> (&File, &RowGroup);
    pub fn rows(&self) -> Vec<u32>;                           // the input of Order::rows
    pub fn locate(&self, row_group: u32) -> (usize, usize);   // (file, row group in the file)
    pub fn fingerprint(&self) -> u64;
}
```

**Design:** `Index::build` does these steps:

1. List the files below the prefix through `object_store`. Keep only the `.parquet` files. If there are none, return `Error::NoFiles`.
2. Sort the files by path.
3. Read the last 4 MiB of each file in one request. Make sure that the footer length is not larger than the file.
4. Decode the footer with `ParquetMetaDataPushDecoder`. Do not read the page indexes. A footer larger than 4 MiB needs a second request.
5. Find the token column. It must be one list level of plain integers, with no date, time, timestamp or decimal type. Its schema must be the same in each file.
6. For each row group, keep the row count and the token column chunk: its first byte, length and codec.
7. Use the dictionary page as the first byte if its offset is from 4 to the data page offset. Some writers set the offset to 0 for no dictionary.

The module reads 16 files at the same time.

The id of a row group is its position in the index: files in path order, then row groups in file order. `locate` finds the file of an id with a binary search over the first id of each file.

`Index::open` lists the files and reads the sidecar. The sidecar must have the same column. If the path, size and ETag of each file are equal to the new list of files, `open` uses the sidecar and reads no footers. If not, it builds the index again and writes a new sidecar.

The sidecar has this layout. All integers are little-endian.

| Field | Type |
|---|---|
| Magic | 8 bytes, `PLTZIDX\0` |
| Version | `u32`, 3 |
| Column | `u32` length, then UTF-8 bytes |
| Schema | `u32` length, then UTF-8 bytes |
| File count | `u32` |
| For each file: path | `u32` length, then UTF-8 bytes |
| For each file: size | `u64` |
| For each file: ETag | `u8` flag (0 or 1), then a length and UTF-8 bytes if the flag is 1 |
| For each file: row groups | `u32` count, then for each row group: rows `u32`, start `u64`, length `u32`, values `u32`, codec `u8` |

A sidecar of an earlier version does not decode, so `open` builds the index again. A row group record has 21 bytes. The ~3 × 10⁶ row groups of FineWeb-Edu need approximately 63 MB, not the 9.4 GB of the footers. The value count gives the decoded size for the loader budget (section 5.7).

The index keeps the schema text of the token column with the root name `tokens`. The schema printer does not quote names, so a real name with a space would not parse back. The index parses the text before it keeps it.

The fingerprint is the 64-bit FNV-1a hash of the same layout, with no ETags and an empty schema.

**Rationale:**

- The footer push decoder does no I/O of its own. The async reader of `parquet` needs a reader object for each file (D-18).
- The index keeps the place of each token column chunk, so the reader needs no footer (D-24).
- The schema of the token column is the same in each file, so one schema decodes each chunk.
- The index uses `object_store` from the start. Thus local disk and S3 use the same code, and the footers of S3 are the expensive case (D-19).
- The list of files from local disk comes in directory order, which is different on each machine. The sort makes the ids the same on each machine.
- The paths are below the prefix and use `/`. Thus a copy of the dataset on S3 and on local disk gives the same index.
- The cache check compares the ETag, because a file can change and keep its size (D-20).
- The fingerprint does not include the ETag. A local ETag holds the inode and the modification time, which are different on each copy (D-21).
- The sidecar uses its own small format. It needs no `serde` dependency, and a golden test pins the bytes (D-22).
- If the sidecar does not exist or is corrupt, `open` builds the index again. This is not an error. The sidecar is only a cache (D-23).
- The module writes the sidecar to a temporary file with a unique name and then renames it. Thus the processes of all GPUs on one node can write at the same time, and no process reads a half-written sidecar.
- Windows does not replace a file while a different process has it open or replaces it. If the rename fails and the sidecar already has the same bytes, a different process wrote the same index, and the write is complete. If not, the module tries again up to five times.
- The first request reads 4 MiB, because a FineWeb-Edu footer is approximately 3.1 MiB. Thus most files need one request, not two.
- The module checks the footer length itself, because `parquet` 60 panics on a footer length that is larger than the file.
- A prefix with no `.parquet` files gives an error. An empty index for a wrong prefix gives no data and no error.

**Limits:**

- `open` reads and writes the sidecar with synchronous file calls. The sidecar is small, and `open` runs one time at the start.
- A row count or the number of row groups above `u32::MAX` gives `Error::TooLarge`.
- `locate` panics if the id is not in the index.
- The sidecar is valid only for the same prefix. Use one sidecar for each dataset.
- If the store gives no ETag, the cache check uses only the path and the size. S3 and local disk give ETags.
- A file name below the prefix that is not UTF-8 stops the build with `Error::Store`.
- The module does not flush the sidecar to the disk before the rename. After a power loss, a damaged sidecar only causes a new build.
- No test shows that `open` skips the page indexes. Code review checks it.

**Verification:**

| Test | Type | What it shows |
|---|---|---|
| `build_reads_row_groups` | Unit | The row groups of Parquet files in nested folders, in path order. Other files are not in the index. `locate` is correct at each file edge. |
| `open_writes_and_reuses_cache` | Unit | `open` writes the sidecar, and a second `open` gives the same index. |
| `open_uses_cache_without_footers` | Unit | A sidecar with false row counts that matches the list of files is used, so no footer is read. A different column or a stale ETag causes a new build. |
| `open_rebuilds_on_change` | Unit | A changed file, a new file and a corrupt sidecar each cause a new build. |
| `bad_footer_names_file` | Unit | A bad magic number, a file that is too short and a footer length larger than the file each give `Error::Footer` with the path. |
| `no_files_is_an_error` | Unit | A prefix with no `.parquet` files gives `Error::NoFiles`. |
| `concurrent_open_shares_cache` | Unit | Eight processes that open a cold cache at the same time all succeed, in ten rounds. |
| `fingerprint_golden` | Unit | The fingerprint and the hash of the sidecar bytes are the same on each OS. The fingerprint does not change with the ETag, but it changes with a row count. |
| `bad_schema_does_not_decode` | Unit | A sidecar with schema text that does not parse gives `None`. |
| `dictionary_offset_only_when_real` | Unit | A dictionary offset of 0, or an offset not before the data page, is not used. |
| `sidecar_roundtrip` | Property | Each index is the same after encode and decode. |
| `decode_never_panics` | Property | A truncated sidecar and random bytes give `None`, not a panic. |
| `locate_past_end_panics` | Unit | `locate` panics for an id after the last row group. |
| `old_sidecar_rebuilds` | Unit | A sidecar of version 2 does not decode, and `open` builds the index again. |

### 5.6 Read

**Function:** The read module gives the tokens of each row of one row group. It reads only the token column chunk of that row group, and never the footer of its file.

**Interface:**

```rust
pub struct Reader { /* private: store, prefix, schema */ }

impl Reader {
    pub fn new(store: Arc<dyn ObjectStore>, prefix: Path, index: &Index) -> Result<Reader, Error>;
    pub async fn read(&self, index: &Index, row_group: u32) -> Result<RowGroupTokens, Error>;
}

pub struct RowGroupTokens { /* private: offsets, values */ }

impl RowGroupTokens {
    pub fn rows(&self) -> usize;
    pub fn row(&self, row: usize) -> &[u32];
    pub fn bytes(&self) -> usize;   // memory of the tokens, for the prefetch budget
}

pub enum Error { Schema(..), Fetch { path, source }, Decode { path, row_group, source }, BadToken { path, row_group } }
```

**Design:** `Reader::read` does these steps:

1. Get the chunk record of the row group from the index: `start`, `len`, `codec` and `rows`.
2. Fetch the bytes `start..start + len`. Cut a chunk larger than 16 MiB into pieces of 16 MiB or less.
3. Fetch the pieces at the same time.
4. Make a `ParquetMetaData` with one row group and one column from the record and the schema in the index.
5. Decode the chunk with `ParquetPushDecoder` into one list array.
6. Make `RowGroupTokens` from the list offsets and the values.

The page reader of `parquet` uses only the codec and the byte range of the chunk from the metadata. It reads all other data from the page headers in the chunk. Thus the metadata that step 4 makes is sufficient.

`RowGroupTokens` keeps `Int32` and `UInt32` values with no copy. It copies other integer types into `u32`. A null row is empty.

**Rationale:**

- A window has row groups from approximately W different files. A new read of a 3.1 MiB footer for each 2 MB row group costs approximately 1.5 times the data. The index records each chunk, so the reader reads no footer (D-24).
- `ParquetPushDecoder` does no I/O of its own. The reader controls each request and knows its size before the fetch, which the prefetch budget needs (D-25).
- The input of v1 is a column of token lists. Tokenization in the loader is a later item (D-26).
- `object_store` does not cut a large range, and AWS advises requests of 8 to 16 MB at the same time. Each piece is a separate `get_ranges` call, so that `object_store` does not merge the pieces again (D-27).
- A negative token, a null token or a token larger than `u32::MAX` gives `Error::BadToken`. A null token would otherwise become token 0.

**Limits:**

- The reader reads one row group at a time. The loader reads many row groups at the same time (section 5.7).
- A list column has 32-bit offsets after the decode, also for a `large_list` column. One row group never has 2³¹ tokens.
- A column of more than one list level, or a column that is not a list of plain integers, gives an index error.

**Verification:**

| Test | Type | What it shows |
|---|---|---|
| `reads_every_variant` | Unit | Each token is correct for 48 files. They have four list types, null rows or not, three codecs, and a dictionary or not. The file name has a space and a percent sign. |
| `reads_without_footer` | Unit | The reader still gives each token after the footer length and magic of the file are set to zero. |
| `bad_columns_are_errors` | Unit | A nested list, text, dates, nanosecond timestamps, and an item name with a space each give `Error::Column`. |
| `missing_or_differing_columns_are_errors` | Unit | A column that is not there, and a column type that differs between files, each give `Error::Column`. |
| `odd_column_name_reads` | Unit | A column named `input ids` reads correctly. |
| `schema_text_golden` | Unit | The schema text of a `list<uint32>` column is pinned, so a change of the `parquet` printer shows. |
| `bad_tokens_are_errors` | Unit | A negative token and a null token each give `Error::BadToken`. |
| `fetch_splits_large_ranges`, `pieces_cover_range` | Unit | The pieces cover the range with no gap and no overlap, and each piece is 16 MiB or less. |
| `index_and_read_over_s3` | Integration | The index, the sidecar and the reader work against versitygw in the `s3` CI job. |

### 5.7 Loader

**Function:** The loader runs the lanes of one GPU. It reads the row groups of these lanes before the lanes need them, and it keeps the decoded tokens in memory below a byte limit. For each step, it gives one sequence for each lane.

**Interface:**

```rust
pub struct Loader { /* private: epoch, lanes, needs, buffer, arrivals, held, driver */ }

impl Loader {
    pub async fn new(epoch: Arc<Epoch>, reader: Arc<Reader>, index: Arc<Index>,
                     lanes: Range<u32>, states: &[LaneState], budget: u64) -> Result<Loader, Error>;
    pub async fn next(&mut self, out: &mut [u32]) -> Result<Option<Vec<u32>>, Error>;
    pub fn states(&self) -> Vec<LaneState>;   // the part of the checkpoint for this GPU
    pub fn resident_bytes(&self) -> usize;
}

pub enum Error { Budget { needed, budget }, Read(read::Error), Join(..), Stopped }
```

The caller gives `lanes = gpu_lanes(B, world, rank)`. Thus the loader does not know world.

**Design:** `Loader::new` makes a need plan for its lanes:

1. Find the row groups of each lane slice with `Order::lane_groups`.
2. The slice of a lane holds the stream positions `[p, p + len)`. The lane takes them at the steps `p / D` to `(p + len − 1) / D`.
3. Give each row group of the slice that step range. The lane shuffles its slice, so a document can come at each step of the range.
4. Merge the ranges of all lanes of the GPU. Each row group gets a first step and a last step.
5. Sort the row groups by first step.

On a resume, the plan starts at the lane state of each lane. The lane reads its leftover again at the resume step, so these documents are necessary at that step.

The cost of a row group is its decoded size: `values · 4 + (rows + 1) · 8` bytes. `peak` finds the largest cost of the row groups that are necessary at the same step. If the budget is less than this peak, `new` returns `Error::Budget`.

A driver task does these steps for each row group of the plan, in order:

1. Get the permits of its cost, in KiB, from a semaphore of the budget size.
2. Get one of 16 read slots.
3. Start a task that reads the row group. The task releases its read slot when the read ends.
4. Send the task to the loader.

The decoded row group keeps its permits. The loader releases the permits when it removes the row group from memory.

`Loader::next` does these steps:

1. Receive each row group with a first step at or before the current step.
2. Fill one sequence for each lane.
3. Remove each row group with this step as its last step.

**Rationale:**

- The plan uses one entry for each lane slice, not for each document. FineWeb-Edu has approximately 10⁹ documents, but only some thousand windows (D-31).
- The driver gets the permits of one row group at a time, in plan order. The semaphore of tokio is first in, first out, so a later row group cannot take the permits that an earlier row group needs (D-32).
- At step s, the loader holds the row groups that are necessary at s, and some that are read early. The first set is not more than the peak. Thus a budget at or above the peak cannot stop the loader (D-33).
- The cost comes from the value count of the chunk. The uncompressed size of a chunk with a dictionary is approximately half of the decoded size (D-34).
- The loader owns its epoch and its lanes. Thus `Lane` holds an `Arc<Epoch>`, and `Epoch` keeps a copy of the row counts (D-35).
- A checkpoint does not hold the row groups in memory. A resume reads them again. torchdata, MosaicML Streaming and LitData do the same (D-36).
- The loader stops all read tasks when it is dropped.
- The loader has a maximum of 16 reads at the same time. Without this limit, a large budget starts thousands of requests at the start. The index has the same limit for footers.
- If a read fails, `next` returns the error, and the next calls return `Error::Stopped`. The lanes did not change in the failed step, so `states()` is a correct point to resume from.

**Limits:**

- The budget does not include the memory of a decode in progress. Research measured approximately 3.5 times the decoded size of the row group during its decode.
- Use a multi-thread tokio runtime. On a single-thread runtime the loader is correct, but the reads progress only while the caller waits for `next`.
- `new` makes the plan synchronously. At FineWeb-Edu scale this takes approximately 0.1 s in a release build.
- The decode of a row group runs in a tokio task and blocks its worker thread for a few milliseconds.
- All lane states must have the same step.
- A plan for a resume can hold row groups that the lane read before the resume, if they are in the same slice.

**Verification:**

| Test | Type | What it shows |
|---|---|---|
| `loader_matches_direct` | Unit | For world 1, 2 and 4, the batches of all ranks are equal to a run over all row groups in memory. |
| `resume_matches_tail` | Unit | A stop at step k on world a and a resume on world b give the same batches as a run with no stop. |
| `budget_is_checked_and_kept` | Unit | A budget below the peak gives `Error::Budget`. A budget equal to the peak runs to the end, and the bytes in memory stay at or below the budget. |
| `evicts_everything` | Unit | After the last step, no row group is in memory. |
| `failed_read_stops_the_loader` | Unit | A corrupt chunk gives `Error::Read` one time and then `Error::Stopped`. The lane states stay at the step before the failure. |
| `prefetches_ahead` | Unit | With a large budget, the loader reads all row groups before `next` needs them. |
| `reads_only_own_row_groups` | Unit | The plans of all ranks cover all row groups. Ranks share only the row groups at the edges of their slices. |
| `plan_matches_lane_streams` | Unit | Each document that a lane takes at step s, also at a resume, is in a row group that the plan needs at s. |
| `lane_groups_match` | Property | `Order::lane_groups` gives the row groups that `Order::lane` uses. |

### 5.8 Planned components

These components are not in the code yet. Their sections will have the full set of parts when they are implemented.

- **Checkpoint:** A checkpoint holds `seed`, `epoch` and the lane state of each lane. It also holds B, W, L, D, C and `Index::fingerprint`. A resume with different values must stop with an error.
- **Python:** The bindings use PyO3 and maturin. Batches go to Python through the Arrow PyCapsule interface. A PyTorch `IterableDataset` gives `state_dict()` and `load_state_dict()`. Users must set `num_workers=0`. The parallel work is in Rust, and a fork of a process with a tokio runtime is not safe.

## 6. Read amplification

**Problem:** A naive split gives each GPU a random part of a shuffled batch. Each GPU then needs a document from almost each row group of the window. Each GPU reads approximately world times its own data.

**Fix:** A GPU owns a contiguous block of lanes (section 5.3). Each lane owns a contiguous stretch of the window in storage order. Thus the lanes of one GPU cover one stretch of storage. The GPU reads that stretch and approximately one shared row group at each edge.

**Estimate:** A simulation gives a read amplification of approximately 1 + world/W. At FineWeb-Edu sizes, the value is 1.01 to 1.25. A cache on each node is a fallback only. Plitzer will add it only if a benchmark shows that it is necessary.

**Trade-off:** A larger W gives a better mix of documents but uses more memory. A change to the slices changes both the shuffle quality and the I/O. Examine both before you change the slices.

## 7. Open questions

| Question | Effect |
|---|---|
| How many tokens does the Pack stage lose to pad tokens and dropped leftover, on real FineWeb-Edu lengths? What part of the sequences are ends of documents only (section 5.4)? | The values of C and D. |
| Must the loader add an EOS token between documents, or must the `Tokens` source do it? | The `Tokens` trait. |
| How well do documents mix inside one sequence on a sorted dataset? | The value of W. |
| What are the statistics of the batch of each GPU? | Possible bias between GPUs. |
| What is the best trade-off between W and memory? | The default W. |

## 8. Decision log

| Id | Decision | Reason | Rejected alternative | PR |
|---|---|---|---|---|
| D-1 | Use `ChaCha8Rng` from `rand_chacha` for each shuffle. | Its algorithm is fixed. | `StdRng`: its algorithm can change between `rand` versions. | #2 |
| D-2 | Use our own `below` for uniform integers. | The output cannot change with a dependency upgrade. | `rand` range sampling: it can change between versions. | #2 |
| D-3 | `shuffle` takes a key, not a generator. | Each shuffle is independent of the shuffles before it. | A shared generator: the order of one window depends on earlier windows. | #2 |
| D-4 | `Key` is a type that only `seed::key` can make. | The compiler stops hand-made keys that can collide. | A raw `[u8; 32]`. | #4 |
| D-5 | Clippy forbids `SeedableRng::from_seed` outside `shuffle`. | No code can make a generator without a `Key`. | A rule in CLAUDE.md only. | #4 |
| D-6 | Lane slices are the fix for read amplification. | Each GPU reads one stretch of storage, at 1.01–1.25×. | A cache on each node as the primary fix. | #4 |
| D-7 | Lanes depend only on B, never on world. | A resume with a different GPU count gives the same sequences. | A split of the batch by GPU. | #4 |
| D-8 | `Window` holds its own index, and only `Order::windows` makes one. | A caller cannot mix the number of one window with the row groups of a different window. | Separate `index` and `&[u32]` inputs to `lane`. | #6 |
| D-9 | Use `proptest` for the properties of the order. | The inputs (row counts, W, B, seed) have a real input space. | Fixed loops over a few cases. | #6 |
| D-10 | Lane slices have equal document counts, not equal token counts. | Simple and exact. Token balance is a Pack problem. | Slices by token count. | #6 |
| D-11 | Invalid configuration causes a panic. | It is a programmer error until the Python bindings exist. | `Result` errors now. | #6 |
| D-12 | The cap removes the oldest leftover tokens. | The leftover is one run to the cursor, so `(doc, offset)` is the full lane state. | Remove the newest tokens: documents stay whole, but the leftover gets gaps and the state becomes a list of ranges. | #8 |
| D-13 | Each epoch ends with `ceil(C / L)` steps that empty the leftovers. | Only the cap drops tokens, and the epoch length is known before the read. | Drop the leftover at the end of the epoch. | #8 |
| D-14 | Each step takes a fixed D documents. | The cursor is a function of the step, so a checkpoint does not store it. | Take documents until the sequence is full: the cursor then depends on token counts and must be in the checkpoint. | #8 |
| D-15 | `fill` writes into a buffer from the caller. | One buffer for each GPU batch, with no copy. | Return a new `Vec` for each sequence. | #8 |
| D-16 | The pack module reads tokens through the `Tokens` trait. | The Read stage does not exist yet. Tests use a corpus in memory, and the Read stage can implement the trait later. | A fixed token source, such as a `Vec` of all tokens. | #8 |
| D-17 | `LaneState` holds the step. | A state cannot go with a different step, which gives wrong sequences with no error. | A separate step input to `Epoch::lane`. | #8 |
| D-18 | Read footers with `ParquetMetaDataPushDecoder`. | It needs no `arrow`, and it does no I/O of its own. | `ParquetMetaDataReader`: its async API needs all of `arrow`. `ParquetObjectReader`: deprecated. | #10 |
| D-19 | Build the index over `object_store` from the start. | Local disk and S3 use the same code, and S3 is the expensive case. | Local files only, with S3 in the Read PR. | #10 |
| D-20 | The cache is valid if the path, size and ETag of each file are the same. | A file can change and keep its size. | Path and size only. | #10 |
| D-21 | The fingerprint does not include the ETag. | A local ETag holds the inode and modification time, so it is different on each copy. | A hash of the full sidecar. | #10 |
| D-30 | The fingerprint does not include the schema text. | The `parquet` printer makes the text, and an upgrade could change it. A changed fingerprint would refuse all saved checkpoints. | A hash of the schema text. | #12 |
| D-31 | The need plan uses the step range of each lane slice. | One entry for each slice, not for each document. | A list of the documents of each step. | #14 |
| D-32 | One driver gets the permits of each row group in plan order. | The tokio semaphore is first in, first out, so no later row group takes the permits of an earlier one. | One task for each row group that gets its own permits. | #14 |
| D-33 | The loader refuses a budget below the peak of the plan. | With such a budget the loader would wait for itself. | Wait and hope that memory becomes free. | #14 |
| D-34 | The budget cost is `values · 4 + (rows + 1) · 8` bytes. | It is the decoded size. | The uncompressed chunk size: approximately half of the decoded size with a dictionary. | #14 |
| D-35 | `Lane` holds an `Arc<Epoch>`, and `Epoch` copies the row counts. | The loader owns the epoch and the lanes together. | A lifetime, which does not let one struct own both. | #14 |
| D-36 | A checkpoint does not hold the prefetched row groups. | The lane state is sufficient, and a resume reads them again. | Save the buffers. | #14 |
| D-22 | The sidecar uses its own binary format and FNV-1a. | No new dependency, and the bytes are fixed by our code. | `serde` with `bincode`, or JSON. | #10 |
| D-23 | A sidecar that does not exist or is corrupt causes a new build. | The sidecar is only a cache. | An error that the user must correct. | #10 |
| D-24 | The index keeps the place of each token column chunk, and the reader reads no footer. | A window has row groups from approximately W files. A footer read for each row group costs approximately 1.5 times the data. | Read the footer of each file again, or keep footers in memory (9.4 GB for FineWeb-Edu). | #12 |
| D-25 | Decode with `ParquetPushDecoder` and metadata that the reader makes. | No I/O in the decoder, so the reader knows each request and its size. | `ParquetRecordBatchStreamBuilder` with an `AsyncFileReader`: it needs the footer metadata. | #12 |
| D-26 | The input of v1 is a column of token lists. | The simplest correct input. The reader takes `Int32` and `UInt32` with no copy. | Text with tokenization in the loader: a large dependency and more CPU load. It is a later item. | #12 |
| D-27 | Fetch a chunk in separate pieces of 16 MiB or less. | `object_store` does not cut a large range, and its default timeout is 30 s. AWS advises 8 to 16 MB requests at the same time. | One request for each chunk. | #12 |
| D-28 | Test S3 against versitygw in a Linux CI job. | MinIO stopped its images in October 2025. versitygw is Apache-2.0, active, and checks request signatures. | MinIO, LocalStack (needs a token since 2026), moto (no signature check). | #12 |
| D-29 | S3 support is the `s3` cargo feature. | The `aws` feature of `object_store` adds an HTTP and TLS stack, which the default build does not need. | S3 in the default build. | #12 |

## 9. Change log

| Date | PR | Change |
|---|---|---|
| 2026-10-17 | #2 | Add the deterministic shuffle (`src/shuffle.rs`). |
| 2026-10-19 | #4 | Add the seed key (`src/seed.rs`) and the `from_seed` ban. Change the planned design to lanes. |
| 2026-10-21 | #6 | Add the order (`src/order.rs`) and `proptest`. Add this document. |
| 2026-10-21 | #8 | Add the packer (`src/pack.rs`) and `Order::lane_len`. |
| 2026-10-22 | #10 | Add the index (`src/index.rs`), `object_store` and `parquet`. Increase the minimum Rust version to 1.88. |
| 2026-10-23 | #12 | Add the reader (`src/read.rs`) and the `s3` feature. Change the index to version 2, with the token column chunk of each row group. |
| 2026-10-23 | #14 | Add the loader (`src/load.rs`) and `Order::lane_groups`. Change the index to version 3, with the value count. `Lane` holds an `Arc<Epoch>`. |

## 10. Maintenance

Each PR that adds or changes a feature, a design or an interface must also change this document. Do these steps in the same PR:

1. Change the status table in section 4.
2. Change or add the component section in section 5. Keep the six parts.
3. Change the status of each requirement that the PR affects (section 3).
4. Add each new decision to section 8, with the rejected alternative.
5. Add one line to section 9.
6. Change the "Last change" field at the top.

Write in ASD-STE100:

- Write one topic in each paragraph, with six sentences or fewer.
- Write sentences with 20 words or fewer for instructions, and 25 words or fewer for descriptions.
- Use the active voice and simple tenses.
- Use each word with only one definition. Section 2 defines the terms.
- Write instructions as commands, with one instruction in each sentence.
- Code names, such as `Order::windows`, are technical names. You can use them as they are.
