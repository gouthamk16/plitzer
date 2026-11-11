# Plitzer engineering design

<!-- ste-allow: engineering, training, streaming, sampling -->

| Field | Value |
|---|---|
| Status | Active. It changes with each feature PR. The code is pre-alpha. |
| Last change | 2026-11-11, PR #28 |
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
| S | The number of steps in an epoch: ⌈T/(B·L)⌉, where T is the number of tokens in the dataset. |
| Stream | The documents of one lane: its slices of the windows of epoch e, then of epoch e + 1, and so on. |
| Slot | One token of a row group, or one row with no token. The slots of a row group are the leaf value count of its token column chunk. |
| Position | The place of a lane in its stream, as `(epoch, window, token)`: after the first `token` tokens of its slice of that window. A checkpoint holds one position for each lane. |
| Key | The 32-byte seed of one shuffle. Only `seed::key` makes a key. |
| Read amplification | The bytes that one GPU reads, divided by the bytes of its own data. |
| Prefix | The folder or key prefix in the store below which the dataset files are. |
| Sidecar | The file that keeps the index between starts. |
| ETag | A version tag that the store gives to each file. It changes when the file changes. |
| Fingerprint | A 64-bit hash of the index without the ETags. |
| Budget | The most bytes of decoded tokens that the loader of one GPU keeps in memory. |
| Token column | The Parquet column that holds the tokens of each document as a list of integers. |
| Checkpoint part | The checkpoint of the lanes of one GPU. |
| Global checkpoint | A checkpoint that holds all B lanes. It resumes on each GPU count. |
| Column chunk | The bytes of one column in one row group. |
| Codec | The compression method of a column chunk, such as Snappy or Zstd. |
| DLPack | A C interface that lets one library use the tensor memory of another library without a copy. |
| Capsule | A Python object that holds a C pointer. DLPack and Arrow give their structs to Python in capsules. |

## 3. Requirements

| Id | Requirement | Verification | Status |
|---|---|---|---|
| REQ-1 | The order is a pure function of `(seed, epoch)` and the index, and the sequences are a pure function of the order and the tokens. | Golden tests in `shuffle`, `seed`, `order` and `pack`. | Met for the order and the packer |
| REQ-2 | Each epoch of the order holds each document exactly once. The sequences hold each token of each lane's stream, in order, and no pad token: no token is lost. | `order::exactly_once` and `pack::lossless` (property tests). | Met |
| REQ-3 | Lanes and their slices depend only on `(seed, epoch, B, W)` and the slots of the index. They do not depend on world. | `Order` has no world input. `order::gpu_lanes_partition` (property test). | Met |
| REQ-4 | A resume at step k gives the same tail as a run without a stop. This is also true when world changes. | `pack::resume_matches_tail` (property test) and `load::resume_matches_tail`. `session::resume_on_another_world_matches_one_run` reads Parquet files through checkpoint bytes. `test_kill_and_resume_on_another_world` and `test_kill_and_resume_with_torch` kill a training run of real processes at a random step and resume it on another number of processes (sections 5.11 and 5.12). | Met |
| REQ-5 | The output is the same on Linux, macOS and Windows. | CI runs the golden tests on the three systems. | Met |
| REQ-6 | Read amplification is 1 + ⌈n/W⌉(world − 1)/n or less, in row groups, for n row groups: approximately 1 + (world − 1)/W. | The reader reads no footer (`read::reads_without_footer`). Each GPU reads only the windows that its lanes reach, each one time (`load::each_window_is_read_once`). Measurement on FineWeb-Edu lengths (section 6). | Met |
| REQ-7 | A dependency upgrade does not change the order without a test failure. | Golden tests pin the shuffle output and the key bytes. | Met |
| REQ-8 | A start with an unchanged dataset reads no Parquet footers. | `index::open_uses_cache_without_footers`. | Met |
| REQ-9 | The reads ahead stay at or below the budget, and a budget that the loader accepts never stops it. | `load::one_window_of_budget_is_enough`, `load::a_budget_below_one_window_is_refused` and `load::passed_windows_are_evicted`. | Met |
| REQ-10 | A resume with different settings or a different dataset stops with an error that names the field. | `checkpoint::check_names_the_field` and `session::resume_refuses_other_runs`. | Met |
| REQ-11 | Python reads each batch without a copy, through DLPack and Arrow. | `test_dlpack_versioned_and_legacy`, `test_arrow_export` and `test_data_outlives_the_batch_and_the_loader`. | Met |
| REQ-12 | Input from Python gives a Python exception, never a panic or a wait with no end. | `test_bad_settings`, `test_resume_refuses_other_runs` and `test_forked_child_makes_its_own_loader`. Manual tests: Ctrl-C during a stopped S3 read gives `KeyboardInterrupt` in 0.51 s. A SIGUSR1 handler that calls `checkpoint()` during a stopped S3 read saves a complete checkpoint. | Met |
| REQ-13 | A PyTorch `state_dict` from each GPU count resumes on each GPU count. | `test_state_dict_gathers_all_ranks` (2 ranks save, 4 ranks and 1 rank resume) and `test_state_dict_resumes` (`torch.save` and `torch.load(weights_only=True)`). | Met |

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
| Pack | `src/pack.rs`, `src/pack/lane.rs`, `src/pack/pieces.rs` | Implemented | [5.4](#54-pack) |
| Index | `src/index.rs` | Implemented | [5.5](#55-index) |
| Read | `src/read.rs` | Implemented | [5.6](#56-read) |
| Loader | `src/load.rs`, `src/load/sequence.rs`, `src/load/prefetch.rs`, `src/load/buffer.rs` | Implemented | [5.7](#57-loader) |
| Checkpoint | `src/checkpoint.rs` | Implemented | [5.8](#58-checkpoint) |
| Session | `src/session.rs` | Implemented | [5.9](#59-session) |
| Store | `src/store.rs`, `src/store/s3.rs` | Implemented | [5.10](#510-store) |
| Python bindings | `plitzer-py/` | Implemented | [5.11](#511-python-bindings) |
| PyTorch dataset | `plitzer-py/python/plitzer/torch.py` | Implemented | [5.12](#512-pytorch-dataset) |
| Benchmark harness | `plitzer-py/python/plitzer/bench/` | Implemented | [5.13](#513-benchmark-harness) |
| Baseline harness | `plitzer-py/python/plitzer/bench/baseline.py`, `mosaic.py`, `hf.py` | Implemented | [5.14](#514-baseline-harness) |
| Packaging and release | `pyproject.toml`, `.github/workflows/release.yml` | Implemented | [5.15](#515-packaging-and-release) |

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

**Function:** The order module tells each lane which part of each window to read in an epoch, and in which sequence. It also tells each GPU which lanes it owns.

**Interface:**

```rust
pub struct Order<'a> {
    pub seed: u64,
    pub epoch: u64,
    pub lanes: u32,       // B
    pub window: u32,      // W
    pub slots: &'a [u32], // the slots of each row group: Index::slots
}

pub struct Window { /* private: index, groups, slots */ }
pub struct Part { pub row_group: u32, pub slots: Range<u64> }

impl Order<'_> {
    pub fn windows(&self) -> Vec<Window>;
    pub fn lane(&self, window: &Window, lane: u32) -> Vec<Part>;   // in storage order
    pub fn shuffle_lane<T>(&self, window: &Window, lane: u32, pieces: &mut [T]);
    pub fn lane_len(&self, window: &Window, lane: u32) -> u64;      // in slots
    pub fn lane_groups(&self, window: &Window, lanes: Range<u32>) -> Vec<Range<usize>>;
}
impl Window {
    pub fn groups(&self) -> &[u32];
}

pub fn gpu_lanes(lanes: u32, world: u32, rank: u32) -> Range<u32>;
```

**Design:** A row group has one slot for each token, and one slot for each row with no token. This is the leaf value count of its token column chunk, which the footer records, so the index gives it with no read. The module does these steps for each epoch:

1. Shuffle the ids of all row groups with `Purpose::RowGroups`.
2. Cut the result into windows of W row groups. The last window can have fewer.
3. Sort the row groups of each window into storage order.
4. Put the slots of the window in a line, in storage order. Call the count N.
5. Give lane `l` the slots `[l·N/B, (l+1)·N/B)`. `lane` gives them as one part for each row group.
6. The pack module makes the pieces of the slice from the rows (section 5.4). `shuffle_lane` shuffles them with `Purpose::Lane { window, lane }`.

GPU `rank` owns the lanes `[rank·B/world, (rank+1)·B/world)`.

A row that crosses the edge of two slices gives its first tokens to one lane and the rest to the next lane. Each `Window` keeps its slot count N, so `lane_len` does not add the counts again.

**Example:** The index has 8 row groups with the slot counts `[3, 1, 4, 1, 5, 9, 2, 6]`. W is 3 and B is 2. Seed 7, epoch 0 gives the windows `[1, 5, 6]`, `[2, 4, 7]` and `[0, 3]`. Window 0 has 1 + 9 + 2 = 12 slots. Each lane gets 6 slots:

```text
slot        0  |  1  2  3  4  5  |  6  7  8  9  |  10  11
row group   1  |  5  5  5  5  5  |  5  5  5  5  |  6   6
            ----- lane 0 -------    ----------- lane 1 -----
```

Lane 0 gets the parts `1: 0..1` and `5: 0..5`. Lane 1 gets `5: 5..9` and `6: 0..2`.

**Rationale:**

- Step 3 is the fix for read amplification. A lane slice covers one stretch of storage. Thus a GPU reads only its own row groups and the row groups at its edges (section 6).
- The lane slices depend on B, W and the slots of the index, but not on world. A change of world changes only which GPU runs a lane (D-7).
- Only `windows` makes a `Window`, and a `Window` holds its own index. Thus a caller cannot give the row groups of one window with the number of a different window. Two windows never share a lane key (D-8).
- The bounds of the slices use slots, not documents. Each lane takes L tokens at each step, so each lane must get the same share of the tokens. With slices of equal document counts, the tokens of the lanes differed by approximately 25 % on FineWeb-Edu lengths, and the lanes went apart by 3.9 windows in one epoch (D-86).
- The bounds use `u128` arithmetic, because `l·N` can be larger than `u64`.

**Limits:**

- `Order::windows` panics if `window` is 0. `Order::lane` panics if `lane` is not less than B. `gpu_lanes` panics if world does not divide B, or if `rank` is not less than world (D-11).
- A lane is empty if N is less than B.
- A row with no token has a slot but gives no token. Thus a lane whose slice has many empty rows gets fewer tokens.
- A `Window` keeps its slot count. Use it only with the `Order` that made it.
- Row group ids are `u32`. FineWeb-Edu has approximately 3 × 10⁶ row groups: approximately 2,900 files of approximately 1,000 row groups.
- The documents of a window mix only inside each lane. The mix quality on sorted datasets is not measured yet (section 7).

**Verification:**

| Test | Type | What it shows |
|---|---|---|
| `exactly_once` | Property | The slices of all lanes, in lane order, are the slots of the window in storage order: each slot is in one slice, and each slice is one stretch. |
| `windows_partition` | Property | Each row group is in one window. Each window is sorted. All windows have W row groups, except the last. |
| `slices_balanced` | Property | The slot counts of the lanes of a window differ by one or less. |
| `lane_len_matches` | Property | `lane_len` is equal to the slots of the parts of `lane`. |
| `lane_groups_match` | Property | `lane_groups` gives the row groups of the parts of `lane`. |
| `gpu_lanes_partition` | Property | The lanes of all GPUs cover `0..B` in order, with no overlap. |
| `gpu_lanes_rejects_uneven_world` | Unit | `gpu_lanes` panics if world does not divide B. |
| `golden` | Unit | The parts above, a lane shuffle, and a hash over 500 row groups are the same on each OS. |
| `epochs_differ` | Unit | The row group shuffle and the lane shuffle both change with the epoch. |

The property tests use small inputs. Thus empty row groups, empty lanes and short last windows occur frequently.

### 5.4 Pack

**Function:** The pack module makes one sequence of exactly L tokens for each lane at each step. It drops no token and adds no pad token. It also gives the position of each lane for a checkpoint, and makes a lane again from that position.

**Interface:**

```rust
pub struct Rows<'a> { pub values: &'a [u32], pub offsets: &'a [usize] }
pub trait Tokens { fn row_group(&self, row_group: u32) -> Option<Rows<'_>>; }  // None: not in memory

pub struct Position { pub epoch: u64, pub window: u32, pub token: u64 }
pub struct Piece { pub row_group: u32, pub start: usize, pub end: usize }   // tokens of `values`

pub struct Orders { pub seed: u64, pub lanes: u32, pub window: u32, pub seq_len: u32, /* slots, epochs */ }
pub struct Epoch { pub number: u64, /* the order inputs, windows */ }
pub struct Lane { /* private */ }
pub enum Need { Ready, Missing { slice: (u64, usize), row_group: u32 }, Bad(String), Empty(String) }

impl Orders {
    pub fn new(seed: u64, lanes: u32, window: u32, seq_len: u32, slots: &[u32]) -> Arc<Orders>;
    pub fn epoch(&self, number: u64) -> Arc<Epoch>;   // made once, shared by the lanes in it
}
impl Epoch {
    pub fn order(&self) -> Order<'_>;
    pub fn windows(&self) -> &[Window];
}
impl Lane {
    pub fn new(orders: &Arc<Orders>, lane: u32, at: Position) -> Result<Lane, String>;
    pub fn position(&self) -> Position;
    pub fn slice(&self) -> (u64, usize);                    // (epoch, window) the lane stands in
    pub fn need(&self, tokens: &impl Tokens) -> Need;       // what the next sequence needs; no change
    pub fn fill(&mut self, tokens: &impl Tokens, out: &mut [u32]);
}
```

**Design:** The stream of a lane is its slice of each window of epoch 0, then of each window of epoch 1, and so on. A position `(epoch, window, token)` is after the first `token` tokens of the lane's slice of that window.

The pieces of a slice are the rows of its parts. A row in a part gives one piece: its tokens in the part. A row that crosses an edge of the slice gives the piece of its tokens inside the slice. A row with no token gives no piece. `shuffle_lane` puts the pieces into the reading order. The lane needs the row offsets to make the pieces, so it makes them when all row groups of the slice are in memory, and keeps them while it is in the slice.

`Lane::fill` does these steps:

1. Copy the remaining tokens of the current piece into `out`.
2. Copy the next pieces, one after the other, until `out` holds L tokens.
3. Stop in the last piece at the token after the last one copied. That is the new position.

Each token is copied one time, from its row group to `out`, with one copy for each piece. A piece can end in one sequence and continue in the next. At the end of a slice, the lane goes to its slice of the next window, and after the last window, to the first window of the next epoch (D-79). It does not stop on a window where its slice has no slots.

`Lane::need` does the same steps without a change to the lane. It gives `Ready` if all row groups of the slices of the next sequence are in memory. Else it gives the first row group that is not in memory, and its window. The loader uses `need` before `fill`, so `fill` never waits (section 5.7).

`Orders::epoch` makes the windows of an epoch one time and shares them with each lane in that epoch. It also keeps the last 4 epochs, so that the loader does not make an epoch again when it releases a window of an epoch that the lanes left.

**Example:** L is 4. A slice has the pieces `a1 a2 a3`, `b1 ... b5` and `c1 ... c6` in reading order, 14 tokens.

| Step | Sequence | Position after |
|---|---|---|
| 0 | a1 a2 a3 b1 | (e, w, 4) |
| 1 | b2 b3 b4 b5 | (e, w, 8) |
| 2 | c1 c2 c3 c4 | (e, w, 12) |
| 3 | c5 c6, then the first 2 tokens of the next slice | (e, w + 1, 2) |

**Rationale:**

- Each step takes tokens until the sequence is full, not a fixed number of documents. A fixed number of documents cannot give L tokens on average: the leftover then grows or empties as a random walk, so a cap drops tokens or a lane pads. On FineWeb-Edu lengths with D = 2 and C = L = 2048, the fixed packer dropped 19.3 % of the tokens (D-77).
- The position counts tokens in a window. The index gives the slots of each slice, so `Lane::new` checks a position and finds its window with no read, and a resume reads only the slice of each lane (D-78).
- A lane goes on into the next epoch, so no lane waits or pads at the end of an epoch (D-79).
- `need` and `fill` are two calls. `need` finds each row group that the step needs while the lane does not change, so the loader can wait for reads and be cancel safe (D-80).
- `need` also checks the row groups of the pieces that the lane keeps. A row group can be in memory for a window of another epoch, and leave with that window.
- The pack module reads tokens through the `Tokens` trait. The loader implements it with the row groups in memory, and the tests use a corpus in memory (D-16).

**Limits:**

- The packer adds no tokens between documents. The `Tokens` source must add an EOS token if the model needs one.
- Each lane takes L tokens at each step, and gets its share of the slots of each window, to one slot. Empty rows and these rounding errors make the lanes move a little apart. This is much less than one window.
- A lane that finds no token in two whole epochs gives `Need::Empty`. This occurs only when a dataset has fewer tokens than the lanes need.
- A position with a token count up to the slots of its slice, but past its tokens, gives `Need::Bad` at the first step: the count of empty rows needs the tokens.
- `Orders::new` panics if B, W or L is 0. `fill` panics if `out` does not have L tokens, or if `need` did not give `Ready` (D-11).

**Verification:**

| Test | Type | What it shows |
|---|---|---|
| `lossless` | Property | The sequences of each lane, joined, are its stream: each token of its slices, in order, across epoch ends. No token is dropped and no pad token occurs. |
| `resume_matches_tail` | Property | For each lane and step k, a lane made from the position after step k supplies the same sequences as the run without a stop. |
| `lanes_stay_together` | Unit | Over row groups of short and long documents, 16 lanes for 2000 steps are always in the same window or the next. |
| `need_names_the_first_missing_row_group` | Unit | With one row group out of memory, `need` gives it, and `Ready` with all row groups. |
| `a_lane_with_no_tokens_is_an_error` | Unit | A lane with no slot, or with only empty rows, gives `Need::Empty`. |
| `bad_positions_are_refused` | Unit | A token count past the slots of the slice, a window past the epoch and a lane out of range give an error. A token count past the tokens of a slice with an empty row gives `Need::Bad`. |
| `golden` | Unit | Two lanes over a small corpus give their streams. A hash over 60 row groups is the same on each OS. |
| `wrong_out_len_panics` | Unit | An `out` of the wrong length causes a panic. |

The property tests use small inputs. Thus empty rows, empty lanes, documents longer than L, L = 1 and steps across many epochs occur frequently.

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
    pub fn rows(&self) -> Vec<u32>;                           // the row count of each row group
    pub fn slots(&self) -> Vec<u32>;                          // the input of Order::slots
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
- Before a build, `open` makes the cache folder and a file in it. It refuses a cache path that is a folder, has no file name or ends in a separator. It also refuses an existing file that does not start with the sidecar magic, such as a dataset file given as the cache by mistake: the write would replace it. The write renames over the old sidecar. On Unix this needs only the folder, so a stale sidecar that is read-only is still replaced. Windows refuses to rename over a read-only file, so on Windows `open` refuses a read-only sidecar before the build. The check changes no file, because another process can have just written a good sidecar. A cache that cannot be written then fails before the footer scan, which takes long on a large dataset (D-57). If the write after the build fails, such as on a full disk, `open` still returns the index: the next start builds it again.
- If the sidecar does not exist or is corrupt, `open` builds the index again. This is not an error. The sidecar is only a cache (D-23).
- The module writes the sidecar to a temporary file with a unique name and then renames it. On Unix the file gets mode 0644 less the umask, as a plain new file, so other users of a shared cache folder can read it. Thus the processes of all GPUs on one node can write at the same time, and no process reads a half-written sidecar.
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
| `unwritable_cache_fails_before_the_build` | Unit | A Parquet file given as the cache is refused and not changed. A cache below a file, a cache that is a folder, a path that ends in a separator and an empty cache path give `Error::Cache`, not the `Error::Footer` of a corrupt file in the dataset. The check does not change a stale sidecar. A missing cache folder is made. |
| `read_only_stale_sidecar_is_replaced` | Unit | A stale sidecar with mode 0444 in a writable folder is replaced by the build (Unix). The new sidecar is readable by the group and others, not 0600 as a temporary file is. |
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
    pub fn bytes_read(&self) -> u64;   // the chunk bytes fetched so far
}

pub struct RowGroupTokens { /* private: offsets, values */ }

impl RowGroupTokens {
    pub fn rows(&self) -> usize;
    pub fn row(&self, row: usize) -> &[u32];
    pub fn slots(&self) -> u64;     // a slot for each token, and one for each row with none
    pub fn view(&self) -> Rows<'_>; // values and offsets, for the packer
    pub fn bytes(&self) -> usize;   // memory of the tokens, for the prefetch budget
}

pub enum Error { Schema(..), Fetch { path, source }, Decode { path, row_group, source }, BadToken { path, row_group },
                 Slots { path, row_group } }
```

**Design:** `Reader::read` does these steps:

1. Get the chunk record of the row group from the index: `start`, `len`, `codec` and `rows`.
2. Fetch the bytes `start..start + len`. Cut a chunk larger than 16 MiB into pieces of 16 MiB or less.
3. Fetch the pieces at the same time.
4. Make a `ParquetMetaData` with one row group and one column from the record and the schema in the index.
5. Decode the chunk with `ParquetPushDecoder` into one list array.
6. Make `RowGroupTokens` from the list offsets and the values.

The page reader of `parquet` uses only the codec and the byte range of the chunk from the metadata. It reads all other data from the page headers in the chunk. Thus the metadata that step 4 makes is sufficient.

After step 3, `read` adds the length of the fetched chunk to an atomic counter. `bytes_read` gives the counter. A chunk that the reader fetches but cannot decode counts, because storage sent its bytes. A failed fetch does not count.

`RowGroupTokens` keeps `Int32` and `UInt32` values with no copy. It copies other integer types into `u32`. A null row is empty.

**Rationale:**

- A window has row groups from approximately W different files. A new read of a 3.1 MiB footer for each 2 MB row group costs approximately 1.5 times the data. The index records each chunk, so the reader reads no footer (D-24).
- `ParquetPushDecoder` does no I/O of its own. The reader controls each request and knows its size before the fetch, which the prefetch budget needs (D-25).
- The input of v1 is a column of token lists. Tokenization in the loader is a later item (D-26).
- `object_store` does not cut a large range, and AWS advises requests of 8 to 16 MB at the same time. Each piece is a separate `get_ranges` call, so that `object_store` does not merge the pieces again (D-27).
- A negative token, a null token or a token larger than `u32::MAX` gives `Error::BadToken`. A null token would otherwise become token 0.
- A chunk that decodes to a different number of rows than its footer gives `Error::Decode`.
- A chunk whose rows do not have the slots of its footer gives `Error::Slots`. The order cuts the slices by these slots, so a lane would otherwise find too few tokens.

**Limits:**

- The reader reads one row group at a time. The loader reads many row groups at the same time (section 5.7).
- A list column has 32-bit offsets after the decode, also for a `large_list` column. One row group never has 2³¹ tokens.
- A column of more than one list level, or a column that is not a list of plain integers, gives an index error.

**Verification:**

| Test | Type | What it shows |
|---|---|---|
| `reads_every_variant` | Unit | Each token is correct for 48 files. They have four list types, null rows or not, three codecs, and a dictionary or not. The file name has a space and a percent sign. |
| `reads_without_footer` | Unit | The reader still gives each token after the footer length and magic of the file are set to zero. |
| `bytes_read_counts_only_fetched_chunks` | Unit | A read adds the chunk length to `bytes_read`. A fetch of a deleted file gives `Error::Fetch` and adds nothing. |
| `bad_columns_are_errors` | Unit | A nested list, text, dates, nanosecond timestamps, and an item name with a space each give `Error::Column`. |
| `missing_or_differing_columns_are_errors` | Unit | A column that is not there, and a column type that differs between files, each give `Error::Column`. |
| `odd_column_name_reads` | Unit | A column named `input ids` reads correctly. |
| `schema_text_golden` | Unit | The schema text of a `list<uint32>` column is pinned, so a change of the `parquet` printer shows. |
| `bad_tokens_are_errors` | Unit | A negative token and a null token each give `Error::BadToken`. |
| `row_count_must_match_the_footer` | Unit | An index with 5 rows for a chunk of 3 rows gives `Error::Decode`. |
| `slots_must_match_the_index` | Unit | `RowGroupTokens::slots` is the value count of the index, and an index with one more gives `Error::Slots`. |
| `fetch_splits_large_ranges`, `pieces_cover_range` | Unit | The pieces cover the range with no gap and no overlap, and each piece is 16 MiB or less. |
| `index_and_read_over_s3` | Integration | The index, the sidecar and the reader work against versitygw in the `s3` CI job. |

### 5.7 Loader

**Function:** The loader runs the lanes of one GPU. It reads the row groups of these lanes before the lanes need them, and it keeps the decoded tokens in memory near the byte budget. For each step, it gives one sequence of L tokens for each lane.

**Interface:**

```rust
pub struct Loader { /* private: orders, lanes, buffer, arrivals, pending, demand, slowest, driver */ }

impl Loader {
    pub fn new(orders: Arc<Orders>, reader: Arc<Reader>, index: Arc<Index>,
               lanes: Range<u32>, at: &[Position], budget: u64) -> Result<Loader, Error>;
    pub async fn next(&mut self, out: &mut [u32]) -> Result<(), Error>;
    pub fn positions(&self) -> Vec<Position>;   // the part of the checkpoint for this GPU
    pub fn resident_bytes(&self) -> usize;
}

pub enum Error { Budget { needed, budget }, Read(read::Error), Join(..), Stopped, State(String), Empty(String) }
```

The caller gives `lanes = gpu_lanes(B, world, rank)`. Thus the loader does not know world.

**Design:** The read order of a GPU is the row groups that its lanes take from, window by window, epoch after epoch. The key of a read is `(epoch, window, position in the window)`. Keys grow along the order. A lane takes tokens at its own pace in documents, so the order does not say at which step a read is necessary (D-81).

A driver task does these steps for each window of the order:

1. Wait until the window is at most 2 windows after the window of the slowest lane, or until a step waits for a read in it.
2. For each row group of the window that a lane of the GPU takes from, and that a lane has not passed: get the permits of its cost, in KiB, from a semaphore of the budget size. If a step waits for this key or a later key, add the permits to the semaphore first, so that the read starts at once (D-82).
3. Get one of 16 read slots, and start a task that reads the row group.
4. Send the task and its key to the loader.

`Loader::next` does these steps:

1. Call `need` for each lane. If a document is not in memory, send its key to the driver as the demand, and take arrivals in order until that key is in memory. Then do step 1 again.
2. Fill one sequence for each lane. This step has no wait.
3. For each lane that left a window, remove the lane from the users of the row groups of its slice there. Remove each row group that has no user.

The loader keeps each arrival for the lanes whose slice in that window holds it and that have not passed the window. A user is `(lane, epoch, window)`. A row group can be in a window of each of two epochs at the same time, at the end of an epoch, so the users of both windows keep it (D-83).

`Loader::new` makes the lanes at their positions and starts the driver at the window of the slowest lane. There is no replay: the leftover of a lane is the rest of one document, and the first step reads it with the other row groups of the slice.

The cost of a row group is its decoded size: `values · 4 + (rows + 1) · 8` bytes. If the budget is less than one window of the lanes of the GPU, `new` returns `Error::Budget`.

**Rationale:**

- The budget caps the reads ahead, not the reads that a step waits for. Lanes can be a window apart, so a fixed budget could hold the row groups of the next window while a slow lane waits for one in the current window. The demand lets the read start, so the loader never waits for itself (D-82).
- The driver reads at most 2 windows after the slowest lane. The stream has no end, so a large budget would otherwise read whole epochs ahead.
- Users are per window, so a release removes only what an arrival added. A lane can pass a window through the copy of a row group from the other epoch, before the read of this window arrives. Then it holds no place in that window (D-83).
- `next` removes row groups only after all lanes are filled. `need` found each row group in memory for the whole step.
- `next` is cancel safe. The lanes change only after the last wait, and the loader keeps the read that it waits for, so a dropped `next` loses no row group (D-44).
- A checkpoint does not hold the row groups in memory. A resume reads them again (D-36).
- If a read fails, `next` returns the error, and the next calls return `Error::Stopped`. The lanes did not change in the failed step, so `positions()` is a correct point to resume from.

**Limits:**

- A step that waits for a read can take the memory past the budget, by the row groups between the last read and the read that the step needs.
- A lane makes the pieces of a slice only when all its row groups are in memory, and the lanes go into the next window at approximately the same step. With a budget of one window, each step into a new window waits for the reads of that window, and the memory is then approximately two windows. Give a budget of two windows or more.
- The budget does not include the memory of a decode in progress.
- Use a multi-thread tokio runtime. On a single-thread runtime the loader is correct, but the reads progress only while the caller waits for `next`.
- At the end of an epoch, a row group in the last window of the epoch and in a window of the next epoch is read two times when the two windows are not in memory at the same time.

**Verification:**

| Test | Type | What it shows |
|---|---|---|
| `loader_matches_direct` | Unit | For three seeds and world 1, 2 and 4, over about three epochs, the batches of all ranks are equal to the packer over all row groups in memory. |
| `resume_matches_tail` | Unit | A stop at step k on world a and a resume on world b from the positions give the same batches as a run with no stop. |
| `one_window_of_budget_is_enough` | Unit | With a budget of one window, world 1 and 2 run to the end in 20 s or less, with the same batches. Without the demand, this test stops. |
| `a_budget_below_one_window_is_refused` | Unit | A budget 1 KiB below one window gives `Error::Budget`. |
| `passed_windows_are_evicted` | Unit | After each step, each row group in memory is in a window from the slowest lane to 4 windows after the fastest lane. |
| `each_window_is_read_once` | Unit | The bytes that a GPU reads are at least the windows before its slowest lane and at most the windows to 2 after its fastest lane. |
| `failed_read_stops_the_loader` | Unit | A corrupt chunk gives `Error::Read` one time and then `Error::Stopped`. The positions stay at the step before the failure. |
| `a_bad_position_is_an_error` | Unit | A token count past the lane's slots gives `Error::State` that names the lane, when the loader is made. |
| `next_is_cancel_safe` | Unit | Over a store with 5 ms reads, `next` futures dropped at their first poll lose no row group. |
| `lane_groups_match` | Property | `Order::lane_groups` gives the row groups that `Order::lane` uses. |

### 5.8 Checkpoint

**Function:** A checkpoint holds the settings, the fingerprint, the epoch, the step and the positions of some lanes. It changes to bytes and back. It refuses a resume with different settings or a different dataset.

**Interface:**

```rust
pub struct Config { pub column: String, pub seed: u64, pub lanes: u32 /* B */,
                    pub window: u32 /* W */, pub seq_len: u32 /* L */ }

pub struct Checkpoint {
    pub config: Config,
    pub fingerprint: u64,     // Index::fingerprint
    pub epoch: u64,
    pub step: u64,            // the next step of the epoch, for all lanes
    pub first_lane: u32,      // lanes[i] is the position of lane first_lane + i
    pub lanes: Vec<Position>,
}

impl Checkpoint {
    pub fn to_bytes(&self) -> Vec<u8>;
    pub fn from_bytes(bytes: &[u8]) -> Result<Checkpoint, Error>;
    pub fn merge(parts: Vec<Checkpoint>) -> Result<Checkpoint, Error>;
    pub fn is_global(&self) -> bool;
    pub fn positions(&self, lanes: Range<u32>) -> Result<Vec<Position>, Error>;
    pub fn check(&self, config: &Config, fingerprint: u64) -> Result<(), Error>;
}

pub enum Error { Bytes(&'static str), Version(u32), Checksum,
                 Mismatch { field, saved, now }, Parts(String), Range(String) }
```

**Design:** The bytes have this layout. All integers are little-endian.

| Field | Type |
|---|---|
| Magic | `PLTZCKPT` (8 bytes) |
| Version | u32, now 2 |
| Column | u32 length, then UTF-8 |
| Seed | u64 |
| B, W, L | 3 × u32 |
| Fingerprint, epoch, step | 3 × u64 |
| First lane | u32 |
| Lane count | u32 |
| Each lane | epoch u64, window u32, token u64 |
| Checksum | u64, FNV-1a of all bytes before it |

Each lane has its own epoch: at the end of an epoch, some lanes are a few documents into the next epoch. Version 1 held the settings and the states of the fixed packer (D, C, pad, a step for each lane), and `from_bytes` refuses it with `Error::Version(1)` (D-84).

Each GPU gives a checkpoint part for its own lanes. `merge` puts the parts of all GPUs into one global checkpoint:

1. Make sure that all parts have the same config, fingerprint, epoch and step.
2. Sort the parts by first lane.
3. Make sure that the parts hold lanes 0 to B − 1, each lane one time.

`check` compares the settings and the fingerprint in a fixed order. It gives the name of the first field that is different.

**Rationale:**

- The checkpoint is bytes. `torch.load(weights_only=True)` accepts bytes, but not a custom class. Rust and Python use the same format (D-37).
- A global checkpoint resumes on each GPU count. Some trainers save the state of rank 0 only, so each rank must be able to make the global checkpoint (D-38).
- Different settings or a different dataset give different batches with no error. Thus a resume with them stops (D-39).
- The format has a version and a checksum, so a damaged or old checkpoint gives a clear error. `from_bytes` checks the checksum first (D-45).
- `from_bytes` does not reserve memory from the lane count in the bytes.

**Limits:**

- The checksum finds damage, not changes by an attacker.
- `from_bytes` does not make sure that each position is correct. `Session::resume` refuses a step past the epoch, a window past the epoch and a token count past the slots of the slice. A token count past the tokens of a slice with empty rows needs the tokens, so the first step gives that error as `load::Error::State`. Python gives both as `CheckpointError`. Any epoch is a place in the stream, and no checkpoint causes a panic.

**Verification:**

| Test | Type | What it shows |
|---|---|---|
| `golden_bytes` | Unit | The bytes of a fixed checkpoint are the same as bytes that Python's `struct` module made. |
| `bytes_round_trip` | Property | `from_bytes(to_bytes(c))` gives `c`. |
| `damaged_bytes_are_refused` | Unit | A changed bit, a changed version without a new checksum, a different version, no magic and too few bytes each give their error. |
| `any_bytes_give_an_error_not_a_panic` | Property | Random bytes never cause a panic, also with a correct checksum. |
| `truncation_is_refused` | Property | A checkpoint with bytes removed from its end gives an error. |
| `merge_tiles_the_lanes` | Unit | Parts in any order give one global checkpoint. |
| `merge_refuses_bad_parts` | Unit | No parts, a missing lane, a gap, an overlap, different steps, configs or epochs, and B = 2³² − 1 in the bytes each give `Error::Parts`, with no large allocation. |
| `positions_of_a_part` | Unit | `positions` gives lanes that the part holds, and refuses other lanes. |
| `check_names_the_field` | Unit | `check` gives the name of the field that is different. |

### 5.9 Session

**Function:** A session runs the lanes of one GPU from one epoch to the next. It opens the dataset, keeps one loader across epochs, and gives the checkpoint part of the GPU.

**Interface:**

```rust
impl Session {
    pub async fn open(store: Arc<dyn ObjectStore>, prefix: Path, cache: &std::path::Path,
                      config: Config, rank: u32, world: u32, budget: u64) -> Result<Session, Error>;
    pub fn set_epoch(&mut self, epoch: u64);
    pub async fn resume(&mut self, checkpoint: &Checkpoint) -> Result<(), Error>;  // makes the loader
    pub async fn next(&mut self, out: &mut [u32]) -> Result<bool, Error>;          // false: the epoch ended
    pub fn checkpoint(&self) -> Checkpoint;    // the part of this GPU
    pub fn ended(&self) -> bool;               // the last step is taken; `next` gives false next
    pub fn epoch(&self) -> u64;
    pub fn steps(&self) -> u64;                // S, the steps of each epoch
    pub fn lanes(&self) -> Range<u32>;
    pub fn bytes_read(&self) -> u64;           // the chunk bytes fetched since `open`
    pub fn bytes_counter(&self) -> Arc<AtomicU64>;
}

pub enum Error { Config(String), Empty, Index(index::Error), Read(read::Error),
                 Load(load::Error), Checkpoint(checkpoint::Error) }
```

**Design:**

- `open` checks the config, opens the index with `Index::open`, and makes the reader and the `Orders`. The lanes are `gpu_lanes(B, world, rank)`. A dataset with no tokens gives `Error::Empty`.
- An epoch has S = ⌈T/(B·L)⌉ steps, where T is the value count of the token column in the index. All GPUs get the same S from the index (D-79).
- `open` refuses more than 2²⁴ lanes, and more than 2³⁰ tokens in one step of one GPU. These sizes are far past any real batch, and a larger value could only use all the memory (D-63).
- `next` makes the loader at the first step, and keeps it across epochs. After S steps, `next` returns false, and the session goes to step 0 of the next epoch. The lanes keep their positions.
- `set_epoch(e)` right after the last step of epoch e − 1 goes to epoch e, and the lanes go on. A later epoch is a jump: the lanes start at the first window of e. An earlier epoch or the current epoch changes nothing.
- `resume` takes the positions from a checkpoint after `check`, refuses a step past the epoch, and makes the loader, which refuses a position past the lane's slots. It drops the old loader first and keeps its positions, and changes the session only when the new loader is made.
- If `next` fails, the session keeps the positions of the loader and drops the loader. The next call makes a new loader and tries the same step again.
- `checkpoint` gives the positions of the loader. After the last step of an epoch, it gives `(epoch + 1, step 0)`, also before the call of `next` that returns false.

**Rationale:**

- After the last step of an epoch, a checkpoint holds `(epoch + 1, step 0)`. The parts of all GPUs agree, also if only some GPUs received the end (D-40).
- Trainers call `set_epoch` at the start of each epoch, also the epoch that they resumed into. A checkpoint after the last step of an epoch is at the next epoch. Thus `set_epoch` only goes forward, and a resume never repeats data (D-43). `__iter__` of the Python `Loader` calls it to start the next epoch, so a call at the end of an epoch must not move the lanes.
- The Python bindings give the config to `open`. Thus `open` returns `Error::Config` for a config that would cause a panic (D-41).
- A failed read leaves the positions before the failed step. Thus a retry can continue (D-42).

**Limits:**

- All GPUs must call `next` the same number of times, as in each data-parallel run.
- T counts one value for each empty row too. Thus S is a little high for a dataset with many empty documents, and an epoch then takes a little more than one pass.

**Verification:**

| Test | Type | What it shows |
|---|---|---|
| `resume_on_another_world_matches_one_run` | Unit | Runs on world a stop at step k, also at and after the end of an epoch. Their merged checkpoint resumes on world b. The batches are the same as one run on world 1 with no stop. |
| `epochs_roll_over` | Unit | After S steps, `ended` is true, and the checkpoint holds `(1, 0)` before and after the end. The lanes keep their positions across the end. Parts from a GPU that received the end and a GPU that did not merge. |
| `set_epoch_starts_that_epoch` | Unit | `set_epoch(1)` after some steps gives the same batches as a session that jumps to epoch 1 at once, and these are not the batches of a run that arrives at epoch 1. |
| `set_epoch_keeps_a_resume_in_that_epoch` | Unit | `set_epoch(0)` after a resume in epoch 0 keeps the resumed position. |
| `set_epoch_never_goes_back` | Unit | `set_epoch(0)` after a resume from the end of epoch 0 gives the batches of epoch 1. |
| `next_is_cancel_safe` | Unit | Over a store with 5 ms reads, calls of `next` that are dropped after one poll give the same batches as a run with no drops. |
| `first_step_after_a_resume_is_cancel_safe` | Unit | Over a store with 20 ms reads, the first `next` after a resume, dropped after one poll, leaves `checkpoint()` unchanged, and the next batches are the same as a run with no stop. |
| `resume_refuses_other_runs` | Unit | A different L, a different seed and an added file each give `Mismatch` with the name of the field. |
| `refused_resume_keeps_the_epoch_end` | Unit | After the last step, a refused `resume` leaves `ended` true, and the next `next` gives false. |
| `resume_needs_this_ranks_lanes` | Unit | A part resumes the rank that holds its lanes, and another rank refuses it. |
| `resume_refuses_positions_outside_the_run` | Unit | A step past the epoch and a token count past the lane's slots give `checkpoint::Error::Range` that names the lane, and `checkpoint()` is unchanged, also on a new session. |
| `open_refuses_bad_settings` | Unit | A world that does not divide B, a rank out of range, no GPUs, W = 0, L = 0, 2²⁵ lanes and L = 2³² − 1 each give `Error::Config`. |
| `next_retries_after_a_failed_read` | Unit | After a failed read and a repair of the file, the next calls give the same batches as a run with no failure. |
| `bytes_read_one_gpu_is_about_the_dataset_an_epoch` | Unit | `bytes_read` is 0 before the first step, and after e epochs it is between e − 1 and e + 1 datasets. |

### 5.10 Store

**Function:** The store module changes a dataset location into an `object_store` store and a prefix.

**Interface:**

```rust
pub struct Location { pub store: Arc<dyn ObjectStore>, pub prefix: Path, pub id: String }

pub fn open(location: &str, options: &[(String, String)]) -> Result<Location, Error>;

pub enum Error { Store(object_store::Error), Scheme(String), Option(String) }
```

**Design:**

| Location | Store | Prefix | `id` |
|---|---|---|---|
| Text with no `://`, such as `data/`, `C:\data` or `runs:2024/data` | `LocalFileSystem::new_with_prefix(path)` | empty | The canonical folder, in `Debug` form, so names that are not UTF-8 stay apart |
| `file:///data` | The same, for the path of the URL | empty | The same |
| `s3://bucket/prefix` (`s3` feature) | `AmazonS3Builder::from_env()` with the URL, then each option | the path of the URL | The bucket, the prefix and the endpoint |
| Other schemes | `Error::Scheme` | — | — |

**Rationale:**

- `object_store::parse_url_opts` does not read the `AWS_*` environment variables. `store` reads them as `from_env` does, and the options override them (D-49). An empty variable counts as unset, as in the AWS CLI. The builder prefers the S3-only endpoint, such as `AWS_ENDPOINT_URL_S3`, to the general one. Thus an `endpoint` option is also set as the S3-only endpoint, unless the options give that endpoint too.
- An unknown option gives `Error::Option`. If the store ignored an option, a wrong endpoint or a wrong key name would fail later with an unclear error.
- The S3 builder fails only on settings, such as a missing bucket. Thus its error is also `Error::Option`, and Python shows it as `ConfigError`.
- A local folder takes no options, so an option for a local folder is an error. A location that is a file, not a folder, is also an error.
- `open` refuses an S3 endpoint that is not an `http://` or `https://` URL, that has spaces, or that holds a user or a password. It also refuses a key, a token or a region with a control character, such as a newline at the end of a file. These values come from the options or the environment. `object_store` stops with a panic on the first request for the first ones, and it repeats the endpoint in its errors.
- Only text with `://` is a URL. `C:\data` and `runs:2024/data` parse as URLs, but they are local paths.
- A URL with `?` or `#`, and an `s3://` URL with a port or a user, give `Error::Option`. These characters can be part of a folder or key name, and the URL parser would drop what follows them, so another dataset would open (D-62). An error message never holds the user, the password, the query or the fragment of a URL, because they can hold keys.
- `id` is the same for each spelling of one dataset, and different for two datasets. It holds no keys or tokens, so new credentials keep the same cache. It holds the endpoint from the options or the environment, the S3-only endpoint first as in the builder, so two servers with the same bucket name get different caches (D-58).

**Limits:**

- Only local disk and S3. Google Cloud Storage and Azure need their `object_store` features and a row in the table above.

**Verification:**

| Test | Type | What it shows |
|---|---|---|
| `opens_local_folders` | Unit | A path, the path with a separator at the end, and a `file://` URL open the same folder with the same `id` (the canonical path). Another folder has another `id`. |
| `a_colon_in_a_path_is_not_a_scheme` | Unit | A folder named `runs:2024` opens as a folder (POSIX). |
| `an_empty_variable_is_unset` | Unit | An empty `AWS_ENDPOINT_URL` is unset, and the location opens. |
| `refuses_settings_that_would_panic` | Unit | An endpoint with no scheme, a space at the end, no text, another scheme or a password, also from the environment, and a token with a newline each give `Error::Option`, with no secret in the message. |
| `s3_id_has_the_endpoint_and_no_keys` | Unit | Two secret keys give one `id`, and two endpoints give two. The `id` holds no key. An `endpoint` option wins over an S3-only endpoint from the environment. |
| `refuses_what_it_cannot_open` | Unit | A missing folder, an unknown scheme, an option for a local folder, and URLs with `?` or `#` each give their error. No error repeats a password or a query. |
| `refuses_unknown_s3_options` | Unit | An unknown S3 option and a URL with no bucket give `Error::Option`. |
| `open_an_s3_url` | Integration | An `s3://` URL with options opens versitygw with the correct prefix. |

### 5.11 Python bindings

**Function:** The `plitzer` Python package runs one `Session` for each Python `Loader`. It gives each step to Python as int32 arrays without a copy, through DLPack and the Arrow PyCapsule interface.

**Interface (`plitzer/_plitzer.pyi`):**

```python
class Loader:
    def __init__(self, location: str | os.PathLike[str], *, seed: int, lanes: int, window: int,
                 seq_len: int, budget: int, rank: int | None = None, world: int | None = None,
                 column: str = "tokens", cache: str | os.PathLike[str] | None = None,
                 storage_options: Mapping[str, str | bool] | None = None) -> None: ...
    def __iter__(self) -> Loader: ...           # the rest of the current epoch
    def __next__(self) -> Batch: ...
    def set_epoch(self, epoch: int) -> None: ...
    def checkpoint(self) -> bytes: ...          # the part of this GPU
    def resume(self, checkpoint: bytes | bytearray) -> None: ...
    epoch: int; lanes: range; bytes_read: int   # read-only properties
class Batch:  tokens: Array                     # shape (lanes, seq_len): each row is full
class Array:  shape; __dlpack__(...); __dlpack_device__(); __arrow_c_array__(...)
def merge(parts: Sequence[bytes | bytearray]) -> bytes: ...
class PlitzerError(Exception); ConfigError, CheckpointError, DataError (PlitzerError)
```

**Design:**

- Each process has one multi-thread tokio runtime with at most 8 worker threads and 32 threads for blocking calls. A loader has at most 16 reads in flight, and each GPU has its own process, so one worker for each CPU would start hundreds of threads in each process (D-60). A forked child makes a new runtime, and it never drops the copy of the parent's runtime: that drop would wait for threads that the child does not have.
- Each `Loader` records the PID of the process that made it. In a forked child, a copied `Loader` gives `RuntimeError` for each method and property: a parent thread can have held one of its locks at the fork. A new `Loader` in the child works.
- The `Loader` is frozen. A lock holds its session for the whole of a step, and a snapshot holds the checkpoint before the next step. `checkpoint()`, `epoch` and `lanes` read only the snapshot, so a signal handler or another thread can read them during a step. A second call that needs the session during a step gets `RuntimeError`, never a deadlock (D-59). No call holds the snapshot lock while Python runs, because Python can release the GIL and `save` takes the lock while it holds the GIL.
- `epoch` is the epoch of the next step, as in `checkpoint()`. After the last step of an epoch, it is the next epoch.
- `bytes_read` reads the counter of the reader, `Session::bytes_counter`, with no lock. Thus it also counts the reads ahead, while they arrive, and a thread can read it during a step (D-85).
- `__next__` releases the GIL and waits on the runtime for 100 ms at a time. Between these waits it checks for signals. Thus Ctrl-C gives `KeyboardInterrupt` in 100 ms or less, and it drops the `next` future. `Loader::next` is cancel safe (D-44), so the next call does the same step again.
- At the end of an epoch, `__next__` raises `StopIteration`, and it raises it again at each later call, as a Python iterator must. `__iter__` starts the next epoch: a new `for` loop gives it. `__iter__` also moves on a loader whose last step is taken but which did not raise `StopIteration` yet, such as the second loader in a `zip` (D-55).
- Each step writes into a new `Vec<u32>`. An arrow `Buffer` holds it, and each export holds a clone of that `Buffer`. Thus an exported array stays valid after the batch, the loader and the export call.
- The default cache is `$PLITZER_CACHE_DIR`, else the user cache folder, then `plitzer/<hash>.idx`. The hash is of the `id` of the store (section 5.10) and the column, not of the location text. Thus `data` from two folders gets two files, and `data/` and `file:///…/data` share one (D-54). The index makes the cache folder if it does not exist. `HOME`, `XDG_CACHE_HOME` and `LOCALAPPDATA` count only when they hold an absolute path: a relative folder would differ for each working directory. `PLITZER_CACHE_DIR` is used as given. With no user cache folder, there is no default, and `Loader` gives `ConfigError` (D-61).
- `rank` defaults to 0 and `world` to 1. A `world` above 1 with no `rank` gives `ConfigError`: rank 0 on each GPU would train all GPUs on the same lanes with no error.
- Integer settings, also the epoch of `set_epoch`, come in as Python objects. Each takes an `int` or another object with `__index__`, such as `np.int64` (the stub types them as `SupportsIndex`). A value that is not an integer, such as `4.0` or `True`, or that does not fit, such as `world=-1`, gives `ConfigError`.
- `resume` and `merge` take `bytes` or `bytearray`.
- A storage option name is a `str`, and a value is a `str` or a `bool` (also `np.bool_`), which becomes `true` or `false`. Another type gives `ConfigError`.
- A local path that is not UTF-8 becomes a `file://` URL, which percent-encodes its bytes.
- If the runtime cannot start its threads, such as at a process limit, `Loader` gives `RuntimeError`. A call from a thread that runs a tokio runtime, such as a callback from another Rust extension, also gives `RuntimeError`, because tokio cannot block that thread.

The errors map to Python exceptions as follows:

| Rust error | Python exception |
|---|---|
| `session::Error::Config`, `load::Error::Budget`, `index::Error::Cache`, `store::Error::Scheme`, `store::Error::Option`, an integer setting that is not an integer or is out of range, no cache folder | `ConfigError` |
| `checkpoint::Error`, `session::Error::Checkpoint` | `CheckpointError` |
| `store::Error::Store`, index, read and other load errors | `DataError` |

The DLPack export is as follows:

| Item | Value |
|---|---|
| Device | CPU, `(1, 0)` |
| Data type | int32 (code 0, 32 bits, 1 lane) |
| Strides | Set, C order, in elements |
| `max_version` ≥ (1, 0) | Capsule `dltensor_versioned`, `DLManagedTensorVersioned` 1.0 |
| No `max_version` | Capsule `dltensor`, `DLManagedTensor` |
| `copy=True` | A copy of the data, with the flag `IS_COPIED` |
| `dl_device` not CPU | `BufferError` |

A consumer takes a DLPack capsule when it changes the capsule name to `used_dltensor` or `used_dltensor_versioned`. It then calls the deleter of the tensor. The capsule destructor calls the deleter only if the name did not change (D-48).

The Arrow export is as follows:

| Array | Arrow type |
|---|---|
| `tokens` | `FixedSizeList<int32, L>`, with the extension `arrow.fixed_shape_tensor` and the metadata `{"shape":[L]}` |

An Arrow consumer moves the structs out of the capsules and sets their `release` to null. The capsule then frees only an empty struct.

**Rationale:**

- Both DLPack ABIs: numpy 2 and current torch ask for the versioned ABI, and older consumers call `__dlpack__()` with no arguments (D-46).
- int32: torch has little support for `uint32`, and vocabularies are much smaller than 2³¹ (D-47).
- Each row is L real tokens, so a batch has no `lengths` and the loader has no `pad` (D-77).
- `set_epoch(e)` opens the iterator for epoch `e` when the loader stands at the start of `e`, also just after `StopIteration`. Thus a trainer that calls `set_epoch` and then `next` for each epoch gets each epoch. The epoch after `u64::MAX` is `u64::MAX`: the count stops there and never wraps to 0.
- The runtime has all its drivers. The S3 client needs the I/O driver, and the waits need the timer.
- The PID check stops a forked child before it waits for runtime threads that it does not have (D-50).
- The 100 ms slices let Ctrl-C work and cost one timer for each slice (D-51).

**Limits:**

- One wheel for each platform covers CPython 3.11 and later (abi3). It does not cover the free-threaded build.
- During a step, only `checkpoint()`, `epoch` and `lanes` work from another thread or a signal handler. `__next__`, `__iter__`, `set_epoch` and `resume` give `RuntimeError` until the step ends.
- If a step stops with a panic, the `Loader` gives `RuntimeError` for each later step: its lanes can be part way through the step. `checkpoint()` still gives the last good state, so a new `Loader` can resume from it.
- The loader checks the budget against the largest window of the epoch where it starts (section 5.7). A later epoch can have a larger window: then the steps that wait for a read take the memory past the budget, but the run does not stop. A jump with `set_epoch` makes a new loader, which can give `ConfigError` with the bytes that are necessary. The budget is not part of a checkpoint, so a resume with a larger budget continues the run.
- The data is writable. Two exports of one array share the same memory.
- Signals are checked only on the main thread, as in Python.
- A `KeyboardInterrupt` can come after `__next__` returns and before the caller gets the batch, as with each Python iterator. Then `checkpoint()` is already after that step, and a resume does not give its batch again. Save checkpoints between training steps, for example from a flag that a signal handler sets, not in an `except KeyboardInterrupt` block around the loop.
- In a forked child, a `Loader` leaks its session when Python frees it. A drop would stop runtime tasks, and a lock that a parent thread held at the fork stays held (D-56).

**Verification:**

| Test | Type | What it shows |
|---|---|---|
| `test_ranks_split_the_global_batch` | pytest | Two loaders on world 2 give the batches of one loader on world 1. |
| `test_iteration_ends_each_epoch` | pytest | A `for` loop ends at the end of each epoch, and the next loop gives the next epoch. |
| `test_set_epoch_after_stop_iteration` | pytest | `set_epoch(e)` and `next` until `StopIteration`, for two epochs, give the batches of both epochs. |
| `test_epoch_is_the_next_step` | pytest | After the last step of an epoch, `epoch` is 1 before and after `StopIteration`. |
| `test_checkpoint_during_a_step` | pytest | Another thread reads `checkpoint()` and `epoch` during all the steps of an epoch with no error. Each checkpoint it reads is complete. |
| `test_iterator_protocol` | pytest | After the end of an epoch, `next` raises `StopIteration` again. The next `for` loop gives the next epoch. |
| `test_non_strict_zip_keeps_ranks_together` | pytest | Two epochs through `zip` (not strict) of two world-2 loaders give the batches of one world-1 loader. |
| `test_batch_shape` | pytest | The shape and the int32 type are correct, and each row is full. |
| `test_dlpack_versioned_and_legacy` | pytest | Both ABIs give the same data, with the correct capsule names. |
| `test_unused_capsules_are_freed` | pytest | 300 000 capsules that nobody takes are freed: the peak memory grows less than 16 MiB (Linux). |
| `test_no_default_cache_without_a_user_folder` | pytest | With no user cache folder and no `cache`, `Loader` gives `ConfigError`. |
| `test_default_cache_is_made_and_shared` | pytest | With no `cache`, the index makes its cache folder. Three spellings of one folder use one cache file. |
| `test_data_outlives_the_batch_and_the_loader` | pytest | DLPack and Arrow data stay correct after the batch and the loader are deleted. |
| `test_copy_and_device` | pytest | `copy=True` gives separate memory. A device that is not the CPU gives `BufferError`. |
| `test_arrow_export` | pytest | pyarrow reads `tokens` as a `FixedShapeTensorArray`. |
| `test_resume_on_another_world` | pytest | Parts from world 2, merged, resume on world 4 with the same tail as one run. |
| `test_resume_refuses_other_runs` | pytest | A different L, bad bytes and an incomplete merge give `CheckpointError`. |
| `test_resume_refuses_a_hand_made_checkpoint` | pytest | Checksummed bytes with a token count or a window past the lane's slots give `CheckpointError` at `resume`, and `checkpoint()` is unchanged. A lane at epoch 2⁶⁴ − 1 gives no panic. |
| `test_bad_settings` | pytest | Bad settings, `world=-1`, `seed=2**200`, `lanes=4.0`, `seq_len=True`, a URL with `#` and a cache below a file give `ConfigError`. A missing folder gives `DataError`. |
| `test_kill_and_resume_on_another_world` | pytest, end to end | Three times with a new seed: a run of 1 to 6 processes is killed at a random step of 3.5 epochs, one time in three at the end of an epoch. Each rank saves its own part at each step. After the kill, the newest step that all ranks saved is merged and resumed on another number of processes, as a real trainer does. The batch of each step and lane, before and after the kill, is the same as one process with no stop. Each rank must end with the exit code of the kill. `PLITZER_E2E_SEED` repeats a seed. |
| `test_resume_with_the_fresh_budget` | pytest | On world 1 and 2, a resume at each step runs with the smallest budget of a fresh run, and 1 KiB less gives `ConfigError`. |
| `test_forked_child_makes_its_own_loader` | pytest | In a child of `os.fork`, the copied loader gives `RuntimeError`. A new loader gives the same first batch as the parent. The child frees the copy with no wait. |
| `test_s3_matches_local` | pytest | The same files on versitygw and on local disk give the same batches. |

### 5.12 PyTorch dataset

**Function:** `plitzer.torch.Dataset` is a PyTorch `IterableDataset` for one rank. Each item is one step of that rank. `state_dict()` gives a global checkpoint, and `load_state_dict()` resumes from it on each GPU count.

**Interface (`plitzer/torch.py`):**

```python
class TokenBatch(NamedTuple):
    tokens: torch.Tensor     # shape (lanes, seq_len), int32
class LoaderSettings(TypedDict): ...  # the keywords of plitzer.Loader, without rank and world
class State(TypedDict):
    plitzer: bytes           # a global checkpoint
class Dataset(IterableDataset[TokenBatch]):
    def __init__(self, location: str | os.PathLike[str], *,
                 group: ProcessGroup | None = None, **settings: Unpack[LoaderSettings]) -> None: ...
    def __iter__(self) -> Iterator[TokenBatch]: ...   # the rest of the current epoch
    def set_epoch(self, epoch: SupportsIndex) -> None: ...
    def state_dict(self) -> State: ...               # a collective over group
    def load_state_dict(self, state: Mapping[str, object]) -> None: ...
    def part(self) -> bytes: ...                     # the part of this rank, not a collective
    epoch: int; lanes: range                         # read-only properties
```

**Design:**

- `Dataset` makes one `plitzer.Loader`. Its rank and world are the rank and size of `group`, the data-parallel group. If `group` is `None`, they come from the default process group. Without `torch.distributed`, the rank is 0 of 1.
- `Dataset` gives `ConfigError` if `WORLD_SIZE` is more than 1 but `torch.distributed` is not initialized, and for a process that is not in `group`. In the first case, each rank would read all lanes and train on the same data, with no error.
- `__iter__` and `state_dict()` make sure that the world size of `group` did not change. A `Dataset` made before `init_process_group`, with no `WORLD_SIZE`, would read all lanes on each rank.
- `__iter__` gives the batches of the loader. `torch.utils.dlpack.from_dlpack` changes each array to a tensor without a copy.
- `state_dict()` gets the checkpoint part of each rank of `group` with `all_gather_object`. Then `plitzer.merge` makes one global checkpoint. Each rank gets the same bytes. With one rank, there is no gather.
- `load_state_dict()` makes sure that the state is `{"plitzer": bytes}`, else it gives `CheckpointError`. Then it calls `Loader.resume`. A global checkpoint resumes on each GPU count, so the GPU count can change.
- `__iter__` in a DataLoader worker gives `RuntimeError`. A pickle of the dataset, for a worker that starts with spawn, gives the same error.

**Rationale:**

- A global checkpoint needs the state of each lane. A rank knows only its own lanes, and the position of a lane depends on its tokens. Thus `state_dict()` must collect the parts from all ranks (D-66).
- Lightning makes the checkpoint on each rank, and then one rank writes it. `torch.distributed.checkpoint` writes an item that each rank holds with the same key only one time. Thus a collective `state_dict()` with the same bytes on each rank works with both (D-66).
- `TokenBatch` is a `NamedTuple` of tensors. `DataLoader` with `batch_size=None` keeps a `NamedTuple` (D-67).
- The state is the position of the next batch that the dataset gives, as for other iterable datasets. It does not guess how many batches the loop holds (D-68).
- The tokens stay int32, as in section 5.11. `nn.Embedding` takes int32 indices.
- The state is bytes, so `torch.load(weights_only=True)` reads it (D-37).
- `plitzer/__init__.py` does not import `plitzer.torch`. Thus `import plitzer` does not load torch. torch is the optional extra `plitzer[torch]`.

**Limits:**

- Each rank of `group` must call `state_dict()`. If only rank 0 calls it, rank 0 waits with no end. Checkpoint code that gathers the parts itself can use `part()` and `plitzer.merge`.
- Use `num_workers=0`. Each item is already a batch, so use `DataLoader(dataset, batch_size=None)`.
- The dataset splits the lanes by rank. Do not let a framework split it again, such as `accelerator.prepare` on its `DataLoader`.
- With NCCL, `all_gather_object` uses the current CUDA device. Call `torch.cuda.set_device` on each rank before `state_dict()`.
- With `pin_memory=True`, the tensors are pinned copies, not views of the export.
- A `Dataset` cannot be pickled or copied, because it holds Rust threads.
- A training loop that holds batches that it did not train yet, such as a loop that prefetches one batch, skips those batches on a resume. Issue #22 is for a fix that knows the loop, such as a Lightning callback.
- A `KeyboardInterrupt` after a step can lose that batch, as in section 5.11. Save the state between training steps.

**Verification:**

| Test | Type | What it shows |
|---|---|---|
| `test_matches_the_loader` | pytest | For two epochs, the tensors are equal to the arrays of `plitzer.Loader`, with int32 and the correct shapes. |
| `test_dataloader_passes_batches_on` | pytest | `DataLoader(batch_size=None, num_workers=0)` gives the same `TokenBatch` items. `num_workers=1` gives `RuntimeError`. |
| `test_state_dict_resumes` | pytest | A state at step 0, at the middle and after the last step, saved with `torch.save` and read with `torch.load(weights_only=True)`, resumes with the same batches as a run with no stop. |
| `test_bad_settings` | pytest | `WORLD_SIZE=2` without `torch.distributed` gives `ConfigError`. A `WORLD_SIZE` that is not a count is ignored. |
| `test_load_state_dict_refuses_bad_states` | pytest | A state with no `plitzer` key, with text, or with damaged bytes gives `CheckpointError`. A group without `torch.distributed` gives `ConfigError`. |
| `test_state_dict_gathers_all_ranks` | pytest | Two gloo ranks give the same state. Four ranks and one rank resume from it with the batches of a run with no stop. |
| `test_kill_and_resume_with_torch` | pytest, end to end | The same kill and resume as `test_kill_and_resume_on_another_world`, over gloo. Each rank calls `state_dict()`, and rank 0 saves it with `torch.save`. The resume reads it with `torch.load(weights_only=True)`. A rank whose gather fails because a peer is killed can end with its own exit code, `PEER_DIED`. Each other end before the kill is a failure. |

### 5.13 Benchmark harness

**Function:** `plitzer.bench` holds the benchmark harnesses. `fineweb` writes a dataset with the shape of FineWeb-Edu. `amplification` measures the bytes that each GPU reads in one epoch. They need the `bench` extra (numpy and pyarrow).

**Interface (`plitzer/bench/`):**

```python
# fineweb: python -m plitzer.bench.fineweb OUT --files 4 --row-groups 128
class RangeFile(io.RawIOBase): ...    # a read-only file over HTTP range requests
def lengths(source: str | Path, row_groups: int) -> Iterator[NDArray[int64]]: ...
def write(groups: Iterator[NDArray[int64]], path: Path, seed: int) -> None: ...
def build(sources: Sequence[str | Path], out: Path, row_groups: int, seed: int = 0) -> list[Path]: ...

# amplification: python -m plitzer.bench.amplification DATA --lanes 512 --window 64 --worlds 1,8,64 ...
class Settings(NamedTuple): lanes; window; seq_len; budget; seed = 0; column = "tokens"
class World(NamedTuple): window; per_gpu: list[int]; dataset: Dataset   # .amplification, .bound
class Dataset(NamedTuple): row_groups: int; bytes: int
def scan(location: Path, column: str) -> Dataset: ...
def measure(location: Path, settings: Settings, worlds: Sequence[int], cache: Path) -> list[World]: ...
def table(results: Sequence[World]) -> str: ...   # Markdown
```

**Design:**

- FineWeb-Edu has a `token_count` column, the GPT-2 tokens of each document, but no token column. `lengths` reads only the footer and the `token_count` chunks of a source file. For a URL, pyarrow reads through `RangeFile`, which makes one HTTP range request for each read. `RangeFile` tries a request again after a connection error or a server error (5xx), up to 4 times. It uses `urllib`, which uses the `HTTPS_PROXY` setting.
- `write` makes one row group for each source row group, with the same documents and lengths. The tokens are random in [0, 50257), from a seeded generator (D-69). It writes a temporary file and renames it, so a failed build leaves no file. It refuses a row group of 2³¹ tokens or more, which int32 offsets cannot hold.
- `measure` runs each rank of each world in turn in one process, for epochs 0 and 1, and records the `Loader.bytes_read` of epoch 1. The prefetch reads a few windows past the slowest lane, also into the next epoch, so epoch 1 counts these reads one time at each end, as each epoch of a long run does. After each epoch, `measure` waits until `bytes_read` does not change for 50 ms: the reads ahead stop at a fixed point after the lanes. All ranks use one index cache. The amplification is the sum of the ranks' bytes divided by `Dataset.bytes` from `scan`: the column chunk bytes of each row group that has rows, from the Parquet metadata.
- `World.bound` is 1 + (world − 1)/W. The lanes of the GPUs split each window into stretches in storage order, and two neighbours share at most the row group at their edge. Thus the GPUs read at most W + world − 1 row groups of a window. The bound counts row groups, not bytes.

**Rationale:** CLAUDE.md puts the harness in the package, with tests, not in a script. A dataset with real lengths gives the real sizes of row groups and chunks, which are the only inputs of the read amplification. A tokenizer run on 10B tokens is not necessary for this.

**Limits:**

- `scan` reads local files only.
- Random tokens compress worse than real tokens. The chunks are larger, by approximately the same ratio in each row group, so the amplification does not change.
- `measure` runs the ranks one after the other, so it does not measure time.

**Verification:**

| Test | Type | Checks |
|---|---|---|
| `test_bytes_read_counts_each_epoch` | pytest | `bytes_read` is 0 before the first step, and after e epochs it is between e − 1 and e + 1 datasets. |
| `test_measure_runs_every_rank` | pytest | `measure` runs each rank of 1, 2 and 4 GPUs, and each amplification is 1 or more. |
| `test_measure_even_row_groups` | pytest | Row groups of equal rows: when world divides W, the amplification of epoch 1 is exactly 1. With half a row group for each lane, 4 GPUs read exactly 2 times the dataset. |
| `test_scan_and_bound` | pytest | `scan` counts the row groups. The bound for 512 row groups, W = 100 and 64 GPUs is 1 + 6 · 63/512. |
| `test_main_prints_a_row_for_each_run` | pytest | The CLI prints one table row for each W and world. |
| `test_build_keeps_lengths_and_row_groups` | pytest | A built file has the source's row groups and lengths, and int32 tokens below the vocabulary size. The same seed gives the same file. |
| `test_build_from_a_url_reads_ranges` | pytest | A build from a local HTTP server that serves only ranges keeps the lengths. No request touches the text chunk of the first row group. |
| `test_range_file_refuses_a_file_url` | pytest | A `file:` URL gives `OSError`. |
| `test_build_retries_a_server_error` | pytest | A build succeeds after 3 responses of 503. After 4, `RangeFile` gives the `HTTPError`. |
| `test_write_refuses_a_row_group_past_int32` | pytest | A row group of 2³¹ tokens gives `ValueError`, and no file. |

### 5.14 Baseline harness

**Function:** `plitzer.bench.baseline` measures how other loaders read the Parquet files that plitzer reads. For each loader, it measures:

- the conversion;
- one shuffled epoch (documents and tokens each second, CPU cores used);
- a resume at the middle of the epoch.

The resume measurements are the wait for the first batch, whether the rest of the epoch is the same as an uninterrupted run, and how many documents the resume loses. The adapters need the `baseline` extra (torch, `mosaicml-streaming`, `datasets`).

**Interface (`plitzer/bench/baseline.py`):**

```python
# python -m plitzer.bench.baseline DATA --loaders mosaic,hf,hf-stream --batch 64 --workers 4
class Run(NamedTuple): seed: int; batch: int; workers: int
class Opened(Protocol):                       # a loader at the start of an epoch or at a state
    def __iter__(self) -> Iterator[list[Doc]]: ...
    def state(self) -> object: ...            # after the batches given so far
class Loader(Protocol):
    def convert(self, data: Path, work: Path) -> Path: ...
    def open(self, data: Path, prepared: Path, run: Run, state: object) -> Opened: ...
class Result(NamedTuple): loader; convert_s; convert_bytes; docs_per_s; tokens_per_s
                          cores: float | None; resume_s; exact: bool; lost: int
LOADERS: dict[str, tuple[str, str]]           # name -> (module, class) of plitzer.bench
def measure(name: str, data: Path, work: Path, run: Run, resume_at: int) -> Result: ...
def table(results: Sequence[Result]) -> str: ...
```

**Design:**

- `mosaic`: `MDSWriter` converts the files to MDS shards with zstd. Each document is one sample, with its tokens as bytes, as llm-foundry writes them (D-75). A shuffled `StreamingDataset` with a `StreamingDataLoader` reads them, with `run.workers` workers. The collate function gives a dict of lists (D-75). The state is `state_dict()`.
- `hf`: `load_dataset("parquet")` converts the files to an Arrow cache. `shuffle(seed)` gives a random order, and each batch is a slice of the shuffled dataset. The state is the index of the next document, as `accelerate.skip_first_batches` uses for a map-style sampler.
- `hf-stream`: `load_dataset(..., streaming=True)` reads the files with no conversion, through a shuffle buffer of 10 000 documents. The state is `state_dict()`, which does not hold the buffer.
- The HF loaders iterate in the main process: `IterableDataset.state_dict()` counts only what the main process takes (D-76).
- Each phase runs in a new `spawn` process (D-73). The phases are the conversion, the timed epoch, the tail and the resume. The process times its own work, so imports and the start of the process do not count.
- The CPU time is `os.times()` of the process, with the workers that it joined. Cores is CPU time divided by wall time. Cores is unknown when `os.times()` cannot count the workers: on Windows, and when the start method is `forkserver` (the Linux default from Python 3.14), whose workers are not children of the phase process.
- The timed epoch counts documents and tokens, and saves the state after `resume_at` batches. It makes no digest, so the time is only the loader's work. Its clock starts after `open`: it measures the pace of the loader. The clock of the resume starts before `open`, because a resume waits for the loader to start.
- The tail phase runs an uninterrupted epoch again, with no clock, and makes a digest of all batches after the first `resume_at`. The resume runs to the end of the epoch, with the same digest. `exact` is true if the two digests are the same. `lost` is the documents of the uninterrupted rest minus the documents of the resumed rest (D-74).
- `measure` refuses an unknown loader and a `resume_at` below 1. The epoch refuses a `resume_at` with no batch after it. The CLI refuses an unknown or repeated loader name before it starts.

**Rationale:** The same files, seed, batch size (in documents), worker count and epoch feed each loader. The input is the FineWeb-Edu-shaped dataset of section 5.13 (D-72). The loaders iterate documents, which is the unit that each loader gives.

**Limits:**

- The numbers depend on the computer. Compare rows of one table, not tables from different computers.
- `cores` is unknown on Windows and with the `forkserver` start method.
- The `hf` rows use no DataLoader workers (D-76).
- The time of the first batch of a resume includes the start of the loader: for Mosaic, the dataset and its workers; for `hf`, the cache and the shuffle. The shuffle of `hf` uses the index file that the earlier phases wrote, as a resumed job uses the file of its first run.
- Each loader reads the data three times after the conversion: the timed epoch, the tail and the resume.

**Verification:**

| Test | Type | Checks |
|---|---|---|
| `test_an_epoch_gives_each_document_once` | pytest | For each loader, one epoch gives each document of the source one time, also the empty ones. |
| `test_resume_is_exact` | pytest | Mosaic and `hf` resume at batch 5 with the rest of the epoch of an uninterrupted run, and lose no document. The other fields are positive, and `cores` is unknown only where `workers_counted()` is false. |
| `test_a_streaming_resume_loses_the_shuffle_buffer` | pytest | `hf-stream` needs no conversion. Its resume is not exact, and it loses the 80 documents that its buffer held. |
| `test_cpu_counts_this_process` | pytest | `cpu()` grows by 0.1 s while this process works, within 30 s of wall time. |
| `test_resume_at_must_be_inside_the_epoch` | pytest | A resume after the last batch, or at 0, gives an error. |
| `test_main_refuses_unknown_or_repeated_loaders` | pytest | The CLI refuses `hf,hf_stream` and `hf,hf` before it starts. |
| `test_main_prints_a_row_for_each_loader` | pytest | The CLI prints one table row for each loader. |

These tests run in a CI job of their own (`baseline`, on Linux), which installs the loaders. `mosaicml-streaming` pins numpy below 2.2, and the other tests run with the newest numpy. Other jobs skip these tests.

**Results:** One run on 4 CPU cores, 15 GiB of RAM and a local disk:
- The data is the 512 row groups of section 6: 512 000 documents, 1104 MiB of Parquet token chunks.
- The settings are batch 64, 4 workers for Mosaic, seed 0, and a resume at batch 4000 of 8000.
- The library versions are `mosaicml-streaming` 0.13.0, `datasets` 5.0.1 and torch 2.15.0 (CPU).

| loader | convert (s) | converted (MiB) | docs/s | Mtokens/s | cores | resume (s) | exact resume | docs lost |
|---|---|---|---|---|---|---|---|---|
| mosaic | 22.3 | 1222 | 28 014 | 28.8 | 2.68 | 2.89 | yes | 0 |
| hf | 4.5 | 2009 | 86 357 | 88.7 | 1.00 | 0.18 | yes | 0 |
| hf-stream | 0.0 | 0 | 2 949 | 3.0 | 1.10 | 3.06 | no | 10 000 |

- The computer is a shared cloud computer. An earlier run timed the digest inside the epoch, so its `hf` speed was 20 % lower. Its Mosaic and `hf-stream` values were the same within 5 %, but its Mosaic conversion took 53.5 s. Thus the conversion time of Mosaic is 22 to 54 s.
- Mosaic needs a conversion, and a second copy of the data. It uses 2.7 cores for 28 000 documents each second. Its resume is exact.
- `hf` makes a second copy of 2 GiB, 1.8 times the Parquet chunks, because the Arrow cache is not compressed. All of the cache fits in the page cache of this computer, so its random access does not wait for the disk. On a dataset larger than RAM, each batch reads from random positions of the disk.
- `hf-stream` needs no copy, but it gives 2900 documents each second, and its resume loses the 10 000 documents of its buffer. Thus it is neither fast nor resumable without loss.
- Section 7 keeps the comparison with plitzer as an open question. The next PR adds plitzer to this table.

### 5.15 Packaging and release

**Function:** The release workflow builds the Python package for each platform, tests each wheel, and publishes the wheels and the sdist to PyPI.

**Interface:**

```sh
pip install plitzer                 # a wheel; "plitzer[torch]" adds the PyTorch dataset
maturin develop --release           # from the repository root, into the active virtualenv
maturin sdist                       # the source package
git tag v0.1.0 && git push --tags   # a release: the tag must be the Cargo version
```

`plitzer.__version__` is the version of the installed package.

**Design:**

- `pyproject.toml` is at the root of the repository. `[tool.maturin]` gives the manifest `plitzer-py/Cargo.toml` and the Python source `plitzer-py/python`. Thus the README and both licence files are in the package, and `pip install` of the repository works (D-90).
- The extension uses the abi3 interface of CPython 3.11. Thus one wheel for each platform covers each CPython from 3.11.
- `.github/workflows/release.yml` builds five artifacts with `maturin-action`:
  - a wheel for Linux x86_64 (manylinux 2.28);
  - a wheel for Linux aarch64, on an arm runner;
  - a universal2 wheel for macOS;
  - a wheel for Windows x64;
  - the sdist.
- Each wheel goes into a new virtualenv on its own system, with no Rust and no source tree. `test_loader.py` and `test_export.py` then run against it. A wheel is also built from the sdist.
- A pull request that changes the workflow runs the builds and the tests. A `v*` tag also runs the publish job. That job refuses a tag that is not the Cargo version, and publishes with PyPI trusted publishing, in the `pypi` environment (D-91).

**Rationale:**

- The aarch64 wheel builds on an arm runner, not with a cross compiler. `aws-lc-sys`, the TLS library of the S3 client, needs a C compiler for its target.
- One universal2 wheel serves both kinds of Mac, and the arm runner tests it.
- Trusted publishing needs no token in the repository. PyPI accepts only this workflow of this repository.

**Limits:**

- The x86_64 half of the macOS wheel is built but not run in CI.
- The first publish needs a pending trusted publisher on pypi.org: project `plitzer`, workflow `release.yml`, environment `pypi`. Only the owner of the repository can add it.
- Linux wheels need glibc 2.28 or later. There is no musl wheel.

**Verification:**

| Check | Type | What it shows |
|---|---|---|
| `wheels` job | CI | Each of the 4 wheels builds, installs into a new virtualenv, and passes `test_loader.py` and `test_export.py` on its own system. |
| `sdist` job | CI | The sdist builds, and a wheel built from it imports and gives `plitzer.__version__`. |
| `twine check` | Manual | The wheel and the sdist have valid metadata. The README renders, and both licence files are in the wheel. |

## 6. Read amplification

**Problem:** A naive split gives each GPU a random part of a shuffled batch. Each GPU then needs a document from almost each row group of the window. Each GPU reads approximately world times its own data.

**Fix:** A GPU owns a contiguous block of lanes (section 5.3). Each lane owns a contiguous stretch of the window in storage order. Thus the lanes of one GPU cover one stretch of storage. The GPU reads that stretch and approximately one shared row group at each edge.

**Bound:** Two neighbour GPUs share at most the row group at their edge. Thus all GPUs read at most world − 1 more row groups than a window holds. An epoch of n row groups has ⌈n/W⌉ windows, so the amplification is 1 + ⌈n/W⌉(world − 1)/n or less, in row groups. This is approximately 1 + (world − 1)/W. The bound counts row groups, not bytes, because an edge row group can be larger than the mean.

**Measurement:** `plitzer.bench` (section 5.13) measured one epoch:
- The dataset has 512 row groups (the first 64 of FineWeb-Edu `sample/10BT` files 000 to 007). It has 512 000 documents and 1104 MiB of token chunks.
- The documents have the real lengths, with random tokens (D-69).
- The settings are B = 512 and L = 2048. The lane slices are cut by slots (section 5.3).
- `measure` counts epoch 1 of each rank.
- Each cell is the measured amplification, with the bound in parentheses:

| W | windows | world 8 | world 64 | world 256 |
|---|---|---|---|---|
| 16 | 32 | 1.442 (1.438) | 4.955 (4.938) | 17.001 (16.938) |
| 64 | 8 | 1.111 (1.109) | 1.994 (1.984) | 5.008 (4.984) |
| 100 | 6 | 1.083 (1.082) | 1.748 (1.738) | 4.025 (3.988) |
| 256 | 2 | 1.027 (1.027) | 1.248 (1.246) | 2.003 (1.996) |
| 300 | 2 | 1.028 (1.027) | 1.247 (1.246) | 2.001 (1.996) |
| 512 | 1 | 1.013 (1.014) | 1.122 (1.123) | 1.502 (1.498) |

**Results:**

- Each measurement is at the bound, in row groups. Some are a little above it in bytes, by 1.3 % or less: an edge row group can be larger than the mean, and a row group in the last window of an epoch and the first window of the next is read two times.
- The slices are cut by tokens, so the edge of a GPU is in a row group, also when world divides W. Thus two neighbour GPUs share one row group in each window. With the slices of equal document counts (#24), each FineWeb-Edu row group had 1000 rows, so the edges were on row groups when world divided W, and the amplification was exactly 1. That cut made the lanes go apart (D-86).
- When world is larger than W, some GPUs have less than one row group. The amplification is then near world/W.
- This dataset has few windows, so its short last window adds to the bound. With the 10 164 row groups of all of `sample/10BT`, the bound is nearer 1 + (world − 1)/W.

**Decision:** Plitzer does not add a cache on each node (D-71). Set W to many times the largest GPU count that you plan to use: the cost is approximately (world − 1)/W. For example, W = 512 costs 1.4 % on 8 GPUs and 12 % on 64 GPUs. A checkpoint holds W, so a resume on a different GPU count keeps it.

**Trade-off:** A larger W gives a better mix of documents but uses more memory. A change to the slices changes both the shuffle quality and the I/O. Examine both before you change the slices.

## 7. Open questions

| Question | Effect |
|---|---|
| Must the loader add an EOS token between documents, or must the `Tokens` source do it? | The `Tokens` trait. |
| How well do documents mix inside one sequence on a sorted dataset? | The value of W. |
| What are the statistics of the batch of each GPU? | Possible bias between GPUs. |
| What is the best trade-off between W and memory? | The default W. |
| How fast is plitzer, with how many cores, against the baseline of section 5.14? | The benchmark table in the README. |

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
| D-10 | Lane slices have equal document counts, not equal token counts. | Simple and exact. Token balance is a Pack problem. D-86 replaces this: lanes cut by documents went apart. | Slices by token count. | #6 |
| D-11 | Invalid configuration causes a panic. | It is a programmer error until the Python bindings exist. D-41 changes this for `Session`. | `Result` errors now. | #6 |
| D-12 | The cap removes the oldest leftover tokens. | The leftover is one run to the cursor, so `(doc, offset)` is the full lane state. D-78 replaces this: there is no cap. | Remove the newest tokens: documents stay whole, but the leftover gets gaps and the state becomes a list of ranges. | #8 |
| D-13 | Each epoch ends with `ceil(C / L)` steps that empty the leftovers. | Only the cap drops tokens, and the epoch length is known before the read. D-79 replaces this: there are no flush steps. | Drop the leftover at the end of the epoch. | #8 |
| D-14 | Each step takes a fixed D documents. | The cursor is a function of the step, so a checkpoint does not store it. D-77 replaces this: the fixed D lost tokens. | Take documents until the sequence is full: the cursor then depends on token counts and must be in the checkpoint. | #8 |
| D-15 | `fill` writes into a buffer from the caller. | One buffer for each GPU batch, with no copy. | Return a new `Vec` for each sequence. | #8 |
| D-16 | The pack module reads tokens through the `Tokens` trait. | The Read stage does not exist yet. Tests use a corpus in memory, and the Read stage can implement the trait later. | A fixed token source, such as a `Vec` of all tokens. | #8 |
| D-17 | `LaneState` holds the step. | A state cannot go with a different step, which gives wrong sequences with no error. D-84 replaces this: one step for all lanes, and a position for each lane. | A separate step input to `Epoch::lane`. | #8 |
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
| D-37 | A checkpoint is versioned bytes with a checksum. | `torch.load(weights_only=True)` accepts bytes. Rust and Python use one format. | A dict of integers, or a custom Python class. | #16 |
| D-38 | Each GPU gives a part, and `merge` makes a global checkpoint. | Some trainers save only rank 0. A global checkpoint resumes on each GPU count. | One file for each rank: a resume needs the same GPU count. | #16 |
| D-39 | A resume with a different config or fingerprint stops, and gives the name of the field. | Other values give other batches with no error. | Accept the resume and show a warning. | #16 |
| D-40 | After the last step of an epoch, the session goes to step 0 of the next epoch. | A checkpoint at the end of an epoch then starts the next epoch. | Stay at the end of the old epoch. | #16 |
| D-41 | `Session::open` returns `Error::Config` for a config that would cause a panic. | Python gives the config. | A panic, which PyO3 shows as `PanicException`. | #16 |
| D-42 | After a failed `next`, the session keeps the lane states and drops the loader. | The next call tries the same step again. | Stop the session, as the loader does. | #16 |
| D-43 | `set_epoch` only goes to a later epoch. | Trainers call it after a resume, also with the epoch before a checkpoint at an epoch end. A reset would repeat steps with no error. | Always go to step 0, or do nothing only for the same epoch. | #16 |
| D-44 | `Loader::next` is cancel safe. The read in progress stays in the loader. | Python drops the future at Ctrl-C, and a timeout drops it too. A lost read stops the loader with a panic. | Document that a dropped `next` breaks the loader. | #16 |
| D-45 | `from_bytes` checks the checksum before the version. | A damaged version field gives `Checksum`, not a false `Version` error. | Check the version first. | #16 |
| D-46 | Export both the versioned (1.0) and the legacy DLPack ABI. | numpy 2 and current torch ask for 1.0. Older consumers call `__dlpack__()` with no arguments. | Only one ABI. | #18 |
| D-47 | Export tokens as int32. | torch has little `uint32` support, and vocabularies are much smaller than 2³¹. | `uint32`, or int64 with a copy. | #18 |
| D-48 | The DLPack capsules use our own destructor through `PyCapsule_New`. | A consumer renames the capsule and deletes the tensor itself. PyO3 destructors cannot see the name, so they would free the tensor a second time. | `PyCapsule::new_with_value`, as for the Arrow capsules. | #18 |
| D-49 | S3 uses `AmazonS3Builder::from_env` and the options, in our `store` module. | `parse_url_opts` does not read the `AWS_*` environment variables. | `object_store::parse_url_opts`. | #18 |
| D-50 | One runtime for each process. A forked child makes its own, and a `Loader` copied into the child refuses to work. | A forked child has none of the parent's runtime threads, so a wait on the copied loader never ends. A new loader in the child must still work. | One runtime that refuses all work after a fork: the child can then never make a loader. | #18 |
| D-51 | Python waits in slices of 100 ms and checks for signals between them. | Ctrl-C works, and `Loader::next` is cancel safe. | One wait with no signal check: Ctrl-C does nothing until the step ends. | #18 |
| D-52 | `pad` has no default. | Many tokenizers use 0 as a real token. `lengths` gives the real tokens, but a silent default hides the choice. D-77 removes `pad`. | A default of 0. | #18 |
| D-53 | The bindings are the `plitzer-py` workspace crate, built by maturin as abi3 wheels for CPython 3.11 and later. | The core crate stays free of PyO3, and one wheel covers each platform. | PyO3 in the core crate behind a feature. | #18 |
| D-54 | The default cache name is a hash of the resolved store and prefix. | The same text can name two datasets, and two texts can name one dataset. | A hash of the location text. | #18 |
| D-55 | `__next__` keeps raising `StopIteration` after an epoch ends, and `__iter__` starts the next epoch. | Python iterators stay ended. A `zip` stops at the first ended loader, so `__iter__` must move the others on. | Start the next epoch at the next `__next__`. | #18 |
| D-56 | In a forked child, a `Loader` leaks its session. | A drop stops runtime tasks, which can wait on a lock that no thread in the child will release. | Drop as usual and risk a hang at exit. | #18 |
| D-57 | Before it reads the footers, `Index::open` makes a file in the cache folder. It changes no file. | A cache that cannot be written then fails at once, not after the full footer scan. Another process can have just written a good sidecar, so the check must not replace it. No file stays open during the build. | Write the cache after the build only, or keep the temporary file open during the build. | #18 |
| D-58 | The default cache name is a hash of `store::Location::id`, which holds the endpoint but no credentials. | New credentials keep the cache. Two servers with the same bucket name do not share it. | A hash of all storage options. | #18 |
| D-59 | A frozen `Loader` with a lock for the session and a checkpoint snapshot that other calls read. | A preemption signal handler or a background thread can save a checkpoint during a step. A second step from another thread gets an error, not a deadlock. | A borrow for each call, which refuses all calls during a step. A lock that each call takes, which deadlocks a signal handler. | #18 |
| D-60 | The runtime has at most 8 worker threads and 32 blocking threads. | Each GPU process has its own runtime, and a loader has at most 16 reads in flight. | One worker for each CPU: hundreds of threads in each process on a large node. | #18 |
| D-61 | No default cache without a user cache folder. | A shared temporary folder lets other users break or poison the index. | A shared folder in the temporary directory. | #18 |
| D-62 | A URL with `?` or `#` is an error. | The parser drops what follows them, and they can be part of a real name, so another dataset would open with no error. | Drop the query and the fragment. | #18 |
| D-63 | `Session::open` refuses more than 2²⁴ lanes, and more than 2³⁰ tokens in one step of one GPU or in its leftovers. | These sizes are far past any real batch. Larger values could only use all memory, which stops the process with no Python exception. D-77 removes the leftovers. | No bound. | #18 |
| D-64 | `Epoch::lane` gives an error, not a panic, for a state that no run makes. `Session::resume` makes the loader at once and keeps the old position on an error. | A checkpoint made by hand passes the checksum, and its states would stop the packer with a panic. A resume that cannot go on must not replace a good position. `Lane::new` now gives this error. | Trust each checksummed checkpoint, or check only some states, or make the loader at the next step. | #18 |
| D-65 | A resume replays the leftover of each lane one window at a time, then plans from the cursor. | The plan counted the replayed row groups together with the row groups of the resume step. A resume, and a retry after a failed read, then could need more than the budget of a run with no stop. D-78 replaces this: there is no replay. | Count the worst resume in the budget of each run, which makes each budget larger. | #20 |
| D-66 | `Dataset.state_dict()` gathers the parts of all ranks and gives the same global checkpoint on each rank. | A global checkpoint needs each lane, and a rank knows only its own lanes. Lightning makes the checkpoint on each rank, and `torch.distributed.checkpoint` writes an item that all ranks hold only one time. | A part for each rank: a resume then needs the same GPU count, or code that merges the parts. | #21 |
| D-67 | Each item is a `TokenBatch` `NamedTuple` of int32 tensors for one step. | `DataLoader` with `batch_size=None` keeps it. The tensors share memory with the export. | A dict, which is not typed, or int64 tensors, which need a copy. | #21 |
| D-68 | `state_dict()` is the position of the next batch that the dataset gives. | A count of fetched batches that the loop did not train is wrong for some loops: at the end of an epoch, for a loop with a step limit, and after a crash during the last batch. | An `ahead` count with heuristics for these cases. | #21 |
| D-69 | Measure the read amplification on FineWeb-Edu lengths, with random tokens. | The amplification depends only on the row counts and chunk bytes. `token_count` gives the real lengths from a few MB of each file. | Tokenize the text: 10B tokens, which is too much for a benchmark run. | #24 |
| D-70 | `Loader.bytes_read` is the count at the end of the last call. | A thread can read it during a step, as it can read `checkpoint()`. D-85 replaces this. | A live count, which needs the session lock during a step, or a new shared counter in `Session`. | #24 |
| D-71 | No cache on each node. | When world divides W, the measured amplification is 1. Otherwise it is near 1 + (world − 1)/W, so a W of several times world keeps it small: 1.016 at W = 300 for 8 GPUs (section 6). With slices cut by slots (D-86), it is near 1 + (world − 1)/W also when world divides W: 1.013 at W = 512 for 8 GPUs. | A cache on each node that GPUs share. | #24 |
| D-72 | The baseline reads the FineWeb-Edu-shaped dataset of section 5.13. | Each loader reads the same files. The speed of a loader does not depend on the token values (D-69). | A tokenized copy of the text: 10B tokens, and a different input from the plitzer measurement. | #25 |
| D-73 | Each phase of the baseline runs in a new process and times itself. | No page of a worker, shared-memory segment or warm object goes from one phase to the next, and imports do not count. | One process for all phases. | #25 |
| D-74 | The resume check runs to the end of the epoch and compares a digest of all later batches. | A resume that keeps its first batch but loses documents later passes a check of the first batch only. `hf-stream` loses its buffer this way. | Compare only the first batch after the resume. | #25 |
| D-75 | The Mosaic adapter writes tokens as bytes, and its collate function gives a dict of lists. | The `ndarray` encoding refuses an empty document. `StreamingDataLoader` counts the samples of a batch from its first value, so a list batch saves a wrong state. | `ndarray:int32` and `collate_fn=list`. | #25 |
| D-76 | The HF adapters iterate in the main process, with no DataLoader workers. | `IterableDataset.state_dict()` counts only what the main process takes. With workers, the state is wrong. | A `DataLoader` with workers, or `torchdata`'s `StatefulDataLoader`, which is one more dependency. | #25 |
| D-77 | Each step takes tokens until the sequence is full: the rest of the current document, then whole documents, cut at exactly L. There is no D, no cap and no pad token. | A fixed D gives L tokens only on average, so the leftover is a random walk: it hits the cap and drops tokens, or it empties and pads. On FineWeb-Edu lengths with D = 2 and C = 2048, 19.3 % of the tokens were lost. | A fixed D with a larger cap: 1.6 % lost at C = 128k, more memory, and the walk grows with the dataset. | #26 |
| D-78 | A lane position is `(epoch, window, token)`, and `fill` copies from the row group into `out`, one copy for each piece. | The index gives the slots of each slice, so a position is checked and found with no read, and a resume reads only the slice of each lane. Pack time went from 1.17 to 0.53 ns for each token. | A leftover buffer, and a replay of it at a resume. | #26 |
| D-79 | The stream of a lane goes on into the next epoch. An epoch is S = ⌈T/(B·L)⌉ steps. | No lane waits or pads at the end of an epoch. Each GPU gets S from the index. | Flush steps with pad tokens at each epoch end. | #26 |
| D-80 | `Lane::need` finds the missing documents of a step, and `fill` runs only when all are in memory. | The loader waits for reads before a lane changes, so `next` stays cancel safe. | A `fill` that waits for reads part way through the lanes. | #26 |
| D-81 | The loader reads in window order across epochs, with keys `(epoch, window, position)`, and no step plan. | The step at which a lane gets to a document depends on token counts, so a plan by step is not known. Window order is the order in which the lanes use row groups. | A plan of step ranges from estimated token counts. | #26 |
| D-82 | The budget caps the reads ahead. A step that waits for a read adds the permits for it, and a budget below one window is refused. | Lanes can be a window apart, so a fixed budget can hold the next window while a slow lane waits. The added permits stop a wait for itself. | A reserve of the budget for the current slice of each lane. | #26 |
| D-83 | The users of a row group are `(lane, epoch, window)`, and the release is after all lanes filled. | At an epoch end, a row group can be in a window of each epoch. Users by lane only freed a row group that another lane still needed in the same step. | Users by lane. | #26 |
| D-84 | Checkpoint version 2: one step, and `(epoch, window, token)` for each lane. Version 1 gives `Error::Version(1)`. | Lanes are in different epochs at an epoch end. Version 1 is from the fixed packer, so its batches cannot continue. | Convert version 1. | #26 |
| D-85 | `Loader.bytes_read` reads the shared counter of the reader, with no lock. | The benchmark must count the reads ahead while they arrive. A thread can read it during a step. | The count at the end of the last call (D-70). | #26 |
| D-86 | Cut the lane slices by slots, the tokens and empty rows that the footer counts, and cut a row at a slice edge. | Each lane takes L tokens a step, so each needs the same share of the tokens. With slices of equal document counts, lanes went apart by 3.9 windows in one epoch and 11.8 in six on FineWeb-Edu lengths, and an epoch took 20 s, not 1.6 s. | Slices by documents (D-10), or slice edges at the nearest document, which need the rows before the slices are known. | #26 |
| D-87 | The reader refuses a chunk whose rows do not have the slots of the index. | The slices are cut by these slots, so a lane would find too few tokens with no clear error. | Trust the footer. | #26 |
| D-90 | `pyproject.toml` is at the root of the repository, and points maturin to `plitzer-py/Cargo.toml`. | The README and the licence files are at the root, and PEP 639 refuses a licence path with `..`. `pip install` of the repository needs a `pyproject.toml` at its root. | Copies of the README and the licences in `plitzer-py/`. | #28 |
| D-91 | Publish from a `v*` tag with PyPI trusted publishing, after each wheel passes its tests on its own system. | No token is in the repository. A wheel that does not import fails before PyPI gets it. | A PyPI token as a secret, and a publish with no test of the wheels. | #28 |
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
| 2026-10-23 | #16 | Add the checkpoint (`src/checkpoint.rs`) and the session (`src/session.rs`). Add `Loader::ended`. Move the binary helpers to `src/codec.rs`. |
| 2026-10-23 | #18 | Add the Python bindings (`plitzer-py/`), the store (`src/store.rs`, `src/store/s3.rs`) and `Session::ended`. Add a Python CI job on the three systems. The index makes the folder of its cache, and checks that it can write the cache before it reads the footers. |
| 2026-10-26 | #20 | A resume replays leftovers one part at a time, then plans from the cursor (`Epoch::replay`, `Replay`, `src/load/replay.rs`). A resume needs no more budget than a run with no stop. |
| 2026-10-26 | #21 | Add the PyTorch dataset (`plitzer/torch.py`): `Dataset`, `TokenBatch` and a collective `state_dict()`. torch is the optional extra `plitzer[torch]`. CI installs the CPU build of torch. |
| 2026-11-05 | #23 | Add the end-to-end tests that kill a training run at a random step and resume it on another number of processes (`plitzer-py/tests/e2e.py`, `e2e_stream.py`, `e2e_torch.py`). |
| 2026-11-09 | #24 | Add `bytes_read` (`Reader`, `Session`, `Loader`) and `plitzer.bench` (section 5.13). Measure the read amplification on FineWeb-Edu lengths (section 6). Correct the bound to 1 + ⌈n/W⌉(world − 1)/n. |
| 2026-11-11 | #25 | Add `plitzer.bench.baseline`, with adapters for MosaicML Streaming and HF `datasets` (section 5.14), and measure the baseline. |
| 2026-11-11 | #26 | Lossless packing: lane slices are cut by slots (tokens and empty rows), so lanes take the same share of the tokens and stay together. Each step fills the sequence from the rest of the current piece and whole pieces, and lanes go on across epochs (`src/pack/lane.rs`, `src/pack/pieces.rs`). The reader checks the slots of each row group. Remove D, C, the pad token and `Batch.lengths`. The loader reads in window order with a budget for the reads ahead (`src/load/sequence.rs`, `prefetch.rs`, `buffer.rs`), and the replay is removed. Checkpoint version 2. `bytes_read` is live. |
| 2026-11-11 | #28 | Add the release workflow (section 5.15): wheels for 4 platforms and an sdist, a test of each wheel on its own system, and PyPI trusted publishing on a `v*` tag. Move `pyproject.toml` to the root, with the README, the licences and the package metadata. Add `plitzer.__version__` and `CHANGELOG.md`. |

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
