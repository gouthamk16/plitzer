# Plitzer engineering design

<!-- ste-allow: engineering, training, streaming, sampling -->

| Field | Value |
|---|---|
| Status | Active. It changes with each feature PR. The code is pre-alpha. |
| Last change | 2026-10-21, PR #8 |
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
| Index | The list of row groups and their row counts. The id of a row group is its position in the index. |
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

## 3. Requirements

| Id | Requirement | Verification | Status |
|---|---|---|---|
| REQ-1 | The order is a pure function of `(seed, epoch, step)` and the index. | Golden tests in `shuffle`, `seed`, `order` and `pack`. | Met for the order and the packer |
| REQ-2 | Each epoch supplies each document exactly once. The only exception is leftover tokens that a lane drops at its cap. | `order::exactly_once` and `pack::exactly_once_tokens` (property tests). | Met for the order and the packer |
| REQ-3 | Lanes and their slices depend only on `(seed, epoch, B, W)`. They do not depend on world. | `Order` has no world input. `order::gpu_lanes_partition` (property test). | Met |
| REQ-4 | A resume at step k gives the same tail as a run without a stop. This is also true when world changes. | `pack::resume_matches_tail` and `pack::world_invariant` (property tests). Planned: end-to-end test. | Met in the packer |
| REQ-5 | The output is the same on Linux, macOS and Windows. | CI runs the golden tests on the three systems. | Met |
| REQ-6 | Read amplification is approximately 1 + world/W or less. | Planned: measurement on FineWeb-Edu. | Planned |
| REQ-7 | A dependency upgrade does not change the order without a test failure. | Golden tests pin the shuffle output and the key bytes. | Met |

## 4. System overview

The pipeline has five stages. Each stage gives its output to the next stage.

```mermaid
flowchart LR
    index["Index<br/>Parquet footers"] --> order["Order<br/>windows and lanes"]
    order --> read["Read<br/>object_store"]
    read --> pack["Pack<br/>L-token sequences"]
    pack --> python["Python<br/>IterableDataset"]
    classDef done fill:#d4edda,stroke:#2e7d32,color:#1b5e20
    class order,pack done
```

| Component | Module | Status | Section |
|---|---|---|---|
| Shuffle | `src/shuffle.rs` | Implemented | [5.1](#51-shuffle) |
| Seed key | `src/seed.rs` | Implemented | [5.2](#52-seed-key) |
| Order | `src/order.rs` | Implemented | [5.3](#53-order) |
| Pack | `src/pack.rs` | Implemented | [5.4](#54-pack) |
| Checkpoint | — | Planned | [5.5](#55-planned-components) |
| Index | — | Planned | [5.5](#55-planned-components) |
| Read (I/O) | — | Planned | [5.5](#55-planned-components) |
| Python bindings | — | Planned | [5.5](#55-planned-components) |

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
- Row group ids and row numbers are `u32`. FineWeb-Edu has approximately 10⁵ row groups.
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

pub struct Epoch<'a> { /* private: order, packing, windows, steps */ }
pub struct Lane<'e> { /* private */ }

impl<'a> Epoch<'a> {
    pub fn new(order: Order<'a>, packing: Packing) -> Self;
    pub fn steps(&self) -> u64;
    pub fn lane(&self, lane: u32, state: LaneState, tokens: &impl Tokens) -> Lane<'_>;
}
impl Lane<'_> {
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
- The pack module reads tokens through the `Tokens` trait. The Read stage does not exist yet, and the tests use a corpus in memory (D-16).
- The lane state holds its step. Thus a caller cannot give the state of one step with a different step (D-17).
- `fill` writes into a buffer from the caller. The Python stage can then give one buffer of B/world × L tokens to each GPU, with no copy (D-15).
- Each `Lane` reads the windows of its `Epoch`. Thus the row group shuffle occurs one time for each epoch, not one time for each lane.

**Limits:**

- If D documents have many more than L + C tokens, the sequences are mostly the ends of documents. The cap removes the starts (section 7).
- The packer adds no tokens between documents. The `Tokens` source must add an EOS token if the model needs one.
- A document can have a maximum of `u32::MAX` tokens.
- A checkpoint is correct only with the same `Order` and `Packing` values. The checkpoint struct must store them (section 5.5).
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

### 5.5 Planned components

These components are not in the code yet. Their sections will have the full set of parts when they are implemented.

- **Checkpoint:** A checkpoint holds `seed`, `epoch` and the lane state of each lane. It also holds B, W, L, D, C and a fingerprint of the index. A resume with different values must stop with an error.
- **Index:** The index reads only the Parquet footers and makes the list of `(file, row_group, num_rows)`. Plitzer keeps the index in a small sidecar file. The footers of FineWeb-Edu are approximately 9.4 GB in total, so Plitzer must not read them again at each start.
- **Read:** The read stage uses `object_store` and the async reader of the `parquet` crate, on tokio. It implements `AsyncFileReader` directly, because `ParquetObjectReader` is deprecated as of parquet 59.2. A byte limit controls the prefetch, not a batch count. It supplies tokens to the pack module through the `Tokens` trait.
- **Python:** The bindings use PyO3 and maturin. Batches go to Python through the Arrow PyCapsule interface. A PyTorch `IterableDataset` gives `state_dict()` and `load_state_dict()`. Users must set `num_workers=0`. The parallel work is in Rust, and a fork of a process with a tokio runtime is not safe.

## 6. Read amplification

**Problem:** A naive split gives each GPU a random part of a shuffled batch. Each GPU then needs a document from almost each row group of the window. Each GPU reads approximately world times its own data.

**Fix:** A GPU owns a contiguous block of lanes (section 5.3). Each lane owns a contiguous stretch of the window in storage order. Thus the lanes of one GPU cover one stretch of storage. The GPU reads that stretch and approximately one shared row group at each edge.

**Estimate:** A simulation gives a read amplification of approximately 1 + world/W. At FineWeb-Edu sizes, the value is 1.01 to 1.25. A cache on each node is a fallback only. Plitzer will add it only if a benchmark shows that it is necessary.

**Trade-off:** A larger W gives a better mix of documents but uses more memory. A change to the slices changes both the shuffle quality and the I/O. Examine both before you change the slices.

## 7. Open questions

| Question | Effect |
|---|---|
| Is the input of v1 raw text, with tokenization in the loader, or pre-tokenized data? | Pack and Read design. |
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

## 9. Change log

| Date | PR | Change |
|---|---|---|
| 2026-10-17 | #2 | Add the deterministic shuffle (`src/shuffle.rs`). |
| 2026-10-19 | #4 | Add the seed key (`src/seed.rs`) and the `from_seed` ban. Change the planned design to lanes. |
| 2026-10-21 | #6 | Add the order (`src/order.rs`) and `proptest`. Add this document. |
| 2026-10-21 | #8 | Add the packer (`src/pack.rs`) and `Order::lane_len`. |

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
