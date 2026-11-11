use arrow_array::builder::{GenericListBuilder, Int32Builder};
use arrow_array::{ArrayRef, RecordBatch};
use object_store::local::LocalFileSystem;
use object_store::path::Path;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use tempfile::TempDir;

use std::collections::{HashMap, HashSet};

use super::*;
use crate::order::gpu_lanes;
use crate::pack::Tokens;
use crate::read::RowGroupTokens;
use sequence::WindowGroups;

mod bytes;
mod fail;

const LANES: u32 = 4;
const SEQ_LEN: u32 = 8;
const PLENTY: u64 = 1 << 30;

struct Data {
    dir: TempDir,
    index: Arc<Index>,
    reader: Arc<Reader>,
    orders: Arc<Orders>,
}

/// Six files of row groups of 5 rows; row `r` of file `f` has
/// `60 * ((f + r) % 9)` tokens, all unique, so row groups span several KiB
/// and the budget's KiB rounding does not hide its cost.
async fn data(seed: u64) -> Data {
    let dir = TempDir::new().unwrap();
    let mut next = 0;
    for f in 0..6 {
        let mut b = GenericListBuilder::<i32, _>::new(Int32Builder::new());
        for r in 0..(7 + 4 * f) {
            for _ in 0..60 * ((f + r) % 9) {
                b.values().append_value(next);
                next += 1;
            }
            b.append(true);
        }
        let column: ArrayRef = Arc::new(b.finish());
        let batch = RecordBatch::try_from_iter([("tokens", column)]).unwrap();
        let props = WriterProperties::builder()
            .set_max_row_group_row_count(Some(5))
            .build();
        let file = std::fs::File::create(dir.path().join(format!("{f}.parquet"))).unwrap();
        let mut w = ArrowWriter::try_new(file, batch.schema(), Some(props)).unwrap();
        w.write(&batch).unwrap();
        w.close().unwrap();
    }
    let index = Index::build(&*store(&dir), &Path::from(""), "tokens")
        .await
        .unwrap();
    let reader = Reader::new(store(&dir), Path::from(""), &index).unwrap();
    let orders = Orders::new(seed, LANES, 3, SEQ_LEN, &index.rows());
    Data {
        dir,
        index: Arc::new(index),
        reader: Arc::new(reader),
        orders,
    }
}

fn store(dir: &TempDir) -> Arc<LocalFileSystem> {
    Arc::new(LocalFileSystem::new_with_prefix(dir.path()).unwrap())
}

/// Every row group in memory, for the packer run without the loader.
struct All(HashMap<u32, RowGroupTokens>);

impl Tokens for All {
    fn tokens(&self, doc: Doc) -> Option<&[u32]> {
        Some(self.0[&doc.row_group].row(doc.row as usize))
    }
}

async fn all(d: &Data) -> All {
    let mut groups = HashMap::new();
    for g in 0..d.index.rows().len() as u32 {
        groups.insert(g, d.reader.read(&d.index, g).await.unwrap());
    }
    All(groups)
}

/// The global batch of each of `steps` steps, from the packer alone.
fn direct(d: &Data, all: &All, steps: usize) -> Vec<Vec<u32>> {
    let mut lanes: Vec<Lane> = (0..LANES)
        .map(|l| Lane::new(&d.orders, l, Position::default()).unwrap())
        .collect();
    let mut seq = vec![0; SEQ_LEN as usize];
    (0..steps)
        .map(|_| {
            let mut batch = Vec::new();
            for lane in &mut lanes {
                assert_eq!(lane.need(all), Need::Ready);
                lane.fill(all, &mut seq);
                batch.extend_from_slice(&seq);
            }
            batch
        })
        .collect()
}

/// About three epochs of steps: lanes cross two epoch ends.
fn steps(d: &Data) -> usize {
    let tokens: u64 = (0..d.index.rows().len() as u32)
        .map(|g| u64::from(d.index.row_group(g).1.values))
        .sum();
    (3 * tokens / u64::from(LANES * SEQ_LEN)) as usize
}

/// Runs a `world`-GPU job from `at` (one position per lane) for `steps`
/// steps. Returns each step's global batch and the positions after.
async fn run(
    d: &Data,
    world: u32,
    at: &[Position],
    steps: usize,
    budget: u64,
) -> (Vec<Vec<u32>>, Vec<Position>) {
    let mut batches = vec![Vec::new(); steps];
    let mut after = Vec::new();
    for rank in 0..world {
        let lanes = gpu_lanes(LANES, world, rank);
        let own = &at[lanes.start as usize..lanes.end as usize];
        let (o, r, i) = (d.orders.clone(), d.reader.clone(), d.index.clone());
        let mut loader = Loader::new(o, r, i, lanes.clone(), own, budget).unwrap();
        let mut out = vec![0; lanes.len() * SEQ_LEN as usize];
        for batch in &mut batches {
            loader.next(&mut out).await.unwrap();
            batch.extend_from_slice(&out);
        }
        after.extend(loader.positions());
    }
    (batches, after)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn loader_matches_direct() {
    for seed in [1, 2, 3] {
        let d = data(seed).await;
        let n = steps(&d);
        let want = direct(&d, &all(&d).await, n);
        let fresh = vec![Position::default(); LANES as usize];
        for world in [1, 2, 4] {
            let (got, _) = run(&d, world, &fresh, n, PLENTY).await;
            assert_eq!(got, want, "seed {seed}, world {world}");
        }
    }
}

// A run stopped at any step and resumed on another GPU count, from the
// positions alone, gives the rest of a run with no stop.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn resume_matches_tail() {
    let d = data(5).await;
    let n = steps(&d);
    let want = direct(&d, &all(&d).await, n);
    let fresh = vec![Position::default(); LANES as usize];
    for (a, b) in [(1, 2), (2, 4), (4, 1)] {
        for k in [0, 1, n / 3, n / 2, n - 1] {
            let (head, at) = run(&d, a, &fresh, k, PLENTY).await;
            let (tail, _) = run(&d, b, &at, n - k, PLENTY).await;
            let joined: Vec<_> = head.into_iter().chain(tail).collect();
            assert_eq!(joined, want, "world {a} then {b}, resume at {k}");
        }
    }
}

// A budget of exactly one window works: lanes a window apart get the reads
// they wait for past the budget, and the run never waits for itself.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn one_window_of_budget_is_enough() {
    let d = data(7).await;
    let n = steps(&d);
    let want = direct(&d, &all(&d).await, n);
    let fresh = vec![Position::default(); LANES as usize];
    for world in [1, 2] {
        let ranks = 0..world;
        let window = |r| sequence::window_kib(&d.orders, &d.index, gpu_lanes(LANES, world, r), 0);
        let budget = ranks.map(window).max().unwrap() * 1024;
        let run = run(&d, world, &fresh, n, budget);
        let got = tokio::time::timeout(std::time::Duration::from_secs(20), run).await;
        assert_eq!(got.expect("no deadlock").0, want, "world {world}");
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_budget_below_one_window_is_refused() {
    let d = data(7).await;
    let needed = sequence::window_kib(&d.orders, &d.index, 0..LANES, 0) * 1024;
    let fresh = vec![Position::default(); LANES as usize];
    let (o, r, i) = (d.orders.clone(), d.reader.clone(), d.index.clone());
    let got = Loader::new(o, r, i, 0..LANES, &fresh, needed - 1024);
    assert!(matches!(got, Err(Error::Budget { .. })), "{:?}", got.err());
}

/// The windows from `from` to `to`, both included, for lanes `lanes`.
fn windows(d: &Data, lanes: Range<u32>, from: (u64, usize), to: (u64, usize)) -> Vec<WindowGroups> {
    let mut seq = Sequence::new(&d.orders, lanes, from);
    let mut out = Vec::new();
    loop {
        let w = seq.next_window();
        if (w.epoch, w.window) > to {
            return out;
        }
        out.push(w);
    }
}

/// `n` windows after `slice`, across the epoch's end.
fn after(d: &Data, (e, w): (u64, usize), n: usize) -> (u64, usize) {
    let per = d.orders.epoch(e).windows().len();
    let ordinal = e as usize * per + w + n;
    ((ordinal / per) as u64, ordinal % per)
}

// Each row group in memory is in a window that some lane has not passed:
// once every lane has passed a window, its row groups are evicted.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn passed_windows_are_evicted() {
    for seed in [9, 10] {
        let d = data(seed).await;
        let fresh = vec![Position::default(); LANES as usize];
        let (o, r, i) = (d.orders.clone(), d.reader.clone(), d.index.clone());
        let mut loader = Loader::new(o, r, i, 0..LANES, &fresh, PLENTY).unwrap();
        let mut out = vec![0; (LANES * SEQ_LEN) as usize];
        for _ in 0..steps(&d) {
            loader.next(&mut out).await.unwrap();
            let slices: Vec<_> = loader.lanes.iter().map(Lane::slice).collect();
            let (slowest, fastest) = (*slices.iter().min().unwrap(), *slices.iter().max().unwrap());
            let live: HashSet<u32> = windows(&d, 0..LANES, slowest, after(&d, fastest, 4))
                .iter()
                .flat_map(|w| w.used().map(|i| w.groups[i]).collect::<Vec<_>>())
                .collect();
            for g in loader.buffer.row_groups() {
                assert!(
                    live.contains(g),
                    "seed {seed}: row group {g} outlives its windows"
                );
            }
        }
    }
}
