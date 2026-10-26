use std::cell::RefCell;
use std::time::Duration;

use arrow_array::builder::{GenericListBuilder, Int32Builder};
use arrow_array::{ArrayRef, RecordBatch};
use object_store::local::LocalFileSystem;
use object_store::path::Path;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use tempfile::TempDir;

use super::plan::{at_cursors, chunks, cursors, plan};
use super::*;

mod budget;
mod cancel;
mod plan;
use crate::order::{Order, gpu_lanes};
use crate::pack::Packing;

const LANES: u32 = 4;
const PACKING: Packing = Packing {
    seq_len: 8,
    docs: 2,
    cap: 10,
    pad: u32::MAX,
};

struct Data {
    dir: TempDir,
    index: Arc<Index>,
    reader: Arc<Reader>,
    epoch: Arc<Epoch>,
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
    let store = Arc::new(LocalFileSystem::new_with_prefix(dir.path()).unwrap());
    let index = Index::build(&*store, &Path::from(""), "tokens")
        .await
        .unwrap();
    let reader = Reader::new(store, Path::from(""), &index).unwrap();
    let rows = index.rows();
    let order = Order {
        seed,
        epoch: 0,
        lanes: LANES,
        window: 3,
        rows: &rows,
    };
    let epoch = Arc::new(Epoch::new(order, PACKING));
    Data {
        dir,
        index: Arc::new(index),
        reader: Arc::new(reader),
        epoch,
    }
}

/// One global batch for each step, run without the loader over every row
/// group in memory. `log` records the row group of each document taken.
struct All {
    groups: HashMap<u32, RowGroupTokens>,
    log: RefCell<Vec<u32>>,
}

impl Tokens for All {
    fn tokens(&self, doc: Doc) -> &[u32] {
        self.log.borrow_mut().push(doc.row_group);
        self.groups[&doc.row_group].row(doc.row as usize)
    }
}

async fn all(d: &Data) -> All {
    let mut groups = HashMap::new();
    for rg in 0..d.index.rows().len() as u32 {
        let tokens = d.reader.read(&d.index, rg).await.unwrap();
        // The budget cost from the index is never below the real size.
        assert!(tokens.bytes() as u64 <= d.index.row_group(rg).1.decoded_bytes());
        groups.insert(rg, tokens);
    }
    All {
        groups,
        log: RefCell::default(),
    }
}

fn direct(d: &Data, all: &All) -> Vec<Vec<u32>> {
    let mut lanes: Vec<Lane> = (0..LANES)
        .map(|l| d.epoch.lane(l, LaneState::default(), all).unwrap())
        .collect();
    let mut seq = vec![0; PACKING.seq_len as usize];
    (0..d.epoch.steps())
        .map(|_| {
            lanes
                .iter_mut()
                .flat_map(|l| {
                    l.fill(all, &mut seq);
                    seq.clone()
                })
                .collect()
        })
        .collect()
}

/// Runs the loaders of a `world`-GPU job from `states` (one per lane) for at
/// most `steps` steps. Returns the global batch of each step and the states
/// after the last.
async fn run(
    d: &Data,
    world: u32,
    states: &[LaneState],
    steps: u64,
    budget: u64,
) -> (Vec<Vec<u32>>, Vec<LaneState>) {
    let mut batches: Vec<Vec<u32>> = Vec::new();
    let mut after = Vec::new();
    for rank in 0..world {
        let lanes = gpu_lanes(LANES, world, rank);
        let own = &states[lanes.start as usize..lanes.end as usize];
        let mut loader = Loader::new(
            d.epoch.clone(),
            d.reader.clone(),
            d.index.clone(),
            lanes.clone(),
            own,
            budget,
        )
        .await
        .unwrap();
        let mut out = vec![0; lanes.len() * PACKING.seq_len as usize];
        for step in 0..steps {
            let Some(_) = loader.next(&mut out).await.unwrap() else {
                break;
            };
            assert!(loader.resident_bytes() as u64 <= budget);
            if batches.len() <= step as usize {
                batches.push(Vec::new());
            }
            batches[step as usize].extend_from_slice(&out);
        }
        after.extend(loader.states());
    }
    (batches, after)
}

const PLENTY: u64 = 1 << 30;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn loader_matches_direct() {
    for seed in [1, 2, 3] {
        let d = data(seed).await;
        let want = direct(&d, &all(&d).await);
        assert!(!want.is_empty());
        let fresh = vec![LaneState::default(); LANES as usize];
        for world in [1, 2, 4] {
            let (got, _) = run(&d, world, &fresh, u64::MAX, PLENTY).await;
            assert_eq!(got, want, "seed {seed}, world {world}");
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn resume_matches_tail() {
    let d = data(5).await;
    let want = direct(&d, &all(&d).await);
    let fresh = vec![LaneState::default(); LANES as usize];
    for (a, b) in [(1, 2), (2, 4), (4, 1)] {
        for k in [0, 1, want.len() as u64 / 2, want.len() as u64] {
            let (head, states) = run(&d, a, &fresh, k, PLENTY).await;
            let (tail, _) = run(&d, b, &states, u64::MAX, PLENTY).await;
            let joined: Vec<_> = head.into_iter().chain(tail).collect();
            assert_eq!(joined, want, "world {a} then {b}, resume at {k}");
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn evicts_everything() {
    let d = data(9).await;
    let fresh = vec![LaneState::default(); LANES as usize];
    let mut loader = Loader::new(
        d.epoch.clone(),
        d.reader.clone(),
        d.index.clone(),
        0..LANES,
        &fresh,
        PLENTY,
    )
    .await
    .unwrap();
    let mut out = vec![0; (LANES * PACKING.seq_len) as usize];
    while loader.next(&mut out).await.unwrap().is_some() {}
    assert_eq!(loader.resident_bytes(), 0);
    assert!(loader.buffer.0.is_empty());
    assert_eq!(loader.next, loader.needs.len());
}

// With room for everything, the prefetch reads every row group ahead of
// use, without any call to `next`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn prefetches_ahead() {
    let d = data(15).await;
    let total: usize = all(&d)
        .await
        .groups
        .values()
        .map(RowGroupTokens::bytes)
        .sum();
    let fresh = vec![LaneState::default(); LANES as usize];
    let loader = Loader::new(
        d.epoch.clone(),
        d.reader.clone(),
        d.index.clone(),
        0..LANES,
        &fresh,
        PLENTY,
    )
    .await
    .unwrap();
    let full = async {
        while loader.resident_bytes() < total {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    };
    tokio::time::timeout(Duration::from_secs(10), full)
        .await
        .expect("every row group prefetched");
}

// A read that fails mid-epoch returns its error once. The lanes have not
// moved, so `states()` is a valid resume point, and later calls report
// `Stopped` instead of reading a row group that never arrived.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn failed_read_stops_the_loader() {
    let d = data(17).await;
    let fresh = vec![LaneState::default(); LANES as usize];
    let needs = plan(&d.epoch, &d.index, 0..LANES, &fresh);
    let bad = needs
        .iter()
        .find(|n| n.first >= 2)
        .expect("a row group needed later");
    let (file, rg) = d.index.row_group(bad.row_group);
    let path = d.dir.path().join(&file.path);
    let mut bytes = std::fs::read(&path).unwrap();
    let chunk = rg.start as usize..rg.start as usize + rg.len as usize;
    bytes[chunk].fill(0xAB);
    std::fs::write(&path, bytes).unwrap();

    let mut loader = Loader::new(
        d.epoch.clone(),
        d.reader.clone(),
        d.index.clone(),
        0..LANES,
        &fresh,
        PLENTY,
    )
    .await
    .unwrap();
    let mut out = vec![0; (LANES * PACKING.seq_len) as usize];
    let mut before = loader.states();
    let err = loop {
        match loader.next(&mut out).await {
            Ok(Some(_)) => before = loader.states(),
            Ok(None) => panic!("the corrupt row group was never read"),
            Err(e) => break e,
        }
    };
    assert!(matches!(err, Error::Read(_)), "{err:?}");
    assert_eq!(loader.states(), before);
    assert!(matches!(loader.next(&mut out).await, Err(Error::Stopped)));
}
