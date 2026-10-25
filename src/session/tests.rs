use arrow_array::builder::{GenericListBuilder, Int32Builder};
use arrow_array::{ArrayRef, RecordBatch};
use object_store::local::LocalFileSystem;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use tempfile::TempDir;

use super::*;

mod cancel;
mod refuse;
use crate::pack::Packing;

fn config() -> Config {
    Config {
        column: "tokens".into(),
        seed: 11,
        lanes: 4,
        window: 3,
        packing: Packing {
            seq_len: 8,
            docs: 2,
            cap: 10,
            pad: u32::MAX,
        },
    }
}

const PLENTY: u64 = 1 << 30;

/// Writes file `f` of row groups of 5 rows; row `r` has `3 * ((f + r) % 7)`
/// unique tokens.
fn write_file(dir: &std::path::Path, f: i32) {
    let mut b = GenericListBuilder::<i32, _>::new(Int32Builder::new());
    for r in 0..(6 + 3 * f) {
        for t in 0..3 * ((f + r) % 7) {
            b.values().append_value(f * 10_000 + r * 100 + t);
        }
        b.append(true);
    }
    let column: ArrayRef = Arc::new(b.finish());
    let batch = RecordBatch::try_from_iter([("tokens", column)]).unwrap();
    let props = WriterProperties::builder()
        .set_max_row_group_row_count(Some(5))
        .build();
    let file = std::fs::File::create(dir.join(format!("{f}.parquet"))).unwrap();
    let mut w = ArrowWriter::try_new(file, batch.schema(), Some(props)).unwrap();
    w.write(&batch).unwrap();
    w.close().unwrap();
}

struct Data {
    files: TempDir,
    cache: TempDir,
}

fn data() -> Data {
    let files = TempDir::new().unwrap();
    for f in 0..4 {
        write_file(files.path(), f);
    }
    let cache = TempDir::new().unwrap();
    Data { files, cache }
}

async fn open(d: &Data, config: Config, rank: u32, world: u32) -> Result<Session, Error> {
    let store = Arc::new(LocalFileSystem::new_with_prefix(d.files.path()).unwrap());
    let cache = d.cache.path().join(format!("index-{rank}-{world}"));
    Session::open(store, Path::from(""), &cache, config, rank, world, PLENTY).await
}

async fn sessions(d: &Data, world: u32) -> Vec<Session> {
    let mut all = Vec::new();
    for rank in 0..world {
        all.push(open(d, config(), rank, world).await.unwrap());
    }
    all
}

/// The next `steps` global steps: each step's batch, all ranks in order, or
/// `None` where an epoch ended.
async fn run(sessions: &mut [Session], steps: usize) -> Vec<Option<Vec<u32>>> {
    let mut out = Vec::new();
    for _ in 0..steps {
        let mut batch = Vec::new();
        let mut ended = Vec::new();
        for s in sessions.iter_mut() {
            let mut seq = vec![0; s.lanes().len() * config().packing.seq_len as usize];
            ended.push(s.next(&mut seq).await.unwrap().is_none());
            batch.extend(seq);
        }
        assert!(ended.iter().all(|&e| e == ended[0]), "ranks end together");
        out.push((!ended[0]).then_some(batch));
    }
    out
}

/// Epochs 0 and 1 and the start of epoch 2 from one GPU, without a stop.
async fn reference(d: &Data) -> Vec<Option<Vec<u32>>> {
    let mut one = sessions(d, 1).await;
    let mut out = Vec::new();
    while out.iter().filter(|b: &&Option<_>| b.is_none()).count() < 2 {
        out.extend(run(&mut one, 1).await);
    }
    out.extend(run(&mut one, 2).await);
    out
}

fn global(sessions: &[Session]) -> Checkpoint {
    let parts = sessions.iter().map(Session::checkpoint).collect();
    let merged = Checkpoint::merge(parts).unwrap();
    Checkpoint::from_bytes(&merged.to_bytes()).unwrap()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn resume_on_another_world_matches_one_run() {
    let d = data();
    let want = reference(&d).await;
    let end = want.iter().position(Option::is_none).unwrap();
    for (a, b) in [(2, 4), (4, 1), (1, 2)] {
        for k in [0, 1, end / 2, end, end + 1, end + 3] {
            let mut first = sessions(&d, a).await;
            let mut got = run(&mut first, k).await;
            let checkpoint = global(&first);
            let mut second = sessions(&d, b).await;
            for s in &mut second {
                s.resume(&checkpoint).await.unwrap();
            }
            // A checkpoint after the last step of an epoch is already at the
            // next epoch, so the resumed run gives no `None` for the old one.
            let mut tail = &want[k..];
            if tail.first() == Some(&None) {
                tail = &tail[1..];
            }
            got.extend(run(&mut second, tail.len()).await);
            assert_eq!(
                got,
                [&want[..k], tail].concat(),
                "world {a} then {b}, resume at {k}"
            );
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn epochs_roll_over() {
    let d = data();
    let want = reference(&d).await;
    let end = want.iter().position(Option::is_none).unwrap();
    assert_ne!(want[..end], want[end + 1..2 * end + 1], "epochs differ");
    let next_epoch =
        |c: Checkpoint| c.epoch == 1 && c.lanes.iter().all(|&s| s == LaneState::default());
    let mut two = sessions(&d, 2).await;
    run(&mut two, end).await;
    assert!(two[0].ended() && two[1].ended());
    assert!(
        next_epoch(two[0].checkpoint()),
        "before the call that gives None"
    );
    let mut seq = vec![0; (2 * config().packing.seq_len) as usize];
    assert_eq!(two[0].next(&mut seq).await.unwrap(), None);
    assert_eq!((two[0].epoch(), two[1].epoch()), (1, 0));
    assert!(!two[0].ended() && two[1].ended());
    // One rank saw `None` and one did not: their parts still agree.
    assert!(next_epoch(global(&two)));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn set_epoch_starts_that_epoch() {
    let d = data();
    let want = reference(&d).await;
    let end = want.iter().position(Option::is_none).unwrap();
    let mut two = sessions(&d, 2).await;
    run(&mut two, 3).await;
    for s in &mut two {
        s.set_epoch(1);
    }
    assert_eq!(run(&mut two, end + 1).await, want[end + 1..2 * end + 2]);
}

// A trainer that saved after an epoch's last step resumes into the next
// epoch, then calls `set_epoch` with the epoch it saved in.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn set_epoch_never_goes_back() {
    let d = data();
    let want = reference(&d).await;
    let end = want.iter().position(Option::is_none).unwrap();
    let mut one = sessions(&d, 1).await;
    run(&mut one, end).await;
    let checkpoint = global(&one);
    let mut again = sessions(&d, 1).await;
    again[0].resume(&checkpoint).await.unwrap();
    again[0].set_epoch(0);
    assert_eq!(run(&mut again, 2).await, want[end + 1..end + 3]);
}

// Trainers call `set_epoch` at the start of each epoch, also the one they
// resumed into, so it must keep the resumed position.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn set_epoch_keeps_a_resume_in_that_epoch() {
    let d = data();
    let want = reference(&d).await;
    let mut two = sessions(&d, 2).await;
    run(&mut two, 3).await;
    let checkpoint = global(&two);
    let mut again = sessions(&d, 2).await;
    for s in &mut again {
        s.resume(&checkpoint).await.unwrap();
        s.set_epoch(0);
    }
    assert_eq!(run(&mut again, 4).await, want[3..7]);
}

// A failed read leaves the session before the failed step, so the next call
// retries that step once the data is readable again.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn next_retries_after_a_failed_read() {
    let d = data();
    let want = reference(&d).await;
    let mut one = sessions(&d, 1).await;
    let mut got = run(&mut one, 2).await;
    let path = d.files.path().join("3.parquet");
    let good = std::fs::read(&path).unwrap();
    // The chunks come first and the footer last; the reader never reads the
    // footer, so damage the chunks.
    let mut bad = good.clone();
    bad[4..good.len() / 2].fill(0xAB);
    std::fs::write(&path, &bad).unwrap();
    // Some later steps may still be served from memory before the failure.
    let mut seq = vec![0; (4 * config().packing.seq_len) as usize];
    loop {
        assert!(got.len() < want.len(), "the damaged file was never read");
        match one[0].next(&mut seq).await {
            Ok(real) => got.push(real.map(|_| seq.clone())),
            Err(e) => {
                assert!(matches!(e, Error::Load(load::Error::Read(_))), "{e}");
                break;
            }
        }
    }
    std::fs::write(&path, &good).unwrap();
    let rest = want.len() - got.len();
    got.extend(run(&mut one, rest).await);
    assert_eq!(got, want);
}
