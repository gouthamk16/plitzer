use std::sync::Arc;

use object_store::local::LocalFileSystem;
use parquet::data_type::Int32Type;
use parquet::file::properties::WriterProperties;
use parquet::file::writer::SerializedFileWriter;
use parquet::schema::parser::parse_message_type;
use tempfile::TempDir;

use super::sidecar::tests::{file, index};
use super::*;
use crate::order::Order;

const SCHEMA: &str = "message m {
    required int32 id;
    required group tokens (LIST) { repeated group list { required int32 element; } }
}";

/// Writes a Parquet file with one row group per entry of `groups`, each row
/// with one token.
fn write_parquet(path: &std::path::Path, groups: &[u32]) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let schema = Arc::new(parse_message_type(SCHEMA).unwrap());
    let props = Arc::new(WriterProperties::builder().build());
    let mut w =
        SerializedFileWriter::new(std::fs::File::create(path).unwrap(), schema, props).unwrap();
    for &n in groups {
        let mut rg = w.next_row_group().unwrap();
        let values: Vec<i32> = (0..n as i32).collect();
        let (def, rep) = (vec![1; n as usize], vec![0; n as usize]);
        for levels in [None, Some((&def[..], &rep[..]))] {
            let mut col = rg.next_column().unwrap().unwrap();
            let (d, r) = levels.unzip();
            col.typed::<Int32Type>().write_batch(&values, d, r).unwrap();
            col.close().unwrap();
        }
        rg.close().unwrap();
    }
    w.close().unwrap();
}

/// A store over a temp dir holding `data/` with three Parquet files (one
/// nested, one with no row groups) and two files the index must skip.
fn dataset() -> (TempDir, LocalFileSystem, Path) {
    let dir = TempDir::new().unwrap();
    let data = dir.path().join("data");
    write_parquet(&data.join("b.parquet"), &[3, 1]);
    write_parquet(&data.join("a/x.parquet"), &[2]);
    write_parquet(&data.join("c.parquet"), &[]);
    write_parquet(&data.join("d.parquet"), &[4, 0, 5]);
    std::fs::write(data.join("notes.txt"), "not parquet").unwrap();
    std::fs::write(dir.path().join("outside.parquet"), "not below the prefix").unwrap();
    let store = LocalFileSystem::new_with_prefix(dir.path()).unwrap();
    (dir, store, Path::from("data"))
}

fn summary(index: &Index) -> Vec<(&str, Vec<u32>)> {
    let rows = |f: &File| f.row_groups.iter().map(|g| g.rows).collect();
    index
        .files()
        .iter()
        .map(|f| (f.path.as_str(), rows(f)))
        .collect()
}

fn build(store: &LocalFileSystem, prefix: &Path) -> impl Future<Output = Result<Index, Error>> {
    Index::build(store, prefix, "tokens")
}

fn open<'a>(
    store: &'a LocalFileSystem,
    prefix: &'a Path,
    cache: &'a std::path::Path,
) -> impl Future<Output = Result<Index, Error>> + 'a {
    Index::open(store, prefix, "tokens", cache)
}

fn expected() -> Vec<(&'static str, Vec<u32>)> {
    vec![
        ("a/x.parquet", vec![2]),
        ("b.parquet", vec![3, 1]),
        ("c.parquet", vec![]),
        ("d.parquet", vec![4, 0, 5]),
    ]
}

#[tokio::test]
async fn build_reads_row_groups() {
    let (_dir, store, prefix) = dataset();
    let index = build(&store, &prefix).await.unwrap();
    assert_eq!(summary(&index), expected());
    assert_eq!(index.rows(), [2, 3, 1, 4, 0, 5]);
    let located: Vec<_> = (0..6).map(|rg| index.locate(rg)).collect();
    assert_eq!(located, [(0, 0), (1, 0), (1, 1), (3, 0), (3, 1), (3, 2)]);

    let rows = index.rows();
    let order = Order {
        seed: 7,
        epoch: 0,
        lanes: 2,
        window: 2,
        rows: &rows,
    };
    let docs: usize = order
        .windows()
        .iter()
        .flat_map(|w| [order.lane(w, 0), order.lane(w, 1)])
        .map(|l| l.len())
        .sum();
    assert_eq!(docs, 15);
}

#[test]
#[should_panic(expected = "not in the index")]
fn locate_past_end_panics() {
    index(vec![file("a", &[1, 1]), file("b", &[])]).locate(2);
}

#[tokio::test]
async fn open_writes_and_reuses_cache() {
    let (dir, store, prefix) = dataset();
    let cache = dir.path().join("index.bin");
    let built = open(&store, &prefix, &cache).await.unwrap();
    assert_eq!(summary(&built), expected());
    let written = std::fs::read(&cache).unwrap();
    assert_eq!(open(&store, &prefix, &cache).await.unwrap(), built);
    assert_eq!(std::fs::read(&cache).unwrap(), written);
}

// A sidecar that matches the listing but holds fake row counts. `open` must
// return the fake counts, which shows that it read no footers.
#[tokio::test]
async fn open_uses_cache_without_footers() {
    let (dir, store, prefix) = dataset();
    let cache = dir.path().join("index.bin");
    let mut fake = build(&store, &prefix).await.unwrap();
    fake.files[0].row_groups[0].rows = 99;
    std::fs::write(&cache, sidecar::encode(&fake, true)).unwrap();
    let got = open(&store, &prefix, &cache).await.unwrap();
    assert_eq!(got.files[0].row_groups[0].rows, 99);

    // A sidecar for a different column is not trusted.
    let other = Index::open(&store, &prefix, "id", &cache).await;
    assert!(matches!(other, Err(Error::Column { .. })), "{other:?}");

    // The same sidecar with a stale ETag is not trusted.
    fake.files[0].e_tag = Some("stale".into());
    std::fs::write(&cache, sidecar::encode(&fake, true)).unwrap();
    assert_eq!(
        summary(&open(&store, &prefix, &cache).await.unwrap()),
        expected()
    );
}

#[tokio::test]
async fn open_rebuilds_on_change() {
    let (dir, store, prefix) = dataset();
    let cache = dir.path().join("index.bin");
    open(&store, &prefix, &cache).await.unwrap();

    write_parquet(&dir.path().join("data/b.parquet"), &[7]);
    write_parquet(&dir.path().join("data/e.parquet"), &[1, 2]);
    let index = open(&store, &prefix, &cache).await.unwrap();
    assert_eq!(index.rows(), [2, 7, 4, 0, 5, 1, 2]);
    assert_eq!(
        sidecar::decode(&std::fs::read(&cache).unwrap()).unwrap(),
        index
    );

    std::fs::write(&cache, b"garbage").unwrap();
    assert_eq!(open(&store, &prefix, &cache).await.unwrap(), index);
}

#[tokio::test]
async fn bad_footer_names_file() {
    // A bad magic number, a file shorter than a footer, and a footer length
    // larger than the file (which panics inside parquet 60 unchecked).
    let mut huge = b"xxxx".to_vec();
    huge.extend(u32::MAX.to_le_bytes());
    huge.extend(b"PAR1");
    for bytes in [&b"PAR1 not a footer"[..], b"PAR1", &huge] {
        let (dir, store, prefix) = dataset();
        std::fs::write(dir.path().join("data/bad.parquet"), bytes).unwrap();
        match build(&store, &prefix).await {
            Err(Error::Footer { path, .. }) => assert_eq!(path, "bad.parquet"),
            other => panic!("expected a footer error, got {other:?}"),
        }
    }
}

#[tokio::test]
async fn no_files_is_an_error() {
    let (_dir, store, _) = dataset();
    let result = build(&store, &Path::from("typo")).await;
    assert!(matches!(result, Err(Error::NoFiles { .. })), "{result:?}");
}

// Processes on one node (one per GPU) open the same cold cache at once.
#[test]
fn concurrent_open_shares_cache() {
    let (dir, store, prefix) = dataset();
    let cache = dir.path().join("index.bin");
    // Rounds on a cold cache, since a race shows up only some of the time.
    for _ in 0..10 {
        let _ = std::fs::remove_file(&cache);
        open_at_once(&store, &prefix, &cache);
    }
    let cached = sidecar::decode(&std::fs::read(&cache).unwrap()).unwrap();
    assert_eq!(summary(&cached), expected());
}

fn open_at_once(store: &LocalFileSystem, prefix: &Path, cache: &std::path::Path) {
    std::thread::scope(|s| {
        let opens: Vec<_> = (0..8)
            .map(|_| {
                s.spawn(|| {
                    let rt = tokio::runtime::Builder::new_current_thread()
                        .build()
                        .unwrap();
                    rt.block_on(open(store, prefix, cache)).unwrap()
                })
            })
            .collect();
        for open in opens {
            assert_eq!(summary(&open.join().unwrap()), expected());
        }
    });
}

#[tokio::test]
async fn old_sidecar_rebuilds() {
    let (dir, store, prefix) = dataset();
    let cache = dir.path().join("index.bin");
    let mut bytes = sidecar::encode(&build(&store, &prefix).await.unwrap(), true);
    bytes[8..12].copy_from_slice(&2u32.to_le_bytes());
    assert_eq!(sidecar::decode(&bytes), None);
    std::fs::write(&cache, bytes).unwrap();
    assert_eq!(
        summary(&open(&store, &prefix, &cache).await.unwrap()),
        expected()
    );
}
