use std::sync::Arc;

use object_store::local::LocalFileSystem;
use parquet::data_type::Int64Type;
use parquet::file::properties::WriterProperties;
use parquet::file::writer::SerializedFileWriter;
use parquet::schema::parser::parse_message_type;
use proptest::prelude::*;
use tempfile::TempDir;

use super::*;
use crate::order::Order;

/// Writes a Parquet file with one row group per entry of `groups`.
fn write_parquet(path: &std::path::Path, groups: &[u32]) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let schema = Arc::new(parse_message_type("message m { required int64 x; }").unwrap());
    let props = Arc::new(WriterProperties::builder().build());
    let mut w =
        SerializedFileWriter::new(std::fs::File::create(path).unwrap(), schema, props).unwrap();
    for &n in groups {
        let mut rg = w.next_row_group().unwrap();
        let mut col = rg.next_column().unwrap().unwrap();
        let values: Vec<i64> = (0..i64::from(n)).collect();
        col.typed::<Int64Type>()
            .write_batch(&values, None, None)
            .unwrap();
        col.close().unwrap();
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

fn summary(index: &Index) -> Vec<(&str, &[u32])> {
    index
        .files()
        .iter()
        .map(|f| (f.path.as_str(), f.rows.as_slice()))
        .collect()
}

const EXPECTED: [(&str, &[u32]); 4] = [
    ("a/x.parquet", &[2]),
    ("b.parquet", &[3, 1]),
    ("c.parquet", &[]),
    ("d.parquet", &[4, 0, 5]),
];

#[tokio::test]
async fn build_reads_row_groups() {
    let (_dir, store, prefix) = dataset();
    let index = Index::build(&store, &prefix).await.unwrap();
    assert_eq!(summary(&index), EXPECTED);
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
    let files = vec![file("a", &[1, 1]), file("b", &[])];
    Index::new(files).unwrap().locate(2);
}

fn file(path: &str, rows: &[u32]) -> File {
    File {
        path: path.into(),
        size: 10,
        e_tag: Some("t".into()),
        rows: rows.to_vec(),
    }
}

#[tokio::test]
async fn open_writes_and_reuses_cache() {
    let (dir, store, prefix) = dataset();
    let cache = dir.path().join("index.bin");
    let built = Index::open(&store, &prefix, &cache).await.unwrap();
    assert_eq!(summary(&built), EXPECTED);
    let written = std::fs::read(&cache).unwrap();
    assert_eq!(Index::open(&store, &prefix, &cache).await.unwrap(), built);
    assert_eq!(std::fs::read(&cache).unwrap(), written);
}

// A sidecar that matches the listing but holds fake row counts. `open` must
// return the fake counts, which shows that it read no footers.
#[tokio::test]
async fn open_uses_cache_without_footers() {
    let (dir, store, prefix) = dataset();
    let cache = dir.path().join("index.bin");
    let mut files = Index::build(&store, &prefix).await.unwrap().files;
    files[0].rows = vec![99];
    std::fs::write(&cache, sidecar::encode(&files, true)).unwrap();
    assert_eq!(
        Index::open(&store, &prefix, &cache).await.unwrap().files[0].rows,
        [99]
    );

    // The same sidecar with a stale ETag is not trusted.
    files[0].e_tag = Some("stale".into());
    std::fs::write(&cache, sidecar::encode(&files, true)).unwrap();
    assert_eq!(
        summary(&Index::open(&store, &prefix, &cache).await.unwrap()),
        EXPECTED
    );
}

#[tokio::test]
async fn open_rebuilds_on_change() {
    let (dir, store, prefix) = dataset();
    let cache = dir.path().join("index.bin");
    Index::open(&store, &prefix, &cache).await.unwrap();

    write_parquet(&dir.path().join("data/b.parquet"), &[7]);
    write_parquet(&dir.path().join("data/e.parquet"), &[1, 2]);
    let index = Index::open(&store, &prefix, &cache).await.unwrap();
    assert_eq!(index.rows(), [2, 7, 4, 0, 5, 1, 2]);
    assert_eq!(
        sidecar::decode(&std::fs::read(&cache).unwrap()).unwrap(),
        index.files
    );

    std::fs::write(&cache, b"garbage").unwrap();
    assert_eq!(Index::open(&store, &prefix, &cache).await.unwrap(), index);
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
        match Index::build(&store, &prefix).await {
            Err(Error::Footer { path, .. }) => assert_eq!(path, "bad.parquet"),
            other => panic!("expected a footer error, got {other:?}"),
        }
    }
}

#[tokio::test]
async fn no_files_is_an_error() {
    let (_dir, store, _) = dataset();
    let result = Index::build(&store, &Path::from("typo")).await;
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
    let files = sidecar::decode(&std::fs::read(&cache).unwrap()).unwrap();
    assert_eq!(files.len(), EXPECTED.len());
}

fn open_at_once(store: &LocalFileSystem, prefix: &Path, cache: &std::path::Path) {
    std::thread::scope(|s| {
        let opens: Vec<_> = (0..8)
            .map(|_| {
                s.spawn(|| {
                    let rt = tokio::runtime::Builder::new_current_thread()
                        .build()
                        .unwrap();
                    rt.block_on(Index::open(store, prefix, cache)).unwrap()
                })
            })
            .collect();
        for open in opens {
            assert_eq!(summary(&open.join().unwrap()), EXPECTED);
        }
    });
}

#[test]
fn fingerprint_golden() {
    let index = Index::new(vec![file("a/x.parquet", &[2]), file("b.parquet", &[3, 1])]).unwrap();
    assert_eq!(index.fingerprint(), 2_109_774_877_995_249_362);
    // The cache bytes, with ETags, are pinned too.
    let cache = sidecar::fnv1a(&sidecar::encode(&index.files, true));
    assert_eq!(cache, 10_182_121_034_693_317_030);

    let mut files = index.files.clone();
    files[0].e_tag = None;
    assert_eq!(
        Index::new(files.clone()).unwrap().fingerprint(),
        index.fingerprint()
    );
    files[1].rows[1] = 2;
    assert_ne!(
        Index::new(files).unwrap().fingerprint(),
        index.fingerprint()
    );
}

fn files() -> impl Strategy<Value = Vec<File>> {
    let f = (
        ".{0,8}",
        any::<u64>(),
        proptest::option::of(".{0,8}"),
        prop::collection::vec(any::<u32>(), 0..5),
    );
    prop::collection::vec(
        f.prop_map(|(path, size, e_tag, rows)| File {
            path,
            size,
            e_tag,
            rows,
        }),
        0..5,
    )
}

proptest! {
    #[test]
    fn sidecar_roundtrip(files in files()) {
        prop_assert_eq!(sidecar::decode(&sidecar::encode(&files, true)), Some(files));
    }

    #[test]
    fn decode_never_panics(files in files(), cut in any::<prop::sample::Index>(), junk in prop::collection::vec(any::<u8>(), 0..64)) {
        let bytes = sidecar::encode(&files, true);
        let cut = cut.index(bytes.len());
        prop_assert_eq!(sidecar::decode(&bytes[..cut]), None);
        sidecar::decode(&junk);
    }
}
