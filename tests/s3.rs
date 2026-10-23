//! The index and the reader against a real S3 server. CI runs this file in
//! its `s3` job against versitygw; run it locally with
//! `PLITZER_S3_ENDPOINT=http://127.0.0.1:7070 cargo test --features s3 --test s3`.
#![cfg(feature = "s3")]

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use arrow_array::builder::{GenericListBuilder, Int32Builder};
use arrow_array::{ArrayRef, RecordBatch};
use object_store::aws::AmazonS3Builder;
use object_store::path::Path;
use object_store::{ObjectStore, ObjectStoreExt, PutPayload};
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use plitzer::index::Index;
use plitzer::read::Reader;

fn env(name: &str, default: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| default.to_owned())
}

/// The store under test. A missing endpoint is a failure, not a skip, so
/// the CI job cannot pass without running.
fn store() -> Arc<dyn ObjectStore> {
    let endpoint = std::env::var("PLITZER_S3_ENDPOINT")
        .expect("set PLITZER_S3_ENDPOINT to an S3 endpoint, such as http://127.0.0.1:7070");
    let store = AmazonS3Builder::new()
        .with_endpoint(endpoint)
        .with_allow_http(true)
        .with_region("us-east-1")
        .with_bucket_name(env("PLITZER_S3_BUCKET", "plitzer-test"))
        .with_access_key_id(env("PLITZER_S3_ACCESS_KEY", "test"))
        .with_secret_access_key(env("PLITZER_S3_SECRET_KEY", "testsecret"))
        .build()
        .unwrap();
    Arc::new(store)
}

/// Row `r` of file `f` holds `r % 5` unique tokens.
fn tokens(f: usize, r: usize) -> Vec<u32> {
    (0..r % 5)
        .map(|i| (f * 10_000 + r * 10 + i) as u32)
        .collect()
}

fn parquet_bytes(f: usize, rows: usize) -> Vec<u8> {
    let mut b = GenericListBuilder::<i32, _>::new(Int32Builder::new());
    for r in 0..rows {
        tokens(f, r)
            .iter()
            .for_each(|&t| b.values().append_value(t as i32));
        b.append(true);
    }
    let column: ArrayRef = Arc::new(b.finish());
    let batch = RecordBatch::try_from_iter([("tokens", column)]).unwrap();
    let props = WriterProperties::builder()
        .set_max_row_group_row_count(Some(7))
        .build();
    let mut out = Vec::new();
    let mut w = ArrowWriter::try_new(&mut out, batch.schema(), Some(props)).unwrap();
    w.write(&batch).unwrap();
    w.close().unwrap();
    out
}

#[tokio::test]
async fn index_and_read_over_s3() {
    let store = store();
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let prefix = Path::from(format!("run-{nanos}"));
    for (f, rows) in [(0, 20), (1, 9)] {
        let location = prefix.clone().join(format!("part-{f}.parquet"));
        let payload = PutPayload::from(parquet_bytes(f, rows));
        store.put(&location, payload).await.unwrap();
    }

    let dir = tempfile::TempDir::new().unwrap();
    let cache = dir.path().join("index.bin");
    let index = Index::open(&*store, &prefix, "tokens", &cache)
        .await
        .unwrap();
    assert_eq!(index.rows(), [7, 7, 6, 7, 2]);
    let again = Index::open(&*store, &prefix, "tokens", &cache)
        .await
        .unwrap();
    assert_eq!(again, index);

    let reader = Reader::new(store.clone(), prefix.clone(), &index).unwrap();
    let mut got = Vec::new();
    for rg in 0..5 {
        let t = reader.read(&index, rg).await.unwrap();
        got.extend((0..t.rows()).map(|r| t.row(r).to_vec()));
    }
    let want: Vec<_> = (0..20)
        .map(|r| tokens(0, r))
        .chain((0..9).map(|r| tokens(1, r)))
        .collect();
    assert_eq!(got, want);
}

// `store::open` reads the URL's bucket and prefix, and the options.
#[tokio::test]
async fn open_an_s3_url() {
    let bucket = env("PLITZER_S3_BUCKET", "plitzer-test");
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let prefix = Path::from(format!("url-{nanos}"));
    let location = prefix.clone().join("part-0.parquet");
    let payload = PutPayload::from(parquet_bytes(0, 9));
    store().put(&location, payload).await.unwrap();

    let options: Vec<(String, String)> = [
        ("endpoint", std::env::var("PLITZER_S3_ENDPOINT").unwrap()),
        ("allow_http", "true".to_string()),
        ("region", "us-east-1".to_string()),
        ("access_key_id", env("PLITZER_S3_ACCESS_KEY", "test")),
        (
            "secret_access_key",
            env("PLITZER_S3_SECRET_KEY", "testsecret"),
        ),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v))
    .collect();
    let url = format!("s3://{bucket}/{prefix}");
    let (store, at) = plitzer::store::open(&url, &options).unwrap();
    assert_eq!(at, prefix);
    let index = Index::build(&*store, &at, "tokens").await.unwrap();
    assert_eq!(index.rows(), [7, 2]);
}
