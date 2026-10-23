use std::sync::Arc;

use arrow_array::builder::{
    Date32Builder, GenericListBuilder, Int32Builder, Int64Builder, TimestampNanosecondBuilder,
    UInt32Builder,
};
use arrow_array::{ArrayRef, Int64Array, RecordBatch};
use object_store::local::LocalFileSystem;
use parquet::arrow::ArrowWriter;
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::WriterProperties;
use tempfile::TempDir;

use super::*;
use crate::index::Error as IndexError;

/// The tokens of row `r`: `r % 7` tokens, unique across rows. Every fifth
/// row is null when the column is nullable, which reads as empty.
fn expected(r: usize, nullable: bool) -> Vec<u32> {
    if nullable && r.is_multiple_of(5) {
        return Vec::new();
    }
    (0..r % 7).map(|i| (r * 100 + i) as u32).collect()
}

#[derive(Clone, Copy, Debug)]
enum Kind {
    ListI32,
    ListU32,
    LargeListI32,
    ListI64,
}

fn token_column(kind: Kind, rows: usize, nullable: bool) -> ArrayRef {
    macro_rules! build {
        ($offset:ty, $values:expr, $cast:ty) => {{
            let mut b = GenericListBuilder::<$offset, _>::new($values);
            for r in 0..rows {
                if nullable && r.is_multiple_of(5) {
                    b.append_null();
                    continue;
                }
                expected(r, false)
                    .iter()
                    .for_each(|&t| b.values().append_value(t as $cast));
                b.append(true);
            }
            Arc::new(b.finish()) as ArrayRef
        }};
    }
    match kind {
        Kind::ListI32 => build!(i32, Int32Builder::new(), i32),
        Kind::ListU32 => build!(i32, UInt32Builder::new(), u32),
        Kind::LargeListI32 => build!(i64, Int32Builder::new(), i32),
        Kind::ListI64 => build!(i32, Int64Builder::new(), i64),
    }
}

/// Writes `rows` rows with an `id` column before `tokens`, in row groups of 9.
fn write(path: &std::path::Path, tokens: ArrayRef, props: WriterProperties) {
    write_named(path, "tokens", tokens, props);
}

fn write_named(path: &std::path::Path, name: &str, tokens: ArrayRef, props: WriterProperties) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let ids: ArrayRef = Arc::new(Int64Array::from_iter_values(0..tokens.len() as i64));
    let batch = RecordBatch::try_from_iter([("id", ids), (name, tokens)]).unwrap();
    let file = std::fs::File::create(path).unwrap();
    let mut w = ArrowWriter::try_new(file, batch.schema(), Some(props)).unwrap();
    w.write(&batch).unwrap();
    w.close().unwrap();
}

fn props(compression: Compression, dictionary: bool) -> WriterProperties {
    WriterProperties::builder()
        .set_compression(compression)
        .set_dictionary_enabled(dictionary)
        .set_max_row_group_row_count(Some(9))
        .build()
}

struct Data {
    dir: TempDir,
    store: Arc<LocalFileSystem>,
    prefix: Path,
}

fn data() -> Data {
    let dir = TempDir::new().unwrap();
    let store = Arc::new(LocalFileSystem::new_with_prefix(dir.path()).unwrap());
    Data {
        dir,
        store,
        prefix: Path::from("data"),
    }
}

impl Data {
    fn path(&self, name: &str) -> std::path::PathBuf {
        self.dir.path().join("data").join(name)
    }

    async fn index(&self) -> Result<Index, IndexError> {
        Index::build(&*self.store, &self.prefix, "tokens").await
    }

    /// Every row of every row group, in index order.
    async fn read_all(&self, index: &Index) -> Vec<Vec<u32>> {
        let reader = Reader::new(self.store.clone(), self.prefix.clone(), index).unwrap();
        let mut rows = Vec::new();
        for rg in 0..index.rows().len() as u32 {
            let tokens = reader.read(index, rg).await.unwrap();
            rows.extend((0..tokens.rows()).map(|r| tokens.row(r).to_vec()));
        }
        rows
    }
}

#[tokio::test]
async fn reads_every_variant() {
    let kinds = [
        Kind::ListI32,
        Kind::ListU32,
        Kind::LargeListI32,
        Kind::ListI64,
    ];
    let codecs = [
        Compression::UNCOMPRESSED,
        Compression::SNAPPY,
        Compression::ZSTD(ZstdLevel::default()),
    ];
    for kind in kinds {
        for nullable in [false, true] {
            for compression in codecs {
                for dictionary in [false, true] {
                    let d = data();
                    let tokens = token_column(kind, 40, nullable);
                    // A space and a percent sign check the path round trip.
                    write(
                        &d.path("a b%.parquet"),
                        tokens,
                        props(compression, dictionary),
                    );
                    let index = d.index().await.unwrap();
                    let want: Vec<_> = (0..40).map(|r| expected(r, nullable)).collect();
                    let case = (kind, nullable, compression, dictionary);
                    assert_eq!(d.read_all(&index).await, want, "{case:?}");
                }
            }
        }
    }
}

// The file's footer length and magic are zeroed after indexing. The reader
// still decodes every row group, so it never reads the footer.
#[tokio::test]
async fn reads_without_footer() {
    let d = data();
    let path = d.path("x.parquet");
    write(
        &path,
        token_column(Kind::ListI32, 40, true),
        props(Compression::SNAPPY, true),
    );
    let index = d.index().await.unwrap();
    let mut bytes = std::fs::read(&path).unwrap();
    let n = bytes.len();
    bytes[n - 8..].fill(0);
    std::fs::write(&path, bytes).unwrap();
    let want: Vec<_> = (0..40).map(|r| expected(r, true)).collect();
    assert_eq!(d.read_all(&index).await, want);
}

// A nested list, text, dates (INT32 underneath), and an item name the
// schema text cannot hold.
#[tokio::test]
async fn bad_columns_are_errors() {
    let nested = {
        let inner = GenericListBuilder::<i32, _>::new(Int32Builder::new());
        let mut b = GenericListBuilder::<i32, _>::new(inner);
        b.values().values().append_value(1);
        b.values().append(true);
        b.append(true);
        Arc::new(b.finish()) as ArrayRef
    };
    let dates = {
        let mut b = GenericListBuilder::<i32, _>::new(Date32Builder::new());
        b.values().append_value(1);
        b.append(true);
        Arc::new(b.finish()) as ArrayRef
    };
    // The printer does not quote names, so this schema would not parse back.
    let spaced = {
        let item = arrow_schema::Field::new("my item", arrow_schema::DataType::Int32, true);
        let mut b = GenericListBuilder::<i32, _>::new(Int32Builder::new()).with_field(item);
        b.values().append_value(1);
        b.append(true);
        Arc::new(b.finish()) as ArrayRef
    };
    // Nanosecond timestamps have a logical type but no converted type.
    let nanos = {
        let mut b = GenericListBuilder::<i32, _>::new(TimestampNanosecondBuilder::new());
        b.values().append_value(1);
        b.append(true);
        Arc::new(b.finish()) as ArrayRef
    };
    let text: ArrayRef = Arc::new(arrow_array::StringArray::from(vec!["a"]));
    for tokens in [nested, text, dates, nanos, spaced] {
        let d = data();
        let p = props(Compression::SNAPPY, false);
        write(&d.path("x.parquet"), tokens, p);
        assert!(matches!(d.index().await, Err(IndexError::Column { .. })));
    }
}

#[tokio::test]
async fn missing_or_differing_columns_are_errors() {
    let d = data();
    let p = props(Compression::SNAPPY, false);
    write(
        &d.path("a.parquet"),
        token_column(Kind::ListI32, 3, false),
        p.clone(),
    );
    write(
        &d.path("b.parquet"),
        token_column(Kind::ListI64, 3, false),
        p,
    );
    match d.index().await {
        Err(IndexError::Column { path, .. }) => assert_eq!(path, "b.parquet"),
        other => panic!("expected a column error, got {other:?}"),
    }
    let missing = Index::build(&*d.store, &d.prefix, "input_ids").await;
    assert!(matches!(missing, Err(IndexError::Column { .. })));
}

// The schema printer does not quote names, so the index stores the column
// under a fixed name. A name with a space must still read.
#[tokio::test]
async fn odd_column_name_reads() {
    let d = data();
    let tokens = token_column(Kind::ListI32, 20, false);
    write_named(
        &d.path("x.parquet"),
        "input ids",
        tokens,
        props(Compression::SNAPPY, false),
    );
    let index = Index::build(&*d.store, &d.prefix, "input ids")
        .await
        .unwrap();
    let want: Vec<_> = (0..20).map(|r| expected(r, false)).collect();
    assert_eq!(d.read_all(&index).await, want);
}

// The schema text comes from parquet's printer and is stored in each
// sidecar. A printer change after an upgrade must show up here.
#[tokio::test]
async fn schema_text_golden() {
    let d = data();
    let tokens = token_column(Kind::ListU32, 3, true);
    write(
        &d.path("x.parquet"),
        tokens,
        props(Compression::SNAPPY, false),
    );
    let index = d.index().await.unwrap();
    let want = "message schema {
  OPTIONAL group tokens (LIST) {
    REPEATED group list {
      OPTIONAL INT32 item (INTEGER(32,false));
    }
  }
}
";
    assert_eq!(index.schema(), want);
}

// A negative token, and a null token inside a list, which would otherwise
// read as token 0.
#[tokio::test]
async fn bad_tokens_are_errors() {
    for tokens in [[Some(3), Some(-1)], [Some(3), None]] {
        let d = data();
        let mut b = GenericListBuilder::<i32, _>::new(Int32Builder::new());
        b.values().extend(tokens);
        b.append(true);
        let p = props(Compression::SNAPPY, false);
        write(&d.path("x.parquet"), Arc::new(b.finish()), p);
        let index = d.index().await.unwrap();
        let reader = Reader::new(d.store.clone(), d.prefix.clone(), &index).unwrap();
        let got = reader.read(&index, 0).await;
        assert!(
            matches!(got, Err(Error::BadToken { .. })),
            "{tokens:?}: {got:?}"
        );
    }
}

#[tokio::test]
async fn fetch_splits_large_ranges() {
    let d = data();
    let path = d.path("x.parquet");
    write(
        &path,
        token_column(Kind::ListI32, 40, false),
        props(Compression::SNAPPY, false),
    );
    let bytes = std::fs::read(&path).unwrap();
    let location = Path::from("data/x.parquet");
    for piece in [64, 1000, 1 << 30] {
        let got = fetch(&*d.store, &location, 3..bytes.len() as u64, piece)
            .await
            .unwrap();
        assert_eq!(got, bytes[3..], "piece {piece}");
    }
}

#[test]
fn pieces_cover_range() {
    assert_eq!(pieces(3..10, 4), [3..7, 7..10]);
    assert_eq!(pieces(3..11, 4), [3..7, 7..11]);
    assert_eq!(pieces(3..4, 16 << 20), vec![3..4]);
    assert!(pieces(5..5, 4).is_empty());
    let big = pieces(0..(40 << 20), PIECE);
    assert_eq!(big.len(), 3);
    assert!(big.iter().all(|p| p.end - p.start <= PIECE));
}
