use std::fmt;
use std::ops::Range;
use std::sync::Arc;

use bytes::{Bytes, BytesMut};
use futures::future::try_join_all;
use object_store::ObjectStore;
use object_store::path::Path;
use parquet::DecodeResult;
use parquet::arrow::arrow_reader::{ArrowReaderMetadata, ArrowReaderOptions};
use parquet::arrow::push_decoder::ParquetPushDecoderBuilder;
use parquet::errors::ParquetError;
use parquet::file::metadata::{
    ColumnChunkMetaData, FileMetaData, ParquetMetaData, RowGroupMetaData,
};
use parquet::schema::types::SchemaDescPtr;

use crate::index::{Index, RowGroup, parse_schema};

mod tokens;

pub use tokens::RowGroupTokens;

/// The largest request for one chunk. `object_store` never splits a range,
/// and AWS advises 8–16 MB range requests in parallel, so larger chunks are
/// fetched in pieces of this size.
const PIECE: u64 = 16 << 20;

#[derive(Debug)]
pub enum Error {
    /// The index's schema text does not parse. The index is corrupt.
    Schema(ParquetError),
    /// A chunk could not be fetched.
    Fetch {
        path: String,
        source: object_store::Error,
    },
    /// A chunk could not be decoded.
    Decode {
        path: String,
        row_group: u32,
        source: ParquetError,
    },
    /// A token is null, negative, or does not fit in `u32`.
    BadToken { path: String, row_group: u32 },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Schema(e) => write!(f, "the index holds a bad column schema: {e}"),
            Error::Fetch { path, source } => write!(f, "fetching from {path} failed: {source}"),
            Error::Decode {
                path,
                row_group,
                source,
            } => {
                write!(
                    f,
                    "decoding row group {row_group} of {path} failed: {source}"
                )
            }
            Error::BadToken { path, row_group } => write!(
                f,
                "row group {row_group} of {path} has a null, negative or too large token"
            ),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Schema(e) | Error::Decode { source: e, .. } => Some(e),
            Error::Fetch { source, .. } => Some(source),
            Error::BadToken { .. } => None,
        }
    }
}

/// Reads the token column of single row groups. It fetches only the bytes
/// of each chunk and never reads a footer: the index has all it needs.
pub struct Reader {
    store: Arc<dyn ObjectStore>,
    prefix: Path,
    schema: SchemaDescPtr,
}

impl Reader {
    pub fn new(store: Arc<dyn ObjectStore>, prefix: Path, index: &Index) -> Result<Self, Error> {
        let schema = parse_schema(index.schema()).map_err(Error::Schema)?;
        Ok(Reader {
            store,
            prefix,
            schema,
        })
    }

    /// The tokens of each row of row group `row_group` of `index`.
    pub async fn read(&self, index: &Index, row_group: u32) -> Result<RowGroupTokens, Error> {
        let (file, rg) = index.row_group(row_group);
        if rg.rows == 0 {
            return Ok(RowGroupTokens::default());
        }
        let location = self.location(&file.path)?;
        let range = rg.start..rg.start + u64::from(rg.len);
        let data = fetch(&*self.store, &location, range.clone(), PIECE)
            .await
            .map_err(|source| Error::Fetch {
                path: file.path.clone(),
                source,
            })?;
        let path = || file.path.clone();
        let column = decode(&self.schema, rg, range, data).map_err(|source| Error::Decode {
            path: path(),
            row_group,
            source,
        })?;
        RowGroupTokens::new(&column).ok_or_else(|| Error::BadToken {
            path: path(),
            row_group,
        })
    }

    fn location(&self, path: &str) -> Result<Path, Error> {
        let relative = Path::parse(path).map_err(|e| Error::Fetch {
            path: path.to_owned(),
            source: e.into(),
        })?;
        Ok(self.prefix.parts().chain(relative.parts()).collect())
    }
}

/// The bytes of `range`, in requests of at most `piece` bytes made at once.
/// Each piece is its own `get_ranges` call, so `object_store` does not
/// merge the pieces back into one request.
async fn fetch(
    store: &dyn ObjectStore,
    location: &Path,
    range: Range<u64>,
    piece: u64,
) -> Result<Bytes, object_store::Error> {
    let requests = pieces(range.clone(), piece)
        .into_iter()
        .map(|p| async move { store.get_ranges(location, &[p]).await });
    let mut parts: Vec<Bytes> = try_join_all(requests)
        .await?
        .into_iter()
        .flatten()
        .collect();
    if parts.len() == 1 {
        return Ok(parts.pop().unwrap());
    }
    let mut joined = BytesMut::with_capacity((range.end - range.start) as usize);
    parts.iter().for_each(|p| joined.extend_from_slice(p));
    Ok(joined.freeze())
}

/// `range` cut into consecutive pieces of at most `piece` bytes.
fn pieces(range: Range<u64>, piece: u64) -> Vec<Range<u64>> {
    let starts = (range.start..range.end).step_by(piece as usize);
    starts.map(|s| s..(s + piece).min(range.end)).collect()
}

/// Decodes one chunk from the index record alone.
fn decode(
    schema: &SchemaDescPtr,
    rg: &RowGroup,
    range: Range<u64>,
    data: Bytes,
) -> Result<arrow_array::ArrayRef, ParquetError> {
    let mut decoder = ParquetPushDecoderBuilder::new_with_metadata(metadata(schema, rg)?)
        .with_batch_size(rg.rows as usize)
        .build()?;
    decoder.push_range(range, data)?;
    match decoder.try_decode()? {
        // The lanes index rows by the footer's count, so fewer would panic.
        DecodeResult::Data(batch) if batch.num_rows() != rg.rows as usize => {
            Err(ParquetError::General(format!(
                "the chunk has {} rows, but its footer says {}",
                batch.num_rows(),
                rg.rows
            )))
        }
        DecodeResult::Data(batch) => Ok(batch.column(0).clone()),
        // The index records the whole chunk, so this means a bad record.
        DecodeResult::NeedsData(_) | DecodeResult::Finished => Err(ParquetError::General(
            "the chunk needs bytes outside its index record".into(),
        )),
    }
}

/// Metadata for a file with one row group of one column. The page reader
/// takes the codec and the chunk range from it, and all else from the page
/// headers, so the index record is enough.
fn metadata(schema: &SchemaDescPtr, rg: &RowGroup) -> Result<ArrowReaderMetadata, ParquetError> {
    let rows = i64::from(rg.rows);
    let chunk = ColumnChunkMetaData::builder(schema.column(0))
        .set_compression(rg.codec.to_parquet())
        .set_data_page_offset(rg.start as i64)
        .set_total_compressed_size(i64::from(rg.len))
        .build()?;
    let group = RowGroupMetaData::builder(schema.clone())
        .set_num_rows(rows)
        .set_column_metadata(vec![chunk])
        .build()?;
    let file = FileMetaData::new(2, rows, None, None, schema.clone(), None);
    let metadata = Arc::new(ParquetMetaData::new(file, vec![group]));
    ArrowReaderMetadata::try_new(metadata, ArrowReaderOptions::new())
}

#[cfg(test)]
mod tests;
