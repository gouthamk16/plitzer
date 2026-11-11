use std::fmt;
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use bytes::{Bytes, BytesMut};
use futures::future::try_join_all;
use object_store::ObjectStore;
use object_store::path::Path;
use parquet::errors::ParquetError;
use parquet::schema::types::SchemaDescPtr;

use crate::index::{Index, parse_schema};

mod column;
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
    /// The rows do not have the slots that the index records: a token for
    /// each slot, and one slot for each row with none. The order cuts
    /// slices by these slots.
    Slots { path: String, row_group: u32 },
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
            Error::Slots { path, row_group } => write!(
                f,
                "row group {row_group} of {path} does not have the tokens that its footer records"
            ),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Schema(e) | Error::Decode { source: e, .. } => Some(e),
            Error::Fetch { source, .. } => Some(source),
            Error::BadToken { .. } | Error::Slots { .. } => None,
        }
    }
}

/// Reads the token column of single row groups. It fetches only the bytes
/// of each chunk and never reads a footer: the index has all it needs.
pub struct Reader {
    store: Arc<dyn ObjectStore>,
    prefix: Path,
    schema: SchemaDescPtr,
    /// The bytes of every chunk fetched so far, shared with those who watch.
    fetched: Arc<AtomicU64>,
}

impl Reader {
    pub fn new(store: Arc<dyn ObjectStore>, prefix: Path, index: &Index) -> Result<Self, Error> {
        let schema = parse_schema(index.schema()).map_err(Error::Schema)?;
        Ok(Reader {
            store,
            prefix,
            schema,
            fetched: Arc::new(AtomicU64::new(0)),
        })
    }

    /// The bytes of the token chunks fetched from storage so far, also those
    /// that then failed to decode. A failed fetch adds nothing.
    pub fn bytes_read(&self) -> u64 {
        self.fetched.load(Ordering::Relaxed)
    }

    /// The counter of `bytes_read`, to read at any time, also during a read.
    pub fn bytes_counter(&self) -> Arc<AtomicU64> {
        self.fetched.clone()
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
        self.fetched.fetch_add(data.len() as u64, Ordering::Relaxed);
        let path = || file.path.clone();
        column::decode(self.schema.column(0), rg, data).map_err(|e| match e {
            column::Fail::Parquet(source) => Error::Decode {
                path: path(),
                row_group,
                source,
            },
            column::Fail::BadToken => Error::BadToken {
                path: path(),
                row_group,
            },
            column::Fail::Slots => Error::Slots {
                path: path(),
                row_group,
            },
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

#[cfg(test)]
mod tests;
