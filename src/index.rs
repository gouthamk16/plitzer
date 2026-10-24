use std::fmt;
use std::future::ready;

use futures::{StreamExt, TryStreamExt, stream};
use object_store::path::Path;
use object_store::{ObjectMeta, ObjectStore};
use parquet::errors::ParquetError;

mod column;
mod footer;
mod sidecar;

pub use column::{Codec, RowGroup, parse as parse_schema};

/// Footers read at once. This bounds the requests in flight against S3.
const CONCURRENT_FOOTERS: usize = 16;

/// One Parquet file of the dataset.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct File {
    /// The path below the dataset prefix, with `/` separators.
    pub path: String,
    pub size: u64,
    /// The store's version tag of the file. Only the cache check uses it.
    pub e_tag: Option<String>,
    /// The token column chunk of each row group, in file order.
    pub row_groups: Vec<RowGroup>,
}

#[derive(Debug)]
pub enum Error {
    /// Listing the dataset failed.
    Store(object_store::Error),
    /// A file's footer could not be fetched or decoded.
    Footer { path: String, source: ParquetError },
    /// The number of row groups does not fit in `u32`.
    TooLarge { path: String },
    /// The token column is missing, has the wrong type, or differs from the
    /// first file's.
    Column { path: String, reason: String },
    /// No `.parquet` file is below the prefix.
    NoFiles { prefix: String },
    /// The sidecar could not be written.
    Cache(std::io::Error),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Store(e) => write!(f, "listing the dataset failed: {e}"),
            Error::Footer { path, source } => {
                write!(f, "reading the footer of {path} failed: {source}")
            }
            Error::TooLarge { path } => write!(f, "{path} has more row groups than fit in u32"),
            Error::Column { path, reason } => write!(f, "{path}: {reason}"),
            Error::NoFiles { prefix } => write!(f, "no .parquet files below {prefix:?}"),
            Error::Cache(e) => write!(f, "writing the index cache failed: {e}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Store(e) => Some(e),
            Error::Footer { source, .. } => Some(source),
            Error::TooLarge { .. } | Error::Column { .. } | Error::NoFiles { .. } => None,
            Error::Cache(e) => Some(e),
        }
    }
}

impl From<object_store::Error> for Error {
    fn from(e: object_store::Error) -> Self {
        Error::Store(e)
    }
}

/// The row groups of a dataset and where their token column is. A row
/// group's id is its position: files in path order, then row groups in file
/// order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Index {
    column: String,
    /// The token column's schema, the same in every file. See `parse_schema`.
    schema: String,
    files: Vec<File>,
    /// The id of each file's first row group.
    starts: Vec<u32>,
}

impl Index {
    /// Reads the footer of every `.parquet` file below `prefix`, and records
    /// where each row group's chunk of the token column `column` is.
    pub async fn build(
        store: &dyn ObjectStore,
        prefix: &Path,
        column: &str,
    ) -> Result<Self, Error> {
        let metas = list(store, prefix).await?;
        Self::from_footers(store, prefix, column, metas).await
    }

    /// The index from the sidecar at `cache` if it still matches the listing,
    /// else a new build, which is then written to `cache`. A missing or
    /// corrupt sidecar only causes a rebuild. A cache folder that cannot be
    /// written is `Error::Cache`, before the build.
    pub async fn open(
        store: &dyn ObjectStore,
        prefix: &Path,
        column: &str,
        cache: &std::path::Path,
    ) -> Result<Self, Error> {
        let metas = list(store, prefix).await?;
        let cached = std::fs::read(cache).ok().and_then(|b| sidecar::decode(&b));
        let fresh = |i: &Index| i.column == column && matches(&i.files, prefix, &metas);
        if let Some(index) = cached.filter(fresh) {
            return Ok(index);
        }
        sidecar::check_writable(cache).map_err(Error::Cache)?;
        let index = Self::from_footers(store, prefix, column, metas).await?;
        // The check above catches a folder that cannot be written. A later
        // failure, such as a full disk, costs a rebuild at the next start;
        // this run keeps the index it built.
        let _ = sidecar::write(cache, &sidecar::encode(&index, true));
        Ok(index)
    }

    pub fn column(&self) -> &str {
        &self.column
    }

    pub fn schema(&self) -> &str {
        &self.schema
    }

    pub fn files(&self) -> &[File] {
        &self.files
    }

    /// The row count of each row group, by id. This is `Order::rows`.
    pub fn rows(&self) -> Vec<u32> {
        self.files
            .iter()
            .flat_map(|f| f.row_groups.iter().map(|g| g.rows))
            .collect()
    }

    /// The file of row group `row_group`, and where its token chunk is.
    pub fn row_group(&self, row_group: u32) -> (&File, &RowGroup) {
        let (f, r) = self.locate(row_group);
        (&self.files[f], &self.files[f].row_groups[r])
    }

    /// The file of row group `row_group`, and its position in that file.
    pub fn locate(&self, row_group: u32) -> (usize, usize) {
        let file = self.starts.partition_point(|&s| s <= row_group);
        let in_file = file
            .checked_sub(1)
            .map(|f| (f, (row_group - self.starts[f]) as usize))
            .filter(|&(f, r)| r < self.files[f].row_groups.len());
        in_file.unwrap_or_else(|| panic!("row group {row_group} is not in the index"))
    }

    /// A hash of the column name, the paths, the sizes and the chunks. It
    /// leaves out the ETags, which hold inodes and modification times on
    /// local disk, so copies of one dataset have the same fingerprint. See
    /// `sidecar::encode`.
    pub fn fingerprint(&self) -> u64 {
        crate::codec::fnv1a(&sidecar::encode(self, false))
    }

    fn new(column: String, schema: String, files: Vec<File>) -> Result<Self, Error> {
        let mut starts = Vec::with_capacity(files.len());
        let mut next = 0u32;
        for f in &files {
            starts.push(next);
            let count = u32::try_from(f.row_groups.len()).ok();
            next = count
                .and_then(|c| next.checked_add(c))
                .ok_or_else(|| Error::TooLarge {
                    path: f.path.clone(),
                })?;
        }
        Ok(Index {
            column,
            schema,
            files,
            starts,
        })
    }

    async fn from_footers(
        store: &dyn ObjectStore,
        prefix: &Path,
        column: &str,
        metas: Vec<ObjectMeta>,
    ) -> Result<Self, Error> {
        let read: Vec<(File, String)> = stream::iter(metas)
            .map(|meta| read_file(store, prefix, column, meta))
            .buffered(CONCURRENT_FOOTERS)
            .try_collect()
            .await?;
        let schema = read[0].1.clone();
        if let Some((f, _)) = read.iter().find(|(_, s)| *s != schema) {
            return Err(Error::Column {
                path: f.path.clone(),
                reason: format!("column {column:?} differs from {}", read[0].0.path),
            });
        }
        let files = read.into_iter().map(|(f, _)| f).collect();
        Self::new(column.to_owned(), schema, files)
    }
}

/// The `.parquet` files below `prefix`, sorted by path. None is an error,
/// since a wrong prefix would otherwise give an empty index. Local listings come
/// back in directory order, which differs between machines.
async fn list(store: &dyn ObjectStore, prefix: &Path) -> Result<Vec<ObjectMeta>, Error> {
    let mut metas: Vec<ObjectMeta> = store
        .list(Some(prefix))
        .try_filter(|m| ready(m.location.extension() == Some("parquet")))
        .try_collect()
        .await?;
    if metas.is_empty() {
        return Err(Error::NoFiles {
            prefix: prefix.to_string(),
        });
    }
    metas.sort_by(|a, b| a.location.cmp(&b.location));
    Ok(metas)
}

fn relative(prefix: &Path, location: &Path) -> String {
    let parts = location
        .prefix_match(prefix)
        .expect("listed below the prefix");
    parts
        .map(|p| p.as_ref().to_owned())
        .collect::<Vec<_>>()
        .join("/")
}

/// True if `files` describe exactly the listed files, unchanged.
fn matches(files: &[File], prefix: &Path, metas: &[ObjectMeta]) -> bool {
    files.len() == metas.len()
        && files.iter().zip(metas).all(|(f, m)| {
            f.path == relative(prefix, &m.location) && f.size == m.size && f.e_tag == m.e_tag
        })
}

/// One file of the index, and its token column's schema.
async fn read_file(
    store: &dyn ObjectStore,
    prefix: &Path,
    column: &str,
    meta: ObjectMeta,
) -> Result<(File, String), Error> {
    let path = relative(prefix, &meta.location);
    let footer = match footer::read(store, &meta).await {
        Ok(footer) => footer,
        Err(source) => return Err(Error::Footer { path, source }),
    };
    let (schema, row_groups) = match column::chunks(&footer, column) {
        Ok(found) => found,
        Err(reason) => return Err(Error::Column { path, reason }),
    };
    let file = File {
        path,
        size: meta.size,
        e_tag: meta.e_tag,
        row_groups,
    };
    Ok((file, schema))
}

#[cfg(test)]
mod tests;
