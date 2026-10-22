use std::fmt;
use std::future::ready;

use futures::{StreamExt, TryStreamExt, stream};
use object_store::path::Path;
use object_store::{ObjectMeta, ObjectStore};
use parquet::DecodeResult;
use parquet::errors::ParquetError;
use parquet::file::metadata::{PageIndexPolicy, ParquetMetaData, ParquetMetaDataPushDecoder};

mod sidecar;

/// Footers read at once. Each footer is one or two range requests, so this
/// bounds the requests in flight against S3.
const CONCURRENT_FOOTERS: usize = 16;

/// One Parquet file of the dataset.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct File {
    /// The path below the dataset prefix, with `/` separators.
    pub path: String,
    pub size: u64,
    /// The store's version tag of the file. Only the cache check uses it.
    pub e_tag: Option<String>,
    /// The row count of each row group, in file order.
    pub rows: Vec<u32>,
}

#[derive(Debug)]
pub enum Error {
    /// Listing the dataset failed.
    Store(object_store::Error),
    /// A file's footer could not be fetched or decoded.
    Footer { path: String, source: ParquetError },
    /// A row count or the number of row groups does not fit in `u32`.
    TooLarge { path: String },
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
            Error::TooLarge { path } => {
                write!(f, "{path} has more rows or row groups than fit in u32")
            }
            Error::Cache(e) => write!(f, "writing the index cache failed: {e}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Store(e) => Some(e),
            Error::Footer { source, .. } => Some(source),
            Error::TooLarge { .. } => None,
            Error::Cache(e) => Some(e),
        }
    }
}

impl From<object_store::Error> for Error {
    fn from(e: object_store::Error) -> Self {
        Error::Store(e)
    }
}

/// The row groups of a dataset. A row group's id is its position: files in
/// path order, then row groups in file order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Index {
    files: Vec<File>,
    /// The id of each file's first row group.
    starts: Vec<u32>,
}

impl Index {
    /// Reads the footer of every `.parquet` file below `prefix`.
    pub async fn build(store: &dyn ObjectStore, prefix: &Path) -> Result<Self, Error> {
        let metas = list(store, prefix).await?;
        Self::from_footers(store, prefix, metas).await
    }

    /// The index from the sidecar at `cache` if it still matches the listing,
    /// else a new build, which is then written to `cache`. A missing or
    /// corrupt sidecar only causes a rebuild.
    pub async fn open(
        store: &dyn ObjectStore,
        prefix: &Path,
        cache: &std::path::Path,
    ) -> Result<Self, Error> {
        let metas = list(store, prefix).await?;
        let cached = std::fs::read(cache).ok().and_then(|b| sidecar::decode(&b));
        if let Some(files) = cached.filter(|files| matches(files, prefix, &metas)) {
            return Self::new(files);
        }
        let index = Self::from_footers(store, prefix, metas).await?;
        sidecar::write(cache, &sidecar::encode(&index.files, true)).map_err(Error::Cache)?;
        Ok(index)
    }

    pub fn files(&self) -> &[File] {
        &self.files
    }

    /// The row count of each row group, by id. This is `Order::rows`.
    pub fn rows(&self) -> Vec<u32> {
        self.files
            .iter()
            .flat_map(|f| f.rows.iter().copied())
            .collect()
    }

    /// The file of row group `row_group`, and its position in that file.
    pub fn locate(&self, row_group: u32) -> (usize, usize) {
        let file = self.starts.partition_point(|&s| s <= row_group);
        let in_file = file
            .checked_sub(1)
            .map(|f| (f, (row_group - self.starts[f]) as usize))
            .filter(|&(f, r)| r < self.files[f].rows.len());
        in_file.unwrap_or_else(|| panic!("row group {row_group} is not in the index"))
    }

    /// A hash of the paths, sizes and row counts. It leaves out the ETags,
    /// which hold inodes and modification times on local disk, so copies of
    /// one dataset on different machines have the same fingerprint.
    pub fn fingerprint(&self) -> u64 {
        sidecar::fnv1a(&sidecar::encode(&self.files, false))
    }

    fn new(files: Vec<File>) -> Result<Self, Error> {
        let mut starts = Vec::with_capacity(files.len());
        let mut next = 0u32;
        for f in &files {
            starts.push(next);
            let count = u32::try_from(f.rows.len()).ok();
            next = count
                .and_then(|c| next.checked_add(c))
                .ok_or_else(|| Error::TooLarge {
                    path: f.path.clone(),
                })?;
        }
        Ok(Index { files, starts })
    }

    async fn from_footers(
        store: &dyn ObjectStore,
        prefix: &Path,
        metas: Vec<ObjectMeta>,
    ) -> Result<Self, Error> {
        let files = stream::iter(metas)
            .map(|meta| read_file(store, prefix, meta))
            .buffered(CONCURRENT_FOOTERS)
            .try_collect()
            .await?;
        Self::new(files)
    }
}

/// The `.parquet` files below `prefix`, sorted by path. Local listings come
/// back in directory order, which differs between machines.
async fn list(store: &dyn ObjectStore, prefix: &Path) -> Result<Vec<ObjectMeta>, Error> {
    let mut metas: Vec<ObjectMeta> = store
        .list(Some(prefix))
        .try_filter(|m| ready(m.location.extension() == Some("parquet")))
        .try_collect()
        .await?;
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

async fn read_file(
    store: &dyn ObjectStore,
    prefix: &Path,
    meta: ObjectMeta,
) -> Result<File, Error> {
    let path = relative(prefix, &meta.location);
    let footer = match footer(store, &meta).await {
        Ok(footer) => footer,
        Err(source) => return Err(Error::Footer { path, source }),
    };
    let rows: Result<Vec<u32>, _> = footer
        .row_groups()
        .iter()
        .map(|rg| u32::try_from(rg.num_rows()))
        .collect();
    let Ok(rows) = rows else {
        return Err(Error::TooLarge { path });
    };
    Ok(File {
        path,
        size: meta.size,
        e_tag: meta.e_tag,
        rows,
    })
}

/// The file's footer, without its page indexes, which the index never needs.
async fn footer(
    store: &dyn ObjectStore,
    meta: &ObjectMeta,
) -> Result<ParquetMetaData, ParquetError> {
    let mut decoder = ParquetMetaDataPushDecoder::try_new(meta.size)?
        .with_page_index_policy(PageIndexPolicy::Skip);
    loop {
        match decoder.try_decode()? {
            DecodeResult::Data(metadata) => return Ok(metadata),
            DecodeResult::NeedsData(ranges) => {
                let data = store
                    .get_ranges(&meta.location, &ranges)
                    .await
                    .map_err(|e| ParquetError::External(Box::new(e)))?;
                decoder.push_ranges(ranges, data)?;
            }
            DecodeResult::Finished => {
                return Err(ParquetError::General("no footer in file".into()));
            }
        }
    }
}

#[cfg(test)]
mod tests;
