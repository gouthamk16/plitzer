//! The index cache file. All integers are little-endian:
//!
//! ```text
//! magic "PLTZIDX\0" | version u32 | column (len u32, UTF-8) | schema (len u32, UTF-8) |
//! file count u32 | each file:
//!   path (len u32, UTF-8) | size u64 | e_tag (flag u8, then len u32, UTF-8) |
//!   row-group count u32 | each row group: rows u32 | start u64 | len u32 | values u32 |
//!   codec u8
//! ```

use std::io::{self, Write};

use super::{Codec, File, Index, RowGroup};
use crate::codec::{Reader, len, put_str};

const MAGIC: &[u8; 8] = b"PLTZIDX\0";
/// Version 1 held only row counts, version 2 no value counts. A sidecar of
/// another version does not decode, so `Index::open` rebuilds it.
const VERSION: u32 = 3;

/// The index in the cache format. Without `cache`, every ETag is written as
/// absent and the schema as empty, for the fingerprint: ETags differ between
/// copies of a dataset, and the schema text comes from `parquet`'s printer,
/// which an upgrade could change.
pub(super) fn encode(index: &Index, cache: bool) -> Vec<u8> {
    let mut out = MAGIC.to_vec();
    out.extend(VERSION.to_le_bytes());
    put_str(&mut out, &index.column);
    put_str(&mut out, if cache { &index.schema } else { "" });
    out.extend(len(index.files.len()).to_le_bytes());
    for f in &index.files {
        put_str(&mut out, &f.path);
        out.extend(f.size.to_le_bytes());
        match f.e_tag.as_deref().filter(|_| cache) {
            Some(tag) => {
                out.push(1);
                put_str(&mut out, tag);
            }
            None => out.push(0),
        }
        out.extend(len(f.row_groups.len()).to_le_bytes());
        for g in &f.row_groups {
            out.extend(g.rows.to_le_bytes());
            out.extend(g.start.to_le_bytes());
            out.extend(g.len.to_le_bytes());
            out.extend(g.values.to_le_bytes());
            out.push(g.codec.byte());
        }
    }
    out
}

/// The index in `bytes`, or `None` if they are not a complete sidecar of
/// this version.
pub(super) fn decode(bytes: &[u8]) -> Option<Index> {
    let mut r = Reader(bytes);
    if r.take(8)? != MAGIC || r.u32()? != VERSION {
        return None;
    }
    let column = r.str()?;
    let schema = r.str()?;
    super::parse_schema(&schema).ok()?;
    // Counts come from the file, so nothing is preallocated from them.
    let mut files = Vec::new();
    for _ in 0..r.u32()? {
        files.push(file(&mut r)?);
    }
    if !r.0.is_empty() {
        return None;
    }
    Index::new(column, schema, files).ok()
}

fn file(r: &mut Reader) -> Option<File> {
    let path = r.str()?;
    let size = r.u64()?;
    let e_tag = match r.take(1)? {
        [0] => None,
        [1] => Some(r.str()?),
        _ => return None,
    };
    let mut row_groups = Vec::new();
    for _ in 0..r.u32()? {
        row_groups.push(RowGroup {
            rows: r.u32()?,
            start: r.u64()?,
            len: r.u32()?,
            values: r.u32()?,
            codec: Codec::from_byte(r.take(1)?[0])?,
        });
    }
    Some(File {
        path,
        size,
        e_tag,
        row_groups,
    })
}

/// Checks, before the build, that `write` can later put a sidecar at
/// `path`, so a cache that cannot be written fails before any footer is
/// read. It refuses a folder, a path with no file name or one that ends in a
/// separator (rename cannot put a file there), and makes a file
/// in the folder: on Unix `write` renames over the old sidecar, which needs
/// only the folder, so a read-only stale sidecar is still replaced. It changes
/// no file, since another process may have just written a good sidecar.
pub(super) fn check_writable(path: &std::path::Path) -> io::Result<()> {
    let names_a_folder = path.to_string_lossy().ends_with(std::path::is_separator);
    if path.is_dir() || path.file_name().is_none() || names_a_folder {
        let why = format!("the index cache {:?} is not a file path", path);
        return Err(io::Error::new(io::ErrorKind::InvalidInput, why));
    }
    // Windows, unlike Unix, refuses to rename over a read-only file.
    #[cfg(windows)]
    if path.metadata().is_ok_and(|m| m.permissions().readonly()) {
        let why = format!("the index cache {:?} is read-only", path);
        return Err(io::Error::new(io::ErrorKind::PermissionDenied, why));
    }
    tempfile::NamedTempFile::new_in(folder(path)?).map(drop)
}

/// The folder of `path`, made if it does not exist, as on a new machine.
fn folder(path: &std::path::Path) -> io::Result<&std::path::Path> {
    let dir = path.parent().filter(|d| !d.as_os_str().is_empty());
    let dir = dir.unwrap_or(".".as_ref());
    std::fs::create_dir_all(dir)?;
    Ok(dir)
}

/// Writes `bytes` to a temporary file with a unique name next to `path`,
/// then renames it. Processes that start together (one per GPU) can then
/// write at once, and no reader sees a half-written sidecar.
pub(super) fn write(path: &std::path::Path, bytes: &[u8]) -> io::Result<()> {
    let mut tmp = tempfile::NamedTempFile::new_in(folder(path)?)?;
    tmp.write_all(bytes)?;
    for attempt in 0.. {
        let Err(e) = tmp.persist(path) else {
            return Ok(());
        };
        // Windows refuses to replace a file while another process has it
        // open or is replacing it. If that process wrote the same index, the
        // sidecar is already right; else retry briefly before giving up.
        if std::fs::read(path).is_ok_and(|b| b == bytes) {
            return Ok(());
        }
        if attempt == 5 || e.error.kind() != io::ErrorKind::PermissionDenied {
            return Err(e.error);
        }
        std::thread::sleep(std::time::Duration::from_millis(10 << attempt));
        tmp = e.file;
    }
    unreachable!()
}

#[cfg(test)]
pub(super) mod tests;
