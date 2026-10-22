//! The index cache file. All integers are little-endian:
//!
//! ```text
//! magic "PLTZIDX\0" | version u32 | file count u32 | each file:
//!   path (len u32, UTF-8) | size u64 | e_tag (flag u8, then len u32, UTF-8) |
//!   row-group count u32 | rows u32 × count
//! ```

use std::io::{self, Write};

use super::File;

const MAGIC: &[u8; 8] = b"PLTZIDX\0";
const VERSION: u32 = 1;

/// The files in the cache format. Without `e_tags`, every ETag is written
/// as absent, which makes the bytes the same on every copy of a dataset.
pub(super) fn encode(files: &[File], e_tags: bool) -> Vec<u8> {
    let mut out = MAGIC.to_vec();
    out.extend(VERSION.to_le_bytes());
    out.extend(len(files.len()).to_le_bytes());
    for f in files {
        put_str(&mut out, &f.path);
        out.extend(f.size.to_le_bytes());
        match f.e_tag.as_deref().filter(|_| e_tags) {
            Some(tag) => {
                out.push(1);
                put_str(&mut out, tag);
            }
            None => out.push(0),
        }
        out.extend(len(f.rows.len()).to_le_bytes());
        for r in &f.rows {
            out.extend(r.to_le_bytes());
        }
    }
    out
}

/// The files in `bytes`, or `None` if they are not a complete sidecar of
/// this version.
pub(super) fn decode(bytes: &[u8]) -> Option<Vec<File>> {
    let mut r = Reader(bytes);
    if r.take(8)? != MAGIC || r.u32()? != VERSION {
        return None;
    }
    // Counts come from the file, so nothing is preallocated from them.
    let mut files = Vec::new();
    for _ in 0..r.u32()? {
        let path = r.str()?;
        let size = r.u64()?;
        let e_tag = match r.take(1)? {
            [0] => None,
            [1] => Some(r.str()?),
            _ => return None,
        };
        let mut rows = Vec::new();
        for _ in 0..r.u32()? {
            rows.push(r.u32()?);
        }
        files.push(File {
            path,
            size,
            e_tag,
            rows,
        });
    }
    r.0.is_empty().then_some(files)
}

/// Writes `bytes` to a temporary file with a unique name next to `path`,
/// then renames it. Processes that start together (one per GPU) can then
/// write at once, and no reader sees a half-written sidecar.
pub(super) fn write(path: &std::path::Path, bytes: &[u8]) -> io::Result<()> {
    let dir = path.parent().filter(|d| !d.as_os_str().is_empty());
    let mut tmp = tempfile::NamedTempFile::new_in(dir.unwrap_or(".".as_ref()))?;
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

/// 64-bit FNV-1a. Its output is fixed by definition, unlike `std`'s hashers.
pub(super) fn fnv1a(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

fn len(n: usize) -> u32 {
    u32::try_from(n).expect("count fits in u32")
}

fn put_str(out: &mut Vec<u8>, s: &str) {
    out.extend(len(s.len()).to_le_bytes());
    out.extend(s.as_bytes());
}

struct Reader<'a>(&'a [u8]);

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let (head, rest) = self.0.split_at_checked(n)?;
        self.0 = rest;
        Some(head)
    }

    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }

    fn u64(&mut self) -> Option<u64> {
        Some(u64::from_le_bytes(self.take(8)?.try_into().ok()?))
    }

    fn str(&mut self) -> Option<String> {
        let n = self.u32()? as usize;
        String::from_utf8(self.take(n)?.to_vec()).ok()
    }
}
