//! Small helpers for the little-endian binary formats of the index sidecar
//! and the checkpoint.

/// 64-bit FNV-1a. Its output is fixed by definition, unlike `std`'s hashers.
pub(crate) fn fnv1a(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

pub(crate) fn len(n: usize) -> u32 {
    u32::try_from(n).expect("count fits in u32")
}

pub(crate) fn put_str(out: &mut Vec<u8>, s: &str) {
    out.extend(len(s.len()).to_le_bytes());
    out.extend(s.as_bytes());
}

/// Reads little-endian fields from the front of a byte slice. Each read
/// gives `None` when too few bytes are left.
pub(crate) struct Reader<'a>(pub(crate) &'a [u8]);

impl<'a> Reader<'a> {
    pub(crate) fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let (head, rest) = self.0.split_at_checked(n)?;
        self.0 = rest;
        Some(head)
    }

    pub(crate) fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }

    pub(crate) fn u64(&mut self) -> Option<u64> {
        Some(u64::from_le_bytes(self.take(8)?.try_into().ok()?))
    }

    pub(crate) fn str(&mut self) -> Option<String> {
        let n = self.u32()? as usize;
        String::from_utf8(self.take(n)?.to_vec()).ok()
    }
}
