/// What a shuffle orders. Each purpose has its own tag in the key, so two
/// shuffles never share a random stream even when their numbers match.
///
/// A lane is one of the global batch's sequence slots. Lanes are fixed by the
/// batch size, never by the GPU count, which is why the same lane gets the
/// same order however many GPUs there are.
#[derive(Clone, Copy, Debug)]
pub enum Purpose {
    RowGroups,
    Lane { window: u64, lane: u32 },
}

/// A shuffle's 32-byte ChaCha key. Only `key` can build one, so hand-packed
/// bytes that might collide with another shuffle's stream don't compile.
#[derive(Clone, Copy, Debug)]
pub struct Key([u8; 32]);

impl Key {
    pub(crate) fn bytes(self) -> [u8; 32] {
        self.0
    }
}

/// Packs `(seed, epoch, purpose)` into a key for `shuffle::shuffle`.
///
/// Layout, all little-endian:
/// `seed[0..8] epoch[8..16] tag[16..20] lane[20..24] window[24..32]`.
/// Every field owns fixed bytes, so distinct inputs always give distinct keys.
pub fn key(seed: u64, epoch: u64, purpose: Purpose) -> Key {
    // Tags are baked into every saved order: never renumber them.
    let (tag, lane, window) = match purpose {
        Purpose::RowGroups => (0u32, 0, 0),
        Purpose::Lane { window, lane } => (1, lane, window),
    };
    let mut k = [0; 32];
    k[0..8].copy_from_slice(&seed.to_le_bytes());
    k[8..16].copy_from_slice(&epoch.to_le_bytes());
    k[16..20].copy_from_slice(&tag.to_le_bytes());
    k[20..24].copy_from_slice(&lane.to_le_bytes());
    k[24..32].copy_from_slice(&window.to_le_bytes());
    Key(k)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Every byte of every field is nonzero and distinct, so a slot that
    // moves, overlaps or narrows changes these bytes.
    #[test]
    fn golden() {
        let lane = Purpose::Lane {
            window: 0x2122_2324_2526_2728,
            lane: 0x3132_3334,
        };
        assert_eq!(
            key(0x0102_0304_0506_0708, 0x1112_1314_1516_1718, lane).bytes(),
            [
                0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01, // seed
                0x18, 0x17, 0x16, 0x15, 0x14, 0x13, 0x12, 0x11, // epoch
                0x01, 0x00, 0x00, 0x00, // tag
                0x34, 0x33, 0x32, 0x31, // lane
                0x28, 0x27, 0x26, 0x25, 0x24, 0x23, 0x22, 0x21, // window
            ]
        );
    }

    #[test]
    fn purposes_never_collide() {
        let lane0 = Purpose::Lane { window: 0, lane: 0 };
        assert_ne!(
            key(42, 1, Purpose::RowGroups).bytes(),
            key(42, 1, lane0).bytes()
        );
    }
}
