/// What a shuffle orders. Each purpose has its own tag in the key, so two
/// shuffles never share a random stream even when their numbers match.
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

    // Every field holds a different nonzero value, so any overlap between
    // slots changes these bytes.
    #[test]
    fn golden() {
        let lane = Purpose::Lane { window: 3, lane: 5 };
        assert_eq!(
            key(42, 1, lane).bytes(),
            [
                42, 0, 0, 0, 0, 0, 0, 0, // seed
                1, 0, 0, 0, 0, 0, 0, 0, // epoch
                1, 0, 0, 0, // tag
                5, 0, 0, 0, // lane
                3, 0, 0, 0, 0, 0, 0, 0, // window
            ]
        );
        assert_eq!(key(42, 1, Purpose::RowGroups).bytes()[16..], [0; 16]);
    }
}
