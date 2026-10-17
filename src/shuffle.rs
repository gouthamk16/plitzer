use rand_chacha::ChaCha8Rng;
use rand_chacha::rand_core::{Rng, SeedableRng};

/// Fisher-Yates shuffle. The output depends only on `seed`, so it is
/// identical on every platform and across dependency upgrades.
///
/// Takes a seed rather than a generator so every call starts fresh: a shared
/// generator would make one window's order depend on all the windows before it.
pub fn shuffle<T>(items: &mut [T], seed: [u8; 32]) {
    let mut rng = ChaCha8Rng::from_seed(seed);
    for i in (1..items.len()).rev() {
        let j = below(&mut rng, i as u64 + 1) as usize;
        items.swap(i, j);
    }
}

/// Uniform integer in `[0, n)`. Hand-rolled because `rand`'s range sampling
/// may change between versions, which would silently change every order.
fn below(rng: &mut ChaCha8Rng, n: u64) -> u64 {
    // 2^64 isn't a multiple of n, so plain `x % n` favours small results.
    // Rejecting the lowest `2^64 mod n` values leaves whole copies of [0, n).
    let skip = n.wrapping_neg() % n;
    loop {
        let x = rng.next_u64();
        if x >= skip {
            return x % n;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shuffled(n: u32, seed: u64) -> Vec<u32> {
        let mut key = [0; 32];
        key[..8].copy_from_slice(&seed.to_le_bytes());
        let mut v: Vec<u32> = (0..n).collect();
        shuffle(&mut v, key);
        v
    }

    #[test]
    fn golden() {
        assert_eq!(
            shuffled(20, 7),
            [
                18, 17, 15, 10, 12, 1, 9, 16, 8, 0, 11, 5, 2, 7, 6, 3, 13, 14, 4, 19
            ]
        );
    }

    #[test]
    fn is_permutation() {
        for seed in 0..256 {
            for n in [0, 1, 2, 1_000] {
                let mut v = shuffled(n, seed);
                v.sort_unstable();
                assert_eq!(v, (0..n).collect::<Vec<_>>(), "seed {seed}, n {n}");
            }
        }
    }

    // Catches the classic off-by-one (`below(i)` instead of `below(i + 1)`),
    // which still yields valid permutations but never leaves an item in place.
    #[test]
    fn all_orders_equally_likely() {
        let mut counts = std::collections::HashMap::new();
        for seed in 0..60_000 {
            *counts.entry(shuffled(3, seed)).or_insert(0) += 1;
        }
        assert_eq!(counts.len(), 6);
        for c in counts.values() {
            assert!((9_500..10_500).contains(c), "skewed: {counts:?}");
        }
    }
}
