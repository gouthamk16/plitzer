use rand_chacha::ChaCha8Rng;
use rand_chacha::rand_core::{Rng, SeedableRng};

use crate::seed::Key;

/// Fisher-Yates shuffle. The output depends only on `key` and is identical
/// on every platform. `rand_chacha` only promises a stable stream within a
/// minor version, so the `golden` test is what catches a breaking upgrade.
///
/// Takes a key rather than a generator so every call starts fresh: a shared
/// generator would make one window's order depend on all the windows before it.
#[expect(
    clippy::disallowed_methods,
    reason = "the one place a Key becomes a generator"
)]
pub fn shuffle<T>(items: &mut [T], key: Key) {
    let mut rng = ChaCha8Rng::from_seed(key.bytes());
    for i in (1..items.len()).rev() {
        let j = below(&mut rng, i as u64 + 1) as usize;
        items.swap(i, j);
    }
}

/// Uniform integer in `[0, n)`. Hand-rolled because `rand`'s range sampling
/// may change between versions, which would silently change every order.
fn below(rng: &mut ChaCha8Rng, n: u64) -> u64 {
    debug_assert!(n > 0, "below(0) has no valid result");
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
#[expect(
    clippy::disallowed_methods,
    reason = "tests drive `below` from a raw seed"
)]
mod tests {
    use super::*;
    use crate::seed::{Purpose, key};

    fn shuffled(n: u32, seed: u64) -> Vec<u32> {
        let mut v: Vec<u32> = (0..n).collect();
        shuffle(&mut v, key(seed, 0, Purpose::RowGroups));
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
        // Long enough to cross many ChaCha block refills. A positional hash
        // covers every element, so a change anywhere in the stream shows up.
        let hash = shuffled(1_000, 7).iter().fold(0u64, |h, &x| {
            h.wrapping_mul(1_000_003).wrapping_add(x as u64)
        });
        assert_eq!(hash, 2_786_988_667_572_105_488);

        // Large n so the rejection branch runs: draws 3 and 4 each reject twice.
        let mut rng = ChaCha8Rng::from_seed([0; 32]);
        let draws: Vec<u64> = (0..4).map(|_| below(&mut rng, 3 << 62)).collect();
        assert_eq!(
            draws,
            [
                1_603_386_510_163_247_166,
                11_647_726_043_916_688_255,
                4_715_024_415_260_232_856,
                4_827_309_107_960_445_752
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

    // With n = 3 * 2^62, plain `x % n` lands below 2^62 half the time instead
    // of a third, so this fails if the rejection step is broken or removed.
    #[test]
    fn below_is_unbiased() {
        let n: u64 = 3 << 62;
        let mut rng = ChaCha8Rng::from_seed([0; 32]);
        let low = (0..30_000).filter(|_| below(&mut rng, n) < 1 << 62).count();
        assert!((9_500..10_500).contains(&low), "low: {low}");
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
