use proptest::prelude::*;

use super::*;

fn config() -> Config {
    Config {
        column: "tokens".into(),
        seed: 7,
        lanes: 4,
        window: 3,
        seq_len: 8,
    }
}

/// `n` lanes from `first_lane` at step `step`, alternately in epochs 2 and
/// 3: lanes stand a few documents apart at an epoch's end.
fn part(first_lane: u32, step: u64, n: usize) -> Checkpoint {
    let lanes = (0..n)
        .map(|i| Position {
            epoch: 2 + i as u64 % 2,
            window: 3 * i as u32 + 1,
            token: i as u64,
        })
        .collect();
    Checkpoint {
        config: config(),
        fingerprint: 0xfeed,
        epoch: 2,
        step,
        first_lane,
        lanes,
    }
}

// Pinned bytes: a change here breaks every saved checkpoint. The same bytes
// were built independently with Python's `struct` module.
#[test]
fn golden_bytes() {
    let bytes = part(1, 5, 2).to_bytes();
    let hex: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
    assert_eq!(bytes.len(), 122);
    assert_eq!(&hex[..24], "504c545a434b505402000000");
    assert_eq!(fnv1a(&bytes), 0xf0ec_5246_e1f9_a518);
}

#[test]
fn damaged_bytes_are_refused() {
    let bytes = part(0, 1, 4).to_bytes();
    let mut flipped = bytes.clone();
    flipped[40] ^= 1;
    assert_eq!(Checkpoint::from_bytes(&flipped), Err(Error::Checksum));
    let mut version = bytes[..bytes.len() - 8].to_vec();
    version[8] = 9;
    assert_eq!(
        Checkpoint::from_bytes(&version),
        Err(Error::Checksum),
        "damage, not a version"
    );
    version.extend(fnv1a(&version).to_le_bytes());
    assert_eq!(Checkpoint::from_bytes(&version), Err(Error::Version(9)));
    assert_eq!(
        Checkpoint::from_bytes(b"PLTZ"),
        Err(Error::Bytes("no magic"))
    );
    assert_eq!(Checkpoint::from_bytes(&bytes[..10]), Err(Error::Checksum));
    let mut magic_only = MAGIC.to_vec();
    magic_only.extend(fnv1a(MAGIC).to_le_bytes());
    assert_eq!(
        Checkpoint::from_bytes(&magic_only),
        Err(Error::Bytes("truncated"))
    );
    // A count larger than the lanes that follow, with a valid checksum.
    let mut short = bytes[..bytes.len() - 8 - 20].to_vec();
    short.extend(fnv1a(&short).to_le_bytes());
    assert_eq!(
        Checkpoint::from_bytes(&short),
        Err(Error::Bytes("truncated"))
    );
}

#[test]
fn merge_tiles_the_lanes() {
    let parts = vec![part(2, 4, 2), part(0, 4, 2)];
    let merged = Checkpoint::merge(parts).unwrap();
    assert!(merged.is_global());
    assert_eq!(merged.lanes[..2], part(0, 4, 2).lanes[..]);
    assert_eq!(merged.lanes[2..], part(2, 4, 2).lanes[..]);
    assert_eq!(merged.positions(1..3).unwrap(), merged.lanes[1..3]);
}

#[test]
fn merge_refuses_bad_parts() {
    let refused = |parts: Vec<Checkpoint>| matches!(Checkpoint::merge(parts), Err(Error::Parts(_)));
    assert!(refused(vec![]));
    let mut huge = part(0, 4, 2);
    huge.config.lanes = u32::MAX;
    assert!(refused(vec![huge]), "B from the bytes is not reserved");
    assert!(refused(vec![part(0, 4, 2)]), "lanes 2..4 missing");
    assert!(refused(vec![part(0, 4, 2), part(3, 4, 2)]), "gap");
    assert!(refused(vec![part(0, 4, 2), part(1, 4, 3)]), "overlap");
    assert!(refused(vec![part(0, 4, 2), part(2, 5, 2)]), "steps differ");
    let mut other = part(2, 4, 2);
    other.config.seed = 8;
    assert!(refused(vec![part(0, 4, 2), other]), "configs differ");
    let mut other = part(2, 4, 2);
    other.epoch = 3;
    assert!(refused(vec![part(0, 4, 2), other]), "epochs differ");
}

#[test]
fn positions_of_a_part() {
    let p = part(2, 4, 2);
    assert!(!p.is_global());
    assert_eq!(p.positions(2..4).unwrap(), p.lanes);
    assert!(matches!(p.positions(0..2), Err(Error::Parts(_))));
    assert!(matches!(p.positions(3..5), Err(Error::Parts(_))));
    #[allow(clippy::reversed_empty_ranges)]
    let reversed = 10..2;
    assert!(matches!(p.positions(reversed), Err(Error::Parts(_))));
}

#[test]
fn check_names_the_field() {
    let c = part(0, 1, 4);
    assert_eq!(c.check(&config(), 0xfeed), Ok(()));
    let field = |config: Config, fingerprint: u64| match c.check(&config, fingerprint) {
        Err(Error::Mismatch { field, .. }) => field,
        other => panic!("{other:?}"),
    };
    let mut other = config();
    other.seq_len = 16;
    assert_eq!(field(other, 0xfeed), "seq_len");
    let mut other = config();
    other.window = 4;
    assert_eq!(field(other, 0xfeed), "window");
    let mut other = config();
    other.column = "ids".into();
    assert_eq!(field(other, 0xfeed), "column");
    assert_eq!(field(config(), 1), "fingerprint");
}

fn checkpoints() -> impl Strategy<Value = Checkpoint> {
    let lane =
        (any::<u64>(), any::<u32>(), any::<u64>()).prop_map(|(epoch, window, token)| Position {
            epoch,
            window,
            token,
        });
    let config = (".{0,12}", any::<u64>(), any::<[u32; 3]>()).prop_map(
        |(column, seed, [lanes, window, seq_len])| Config {
            column,
            seed,
            lanes,
            window,
            seq_len,
        },
    );
    let lanes = proptest::collection::vec(lane, 0..8);
    (config, any::<[u64; 3]>(), any::<u32>(), lanes).prop_map(
        |(config, [fingerprint, epoch, step], first_lane, lanes)| Checkpoint {
            config,
            fingerprint,
            epoch,
            step,
            first_lane,
            lanes,
        },
    )
}

proptest! {
    #[test]
    fn bytes_round_trip(c in checkpoints()) {
        prop_assert_eq!(Checkpoint::from_bytes(&c.to_bytes()), Ok(c));
    }

    #[test]
    fn any_bytes_give_an_error_not_a_panic(
        bytes in proptest::collection::vec(any::<u8>(), 0..200),
    ) {
        let _ = Checkpoint::from_bytes(&bytes);
        let mut framed = MAGIC.to_vec();
        framed.extend(VERSION.to_le_bytes());
        framed.extend(&bytes);
        framed.extend(fnv1a(&framed).to_le_bytes());
        let _ = Checkpoint::from_bytes(&framed);
    }

    #[test]
    fn truncation_is_refused(c in checkpoints(), cut in 1usize..40) {
        let bytes = c.to_bytes();
        let keep = bytes.len().saturating_sub(cut);
        prop_assert!(Checkpoint::from_bytes(&bytes[..keep]).is_err());
    }
}
