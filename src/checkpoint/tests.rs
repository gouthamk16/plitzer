use proptest::prelude::*;

use super::*;

fn config() -> Config {
    Config {
        column: "tokens".into(),
        seed: 7,
        lanes: 4,
        window: 3,
        packing: Packing {
            seq_len: 8,
            docs: 2,
            cap: 10,
            pad: u32::MAX,
        },
    }
}

fn part(first_lane: u32, steps: &[u64]) -> Checkpoint {
    let lanes = steps
        .iter()
        .enumerate()
        .map(|(i, &step)| LaneState {
            step,
            doc: 3 * step + i as u64,
            offset: i as u32,
        })
        .collect();
    Checkpoint {
        config: config(),
        fingerprint: 0xfeed,
        epoch: 2,
        first_lane,
        lanes,
    }
}

// Pinned bytes: a change here breaks every saved checkpoint. The same bytes
// were built independently with Python's `struct` module.
#[test]
fn golden_bytes() {
    let bytes = part(1, &[5, 5]).to_bytes();
    let hex: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
    assert_eq!(bytes.len(), 126);
    assert_eq!(&hex[..24], "504c545a434b505401000000");
    assert_eq!(fnv1a(&bytes), 0xe604_36cc_0508_3891);
}

#[test]
fn damaged_bytes_are_refused() {
    let bytes = part(0, &[1, 1, 1, 1]).to_bytes();
    let mut flipped = bytes.clone();
    flipped[40] ^= 1;
    assert_eq!(Checkpoint::from_bytes(&flipped), Err(Error::Checksum));
    let mut version = bytes.clone();
    version[8] = 9;
    assert_eq!(Checkpoint::from_bytes(&version), Err(Error::Version(9)));
    assert_eq!(
        Checkpoint::from_bytes(b"PLTZ"),
        Err(Error::Bytes("no magic"))
    );
    assert_eq!(
        Checkpoint::from_bytes(&bytes[..10]),
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
    let parts = vec![part(2, &[4, 4]), part(0, &[4, 4])];
    let merged = Checkpoint::merge(parts).unwrap();
    assert!(merged.is_global());
    assert_eq!(merged.lanes[..2], part(0, &[4, 4]).lanes[..]);
    assert_eq!(merged.lanes[2..], part(2, &[4, 4]).lanes[..]);
    assert_eq!(merged.states(1..3).unwrap(), merged.lanes[1..3]);
}

#[test]
fn merge_refuses_bad_parts() {
    let refused = |parts: Vec<Checkpoint>| matches!(Checkpoint::merge(parts), Err(Error::Parts(_)));
    assert!(refused(vec![]));
    assert!(refused(vec![part(0, &[4, 4])]), "lanes 2..4 missing");
    assert!(refused(vec![part(0, &[4, 4]), part(3, &[4, 4])]), "gap");
    assert!(
        refused(vec![part(0, &[4, 4]), part(1, &[4, 4, 4])]),
        "overlap"
    );
    assert!(
        refused(vec![part(0, &[4, 4]), part(2, &[5, 5])]),
        "steps differ"
    );
    let mut other = part(2, &[4, 4]);
    other.config.seed = 8;
    assert!(refused(vec![part(0, &[4, 4]), other]), "configs differ");
    let mut other = part(2, &[4, 4]);
    other.epoch = 3;
    assert!(refused(vec![part(0, &[4, 4]), other]), "epochs differ");
}

#[test]
fn states_of_a_part() {
    let p = part(2, &[4, 4]);
    assert!(!p.is_global());
    assert_eq!(p.states(2..4).unwrap(), p.lanes);
    assert!(matches!(p.states(0..2), Err(Error::Parts(_))));
    assert!(matches!(p.states(3..5), Err(Error::Parts(_))));
    #[allow(clippy::reversed_empty_ranges)]
    let reversed = 10..2;
    assert!(matches!(p.states(reversed), Err(Error::Parts(_))));
    let mixed = part(0, &[4, 5]);
    assert!(matches!(mixed.states(0..2), Err(Error::Parts(_))));
    assert_eq!(mixed.states(1..2).unwrap(), mixed.lanes[1..]);
}

#[test]
fn check_names_the_field() {
    let c = part(0, &[1, 1, 1, 1]);
    assert_eq!(c.check(&config(), 0xfeed), Ok(()));
    let field = |config: Config, fingerprint: u64| match c.check(&config, fingerprint) {
        Err(Error::Mismatch { field, .. }) => field,
        other => panic!("{other:?}"),
    };
    let mut other = config();
    other.packing.seq_len = 16;
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
    let lane = (any::<u64>(), any::<u64>(), any::<u32>())
        .prop_map(|(step, doc, offset)| LaneState { step, doc, offset });
    let packing = any::<[u32; 4]>().prop_map(|[seq_len, docs, cap, pad]| Packing {
        seq_len,
        docs,
        cap,
        pad,
    });
    let config = (".{0,12}", any::<u64>(), any::<[u32; 2]>(), packing).prop_map(
        |(column, seed, [lanes, window], packing)| Config {
            column,
            seed,
            lanes,
            window,
            packing,
        },
    );
    let lanes = proptest::collection::vec(lane, 0..8);
    (config, any::<[u64; 2]>(), any::<u32>(), lanes).prop_map(
        |(config, [fingerprint, epoch], first_lane, lanes)| Checkpoint {
            config,
            fingerprint,
            epoch,
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
