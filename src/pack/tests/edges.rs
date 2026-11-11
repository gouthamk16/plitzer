use super::*;

#[test]
fn need_names_the_first_missing_row_group() {
    struct Without<'a>(&'a Corpus, u32);
    impl Tokens for Without<'_> {
        fn row_group(&self, g: u32) -> Option<Rows<'_>> {
            (g != self.1).then(|| self.0.row_group(g)).flatten()
        }
    }
    // L is all 12 tokens, so the first sequence needs both row groups.
    let c = Corpus::new(&[vec![3, 4], vec![5]]);
    let o = Orders::new(1, 1, 2, 12, &c.slots());
    let lane = Lane::new(&o, 0, Position::default()).unwrap();
    for skip in 0..2 {
        let missing = Need::Missing {
            slice: (0, 0),
            row_group: skip,
        };
        assert_eq!(lane.need(&Without(&c, skip)), missing);
    }
    assert_eq!(lane.need(&c), Need::Ready);
}

#[test]
fn lanes_stay_together() {
    // Row groups whose documents are short in some and long in others, as
    // in FineWeb-Edu: lanes cut by documents drift apart by whole windows,
    // and lanes cut by slots stay in the same window or the next.
    let lens: Vec<Vec<usize>> = (0..64)
        .map(|g| (0..50).map(|r| 1 + (g * 13 % 40) * (r % 3 + 1)).collect())
        .collect();
    let c = Corpus::new(&lens);
    let (lanes, window) = (16, 4);
    let o = Orders::new(3, lanes, window, 64, &c.slots());
    let mut all: Vec<Lane> = (0..lanes)
        .map(|l| Lane::new(&o, l, Position::default()).unwrap())
        .collect();
    let windows = o.epoch(0).windows().len() as u64;
    let mut out = vec![0; 64];
    for _ in 0..2000 {
        let at: Vec<u64> = all
            .iter()
            .map(|l| l.slice().0 * windows + l.slice().1 as u64)
            .collect();
        assert!(
            at.iter().max().unwrap() - at.iter().min().unwrap() <= 1,
            "{at:?}"
        );
        for lane in &mut all {
            assert_eq!(lane.need(&c), Need::Ready);
            lane.fill(&c, &mut out);
        }
    }
}

#[test]
fn a_lane_with_no_tokens_is_an_error() {
    // Three slots and four lanes: lane 0 never gets one.
    let c = Corpus::new(&[vec![1, 1, 1]]);
    let o = Orders::new(1, 4, 1, 2, &c.slots());
    let lane = Lane::new(&o, 0, Position::default()).unwrap();
    assert!(matches!(lane.need(&c), Need::Empty(why) if why.contains("no token")));
    // Only empty documents: lane 0 gets them, but never a token.
    let c = Corpus::new(&[vec![0, 0]]);
    let o = Orders::new(1, 1, 1, 2, &c.slots());
    let lane = Lane::new(&o, 0, Position::default()).unwrap();
    assert!(matches!(lane.need(&c), Need::Empty(why) if why.contains("no token")));
}

#[test]
fn bad_positions_are_refused() {
    // One window of 12 slots: 3 + 4 + 5 tokens.
    let c = Corpus::new(&[vec![3, 4], vec![5]]);
    let o = Orders::new(1, 1, 2, 2, &c.slots());
    let at = |epoch, window, token| Position {
        epoch,
        window,
        token,
    };
    assert!(Lane::new(&o, 0, at(0, 0, 13)).is_err(), "past the 12 slots");
    assert!(
        Lane::new(&o, 0, at(0, 2, 0)).is_err(),
        "past the one window"
    );
    assert!(
        Lane::new(&o, 1, at(0, 0, 0)).is_err(),
        "a lane out of range"
    );
    // At the end of epoch 0's windows it stands at the start of epoch 1.
    assert_eq!(
        Lane::new(&o, 0, at(0, 1, 0)).unwrap().position(),
        at(1, 0, 0)
    );
    // An empty row has a slot and no token, so the window has 11 tokens: a
    // position past them is found once the tokens are there.
    let c = Corpus::new(&[vec![3, 0], vec![8]]);
    let o = Orders::new(1, 1, 2, 2, &c.slots());
    let lane = Lane::new(&o, 0, at(0, 0, 12)).unwrap();
    assert!(matches!(lane.need(&c), Need::Bad(why) if why.contains("past its slice")));
}

#[test]
#[should_panic(expected = "out must hold L tokens")]
fn wrong_out_len_panics() {
    let c = Corpus::new(&[vec![3]]);
    let o = Orders::new(1, 1, 1, 2, &c.slots());
    Lane::new(&o, 0, Position::default())
        .unwrap()
        .fill(&c, &mut [0; 3]);
}

#[test]
fn golden() {
    // Two lanes over row groups of [3, 5], [0], [4, 1, 6], [2, 7] tokens:
    // documents longer than L, an empty one, and steps across epoch ends.
    let lens = vec![vec![3, 5], vec![0], vec![4, 1, 6], vec![2, 7]];
    let c = Corpus::new(&lens);
    let o = Orders::new(7, 2, 2, 4, &c.slots());
    let seqs = |lane| -> Vec<Vec<u32>> {
        run(&o, &c, lane, Position::default(), 6)
            .into_iter()
            .map(|(s, _)| s)
            .collect()
    };
    for lane in 0..2 {
        let flat: Vec<u32> = seqs(lane).concat();
        assert_eq!(flat, stream(&o, &c, lane, 24));
    }
    // A larger corpus, so a change anywhere in the packer changes the hash.
    let lens: Vec<Vec<usize>> = (0..60)
        .map(|g| (1..=g % 7 + 1).map(|r| (g * 7 + r * 3) % 23).collect())
        .collect();
    let c = Corpus::new(&lens);
    let o = Orders::new(7, 4, 8, 16, &c.slots());
    let hash = (0..4)
        .flat_map(|l| run(&o, &c, l, Position::default(), 200))
        .flat_map(|(seq, _)| seq)
        .fold(0u64, |h, t| {
            h.wrapping_mul(1_000_003).wrapping_add(t.into())
        });
    assert_eq!(hash, 1_108_164_981_550_171_362);
}
