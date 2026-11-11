use super::*;

#[test]
fn need_names_the_first_missing_document() {
    struct Without<'a>(&'a Corpus, u32);
    impl Tokens for Without<'_> {
        fn tokens(&self, doc: Doc) -> Option<&[u32]> {
            (doc.row_group != self.1)
                .then(|| self.0.tokens(doc))
                .flatten()
        }
    }
    // L is all 12 tokens, so the first sequence needs every document.
    let c = Corpus::new(&[vec![3, 4], vec![5]]);
    let o = Orders::new(1, 1, 2, 12, &c.rows());
    let lane = Lane::new(&o, 0, Position::default()).unwrap();
    let epoch = o.epoch(0);
    let docs = epoch.order().lane(&epoch.windows()[0], 0);
    for skip in 0..2 {
        let first = *docs.iter().find(|d| d.row_group == skip).unwrap();
        let missing = Need::Missing {
            slice: (0, 0),
            doc: first,
        };
        assert_eq!(lane.need(&Without(&c, skip)), missing);
    }
    assert_eq!(lane.need(&c), Need::Ready);
}

#[test]
fn a_lane_with_no_tokens_is_an_error() {
    // Three documents and four lanes: lane 0 never gets a document.
    let c = Corpus::new(&[vec![2, 2, 2]]);
    let o = Orders::new(1, 4, 1, 2, &c.rows());
    let lane = Lane::new(&o, 0, Position::default()).unwrap();
    assert!(matches!(lane.need(&c), Need::Empty(why) if why.contains("no token")));
    // Only empty documents: lane 0 gets them, but never a token.
    let c = Corpus::new(&[vec![0, 0]]);
    let o = Orders::new(1, 1, 1, 2, &c.rows());
    let lane = Lane::new(&o, 0, Position::default()).unwrap();
    assert!(matches!(lane.need(&c), Need::Empty(why) if why.contains("no token")));
}

#[test]
fn bad_positions_are_refused() {
    let c = Corpus::new(&[vec![3, 4], vec![5]]);
    let o = Orders::new(1, 1, 1, 2, &c.rows());
    let at = |epoch, doc, offset| Position { epoch, doc, offset };
    assert!(
        Lane::new(&o, 0, at(0, 4, 0)).is_err(),
        "past the 3 documents"
    );
    assert!(
        Lane::new(&o, 0, at(0, 3, 1)).is_err(),
        "an offset after the end"
    );
    assert!(
        Lane::new(&o, 1, at(0, 0, 0)).is_err(),
        "a lane out of range"
    );
    // At the end of epoch 0 it stands at the start of epoch 1.
    assert_eq!(
        Lane::new(&o, 0, at(0, 3, 0)).unwrap().position(),
        at(1, 0, 0)
    );
    // An offset past the document is found once the tokens are there.
    let lane = Lane::new(&o, 0, at(0, 0, 9)).unwrap();
    assert!(matches!(lane.need(&c), Need::Bad(why) if why.contains("past its document")));
}

#[test]
#[should_panic(expected = "out must hold L tokens")]
fn wrong_out_len_panics() {
    let c = Corpus::new(&[vec![3]]);
    let o = Orders::new(1, 1, 1, 2, &c.rows());
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
    let o = Orders::new(7, 2, 2, 4, &c.rows());
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
    let o = Orders::new(7, 4, 8, 16, &c.rows());
    let hash = (0..4)
        .flat_map(|l| run(&o, &c, l, Position::default(), 200))
        .flat_map(|(seq, _)| seq)
        .fold(0u64, |h, t| {
            h.wrapping_mul(1_000_003).wrapping_add(t.into())
        });
    assert_eq!(hash, 5_178_903_507_416_173_685);
}
