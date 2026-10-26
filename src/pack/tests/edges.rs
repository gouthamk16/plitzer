//! Small fixed cases: inputs that panic or are refused.

use super::*;

fn small() -> (Corpus, Vec<u32>) {
    let c = Corpus::new(&[vec![3, 5, 2], vec![4]]);
    let rows = c.rows();
    (c, rows)
}

const P4: Packing = Packing {
    seq_len: 4,
    docs: 1,
    cap: 8,
    pad: PAD,
};

#[test]
#[should_panic(expected = "epoch has ended")]
fn fill_past_end_panics() {
    let (c, rows) = small();
    let e = epoch(&rows, 1, 2, 7, P4);
    let end = LaneState {
        step: e.steps(),
        doc: 4,
        offset: 0,
    };
    let mut l = e.lane(0, end, &c).unwrap();
    l.fill(&c, &mut [0; 4]);
}

#[test]
#[should_panic(expected = "out must hold seq_len tokens")]
fn wrong_out_len_panics() {
    let (c, rows) = small();
    let e = epoch(&rows, 1, 2, 7, P4);
    e.lane(0, LaneState::default(), &c)
        .unwrap()
        .fill(&c, &mut [0; 3]);
}

#[test]
fn state_past_cursor_is_refused() {
    let (c, rows) = small();
    let e = epoch(&rows, 1, 2, 7, P4);
    let got = e.lane(
        0,
        LaneState {
            step: 1,
            doc: 2,
            offset: 0,
        },
        &c,
    );
    assert!(got.is_err_and(|why| why.contains("past the cursor")));
}

#[test]
fn leftover_after_the_last_step_is_refused() {
    let (c, rows) = small();
    let e = epoch(&rows, 1, 2, 7, P4);
    let end = LaneState {
        step: e.steps(),
        doc: e.lane_docs(0) - 1,
        offset: 0,
    };
    let got = e.lane(0, end, &c);
    assert!(got.is_err_and(|why| why.contains("after the last step")));
}

#[test]
fn offset_past_doc_is_refused() {
    let (c, rows) = small();
    let e = epoch(&rows, 1, 2, 7, P4);
    let got = e.lane(
        0,
        LaneState {
            step: 1,
            doc: 0,
            offset: 99,
        },
        &c,
    );
    assert!(got.is_err_and(|why| why.contains("offset is past its document")));
}

#[test]
#[should_panic(expected = "must be positive")]
fn zero_seq_len_panics() {
    let rows = [1];
    epoch(&rows, 1, 1, 7, Packing { seq_len: 0, ..P4 });
}

#[test]
fn lane_out_of_range_is_refused() {
    let (_, rows) = small();
    let e = epoch(&rows, 1, 2, 7, P4);
    assert!(
        e.check(2, LaneState::default())
            .is_err_and(|why| why.contains("out of range"))
    );
}

/// Counts the documents a lane takes.
struct Counted<'a>(&'a Corpus, std::cell::Cell<u32>);

impl Tokens for Counted<'_> {
    fn tokens(&self, doc: Doc) -> &[u32] {
        self.1.set(self.1.get() + 1);
        self.0.tokens(doc)
    }
}

// A state far behind its cursor is refused as soon as its leftover cannot
// fit the cap, not after every document before the cursor is in memory.
#[test]
fn far_behind_state_stops_early() {
    let c = Corpus::new(&[vec![5; 20]]);
    let rows = c.rows();
    let p = Packing {
        seq_len: 4,
        docs: 1,
        cap: 8,
        pad: PAD,
    };
    let e = epoch(&rows, 1, 1, 7, p);
    let behind = LaneState {
        step: 10,
        doc: 0,
        offset: 0,
    };
    let counted = Counted(&c, std::cell::Cell::new(0));
    let mut replay = e.replay(0, behind).unwrap();
    let got = replay.take(&counted, u64::MAX);
    assert!(got.is_err_and(|why| why.contains("more than the cap")));
    // 5 + 5 tokens fit after the first document's 5; the third does not.
    assert_eq!(counted.1.get(), 3);
}
