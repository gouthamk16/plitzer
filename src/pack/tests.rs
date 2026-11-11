use super::*;

mod edges;
use proptest::prelude::*;

/// Documents whose tokens are unique ids, numbered in storage order.
struct Corpus(Vec<Vec<Vec<u32>>>);

impl Corpus {
    fn new(lens: &[Vec<usize>]) -> Self {
        let mut id = 0;
        let mut token = || {
            id += 1;
            id - 1
        };
        let groups = lens
            .iter()
            .map(|g| {
                g.iter()
                    .map(|&n| (0..n).map(|_| token()).collect())
                    .collect()
            })
            .collect();
        Corpus(groups)
    }

    fn rows(&self) -> Vec<u32> {
        self.0.iter().map(|g| g.len() as u32).collect()
    }
}

impl Tokens for Corpus {
    fn tokens(&self, doc: Doc) -> Option<&[u32]> {
        Some(&self.0[doc.row_group as usize][doc.row as usize])
    }
}

/// `steps` sequences of lane `lane` from `at`, and the position after each.
fn run(
    o: &Arc<Orders>,
    c: &Corpus,
    lane: u32,
    at: Position,
    steps: usize,
) -> Vec<(Vec<u32>, Position)> {
    let mut l = Lane::new(o, lane, at).unwrap();
    let mut out = vec![0; o.seq_len as usize];
    (0..steps)
        .map(|_| {
            assert_eq!(l.need(c), Need::Ready);
            l.fill(c, &mut out);
            (out.clone(), l.position())
        })
        .collect()
}

/// The first `n` tokens of lane `lane`'s stream: its documents of epoch 0,
/// then of epoch 1, and so on, in the order the lane takes them.
fn stream(o: &Orders, c: &Corpus, lane: u32, n: usize) -> Vec<u32> {
    let mut tokens = Vec::new();
    for e in 0.. {
        let epoch = o.epoch(e);
        for w in epoch.windows() {
            for d in epoch.order().lane(w, lane) {
                tokens.extend_from_slice(c.tokens(d).unwrap());
            }
        }
        if tokens.len() >= n {
            break;
        }
    }
    tokens.truncate(n);
    tokens
}

/// Doc lengths per row group, then lanes, window, seed and L. Small values,
/// so empty documents, empty lanes, documents longer than L, L = 1 and steps
/// across many epochs all come up often.
fn inputs() -> impl Strategy<Value = (Vec<Vec<usize>>, u32, u32, u64, u32)> {
    let lens = prop::collection::vec(prop::collection::vec(0usize..12, 1..6), 1..10);
    (lens, 1u32..6, 1u32..4, any::<u64>(), 1u32..8)
}

/// The orders, if every lane gets a token in each epoch that a run of
/// `steps` steps reaches: a lane with none is an error, which `edges` tests.
fn orders(
    lens: &[Vec<usize>],
    lanes: u32,
    window: u32,
    seed: u64,
    seq_len: u32,
    steps: usize,
) -> Option<Arc<Orders>> {
    let c = Corpus::new(lens);
    let o = Orders::new(seed, lanes, window, seq_len, &c.rows());
    let need = (steps + 1) * seq_len as usize;
    let fed = (0..lanes).all(|l| {
        let mut got = 0;
        (0..).take(need + 1).any(|e| {
            let epoch = o.epoch(e);
            let docs = epoch
                .windows()
                .iter()
                .flat_map(|w| epoch.order().lane(w, l));
            let n: usize = docs.map(|d| c.tokens(d).unwrap().len()).sum();
            got += n;
            n == 0 || got >= need
        }) && got >= need
    });
    fed.then_some(o)
}

/// Enough steps for a few epochs of every lane.
fn steps(lens: &[Vec<usize>], seq_len: u32) -> usize {
    2 * lens.iter().flatten().sum::<usize>() / seq_len as usize + 3
}

proptest! {
    // Each lane gives its stream cut into sequences of exactly L tokens:
    // nothing is dropped, nothing is padded, across epoch ends.
    #[test]
    fn lossless((lens, lanes, window, seed, l) in inputs()) {
        let c = Corpus::new(&lens);
        let steps = steps(&lens, l);
        let Some(o) = orders(&lens, lanes, window, seed, l, steps) else { return Ok(()) };
        for lane in 0..lanes {
            let got: Vec<u32> = run(&o, &c, lane, Position::default(), steps)
                .into_iter()
                .flat_map(|(seq, _)| seq)
                .collect();
            prop_assert_eq!(&got, &stream(&o, &c, lane, got.len()));
        }
    }

    // A lane made at the position after any step gives the rest of the run.
    #[test]
    fn resume_matches_tail((lens, lanes, window, seed, l) in inputs()) {
        let c = Corpus::new(&lens);
        let steps = steps(&lens, l);
        let Some(o) = orders(&lens, lanes, window, seed, l, steps) else { return Ok(()) };
        for lane in 0..lanes {
            let full = run(&o, &c, lane, Position::default(), steps);
            for k in 0..steps {
                let tail = run(&o, &c, lane, full[k].1, steps - k - 1);
                prop_assert_eq!(&tail[..], &full[k + 1..]);
            }
        }
    }
}
