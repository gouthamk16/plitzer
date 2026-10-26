use super::*;

mod edges;
mod replay;
use crate::order::gpu_lanes;
use proptest::prelude::*;

const PAD: u32 = u32::MAX;

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
    fn tokens(&self, doc: Doc) -> &[u32] {
        &self.0[doc.row_group as usize][doc.row as usize]
    }
}

/// One step of one lane: the sequence, its real-token count, the state after.
type Out = (Vec<u32>, u32, LaneState);

/// Runs `lane` from `state` to the end of the epoch.
fn run(e: &Arc<Epoch>, c: &Corpus, lane: u32, state: LaneState) -> Vec<Out> {
    let mut l = e.lane(lane, state, c).unwrap();
    let mut out = vec![0; e.packing.seq_len as usize];
    (state.step..e.steps())
        .map(|_| {
            let real = l.fill(c, &mut out);
            (out.clone(), real, l.state())
        })
        .collect()
}

/// Every token of lane `lane`'s documents, in the order the lane takes them.
fn stream(e: &Epoch, c: &Corpus, lane: u32) -> Vec<u32> {
    let docs = e.windows.iter().flat_map(|w| e.order().lane(w, lane));
    docs.flat_map(|d| c.tokens(d).to_vec()).collect()
}

/// Doc lengths per row group, then lanes, window, seed and packing. Small
/// values so zero-length docs, empty lanes, D > lane size, C = 0, C < L and
/// L = 1 all come up often.
fn inputs() -> impl Strategy<Value = (Vec<Vec<usize>>, u32, u32, u64, Packing)> {
    let lens = prop::collection::vec(prop::collection::vec(0usize..12, 0..6), 0..10);
    let packing = (1u32..8, 1u32..4, 0u32..12).prop_map(|(seq_len, docs, cap)| Packing {
        seq_len,
        docs,
        cap,
        pad: PAD,
    });
    (lens, 1u32..6, 1u32..4, any::<u64>(), packing)
}

fn epoch(rows: &[u32], lanes: u32, window: u32, seed: u64, p: Packing) -> Arc<Epoch> {
    let order = Order {
        seed,
        epoch: 1,
        lanes,
        window,
        rows,
    };
    Arc::new(Epoch::new(order, p))
}

proptest! {
    #[test]
    fn resume_matches_tail((lens, lanes, window, seed, p) in inputs()) {
        let c = Corpus::new(&lens);
        let rows = c.rows();
        let e = epoch(&rows, lanes, window, seed, p);
        for lane in 0..lanes {
            let full = run(&e, &c, lane, LaneState::default());
            for k in 0..full.len() {
                prop_assert_eq!(&run(&e, &c, lane, full[k].2)[..], &full[k + 1..]);
            }
        }
    }

    // Each sequence's real tokens are one run of the lane's stream, and runs
    // only move forward. With a cap that never overflows, nothing is lost.
    #[test]
    fn exactly_once_tokens((lens, lanes, window, seed, p) in inputs()) {
        let c = Corpus::new(&lens);
        let rows = c.rows();
        let total = lens.iter().flatten().sum::<usize>() as u32;
        for cap in [p.cap, total] {
            let e = epoch(&rows, lanes, window, seed, Packing { cap, ..p });
            let mut all = Vec::new();
            for lane in 0..lanes {
                let s = stream(&e, &c, lane);
                let pos = |t: u32| s.iter().position(|&x| x == t).unwrap();
                let mut next = 0;
                for (seq, real, _) in run(&e, &c, lane, LaneState::default()) {
                    let seq = &seq[..real as usize];
                    if let Some(&first) = seq.first() {
                        let start = pos(first);
                        prop_assert!(start >= next);
                        prop_assert_eq!(seq, &s[start..start + seq.len()]);
                        next = start + seq.len();
                    }
                    all.extend_from_slice(seq);
                }
            }
            if cap == total {
                all.sort_unstable();
                prop_assert_eq!(all, (0..total).collect::<Vec<_>>());
            }
        }
    }

    #[test]
    fn cap_and_padding((lens, lanes, window, seed, p) in inputs()) {
        let c = Corpus::new(&lens);
        let rows = c.rows();
        let e = epoch(&rows, lanes, window, seed, p);
        let mut out = vec![0; p.seq_len as usize];
        for lane in 0..lanes {
            let mut l = e.lane(lane, LaneState::default(), &c).unwrap();
            for _ in 0..e.steps() {
                let real = l.fill(&c, &mut out) as usize;
                prop_assert!(out[real..].iter().all(|&t| t == PAD));
                prop_assert!(l.leftover.tokens.len() <= p.cap as usize);
                if real < out.len() {
                    prop_assert!(l.leftover.tokens.is_empty());
                }
            }
            prop_assert!(l.leftover.tokens.is_empty());
            let end = LaneState { step: e.steps(), doc: e.lane_docs(lane), offset: 0 };
            prop_assert_eq!(l.state(), end);
        }
    }

    // A run at world a, checkpointed at step k and resumed at world b, gives
    // the same global batches as one uninterrupted run at world 1.
    #[test]
    fn world_invariant(
        (lens, _, window, seed, p) in inputs(),
        a in 0usize..6,
        b in 0usize..6,
        k in 0u64..20,
    ) {
        const LANES: u32 = 12;
        let worlds = [1, 2, 3, 4, 6, 12];
        let c = Corpus::new(&lens);
        let rows = c.rows();
        let e = epoch(&rows, LANES, window, seed, p);
        let k = k.min(e.steps());
        let per_lane: Vec<Vec<Out>> =
            (0..LANES).map(|l| run(&e, &c, l, LaneState::default())).collect();
        let batch = |step: usize| -> Vec<u32> {
            per_lane.iter().flat_map(|r| r[step].0.clone()).collect()
        };

        let mut got: Vec<Vec<u32>> = vec![Vec::new(); e.steps() as usize];
        let mut states = vec![LaneState::default(); LANES as usize];
        for (world, to) in [(worlds[a], k), (worlds[b], e.steps())] {
            for rank in 0..world {
                for lane in gpu_lanes(LANES, world, rank) {
                    let mut l = e.lane(lane, states[lane as usize], &c).unwrap();
                    let mut out = vec![0; p.seq_len as usize];
                    for step in states[lane as usize].step..to {
                        l.fill(&c, &mut out);
                        got[step as usize].extend_from_slice(&out);
                    }
                    states[lane as usize] = l.state();
                }
            }
        }
        for (step, g) in got.iter().enumerate() {
            prop_assert_eq!(g, &batch(step));
        }
    }
}

#[test]
fn golden() {
    // Rows per group: [2, 1, 3, 2]. Doc lengths chosen so a step overflows
    // the cap, a lane runs short, and one doc is empty.
    let lens = vec![vec![3, 5], vec![0], vec![4, 1, 6], vec![2, 7]];
    let c = Corpus::new(&lens);
    let rows = c.rows();
    let p = Packing {
        seq_len: 4,
        docs: 2,
        cap: 3,
        pad: PAD,
    };
    let e = epoch(&rows, 2, 2, 7, p);
    // Lane 1 takes docs [19 20] [21..=27] [13..=18] [3..=7] [], two a step.
    // Step 0 emits 19..=22 and drops 23 24 to keep the newest 3 tokens. Step 1
    // drops 14..=18 3 4. Step 2 runs short. The last step only flushes.
    assert_eq!(e.steps(), 4);
    let s = |step, doc, offset| LaneState { step, doc, offset };
    const P: u32 = PAD;
    #[rustfmt::skip]
    assert_eq!(run(&e, &c, 0, LaneState::default()), [
        (vec![12, 8, 9, 10], 4, s(1, 1, 3)),
        (vec![11, 0, 1, 2], 4, s(2, 3, 0)),
        (vec![P, P, P, P], 0, s(3, 3, 0)),
        (vec![P, P, P, P], 0, s(4, 3, 0)),
    ]);
    #[rustfmt::skip]
    assert_eq!(run(&e, &c, 1, LaneState::default()), [
        (vec![19, 20, 21, 22], 4, s(1, 1, 4)),
        (vec![25, 26, 27, 13], 4, s(2, 3, 2)),
        (vec![5, 6, 7, P], 3, s(3, 5, 0)),
        (vec![P, P, P, P], 0, s(4, 5, 0)),
    ]);

    // A larger corpus, so a change anywhere in the packer changes the hash.
    let lens: Vec<Vec<usize>> = (0..60)
        .map(|g| (0..g % 7).map(|r| (g * 7 + r * 3) % 23).collect())
        .collect();
    let c = Corpus::new(&lens);
    let rows = c.rows();
    let p = Packing {
        seq_len: 16,
        docs: 2,
        cap: 24,
        pad: PAD,
    };
    let e = epoch(&rows, 4, 8, 7, p);
    let hash = (0..4)
        .flat_map(|l| run(&e, &c, l, LaneState::default()))
        .flat_map(|(seq, _, _)| seq)
        .fold(0u64, |h, t| {
            h.wrapping_mul(1_000_003).wrapping_add(t.into())
        });
    assert_eq!(hash, 4_952_267_354_457_612_743);
}
