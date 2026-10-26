use super::*;

/// Records the documents a lane takes.
struct Seen<'a>(&'a Corpus, std::cell::RefCell<Vec<Doc>>);

impl Tokens for Seen<'_> {
    fn tokens(&self, doc: Doc) -> &[u32] {
        self.1.borrow_mut().push(doc);
        self.0.tokens(doc)
    }
}

proptest! {
    // A leftover replayed one document at a time, as the loader replays it
    // one window at a time, gives the same lane as one replay of it all.
    // `finish` before the last document is an error.
    #[test]
    fn replay_in_parts((lens, lanes, window, seed, p) in inputs()) {
        let c = Corpus::new(&lens);
        let rows = c.rows();
        let e = epoch(&rows, lanes, window, seed, p);
        for lane in 0..lanes {
            for (_, _, state) in run(&e, &c, lane, LaneState::default()) {
                let left = e.check(lane, state).unwrap() - state.doc;
                if left > 0 {
                    let mut short = e.replay(lane, state).unwrap();
                    short.take(&c, left - 1).unwrap();
                    prop_assert!(short.finish().is_err());
                }
                let mut parts = e.replay(lane, state).unwrap();
                for _ in 0..left {
                    parts.take(&c, 1).unwrap();
                }
                let mut l = parts.finish().unwrap();
                prop_assert_eq!(l.state(), state);
                // By `part`: each is in one window, and is what `take` takes.
                let mut by_part = e.replay(lane, state).unwrap();
                loop {
                    let part = by_part.part().to_vec();
                    let Some(first) = part.first() else { break };
                    let w = e.windows.iter().find(|w| w.groups().contains(&first.row_group));
                    let w = w.unwrap();
                    prop_assert!(part.iter().all(|d| w.groups().contains(&d.row_group)));
                    let seen = Seen(&c, std::cell::RefCell::default());
                    by_part.take(&seen, part.len() as u64).unwrap();
                    prop_assert_eq!(seen.1.into_inner(), part);
                }
                prop_assert_eq!(by_part.finish().unwrap().state(), state);
                let mut out = vec![0; p.seq_len as usize];
                let tail: Vec<Out> = (state.step..e.steps())
                    .map(|_| {
                        let real = l.fill(&c, &mut out);
                        (out.clone(), real, l.state())
                    })
                    .collect();
                prop_assert_eq!(&tail[..], &run(&e, &c, lane, state)[..]);
            }
        }
    }
}
