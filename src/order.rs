use std::ops::Range;

use crate::seed::{Purpose, key};
use crate::shuffle::shuffle;

/// One document: row `row` of row group `row_group`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Doc {
    pub row_group: u32,
    pub row: u32,
}

/// One epoch's order over a dataset. Everything is recomputed from these
/// fields, so a checkpoint never stores the order itself.
pub struct Order<'a> {
    pub seed: u64,
    pub epoch: u64,
    /// B: sequence slots in the global batch.
    pub lanes: u32,
    /// W: row groups per window.
    pub window: u32,
    /// Row count of each row group. A row group's id is its index.
    pub rows: &'a [u32],
}

/// One window of an epoch. Only `Order::windows` builds one, so a window's
/// number always matches its row groups and no two windows share a lane key.
/// It caches its document count, so use it only with the `Order` that built it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Window {
    index: u64,
    groups: Vec<u32>,
    docs: u64,
}

impl Window {
    /// The window's row-group ids, in storage order.
    pub fn groups(&self) -> &[u32] {
        &self.groups
    }
}

impl Order<'_> {
    /// The epoch's windows. The last may hold fewer than W row groups.
    ///
    /// Row groups are shuffled across the dataset, then each window is sorted
    /// back into storage order, so a lane's contiguous slice of the window is
    /// a contiguous stretch of storage. That keeps each GPU's reads close to
    /// its own share instead of the whole window.
    pub fn windows(&self) -> Vec<Window> {
        assert!(self.window > 0, "window must be positive");
        let n = u32::try_from(self.rows.len()).expect("more than u32::MAX row groups");
        let mut ids: Vec<u32> = (0..n).collect();
        shuffle(&mut ids, key(self.seed, self.epoch, Purpose::RowGroups));
        let window = |(i, chunk): (usize, &[u32])| {
            let mut groups = chunk.to_vec();
            groups.sort_unstable();
            let docs = groups
                .iter()
                .map(|&g| u64::from(self.rows[g as usize]))
                .sum();
            Window {
                index: i as u64,
                groups,
                docs,
            }
        };
        ids.chunks(self.window as usize)
            .enumerate()
            .map(window)
            .collect()
    }

    /// Lane `lane`'s documents in `window`, in reading order.
    ///
    /// Lane `l` owns rows `[l·N/B, (l+1)·N/B)` of the window, so lane sizes
    /// differ by at most one and lanes are empty when N < B.
    pub fn lane(&self, window: &Window, lane: u32) -> Vec<Doc> {
        let mut docs = slice(&window.groups, self.rows, self.bounds(window, lane));
        let purpose = Purpose::Lane {
            window: window.index,
            lane,
        };
        shuffle(&mut docs, key(self.seed, self.epoch, purpose));
        docs
    }

    /// The number of documents `lane` gets from `window`, without the shuffle.
    pub fn lane_len(&self, window: &Window, lane: u32) -> u64 {
        let r = self.bounds(window, lane);
        r.end - r.start
    }

    fn bounds(&self, window: &Window, lane: u32) -> Range<u64> {
        assert!(
            lane < self.lanes,
            "lane {lane} out of range for {} lanes",
            self.lanes
        );
        // `l · N` can exceed u64; the quotient never does.
        let bound =
            |l: u32| (u128::from(l) * u128::from(window.docs) / u128::from(self.lanes)) as u64;
        bound(lane)..bound(lane + 1)
    }
}

/// The documents at positions `range` of the window's rows laid end to end.
fn slice(window: &[u32], rows: &[u32], range: Range<u64>) -> Vec<Doc> {
    let mut docs = Vec::with_capacity((range.end - range.start) as usize);
    let mut start = 0;
    for &g in window {
        let end = start + u64::from(rows[g as usize]);
        for pos in range.start.max(start)..range.end.min(end) {
            let row = (pos - start) as u32;
            docs.push(Doc { row_group: g, row });
        }
        start = end;
    }
    docs
}

/// The lanes GPU `rank` of `world` owns. Lanes depend only on B, so changing
/// `world` changes which GPU runs a lane, never what the lane reads.
pub fn gpu_lanes(lanes: u32, world: u32, rank: u32) -> Range<u32> {
    assert!(
        world > 0 && lanes % world == 0,
        "world {world} must divide {lanes} lanes"
    );
    assert!(rank < world, "rank {rank} out of range for world {world}");
    let per = lanes / world;
    rank * per..(rank + 1) * per
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    /// The given row groups' documents in storage order.
    fn storage(rows: &[u32], groups: impl IntoIterator<Item = u32>) -> Vec<Doc> {
        groups
            .into_iter()
            .flat_map(|g| (0..rows[g as usize]).map(move |row| Doc { row_group: g, row }))
            .collect()
    }

    /// Every lane's documents, window by window.
    fn all_lanes(o: &Order) -> Vec<Vec<Vec<Doc>>> {
        let lanes = |w: &Window| (0..o.lanes).map(|l| o.lane(w, l)).collect();
        o.windows().iter().map(lanes).collect()
    }

    // Small row groups and lane counts close to them, so empty row groups,
    // empty lanes (N < B) and short last windows all come up often.
    fn inputs() -> impl Strategy<Value = (Vec<u32>, u32, u32, u64)> {
        (
            prop::collection::vec(0u32..20, 0..30),
            1u32..12,
            1u32..6,
            any::<u64>(),
        )
    }

    proptest! {
        #[test]
        fn exactly_once((rows, lanes, window, seed) in inputs()) {
            let o = Order { seed, epoch: 3, lanes, window, rows: &rows };
            let mut seen: Vec<Doc> = all_lanes(&o).into_iter().flatten().flatten().collect();
            seen.sort_unstable();
            prop_assert_eq!(seen, storage(&rows, 0..rows.len() as u32));
        }

        #[test]
        fn windows_partition((rows, lanes, window, seed) in inputs()) {
            let o = Order { seed, epoch: 3, lanes, window, rows: &rows };
            let windows = o.windows();
            let groups: Vec<&[u32]> = windows.iter().map(Window::groups).collect();
            for (i, g) in groups.iter().enumerate() {
                prop_assert!(g.is_sorted());
                if i + 1 < groups.len() {
                    prop_assert_eq!(g.len(), window as usize);
                }
            }
            let mut all = groups.concat();
            all.sort_unstable();
            prop_assert_eq!(all, (0..rows.len() as u32).collect::<Vec<_>>());
        }

        // Sorted lane slices laid end to end in lane order must rebuild the
        // window's storage order: each slice is one contiguous run.
        #[test]
        fn slices_contiguous((rows, lanes, window, seed) in inputs()) {
            let o = Order { seed, epoch: 3, lanes, window, rows: &rows };
            for (w, lanes) in o.windows().iter().zip(all_lanes(&o)) {
                let sizes: Vec<usize> = lanes.iter().map(Vec::len).collect();
                let spread = sizes.iter().max().unwrap() - sizes.iter().min().unwrap();
                prop_assert!(spread <= 1, "sizes {:?}", sizes);
                let mut joined = Vec::new();
                for mut lane in lanes {
                    lane.sort_unstable();
                    joined.extend(lane);
                }
                prop_assert_eq!(joined, storage(&rows, w.groups().iter().copied()));
            }
        }

        #[test]
        fn lane_len_matches((rows, lanes, window, seed) in inputs()) {
            let o = Order { seed, epoch: 3, lanes, window, rows: &rows };
            for w in o.windows() {
                for l in 0..lanes {
                    prop_assert_eq!(o.lane_len(&w, l), o.lane(&w, l).len() as u64);
                }
            }
        }

        #[test]
        fn gpu_lanes_partition(per in 1u32..50, world in 1u32..20) {
            let lanes = per * world;
            let joined: Vec<u32> = (0..world).flat_map(|r| gpu_lanes(lanes, world, r)).collect();
            prop_assert_eq!(joined, (0..lanes).collect::<Vec<_>>());
        }
    }

    #[test]
    #[should_panic(expected = "must divide")]
    fn gpu_lanes_rejects_uneven_world() {
        gpu_lanes(10, 4, 0);
    }

    #[test]
    fn golden() {
        let rows = [3, 1, 4, 1, 5, 9, 2, 6];
        let o = Order {
            seed: 7,
            epoch: 0,
            lanes: 2,
            window: 3,
            rows: &rows,
        };
        let pairs = |docs: &[Doc]| {
            docs.iter()
                .map(|d| (d.row_group, d.row))
                .collect::<Vec<_>>()
        };
        let got: Vec<Vec<Vec<(u32, u32)>>> = all_lanes(&o)
            .iter()
            .map(|w| w.iter().map(|l| pairs(l)).collect())
            .collect();
        let groups: Vec<Vec<u32>> = o.windows().iter().map(|w| w.groups().to_vec()).collect();
        assert_eq!(groups, [[1, 5, 6].as_slice(), &[2, 4, 7], &[0, 3]]);
        #[rustfmt::skip]
        assert_eq!(got, [
            [&[(5, 1), (1, 0), (5, 0), (5, 4), (5, 2), (5, 3)][..],
             &[(6, 1), (6, 0), (5, 8), (5, 5), (5, 6), (5, 7)]],
            [&[(4, 0), (2, 0), (2, 3), (2, 1), (2, 2), (4, 2), (4, 1)],
             &[(7, 3), (7, 5), (4, 3), (7, 2), (4, 4), (7, 0), (7, 4), (7, 1)]],
            [&[(0, 0), (0, 1)],
             &[(3, 0), (0, 2)]],
        ]);

        // Many windows and lanes, so a key built from the wrong window or
        // lane number changes the hash.
        let rows: Vec<u32> = (0..500).map(|i| i % 37).collect();
        let o = Order {
            seed: 7,
            epoch: 0,
            lanes: 8,
            window: 16,
            rows: &rows,
        };
        let hash = all_lanes(&o).iter().flatten().flatten().fold(0u64, |h, d| {
            let x = u64::from(d.row_group) << 32 | u64::from(d.row);
            h.wrapping_mul(1_000_003).wrapping_add(x)
        });
        assert_eq!(hash, 2_028_779_180_813_407_981);
    }

    #[test]
    fn epochs_differ() {
        let rows = [1; 100];
        let at = |epoch| Order {
            seed: 7,
            epoch,
            lanes: 1,
            window: 10,
            rows: &rows,
        };
        assert_ne!(at(0).windows(), at(1).windows());

        let rows = [100];
        let at = |epoch| Order {
            seed: 7,
            epoch,
            lanes: 1,
            window: 1,
            rows: &rows,
        };
        let lane = |epoch| {
            let o = at(epoch);
            o.lane(&o.windows()[0], 0)
        };
        assert_ne!(lane(0), lane(1));
    }
}
