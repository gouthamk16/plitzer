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

    /// For each lane in `lanes`, the positions in `window.groups()` of the row
    /// groups its slice touches. Empty for an empty slice.
    pub fn lane_groups(&self, window: &Window, lanes: Range<u32>) -> Vec<Range<usize>> {
        // ends[i]: the window's documents up to the end of its i-th row group.
        let ends: Vec<u64> = window
            .groups
            .iter()
            .scan(0, |end, &g| {
                *end += u64::from(self.rows[g as usize]);
                Some(*end)
            })
            .collect();
        let groups = |l| {
            let r = self.bounds(window, l);
            let first = ends.partition_point(|&e| e <= r.start);
            let last = ends.partition_point(|&e| e < r.end);
            if r.is_empty() {
                first..first
            } else {
                first..last + 1
            }
        };
        lanes.map(groups).collect()
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
        world > 0 && lanes.is_multiple_of(world),
        "world {world} must divide {lanes} lanes"
    );
    assert!(rank < world, "rank {rank} out of range for world {world}");
    let per = lanes / world;
    rank * per..(rank + 1) * per
}

#[cfg(test)]
mod tests;
