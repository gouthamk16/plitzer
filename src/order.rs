use std::ops::Range;

use crate::seed::{Purpose, key};
use crate::shuffle::shuffle;

/// One epoch's order over a dataset. Everything is recomputed from these
/// fields, so a checkpoint never stores the order itself.
///
/// The order works in slots: each row group has `slots[g]` of them, one per
/// token and one per row with no tokens, which is the leaf value count
/// that Parquet records in the footer. Slices are cut by slots, so each
/// lane gets the same share of every window's tokens, and lanes that each
/// take L tokens a step stay together.
pub struct Order<'a> {
    pub seed: u64,
    pub epoch: u64,
    /// B: sequence slots in the global batch.
    pub lanes: u32,
    /// W: row groups per window.
    pub window: u32,
    /// The slots of each row group. A row group's id is its index.
    pub slots: &'a [u32],
}

/// One window of an epoch. Only `Order::windows` builds one, so a window's
/// number always matches its row groups and no two windows share a lane key.
/// It caches its slot count, so use it only with the `Order` that built it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Window {
    index: u64,
    groups: Vec<u32>,
    slots: u64,
}

impl Window {
    /// The window's row-group ids, in storage order.
    pub fn groups(&self) -> &[u32] {
        &self.groups
    }
}

/// The part of one row group in a lane's slice: slots `slots` of row group
/// `row_group`, counted from the row group's first slot.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Part {
    pub row_group: u32,
    pub slots: Range<u64>,
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
        let n = u32::try_from(self.slots.len()).expect("more than u32::MAX row groups");
        let mut ids: Vec<u32> = (0..n).collect();
        shuffle(&mut ids, key(self.seed, self.epoch, Purpose::RowGroups));
        let window = |(i, chunk): (usize, &[u32])| {
            let mut groups = chunk.to_vec();
            groups.sort_unstable();
            let slots = groups.iter().map(|&g| self.size(g)).sum();
            Window {
                index: i as u64,
                groups,
                slots,
            }
        };
        ids.chunks(self.window as usize)
            .enumerate()
            .map(window)
            .collect()
    }

    /// The row groups of lane `lane`'s slice of `window`, in storage order,
    /// with the slots of each that the slice holds.
    ///
    /// Lane `l` owns slots `[l·N/B, (l+1)·N/B)` of the window's N, so lane
    /// sizes differ by at most one slot, and lanes are empty when N < B. A
    /// row that crosses an edge gives its first tokens to one lane and the
    /// rest to the next.
    pub fn lane(&self, window: &Window, lane: u32) -> Vec<Part> {
        let r = self.bounds(window, lane);
        let mut parts = Vec::new();
        let mut start = 0;
        for &g in &window.groups {
            let end = start + self.size(g);
            let (from, to) = (r.start.max(start), r.end.min(end));
            if from < to {
                let slots = from - start..to - start;
                parts.push(Part {
                    row_group: g,
                    slots,
                });
            }
            start = end;
        }
        parts
    }

    /// Puts lane `lane`'s pieces of `window` into its reading order.
    pub fn shuffle_lane<T>(&self, window: &Window, lane: u32, pieces: &mut [T]) {
        let purpose = Purpose::Lane {
            window: window.index,
            lane,
        };
        shuffle(pieces, key(self.seed, self.epoch, purpose));
    }

    /// The number of slots `lane` gets from `window`.
    pub fn lane_len(&self, window: &Window, lane: u32) -> u64 {
        let r = self.bounds(window, lane);
        r.end - r.start
    }

    /// For each lane in `lanes`, the positions in `window.groups()` of the row
    /// groups its slice touches. Empty for an empty slice.
    pub fn lane_groups(&self, window: &Window, lanes: Range<u32>) -> Vec<Range<usize>> {
        // ends[i]: the window's slots up to the end of its i-th row group.
        let ends: Vec<u64> = window
            .groups
            .iter()
            .scan(0, |end, &g| {
                *end += self.size(g);
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

    fn size(&self, g: u32) -> u64 {
        u64::from(self.slots[g as usize])
    }

    fn bounds(&self, window: &Window, lane: u32) -> Range<u64> {
        assert!(
            lane < self.lanes,
            "lane {lane} out of range for {} lanes",
            self.lanes
        );
        // `l · N` can exceed u64; the quotient never does.
        let bound =
            |l: u32| (u128::from(l) * u128::from(window.slots) / u128::from(self.lanes)) as u64;
        bound(lane)..bound(lane + 1)
    }
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
