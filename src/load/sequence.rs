//! The order of a GPU's reads: the row groups its lanes take from, window by
//! window, epoch after epoch. Lanes use their tokens at their own pace, so a
//! read's key says where it is in this order, not at which step it is needed.

use std::ops::Range;
use std::sync::Arc;

use crate::index::Index;
use crate::pack::{Epoch, Orders};

/// A read's place in the order: `(epoch, window, position in the window's
/// row groups)`. Keys only grow along the order.
pub(super) type Key = (u64, usize, usize);

/// One window of the order, for the lanes of one GPU.
pub(super) struct WindowGroups {
    pub epoch: u64,
    pub window: usize,
    /// The window's row groups, in storage order.
    pub groups: Vec<u32>,
    /// For each of the GPU's lanes, the positions in `groups` of its slice.
    pub lanes: Vec<Range<usize>>,
}

impl WindowGroups {
    /// The positions of the row groups that some lane uses. The GPU's lanes
    /// are contiguous, so their slices are one stretch of the window.
    pub fn used(&self) -> Range<usize> {
        let used = self.lanes.iter().filter(|r| !r.is_empty());
        let start = used.clone().map(|r| r.start).min().unwrap_or(0);
        let end = used.map(|r| r.end).max().unwrap_or(0);
        start..end.max(start)
    }

    /// The GPU's lanes, as indices from 0, whose slice holds position `i`.
    pub fn users(&self, i: usize) -> impl Iterator<Item = usize> + '_ {
        let holds = move |(l, r): (usize, &Range<usize>)| r.contains(&i).then_some(l);
        self.lanes.iter().enumerate().filter_map(holds)
    }

    pub fn key(&self, i: usize) -> Key {
        (self.epoch, self.window, i)
    }
}

/// The windows of the order, from a start, for lanes `lanes`.
pub(super) struct Sequence {
    orders: Arc<Orders>,
    lanes: Range<u32>,
    epoch: Arc<Epoch>,
    window: usize,
}

impl Sequence {
    /// From window `start.1` of epoch `start.0`. The dataset must have row
    /// groups, or the order would have no windows to give.
    pub fn new(orders: &Arc<Orders>, lanes: Range<u32>, start: (u64, usize)) -> Self {
        assert!(orders.row_groups() > 0, "the dataset has row groups");
        Sequence {
            orders: orders.clone(),
            lanes,
            epoch: orders.epoch(start.0),
            window: start.1,
        }
    }

    pub fn next_window(&mut self) -> WindowGroups {
        if self.window >= self.epoch.windows().len() {
            self.epoch = self.orders.epoch(self.epoch.number.saturating_add(1));
            self.window = 0;
        }
        let w = &self.epoch.windows()[self.window];
        let lanes = self.epoch.order().lane_groups(w, self.lanes.clone());
        let groups = WindowGroups {
            epoch: self.epoch.number,
            window: self.window,
            groups: w.groups().to_vec(),
            lanes,
        };
        self.window += 1;
        groups
    }
}

/// The budget cost of row group `g`: its decoded size, in KiB.
pub(super) fn kib(index: &Index, g: u32) -> u64 {
    index.row_group(g).1.decoded_bytes().div_ceil(1024)
}

/// The most KiB that one window of epoch `epoch` holds for lanes `lanes`:
/// a run holds about one window at a time, so a smaller budget cannot work.
pub(super) fn window_kib(
    orders: &Arc<Orders>,
    index: &Index,
    lanes: Range<u32>,
    epoch: u64,
) -> u64 {
    let windows = orders.epoch(epoch).windows().len();
    let mut seq = Sequence::new(orders, lanes, (epoch, 0));
    (0..windows)
        .map(|_| {
            let w = seq.next_window();
            w.used().map(|i| kib(index, w.groups[i])).sum::<u64>()
        })
        .max()
        .unwrap_or(0)
}
