use std::collections::HashMap;
use std::ops::Range;

use crate::index::Index;
use crate::pack::{Epoch, LaneState};

/// One row group a GPU needs, and the steps it is needed over.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Need {
    pub row_group: u32,
    pub first: u64,
    pub last: u64,
    /// The budget cost: the decoded size, in KiB.
    pub kib: u64,
}

/// Every row group that lanes `lanes` take a document from, from the steps
/// in `states` to the end of the epoch, in the order the loader needs them.
///
/// A lane's slice of a window holds stream positions `[p, p + len)`, which
/// it takes at steps `p / D ..= (p + len - 1) / D`. The lane shuffles its
/// slice, so each row group of the slice counts as needed over that whole
/// range. On resume, the positions before the cursor are replayed into the
/// leftover at the resume step, so they count as needed then.
pub(super) fn plan(
    epoch: &Epoch,
    index: &Index,
    lanes: Range<u32>,
    states: &[LaneState],
) -> Vec<Need> {
    let order = epoch.order();
    let d = u64::from(epoch.packing().docs);
    let mut steps: HashMap<u32, (u64, u64)> = HashMap::new();
    let mut starts = vec![0u64; lanes.len()];
    for w in epoch.windows() {
        for (i, groups) in order.lane_groups(w, lanes.clone()).into_iter().enumerate() {
            let lane = lanes.start + i as u32;
            let state = states[i];
            let start = starts[i];
            let end = start + order.lane_len(w, lane);
            starts[i] = end;
            let from = start.max(state.doc);
            if from >= end {
                continue;
            }
            let first = (from / d).max(state.step);
            let last = ((end - 1) / d).max(state.step);
            for &g in &w.groups()[groups] {
                let range = steps.entry(g).or_insert((first, last));
                *range = (range.0.min(first), range.1.max(last));
            }
        }
    }
    let mut needs: Vec<Need> = steps
        .into_iter()
        .map(|(row_group, (first, last))| Need {
            row_group,
            first,
            last,
            kib: index.row_group(row_group).1.decoded_bytes().div_ceil(1024),
        })
        .collect();
    needs.sort_unstable_by_key(|n| (n.first, n.row_group));
    needs
}

/// The most KiB live at one step: row groups needed from `first` to `last`.
/// A budget below this deadlocks, since the loader would need to hold more
/// than the budget allows at that step.
pub(super) fn peak(needs: &[Need]) -> u64 {
    // A row group is live from `first` through `last`. Removals at `last + 1`
    // sort before additions at the same step (`false < true`), so a row group
    // that ends never counts together with one that starts after it.
    let mut events: Vec<(u64, bool, u64)> = needs
        .iter()
        .flat_map(|n| [(n.first, true, n.kib), (n.last + 1, false, n.kib)])
        .collect();
    events.sort_unstable();
    let (mut live, mut most) = (0u64, 0u64);
    for (_, add, kib) in events {
        if add {
            live += kib;
            most = most.max(live);
        } else {
            live -= kib;
        }
    }
    most
}
