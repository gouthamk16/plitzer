use std::collections::HashMap;
use std::ops::Range;

use super::Error;
use crate::index::Index;
use crate::order::Window;
use crate::pack::{Epoch, LaneState, at_lane};

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
/// Each state's `doc` is the lane's cursor: the leftover before it is
/// replayed apart, by `replay::lanes`.
///
/// A lane's slice of a window holds stream positions `[p, p + len)`, which
/// it takes at steps `p / D ..= (p + len - 1) / D`. The lane shuffles its
/// slice, so each row group of the slice counts as needed over that whole
/// range.
pub(super) fn plan(
    epoch: &Epoch,
    index: &Index,
    lanes: Range<u32>,
    states: &[LaneState],
) -> Vec<Need> {
    let mut steps: HashMap<u32, (u64, u64)> = HashMap::new();
    let mut starts = vec![0u64; lanes.len()];
    for w in epoch.windows() {
        add_window(epoch, w, lanes.clone(), states, &mut starts, &mut steps);
    }
    let mut needs: Vec<Need> = steps
        .into_iter()
        .map(|(row_group, (first, last))| Need {
            row_group,
            first,
            last,
            kib: kib(index, row_group),
        })
        .collect();
    needs.sort_unstable_by_key(|n| (n.first, n.row_group));
    needs
}

/// Widens the step range of each row group of window `w` that the lanes
/// take from. `starts` holds each lane's stream position at the window.
fn add_window(
    epoch: &Epoch,
    w: &Window,
    lanes: Range<u32>,
    states: &[LaneState],
    starts: &mut [u64],
    steps: &mut HashMap<u32, (u64, u64)>,
) {
    let order = epoch.order();
    let d = u64::from(epoch.packing().docs);
    for (i, groups) in order.lane_groups(w, lanes.clone()).into_iter().enumerate() {
        let state = states[i];
        let start = starts[i];
        let end = start + order.lane_len(w, lanes.start + i as u32);
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

/// The step of `states` and the plan from their cursors, or `Error::Budget`
/// if its peak is more than `budget`.
pub(super) fn needs(
    epoch: &Epoch,
    index: &Index,
    lanes: Range<u32>,
    states: &[LaneState],
    budget: u64,
) -> Result<(u64, Vec<Need>), Error> {
    assert_eq!(states.len(), lanes.len(), "one state for each lane");
    let step = states.first().map_or(0, |s| s.step);
    assert!(
        states.iter().all(|s| s.step == step),
        "lane states from one step"
    );
    let needs = resume(epoch, index, lanes, states).map_err(Error::State)?;
    let needed = peak(&needs);
    if needed > budget / 1024 {
        let needed = needed * 1024;
        return Err(Error::Budget { needed, budget });
    }
    Ok((step, needs))
}

/// The plan of lanes `lanes` at `states`. It starts at each lane's cursor,
/// since the leftover before it is replayed first, apart from the plan.
pub(super) fn resume(
    epoch: &Epoch,
    index: &Index,
    lanes: Range<u32>,
    states: &[LaneState],
) -> Result<Vec<Need>, String> {
    let cursors = cursors(epoch, lanes.clone(), states)?;
    Ok(plan(epoch, index, lanes, &at_cursors(states, &cursors)))
}

/// `states` with each leftover start moved to the cursor: what a lane still
/// takes after its replay.
pub(super) fn at_cursors(states: &[LaneState], cursors: &[u64]) -> Vec<LaneState> {
    (states.iter().zip(cursors))
        .map(|(s, &doc)| LaneState {
            doc,
            offset: 0,
            ..*s
        })
        .collect()
}

/// The cursor of each lane at its state, or the error of a state that no
/// run makes, which names the lane.
pub(super) fn cursors(
    epoch: &Epoch,
    lanes: Range<u32>,
    states: &[LaneState],
) -> Result<Vec<u64>, String> {
    let number = epoch.order().epoch;
    (lanes.zip(states))
        .map(|(l, &s)| (epoch.check(l, s)).map_err(|why| at_lane(l, number, &why)))
        .collect()
}

/// The budget cost of row group `g`: its decoded size, in KiB.
pub(super) fn kib(index: &Index, g: u32) -> u64 {
    index.row_group(g).1.decoded_bytes().div_ceil(1024)
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
