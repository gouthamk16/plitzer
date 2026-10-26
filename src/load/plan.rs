use std::collections::HashMap;
use std::ops::Range;

use crate::index::Index;
use crate::order::Window;
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
/// Each state's `doc` is the lane's cursor: the leftover before it is
/// replayed apart, by `chunks`.
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

/// The plan and the replay chunks of lanes `lanes` at `states`. The plan
/// starts at each lane's cursor, since the leftover before it is replayed
/// first, apart from the plan. With no leftover there are no chunks, and a
/// fresh loader skips their walk over the windows.
pub(super) fn resume(
    epoch: &Epoch,
    index: &Index,
    lanes: Range<u32>,
    states: &[LaneState],
) -> Result<(Vec<Need>, Vec<Chunk>), String> {
    let cursors = cursors(epoch, lanes.clone(), states)?;
    let replayed = (states.iter().zip(&cursors)).any(|(s, &c)| s.doc < c);
    let chunks = match replayed {
        true => chunks(epoch, index, lanes.clone(), states, &cursors),
        false => Vec::new(),
    };
    Ok((
        plan(epoch, index, lanes, &at_cursors(states, &cursors)),
        chunks,
    ))
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
        .map(|(l, &s)| {
            (epoch.check(l, s)).map_err(|why| format!("lane {l} of epoch {number}: {why}"))
        })
        .collect()
}

/// The row groups of one lane's leftover documents in one window, and how
/// many documents those are.
#[derive(Debug)]
pub(super) struct Chunk {
    /// The lane's index in the GPU's lanes.
    pub lane: usize,
    pub groups: Vec<u32>,
    pub docs: u64,
    pub kib: u64,
}

/// The parts of each lane's leftover, the stream positions from its state's
/// `doc` to `cursors`, split by window, in stream order. Each chunk holds
/// only the row groups of its own documents. A fresh run holds all of a
/// lane's row groups of a window at once while it takes from them, so no
/// chunk needs more budget than that run.
pub(super) fn chunks(
    epoch: &Epoch,
    index: &Index,
    lanes: Range<u32>,
    states: &[LaneState],
    cursors: &[u64],
) -> Vec<Chunk> {
    let order = epoch.order();
    let mut starts = vec![0u64; lanes.len()];
    let mut chunks = Vec::new();
    for w in epoch.windows() {
        // Every leftover ends at its cursor; later windows hold none.
        if starts.iter().zip(cursors).all(|(s, c)| s >= c) {
            break;
        }
        for (i, lane) in lanes.clone().enumerate() {
            let start = starts[i];
            let end = start + order.lane_len(w, lane);
            starts[i] = end;
            let (lo, hi) = (start.max(states[i].doc), end.min(cursors[i]));
            if lo >= hi {
                continue;
            }
            let docs = &order.lane(w, lane)[(lo - start) as usize..(hi - start) as usize];
            let mut groups: Vec<u32> = docs.iter().map(|d| d.row_group).collect();
            groups.sort_unstable();
            groups.dedup();
            let kib = groups.iter().map(|&g| kib(index, g)).sum();
            let docs = hi - lo;
            chunks.push(Chunk {
                lane: i,
                groups,
                docs,
                kib,
            });
        }
    }
    chunks
}

/// The budget cost of row group `g`: its decoded size, in KiB.
fn kib(index: &Index, g: u32) -> u64 {
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
