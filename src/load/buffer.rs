//! The row groups in memory, and the lanes that still take from each.

use std::collections::{HashMap, HashSet};

use super::prefetch::{Arrival, Permit};
use super::sequence::WindowGroups;
use crate::pack::{Lane, Orders, Rows, Tokens};
use crate::read::RowGroupTokens;

/// A lane that still takes from a row group, in one window: `(lane index,
/// epoch, window)`. A row group can be in a window of each of two epochs
/// at once, at an epoch's end, for different lanes.
type User = (usize, u64, usize);

/// A row group in memory, the lanes that still take from it, and its share
/// of the budget, which it gives back when it is evicted.
struct Held {
    tokens: RowGroupTokens,
    users: HashSet<User>,
    _permit: Permit,
}

#[derive(Default)]
pub(super) struct Buffer {
    held: HashMap<u32, Held>,
    /// The window of the last arrival, for its users.
    window: Option<WindowGroups>,
}

impl Tokens for Buffer {
    fn row_group(&self, row_group: u32) -> Option<Rows<'_>> {
        Some(self.held.get(&row_group)?.tokens.view())
    }
}

impl Buffer {
    pub fn bytes(&self) -> usize {
        self.held.values().map(|h| h.tokens.bytes()).sum()
    }

    #[cfg(test)]
    pub fn row_groups(&self) -> impl Iterator<Item = &u32> {
        self.held.keys()
    }

    /// Keeps an arrival for the lanes that will still take from it: those
    /// whose slice in its window holds it, and that have not passed the
    /// window. A row group needed in two windows at once, at an epoch's end,
    /// is kept once for the users of both.
    pub fn hold(
        &mut self,
        orders: &Orders,
        lanes: &[Lane],
        arrival: &Arrival,
        tokens: RowGroupTokens,
        permit: Permit,
    ) {
        let (epoch, window, i) = arrival.key;
        if (self.window.as_ref()).is_none_or(|w| (w.epoch, w.window) != (epoch, window)) {
            let e = orders.epoch(epoch);
            let first = lanes[0].lane();
            let w = &e.windows()[window];
            self.window = Some(WindowGroups {
                epoch,
                window,
                groups: w.groups().to_vec(),
                lanes: e.order().lane_groups(w, first..first + lanes.len() as u32),
            });
        }
        let w = self.window.as_ref().expect("set above");
        let users: HashSet<User> = (w.users(i))
            .filter(|&l| lanes[l].slice() <= (epoch, window))
            .map(|l| (l, epoch, window))
            .collect();
        if users.is_empty() {
            return;
        }
        match self.held.get_mut(&arrival.row_group) {
            Some(held) => held.users.extend(users),
            None => {
                let held = Held {
                    tokens,
                    users,
                    _permit: permit,
                };
                self.held.insert(arrival.row_group, held);
            }
        }
    }

    /// `lane`, the GPU's `i`-th, moved on from slice `before`: it no longer
    /// takes from the row groups of the slices it passed, and those that no
    /// lane takes from are evicted.
    pub fn release(&mut self, orders: &Orders, lane: &Lane, i: usize, before: (u64, usize)) {
        let (l, now) = (lane.lane(), lane.slice());
        let mut slice = before;
        while slice < now {
            let e = orders.epoch(slice.0);
            let Some(w) = e.windows().get(slice.1) else {
                slice = (slice.0 + 1, 0);
                continue;
            };
            let range = e.order().lane_groups(w, l..l + 1).remove(0);
            // A lane can pass a window through another window's copy of a
            // row group, before this window's read arrives: then it holds
            // no place in it.
            for g in &w.groups()[range] {
                if let Some(held) = self.held.get_mut(g) {
                    held.users.remove(&(i, slice.0, slice.1));
                    if held.users.is_empty() {
                        self.held.remove(g);
                    }
                }
            }
            slice = (slice.0, slice.1 + 1);
        }
    }
}
