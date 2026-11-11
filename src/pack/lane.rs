//! One lane's packer, which walks its slices window by window.

use std::sync::Arc;

use super::pieces::{self, Piece, Unbuilt};
use super::{Epoch, Need, Orders, Position, Tokens, at_lane};
use crate::order::Window;

const NO_TOKENS: &str = "finds no token in two whole epochs: the dataset has too few tokens \
     for this many lanes";

/// The window a lane stands in. It never stands on a window whose slice has
/// no slots: it moves on to the next one that has some.
#[derive(Clone)]
struct At {
    epoch: Arc<Epoch>,
    window: usize,
    /// Set when two whole epochs gave the lane no slot: it never gets one.
    empty: bool,
}

impl At {
    fn window(&self) -> &Window {
        &self.epoch.windows()[self.window]
    }

    fn advance(&mut self, orders: &Orders, lane: u32) {
        self.window += 1;
        self.settle(orders, lane);
    }

    fn settle(&mut self, orders: &Orders, lane: u32) {
        let from = self.epoch.number;
        loop {
            let windows = self.epoch.windows();
            match windows.get(self.window) {
                Some(w) if self.epoch.order().lane_len(w, lane) > 0 => return,
                Some(_) => self.window += 1,
                None => {
                    let n = self.epoch.number;
                    if n >= from.saturating_add(2) || n == u64::MAX {
                        self.empty = true;
                        return;
                    }
                    (self.epoch, self.window) = (orders.epoch(n + 1), 0);
                }
            }
        }
    }
}

/// The current slice's pieces, and where the lane stands in them.
struct Built {
    pieces: Vec<Piece>,
    /// The row groups of the pieces, and their tokens.
    groups: Vec<u32>,
    tokens: u64,
    next: usize,
    offset: usize,
}

impl Built {
    fn missing(&self, tokens: &impl Tokens) -> Option<u32> {
        self.groups
            .iter()
            .copied()
            .find(|&g| tokens.row_group(g).is_none())
    }
}

/// One lane's packer.
pub struct Lane {
    orders: Arc<Orders>,
    lane: u32,
    at: At,
    /// The tokens taken from the current slice.
    token: u64,
    built: Option<Built>,
}

impl Lane {
    /// Lane `lane` at `at`. A position past the lane's slots is an error; one
    /// at the start of a window whose slice has none stands at the next.
    pub fn new(orders: &Arc<Orders>, lane: u32, at: Position) -> Result<Lane, String> {
        if lane >= orders.lanes {
            let why = format!("lane {lane} out of range for {} lanes", orders.lanes);
            return Err(why);
        }
        let epoch = orders.epoch(at.epoch);
        let (window, windows) = (at.window as usize, epoch.windows().len());
        let slots = match epoch.windows().get(window) {
            Some(w) => epoch.order().lane_len(w, lane),
            None => 0,
        };
        if window > windows || at.token > slots {
            return Err(at_lane(lane, &format!("{at:?} is past the lane's slots")));
        }
        let mut lane = Lane {
            orders: orders.clone(),
            lane,
            at: At {
                epoch,
                window,
                empty: false,
            },
            token: at.token,
            built: None,
        };
        if at.token == 0 {
            lane.at.settle(orders, lane.lane);
        }
        Ok(lane)
    }

    pub fn lane(&self) -> u32 {
        self.lane
    }

    pub fn position(&self) -> Position {
        Position {
            epoch: self.at.epoch.number,
            window: self.at.window as u32,
            token: self.token,
        }
    }

    /// The window the lane stands in: `(epoch, window)`.
    pub fn slice(&self) -> (u64, usize) {
        (self.at.epoch.number, self.at.window)
    }

    /// What the next sequence needs, without moving the lane.
    pub fn need(&self, tokens: &impl Tokens) -> Need {
        let (mut at, mut token) = (self.at.clone(), self.token);
        let mut want = u64::from(self.orders.seq_len);
        // Epochs entered with no token since: at 3, two whole epochs gave
        // none, and the lane would look for ever.
        let (mut epoch, mut dry, mut first) = (at.epoch.number, 0, true);
        loop {
            if at.epoch.number != epoch {
                (epoch, dry) = (at.epoch.number, dry + 1);
            }
            if at.empty || dry == 3 {
                return Need::Empty(at_lane(self.lane, NO_TOKENS));
            }
            // A row group can be in memory for another epoch's window and
            // leave with it, so the cached slice is checked too.
            let cached = self.built.as_ref().filter(|_| first);
            first = false;
            let total = match cached {
                Some(b) => match b.missing(tokens) {
                    None => b.tokens,
                    Some(row_group) => {
                        return Need::Missing {
                            slice: self.slice(),
                            row_group,
                        };
                    }
                },
                None => match self.count(&at, tokens) {
                    Ok(total) => total,
                    Err(need) => return need,
                },
            };
            if token > total {
                let why = format!("{:?} is past its slice's tokens", self.position());
                return Need::Bad(at_lane(self.lane, &why));
            }
            let rest = total - token;
            if rest >= want {
                return Need::Ready;
            }
            if rest > 0 {
                dry = 0;
            }
            (want, token) = (want - rest, 0);
            at.advance(&self.orders, self.lane);
        }
    }

    /// Writes the next sequence into `out`, of exactly L tokens. Every row
    /// group it takes from must be in `tokens`: call `need` first.
    pub fn fill(&mut self, tokens: &impl Tokens, out: &mut [u32]) {
        assert_eq!(
            out.len(),
            self.orders.seq_len as usize,
            "out must hold L tokens"
        );
        let mut filled = 0;
        while filled < out.len() {
            let b = self.built(tokens);
            let Some(&p) = b.pieces.get(b.next) else {
                // A slice whose rows are all empty.
                self.move_on();
                continue;
            };
            let rows = (tokens.row_group(p.row_group)).expect("`need` found the tokens in memory");
            let rest = &rows.values[p.start + b.offset..p.end];
            let n = rest.len().min(out.len() - filled);
            out[filled..filled + n].copy_from_slice(&rest[..n]);
            filled += n;
            if n == rest.len() {
                (b.next, b.offset) = (b.next + 1, 0);
            } else {
                b.offset += n;
            }
            self.token += n as u64;
            self.move_on();
        }
    }

    /// The tokens of the slice at `at`, or what stops their count.
    fn count(&self, at: &At, tokens: &impl Tokens) -> Result<u64, Need> {
        let (order, w) = (at.epoch.order(), at.window());
        let slice = (at.epoch.number, at.window);
        match pieces::pieces(&order, w, self.lane, tokens) {
            Ok(p) => Ok(p.iter().map(Piece::tokens).sum::<usize>() as u64),
            Err(Unbuilt::Missing(row_group)) => Err(Need::Missing { slice, row_group }),
            Err(Unbuilt::Mismatch(g)) => {
                let why = format!("row group {g} has fewer tokens than its slots in the index");
                Err(Need::Bad(at_lane(self.lane, &why)))
            }
        }
    }

    /// The current slice's pieces, built at the first use, with the lane
    /// standing after `token` of their tokens.
    fn built(&mut self, tokens: &impl Tokens) -> &mut Built {
        if self.built.is_none() {
            let order = self.at.epoch.order();
            let pieces = pieces::pieces(&order, self.at.window(), self.lane, tokens)
                .unwrap_or_else(|_| panic!("`need` found the slice in memory"));
            let (mut next, mut left) = (0, self.token as usize);
            while left > 0 && left >= pieces[next].tokens() {
                (next, left) = (next + 1, left - pieces[next].tokens());
            }
            let mut groups: Vec<u32> = pieces.iter().map(|p| p.row_group).collect();
            groups.sort_unstable();
            groups.dedup();
            let total = pieces.iter().map(Piece::tokens).sum::<usize>() as u64;
            self.built = Some(Built {
                pieces,
                groups,
                tokens: total,
                next,
                offset: left,
            });
        }
        self.built.as_mut().expect("set above")
    }

    /// Moves to the next slice when this one has no tokens left.
    fn move_on(&mut self) {
        if self
            .built
            .as_ref()
            .is_some_and(|b| b.next == b.pieces.len())
        {
            (self.built, self.token) = (None, 0);
            self.at.advance(&self.orders, self.lane);
        }
    }
}
