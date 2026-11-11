//! A lane's slice of a window, as pieces of rows in reading order.

use super::{Rows, Tokens};
use crate::order::{Order, Part, Window};

/// Tokens `start..end` of row group `row_group`'s values: one row, or the
/// part of it in the slice when the row crosses a slice edge.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Piece {
    pub row_group: u32,
    pub start: usize,
    pub end: usize,
}

impl Piece {
    pub fn tokens(&self) -> usize {
        self.end - self.start
    }
}

/// Why a slice cannot be built yet.
pub(super) enum Unbuilt {
    /// This row group is not in memory.
    Missing(u32),
    /// The row group's tokens do not match its slots in the index.
    Mismatch(u32),
}

/// Lane `lane`'s pieces of `window`, shuffled into reading order, and the
/// row groups they come from.
pub(super) fn pieces(
    order: &Order,
    window: &Window,
    lane: u32,
    tokens: &impl Tokens,
) -> Result<(Vec<Piece>, Vec<u32>), Unbuilt> {
    let parts = order.lane(window, lane);
    let mut pieces = Vec::new();
    for part in &parts {
        of_part(part, rows(part, tokens)?, |p| pieces.push(p))?;
    }
    order.shuffle_lane(window, lane, &mut pieces);
    Ok((pieces, parts.iter().map(|p| p.row_group).collect()))
}

/// The tokens of lane `lane`'s slice of `window`, with no pieces made.
pub(super) fn count(
    order: &Order,
    window: &Window,
    lane: u32,
    tokens: &impl Tokens,
) -> Result<u64, Unbuilt> {
    let mut total = 0;
    for part in &order.lane(window, lane) {
        of_part(part, rows(part, tokens)?, |p| total += p.tokens() as u64)?;
    }
    Ok(total)
}

fn rows<'a>(part: &Part, tokens: &'a impl Tokens) -> Result<Rows<'a>, Unbuilt> {
    (tokens.row_group(part.row_group)).ok_or(Unbuilt::Missing(part.row_group))
}

/// Gives the pieces of the rows in `part`, in storage order. A row has a
/// slot for each token, or one slot if it has none, which gives no piece.
/// An error if the rows have fewer slots than the part needs.
fn of_part(part: &Part, rows: Rows, mut each: impl FnMut(Piece)) -> Result<(), Unbuilt> {
    let (from, to) = (part.slots.start, part.slots.end);
    let mut slot = 0u64;
    for r in rows.offsets.windows(2) {
        if slot >= to {
            return Ok(());
        }
        let len = (r[1] - r[0]) as u64;
        let next = slot + len.max(1);
        if len > 0 && next > from {
            let start = r[0] + (from.max(slot) - slot) as usize;
            let end = r[0] + (to.min(next) - slot) as usize;
            each(Piece {
                row_group: part.row_group,
                start,
                end,
            });
        }
        slot = next;
    }
    match slot >= to {
        true => Ok(()),
        false => Err(Unbuilt::Mismatch(part.row_group)),
    }
}
