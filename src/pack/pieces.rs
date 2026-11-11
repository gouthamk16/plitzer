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

/// Lane `lane`'s pieces of `window`, shuffled into reading order.
pub(super) fn pieces(
    order: &Order,
    window: &Window,
    lane: u32,
    tokens: &impl Tokens,
) -> Result<Vec<Piece>, Unbuilt> {
    let parts = order.lane(window, lane);
    let mut pieces = Vec::new();
    for part in &parts {
        let rows = tokens
            .row_group(part.row_group)
            .ok_or(Unbuilt::Missing(part.row_group))?;
        of_part(part, rows, &mut pieces).ok_or(Unbuilt::Mismatch(part.row_group))?;
    }
    order.shuffle_lane(window, lane, &mut pieces);
    Ok(pieces)
}

/// Appends the pieces of the rows in `part`, in storage order. A row has a
/// slot for each token, or one slot if it has none, which gives no piece.
/// `None` if the rows have fewer slots than the part needs.
fn of_part(part: &Part, rows: Rows, out: &mut Vec<Piece>) -> Option<()> {
    let (from, to) = (part.slots.start, part.slots.end);
    let mut slot = 0u64;
    for r in rows.offsets.windows(2) {
        if slot >= to {
            return Some(());
        }
        let len = (r[1] - r[0]) as u64;
        let next = slot + len.max(1);
        if len > 0 && next > from {
            let start = r[0] + (from.max(slot) - slot) as usize;
            let end = r[0] + (to.min(next) - slot) as usize;
            out.push(Piece {
                row_group: part.row_group,
                start,
                end,
            });
        }
        slot = next;
    }
    (slot >= to).then_some(())
}
