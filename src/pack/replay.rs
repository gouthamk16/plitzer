//! A lane rebuilt from its state in parts, so a caller can supply the tokens
//! of each part and free them before the next.

use std::sync::Arc;

use super::{Epoch, Lane, LaneState, Leftover, Stream, Tokens};

/// A lane whose leftover is not yet replayed: the documents from the state's
/// `doc` to the cursor, taken in order by calls of `take`.
pub struct Replay {
    lane: Lane,
    state: LaneState,
    /// Documents not yet taken before the cursor.
    left: u64,
}

impl Epoch {
    /// Lane `lane` at `state`, before its leftover is replayed. A state that
    /// no run makes is an error that names what is wrong.
    pub fn replay(self: &Arc<Self>, lane: u32, state: LaneState) -> Result<Replay, String> {
        let cursor = self.check(lane, state)?;
        let lane = Lane {
            epoch: self.clone(),
            lane,
            step: state.step,
            stream: Stream::seek(self, lane, state.doc),
            leftover: Leftover::new(state.doc),
        };
        let left = cursor - state.doc;
        Ok(Replay { lane, state, left })
    }
}

impl Replay {
    /// Takes the next `n` documents of the leftover, or as many as are left.
    pub fn take(&mut self, tokens: &impl Tokens, n: u64) {
        let n = n.min(self.left);
        self.lane.take(tokens, n);
        self.left -= n;
    }

    /// The lane, once every document of the leftover is taken.
    pub fn finish(mut self) -> Result<Lane, String> {
        if self.left > 0 {
            return Err(format!(
                "{} documents of the leftover are not replayed",
                self.left
            ));
        }
        self.lane.start_at(self.state)?;
        Ok(self.lane)
    }
}
