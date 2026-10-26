//! A lane rebuilt from its state in parts, so a caller can supply the tokens
//! of each part and free them before the next.

use std::sync::Arc;

use super::{Doc, Epoch, Lane, LaneState, Leftover, Stream, Tokens};

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
    /// The documents of the leftover still to take that are in one window:
    /// the next part to give `take`. Empty once the leftover is taken.
    pub fn part(&mut self) -> &[Doc] {
        let lane = &mut self.lane;
        let rest = lane.stream.rest(&lane.epoch, lane.lane);
        &rest[..rest.len().min(self.left as usize)]
    }

    /// Takes the next `n` documents of the leftover, or as many as are left.
    /// The state's offset lies in the first document, so once the tokens
    /// after it are more than the cap, the state can never fit: that is an
    /// error at once, before a state far behind its cursor fills memory.
    pub fn take(&mut self, tokens: &impl Tokens, n: u64) -> Result<(), String> {
        let cap = self.lane.epoch.packing.cap as usize;
        for _ in 0..n.min(self.left) {
            self.lane.take(tokens, 1);
            self.left -= 1;
            let leftover = &self.lane.leftover;
            // An empty first document is skipped at once: the state cannot
            // start there, and `finish` would say so after the whole replay.
            if leftover.doc != self.state.doc {
                let why = "does not start on a document with tokens left";
                return Err(format!("state {:?} {why}", self.state));
            }
            let first = leftover.lens.front().copied().unwrap_or(0) as usize;
            if leftover.tokens.len() - first > cap {
                return Err(format!("state {:?} leaves more than the cap", self.state));
            }
        }
        Ok(())
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
