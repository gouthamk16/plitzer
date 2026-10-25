use std::sync::Arc;

use crate::order::{Doc, Order, Window};

mod leftover;

use leftover::Leftover;

/// Where a lane gets each document's tokens. The packer adds no separators,
/// so a source that wants EOS between documents includes it in the tokens.
pub trait Tokens {
    fn tokens(&self, doc: Doc) -> &[u32];
}

/// How lanes pack documents into sequences.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Packing {
    /// L: tokens per sequence.
    pub seq_len: u32,
    /// D: documents a lane takes each step.
    pub docs: u32,
    /// C: the most leftover tokens a lane carries into the next step.
    pub cap: u32,
    /// The token that fills the end of a short sequence.
    pub pad: u32,
}

/// A lane's checkpoint before step `step`: its leftover starts at token
/// `offset` of the lane's `doc`-th document of the epoch and runs to the
/// cursor, so nothing else is needed to resume. The default starts the epoch.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LaneState {
    pub step: u64,
    pub doc: u64,
    pub offset: u32,
}

/// One epoch of packing. It owns a copy of the order's inputs, and lanes
/// share it through an `Arc`, so the windows are computed once and a loader
/// can own an epoch together with its lanes.
pub struct Epoch {
    seed: u64,
    number: u64,
    lanes: u32,
    window: u32,
    rows: Vec<u32>,
    packing: Packing,
    windows: Vec<Window>,
    steps: u64,
}

impl Epoch {
    pub fn new(order: Order<'_>, packing: Packing) -> Self {
        assert!(
            packing.seq_len > 0 && packing.docs > 0,
            "seq_len and docs must be positive"
        );
        let windows = order.windows();
        let mut epoch = Epoch {
            seed: order.seed,
            number: order.epoch,
            lanes: order.lanes,
            window: order.window,
            rows: order.rows.to_vec(),
            packing,
            windows,
            steps: 0,
        };
        let longest = (0..epoch.lanes).map(|l| epoch.lane_docs(l)).max();
        // After the longest lane takes its last documents, its leftover holds
        // at most C tokens, so ceil(C/L) more steps empty every lane.
        epoch.steps = match longest {
            None | Some(0) => 0,
            Some(n) => {
                n.div_ceil(packing.docs.into()) + u64::from(packing.cap.div_ceil(packing.seq_len))
            }
        };
        epoch
    }

    /// The order of this epoch, as a view of the copied inputs.
    pub fn order(&self) -> Order<'_> {
        Order {
            seed: self.seed,
            epoch: self.number,
            lanes: self.lanes,
            window: self.window,
            rows: &self.rows,
        }
    }

    pub fn windows(&self) -> &[Window] {
        &self.windows
    }

    pub fn packing(&self) -> Packing {
        self.packing
    }

    /// Steps in the epoch. Every lane supplies one sequence per step.
    pub fn steps(&self) -> u64 {
        self.steps
    }

    /// Lane `lane`, resumed from `state`.
    /// Lane `lane` at `state`. A state that no run makes, such as one from a
    /// checkpoint made by hand, is an error that names what is wrong.
    pub fn lane(
        self: &Arc<Self>,
        lane: u32,
        state: LaneState,
        tokens: &impl Tokens,
    ) -> Result<Lane, String> {
        let cursor = self.check(lane, state)?;
        let mut l = Lane {
            epoch: self.clone(),
            lane,
            step: state.step,
            stream: Stream::seek(self, lane, state.doc),
            leftover: Leftover::new(state.doc),
        };
        l.take(tokens, cursor - state.doc);
        l.start_at(state)?;
        Ok(l)
    }

    /// The documents lane `lane` takes in the whole epoch.
    pub fn lane_docs(&self, lane: u32) -> u64 {
        self.windows
            .iter()
            .map(|w| self.order().lane_len(w, lane))
            .sum()
    }

    /// Whether `state` can stand in lane `lane` of this epoch: no later than
    /// its last step, with its leftover not past the documents it has taken.
    /// Gives the lane's cursor at that step. An offset past its document is
    /// found only with the tokens, by `lane`.
    pub fn check(&self, lane: u32, state: LaneState) -> Result<u64, String> {
        if state.step > self.steps {
            return Err(format!(
                "state {state:?} is past the epoch's {} steps",
                self.steps
            ));
        }
        let cursor = self.cursor(lane, state.step);
        if state.doc > cursor {
            return Err(format!("state {state:?} is past the cursor {cursor}"));
        }
        Ok(cursor)
    }

    /// The documents lane `lane` has taken before step `step`.
    fn cursor(&self, lane: u32, step: u64) -> u64 {
        assert!(
            lane < self.lanes,
            "lane {lane} out of range for {} lanes",
            self.lanes
        );
        step.saturating_mul(self.packing.docs.into())
            .min(self.lane_docs(lane))
    }
}

/// One lane's packer for one epoch.
pub struct Lane {
    epoch: Arc<Epoch>,
    lane: u32,
    step: u64,
    stream: Stream,
    leftover: Leftover,
}

impl Lane {
    /// Writes the next sequence into `out`, pads its end, and returns the
    /// number of real tokens.
    pub fn fill(&mut self, tokens: &impl Tokens, out: &mut [u32]) -> u32 {
        let p = self.epoch.packing;
        assert_eq!(
            out.len(),
            p.seq_len as usize,
            "out must hold seq_len tokens"
        );
        assert!(self.step < self.epoch.steps, "the epoch has ended");
        self.take(tokens, p.docs.into());
        let real = self.leftover.tokens.len().min(out.len());
        for (o, &t) in out.iter_mut().zip(&self.leftover.tokens) {
            *o = t;
        }
        out[real..].fill(p.pad);
        self.leftover.advance(real);
        // Drop the oldest tokens, not the newest, so the leftover stays one
        // run that ends at the cursor and `LaneState` can describe it.
        let over = self.leftover.tokens.len().saturating_sub(p.cap as usize);
        self.leftover.advance(over);
        self.step += 1;
        real as u32
    }

    pub fn state(&self) -> LaneState {
        LaneState {
            step: self.step,
            doc: self.leftover.doc,
            offset: self.leftover.offset,
        }
    }

    /// Skips the replayed leftover's first `state.offset` tokens and checks
    /// that the lane now stands exactly at `state`.
    fn start_at(&mut self, state: LaneState) -> Result<(), String> {
        let first = self.leftover.lens.front().copied().unwrap_or(0);
        if state.offset != 0 && state.offset >= first {
            return Err(format!("state {state:?} offset is past its document"));
        }
        self.leftover.advance(state.offset as usize);
        if self.state() != state {
            return Err(format!(
                "state {state:?} does not start on a document with tokens left"
            ));
        }
        if self.leftover.tokens.len() > self.epoch.packing.cap as usize {
            return Err(format!("state {state:?} leaves more than the cap"));
        }
        Ok(())
    }

    /// Adds the next `n` documents, or as many as are left, to the leftover.
    fn take(&mut self, tokens: &impl Tokens, n: u64) {
        for _ in 0..n {
            let Some(doc) = self.stream.next(&self.epoch, self.lane) else {
                break;
            };
            self.leftover.push(tokens.tokens(doc));
        }
    }
}

/// A lane's documents for the epoch, one window's slice at a time.
struct Stream {
    window: usize,
    docs: Vec<Doc>,
    next: usize,
}

impl Stream {
    /// The stream positioned at the lane's `pos`-th document.
    fn seek(epoch: &Epoch, lane: u32, mut pos: u64) -> Self {
        let mut window = 0;
        while let Some(w) = epoch.windows.get(window) {
            let len = epoch.order().lane_len(w, lane);
            if pos < len {
                let docs = epoch.order().lane(w, lane);
                return Stream {
                    window,
                    docs,
                    next: pos as usize,
                };
            }
            pos -= len;
            window += 1;
        }
        Stream {
            window,
            docs: Vec::new(),
            next: 0,
        }
    }

    fn next(&mut self, epoch: &Epoch, lane: u32) -> Option<Doc> {
        while self.next == self.docs.len() {
            if self.window + 1 >= epoch.windows.len() {
                return None;
            }
            self.window += 1;
            self.docs = epoch.order().lane(&epoch.windows[self.window], lane);
            self.next = 0;
        }
        self.next += 1;
        Some(self.docs[self.next - 1])
    }
}

#[cfg(test)]
mod tests;
