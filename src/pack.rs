use std::collections::VecDeque;

use crate::order::{Doc, Order, Window};

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

/// Where a lane's leftover starts: token `offset` of the lane's `doc`-th
/// document of the epoch. The leftover always runs from here to the lane's
/// cursor, so this and the step are all a resume needs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LaneState {
    pub doc: u64,
    pub offset: u32,
}

/// One epoch of packing. Lanes borrow its windows, so they are computed once.
pub struct Epoch<'a> {
    order: Order<'a>,
    packing: Packing,
    windows: Vec<Window>,
    steps: u64,
}

impl<'a> Epoch<'a> {
    pub fn new(order: Order<'a>, packing: Packing) -> Self {
        assert!(
            packing.seq_len > 0 && packing.docs > 0,
            "seq_len and docs must be positive"
        );
        let windows = order.windows();
        let mut epoch = Epoch {
            order,
            packing,
            windows,
            steps: 0,
        };
        let longest = (0..epoch.order.lanes).map(|l| epoch.lane_docs(l)).max();
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

    /// Steps in the epoch. Every lane supplies one sequence per step.
    pub fn steps(&self) -> u64 {
        self.steps
    }

    /// Lane `lane` before step `step`, with `state` from `Lane::state` at that
    /// step. Step 0 with the default state starts the epoch.
    pub fn lane(&self, lane: u32, step: u64, state: LaneState, tokens: &impl Tokens) -> Lane<'_> {
        assert!(
            step <= self.steps,
            "step {step} is past the epoch's {} steps",
            self.steps
        );
        let cursor = self.cursor(lane, step);
        assert!(
            state.doc <= cursor,
            "state {state:?} is past the cursor {cursor}"
        );
        let mut lane = Lane {
            epoch: self,
            lane,
            step,
            stream: Stream::seek(self, lane, state.doc),
            leftover: Leftover::new(state.doc),
        };
        for _ in state.doc..cursor {
            let doc = lane
                .stream
                .next(self, lane.lane)
                .expect("cursor within the lane");
            lane.leftover.push(tokens.tokens(doc));
        }
        let first = lane.leftover.lens.front().copied().unwrap_or(0);
        assert!(
            state.offset < first || (state.doc == cursor && state.offset == 0),
            "state {state:?} offset is past its document"
        );
        lane.leftover.advance(state.offset as usize);
        assert_eq!(
            lane.state(),
            state,
            "state does not start on a document with tokens left"
        );
        assert!(
            lane.leftover.tokens.len() <= self.packing.cap as usize,
            "state {state:?} leaves more than the cap"
        );
        lane
    }

    /// The documents lane `lane` takes in the whole epoch.
    fn lane_docs(&self, lane: u32) -> u64 {
        self.windows
            .iter()
            .map(|w| self.order.lane_len(w, lane))
            .sum()
    }

    /// The documents lane `lane` has taken before step `step`.
    fn cursor(&self, lane: u32, step: u64) -> u64 {
        assert!(
            lane < self.order.lanes,
            "lane {lane} out of range for {} lanes",
            self.order.lanes
        );
        step.saturating_mul(self.packing.docs.into())
            .min(self.lane_docs(lane))
    }
}

/// One lane's packer for one epoch.
pub struct Lane<'e> {
    epoch: &'e Epoch<'e>,
    lane: u32,
    step: u64,
    stream: Stream,
    leftover: Leftover,
}

impl Lane<'_> {
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
        for _ in 0..p.docs {
            let Some(doc) = self.stream.next(self.epoch, self.lane) else {
                break;
            };
            self.leftover.push(tokens.tokens(doc));
        }
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
        self.leftover.start
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
            let len = epoch.order.lane_len(w, lane);
            if pos < len {
                let docs = epoch.order.lane(w, lane);
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
            self.docs = epoch.order.lane(&epoch.windows[self.window], lane);
            self.next = 0;
        }
        self.next += 1;
        Some(self.docs[self.next - 1])
    }
}

/// The tokens a lane has taken but not yet emitted or dropped.
struct Leftover {
    tokens: VecDeque<u32>,
    /// Remaining tokens of each document from `start.doc` to the cursor.
    lens: VecDeque<u32>,
    start: LaneState,
}

impl Leftover {
    fn new(doc: u64) -> Self {
        Leftover {
            tokens: VecDeque::new(),
            lens: VecDeque::new(),
            start: LaneState { doc, offset: 0 },
        }
    }

    fn push(&mut self, doc: &[u32]) {
        self.tokens.extend(doc);
        self.lens
            .push_back(u32::try_from(doc.len()).expect("document longer than u32::MAX tokens"));
        self.skip_empty();
    }

    /// Removes the oldest `n` tokens.
    fn advance(&mut self, mut n: usize) {
        self.tokens.drain(..n);
        while n > 0 {
            let front = &mut self.lens[0];
            let k = n.min(*front as usize);
            *front -= k as u32;
            self.start.offset += k as u32;
            n -= k;
            self.skip_empty();
        }
    }

    /// Keeps `start` on a document with tokens left, or on the cursor when
    /// the leftover is empty.
    fn skip_empty(&mut self) {
        while self.lens.front() == Some(&0) {
            self.lens.pop_front();
            self.start = LaneState {
                doc: self.start.doc + 1,
                offset: 0,
            };
        }
    }
}

#[cfg(test)]
mod tests;
