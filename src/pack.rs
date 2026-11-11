//! Lanes: each lane packs its stream of documents into sequences of exactly
//! L tokens, with nothing dropped and no padding. A lane's stream is its
//! slice of each window of epoch 0, then of epoch 1, and so on, so a lane
//! that ends an epoch goes on into the next.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, Weak};

use crate::order::{Doc, Order, Window};

mod stream;
#[cfg(test)]
mod tests;

pub use stream::Stream;

/// A lane state error, named by its lane: the session and the loader both
/// give it, and Python users see it in `CheckpointError`.
pub(crate) fn at_lane(lane: u32, why: &str) -> String {
    format!("lane {lane}: {why}")
}

/// Where a lane gets each document's tokens: `None` when they are not in
/// memory. The packer adds no separators, so a source that wants EOS between
/// documents includes it in the tokens.
pub trait Tokens {
    fn tokens(&self, doc: Doc) -> Option<&[u32]>;
}

/// Where a lane stands: before token `offset` of the `doc`-th document of its
/// stream in epoch `epoch`. The default is the start of epoch 0.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Position {
    pub epoch: u64,
    pub doc: u64,
    pub offset: u32,
}

/// The inputs of the order of every epoch, and each epoch's windows, made
/// once and shared by the lanes that are in it.
pub struct Orders {
    pub seed: u64,
    /// B: the lanes of the global batch.
    pub lanes: u32,
    /// W: row groups per window.
    pub window: u32,
    /// L: tokens per sequence.
    pub seq_len: u32,
    rows: Arc<[u32]>,
    epochs: Mutex<HashMap<u64, Weak<Epoch>>>,
}

impl Orders {
    pub fn new(seed: u64, lanes: u32, window: u32, seq_len: u32, rows: &[u32]) -> Arc<Self> {
        assert!(
            lanes > 0 && window > 0 && seq_len > 0,
            "B, W and L must be positive"
        );
        Arc::new(Orders {
            seed,
            lanes,
            window,
            seq_len,
            rows: rows.into(),
            epochs: Mutex::new(HashMap::new()),
        })
    }

    /// Epoch `number`, made when no lane holds it.
    pub fn epoch(&self, number: u64) -> Arc<Epoch> {
        let mut epochs = self.epochs.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(epoch) = epochs.get(&number).and_then(Weak::upgrade) {
            return epoch;
        }
        let epoch = Arc::new(Epoch::new(self, number));
        epochs.retain(|_, e| e.strong_count() > 0);
        epochs.insert(number, Arc::downgrade(&epoch));
        epoch
    }

    pub fn rows(&self) -> &[u32] {
        &self.rows
    }
}

/// One epoch's windows, and the order that made them.
pub struct Epoch {
    pub number: u64,
    seed: u64,
    lanes: u32,
    window: u32,
    rows: Arc<[u32]>,
    windows: Vec<Window>,
}

impl Epoch {
    fn new(orders: &Orders, number: u64) -> Self {
        let mut epoch = Epoch {
            number,
            seed: orders.seed,
            lanes: orders.lanes,
            window: orders.window,
            rows: orders.rows.clone(),
            windows: Vec::new(),
        };
        epoch.windows = epoch.order().windows();
        epoch
    }

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

    /// The documents of lane `lane` in the whole epoch.
    pub fn lane_docs(&self, lane: u32) -> u64 {
        let order = self.order();
        self.windows.iter().map(|w| order.lane_len(w, lane)).sum()
    }
}

/// What the next sequence of a lane needs.
#[derive(Debug, PartialEq, Eq)]
pub enum Need {
    /// Every document it takes tokens from is in memory.
    Ready,
    /// The first document it needs that is not in memory, and the
    /// `(epoch, window)` it is in.
    Missing { slice: (u64, usize), doc: Doc },
    /// A position that no run makes, such as from a hand-made checkpoint.
    Bad(String),
    /// A lane that finds no token in two whole epochs: it would never fill
    /// a sequence.
    Empty(String),
}

const NO_TOKENS: &str = "finds no token in two whole epochs: the dataset has too few documents \
     or tokens for this many lanes";

/// One lane's packer.
pub struct Lane {
    lane: u32,
    stream: Stream,
    offset: u32,
}

impl Lane {
    /// Lane `lane` at `at`. A position past the lane's documents of its
    /// epoch is an error; one at their end stands at the next epoch.
    pub fn new(orders: &Arc<Orders>, lane: u32, at: Position) -> Result<Lane, String> {
        if lane >= orders.lanes {
            return Err(format!(
                "lane {lane} out of range for {} lanes",
                orders.lanes
            ));
        }
        let named = |why: String| at_lane(lane, &why);
        let stream = Stream::seek(orders, lane, at.epoch, at.doc).map_err(named)?;
        if at.offset > 0 && stream.position().epoch != at.epoch {
            return Err(named(format!("{at:?} is past the lane's documents")));
        }
        Ok(Lane {
            lane,
            stream,
            offset: at.offset,
        })
    }

    pub fn lane(&self) -> u32 {
        self.lane
    }

    pub fn position(&self) -> Position {
        Position {
            offset: self.offset,
            ..self.stream.position()
        }
    }

    /// The window the lane stands in: `(epoch, window)`.
    pub fn slice(&self) -> (u64, usize) {
        self.stream.slice()
    }

    /// What the next sequence needs, without moving the lane.
    pub fn need(&self, tokens: &impl Tokens) -> Need {
        let mut stream = self.stream.clone();
        let (mut offset, mut need) = (self.offset as usize, self.stream.seq_len());
        // Epochs entered with no token since: at 3, two whole epochs gave
        // none, and the lane would look for ever.
        let (mut epoch, mut dry) = (stream.position().epoch, 0);
        loop {
            let Some(doc) = stream.current() else {
                return Need::Empty(at_lane(self.lane, NO_TOKENS));
            };
            if stream.position().epoch != epoch {
                (epoch, dry) = (stream.position().epoch, dry + 1);
                if dry == 3 {
                    return Need::Empty(at_lane(self.lane, NO_TOKENS));
                }
            }
            let Some(t) = tokens.tokens(doc) else {
                let slice = stream.slice();
                return Need::Missing { slice, doc };
            };
            if offset > t.len() {
                let why = format!("{:?} is past its document", self.position());
                return Need::Bad(at_lane(self.lane, &why));
            }
            let rest = t.len() - offset;
            if rest >= need {
                return Need::Ready;
            }
            if rest > 0 {
                dry = 0;
            }
            (need, offset) = (need - rest, 0);
            stream.advance();
        }
    }

    /// Writes the next sequence into `out`, of exactly L tokens. Every
    /// document it takes must be in `tokens`: call `need` first.
    pub fn fill(&mut self, tokens: &impl Tokens, out: &mut [u32]) {
        assert_eq!(out.len(), self.stream.seq_len(), "out must hold L tokens");
        let mut filled = 0;
        while filled < out.len() {
            let doc = self.stream.current().expect("`need` found the documents");
            let t = tokens
                .tokens(doc)
                .expect("`need` found the tokens in memory");
            let rest = &t[self.offset as usize..];
            let n = rest.len().min(out.len() - filled);
            out[filled..filled + n].copy_from_slice(&rest[..n]);
            filled += n;
            if n == rest.len() {
                self.stream.advance();
                self.offset = 0;
            } else {
                self.offset += n as u32;
            }
        }
    }
}
