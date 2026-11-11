//! Lanes: each lane packs its stream of documents into sequences of exactly
//! L tokens, with nothing dropped and no padding. A lane's stream is its
//! slice of each window of epoch 0, then of epoch 1, and so on, so a lane
//! that ends an epoch goes on into the next.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, Weak};

use crate::order::{Order, Window};

mod lane;
mod pieces;
#[cfg(test)]
mod tests;

pub use lane::Lane;
pub use pieces::Piece;

/// A lane state error, named by its lane: the session and the loader both
/// give it, and Python users see it in `CheckpointError`.
pub(crate) fn at_lane(lane: u32, why: &str) -> String {
    format!("lane {lane}: {why}")
}

/// The tokens of one row group: row `r` is `values[offsets[r]..offsets[r + 1]]`.
#[derive(Clone, Copy)]
pub struct Rows<'a> {
    pub values: &'a [u32],
    pub offsets: &'a [usize],
}

/// Where a lane gets each row group's tokens: `None` when they are not in
/// memory. The packer adds no separators, so a source that wants EOS between
/// documents includes it in the tokens.
pub trait Tokens {
    fn row_group(&self, row_group: u32) -> Option<Rows<'_>>;
}

/// Where a lane stands: after the first `token` tokens of its slice of
/// window `window` of epoch `epoch`. The default is the start of epoch 0.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Position {
    pub epoch: u64,
    pub window: u32,
    pub token: u64,
}

/// The epochs kept alive after no lane holds them: the loader still releases
/// the windows of the epoch that its lanes just left.
const RECENT: usize = 4;

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
    slots: Arc<[u32]>,
    epochs: Mutex<Epochs>,
}

#[derive(Default)]
struct Epochs {
    made: HashMap<u64, Weak<Epoch>>,
    recent: VecDeque<Arc<Epoch>>,
}

impl Orders {
    /// `slots` is `Order::slots`: each row group's leaf value count.
    pub fn new(seed: u64, lanes: u32, window: u32, seq_len: u32, slots: &[u32]) -> Arc<Self> {
        assert!(
            lanes > 0 && window > 0 && seq_len > 0,
            "B, W and L must be positive"
        );
        Arc::new(Orders {
            seed,
            lanes,
            window,
            seq_len,
            slots: slots.into(),
            epochs: Mutex::default(),
        })
    }

    /// Epoch `number`, made when no lane holds it and it is not recent.
    pub fn epoch(&self, number: u64) -> Arc<Epoch> {
        let mut epochs = self.epochs.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(epoch) = epochs.made.get(&number).and_then(Weak::upgrade) {
            return epoch;
        }
        let epoch = Arc::new(Epoch::new(self, number));
        epochs.made.retain(|_, e| e.strong_count() > 0);
        epochs.made.insert(number, Arc::downgrade(&epoch));
        if epochs.recent.len() == RECENT {
            epochs.recent.pop_front();
        }
        epochs.recent.push_back(epoch.clone());
        epoch
    }

    pub fn row_groups(&self) -> usize {
        self.slots.len()
    }
}

/// One epoch's windows, and the order that made them.
pub struct Epoch {
    pub number: u64,
    seed: u64,
    lanes: u32,
    window: u32,
    slots: Arc<[u32]>,
    windows: Vec<Window>,
}

impl Epoch {
    fn new(orders: &Orders, number: u64) -> Self {
        let mut epoch = Epoch {
            number,
            seed: orders.seed,
            lanes: orders.lanes,
            window: orders.window,
            slots: orders.slots.clone(),
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
            slots: &self.slots,
        }
    }

    pub fn windows(&self) -> &[Window] {
        &self.windows
    }

    /// The slots of lane `lane` in the whole epoch.
    pub fn lane_slots(&self, lane: u32) -> u64 {
        let order = self.order();
        self.windows.iter().map(|w| order.lane_len(w, lane)).sum()
    }
}

/// What the next sequence of a lane needs.
#[derive(Debug, PartialEq, Eq)]
pub enum Need {
    /// Every row group it takes tokens from is in memory.
    Ready,
    /// The first row group it needs that is not in memory, and the
    /// `(epoch, window)` it is in.
    Missing { slice: (u64, usize), row_group: u32 },
    /// A position that no run makes, such as from a hand-made checkpoint,
    /// or tokens that do not match the index.
    Bad(String),
    /// A lane that finds no token in two whole epochs: it would never fill
    /// a sequence.
    Empty(String),
}
