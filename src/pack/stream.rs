//! A lane's documents, slice by slice and epoch after epoch.

use std::sync::Arc;

use super::{Epoch, Orders, Position};
use crate::order::Doc;

/// A lane's place in its stream. It stands on a document, never on the end
/// of a slice: `current` is the next document the lane takes. Cloning is
/// cheap, so a lane can look ahead without moving.
#[derive(Clone)]
pub struct Stream {
    orders: Arc<Orders>,
    lane: u32,
    epoch: Arc<Epoch>,
    window: usize,
    docs: Arc<[Doc]>,
    next: usize,
    /// The position of `docs[next]` in the lane's documents of the epoch.
    doc: u64,
    /// Set when the lane went through two whole epochs and found no
    /// document: it never gets one.
    empty: bool,
}

impl Stream {
    /// Lane `lane` before its `doc`-th document of epoch `epoch`. A `doc` at
    /// the end of the epoch's documents stands at the start of the next.
    pub fn seek(orders: &Arc<Orders>, lane: u32, epoch: u64, doc: u64) -> Result<Self, String> {
        let e = orders.epoch(epoch);
        let total = e.lane_docs(lane);
        if doc > total {
            return Err(format!(
                "document {doc} of epoch {epoch} is past its {total}"
            ));
        }
        let (mut window, mut start) = (0, 0);
        let order = e.order();
        while let Some(w) = e.windows().get(window) {
            let len = order.lane_len(w, lane);
            if doc < start + len {
                break;
            }
            (window, start) = (window + 1, start + len);
        }
        let docs = match e.windows().get(window) {
            Some(w) => order.lane(w, lane).into(),
            None => Arc::from([]),
        };
        let mut stream = Stream {
            orders: orders.clone(),
            lane,
            epoch: e,
            window,
            docs,
            next: (doc - start) as usize,
            doc,
            empty: false,
        };
        stream.settle();
        Ok(stream)
    }

    pub fn seq_len(&self) -> usize {
        self.orders.seq_len as usize
    }

    pub fn position(&self) -> Position {
        Position {
            epoch: self.epoch.number,
            doc: self.doc,
            offset: 0,
        }
    }

    /// `(epoch, window)` of the current document.
    pub fn slice(&self) -> (u64, usize) {
        (self.epoch.number, self.window)
    }

    /// The next document the lane takes, or `None` if it never gets one.
    pub fn current(&self) -> Option<Doc> {
        (!self.empty).then(|| self.docs[self.next])
    }

    /// Moves past the current document.
    pub fn advance(&mut self) {
        self.next += 1;
        self.doc += 1;
        self.settle();
    }

    /// Moves past slices with no documents left, into later windows and
    /// epochs, until it stands on a document.
    fn settle(&mut self) {
        let from = self.epoch.number;
        while self.next == self.docs.len() {
            self.window += 1;
            if self.window >= self.epoch.windows().len() {
                if self.epoch.number >= from + 2 {
                    self.empty = true;
                    return;
                }
                self.epoch = self.orders.epoch(self.epoch.number + 1);
                (self.window, self.doc) = (0, 0);
            }
            let Some(w) = self.epoch.windows().get(self.window) else {
                continue;
            };
            self.docs = self.epoch.order().lane(w, self.lane).into();
            self.next = 0;
        }
    }
}
