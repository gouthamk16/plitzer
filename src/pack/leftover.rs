use std::collections::VecDeque;

/// The tokens a lane has taken but not yet emitted or dropped.
pub(super) struct Leftover {
    pub(super) tokens: VecDeque<u32>,
    /// Remaining tokens of each document from `doc` to the cursor.
    pub(super) lens: VecDeque<u32>,
    /// Where the leftover starts.
    pub(super) doc: u64,
    pub(super) offset: u32,
}

impl Leftover {
    pub(super) fn new(doc: u64) -> Self {
        Leftover {
            tokens: VecDeque::new(),
            lens: VecDeque::new(),
            doc,
            offset: 0,
        }
    }

    pub(super) fn push(&mut self, doc: &[u32]) {
        self.tokens.extend(doc);
        self.lens
            .push_back(u32::try_from(doc.len()).expect("document longer than u32::MAX tokens"));
        self.skip_empty();
    }

    /// Removes the oldest `n` tokens.
    pub(super) fn advance(&mut self, mut n: usize) {
        self.tokens.drain(..n);
        while n > 0 {
            let front = &mut self.lens[0];
            let k = n.min(*front as usize);
            *front -= k as u32;
            self.offset += k as u32;
            n -= k;
            self.skip_empty();
        }
    }

    /// Keeps the start on a document with tokens left, or on the cursor when
    /// the leftover is empty.
    fn skip_empty(&mut self) {
        while self.lens.front() == Some(&0) {
            self.lens.pop_front();
            self.doc += 1;
            self.offset = 0;
        }
    }
}
