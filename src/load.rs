use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use tokio::sync::mpsc::{UnboundedReceiver, unbounded_channel};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tokio::task::{JoinError, JoinHandle};

use crate::index::Index;
use crate::order::Doc;
use crate::pack::{Epoch, Lane, LaneState, Tokens};
use crate::read::{self, Reader, RowGroupTokens};

mod plan;

use plan::{Need, peak, plan};

#[derive(Debug)]
pub enum Error {
    /// The budget is below the most the GPU's lanes need at one step, so
    /// the prefetch would wait for itself.
    Budget {
        needed: u64,
        budget: u64,
    },
    Read(read::Error),
    /// A read task panicked.
    Join(JoinError),
    /// An earlier read failed. `states()` still holds the state before the
    /// failed step, so build a new loader from it.
    Stopped,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Budget { needed, budget } => write!(
                f,
                "the prefetch budget is {budget} bytes, but one step needs {needed}"
            ),
            Error::Read(e) => write!(f, "{e}"),
            Error::Join(e) => write!(f, "a read task failed: {e}"),
            Error::Stopped => write!(f, "an earlier read failed; resume from states()"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Budget { .. } | Error::Stopped => None,
            Error::Read(e) => Some(e),
            Error::Join(e) => Some(e),
        }
    }
}

/// A task that is stopped when its handle is dropped, so a dropped loader
/// leaves no reads running.
struct Abort<T>(JoinHandle<T>);

impl<T> Drop for Abort<T> {
    fn drop(&mut self) {
        self.0.abort();
    }
}

type Read = Result<(RowGroupTokens, OwnedSemaphorePermit), read::Error>;

/// Reads in flight at once. The budget alone would allow thousands of
/// concurrent requests at start; the index caps footer reads the same way.
const CONCURRENT_READS: usize = 16;

/// The decoded row groups in memory. Each holds its budget permits until it
/// is evicted.
#[derive(Default)]
struct Buffer(HashMap<u32, (RowGroupTokens, OwnedSemaphorePermit)>);

impl Tokens for Buffer {
    fn tokens(&self, doc: Doc) -> &[u32] {
        let (tokens, _) = &self.0[&doc.row_group];
        tokens.row(doc.row as usize)
    }
}

/// The lanes of one GPU, fed by a prefetch of their row groups that stays
/// within a byte budget.
pub struct Loader {
    epoch: Arc<Epoch>,
    lanes: Vec<Lane>,
    needs: Vec<Need>,
    /// The next entry of `needs` to take from `arrivals`.
    next: usize,
    /// The row groups to evict after each step.
    evict: BTreeMap<u64, Vec<u32>>,
    buffer: Buffer,
    arrivals: UnboundedReceiver<(u32, Abort<Read>)>,
    /// Decoded bytes of every finished read not yet evicted, also those
    /// still waiting in `arrivals`.
    held: Arc<AtomicUsize>,
    /// Set when a read failed: the lanes may be mid-step, so `next` stops.
    stopped: bool,
    _driver: Abort<()>,
}

impl Loader {
    /// The loader for lanes `lanes`, each resumed from its state in
    /// `states`. `budget` caps the bytes of decoded tokens in memory. The
    /// reads run as tokio tasks; on a multi-thread runtime they continue in
    /// the background while the caller trains.
    pub async fn new(
        epoch: Arc<Epoch>,
        reader: Arc<Reader>,
        index: Arc<Index>,
        lanes: Range<u32>,
        states: &[LaneState],
        budget: u64,
    ) -> Result<Self, Error> {
        assert_eq!(states.len(), lanes.len(), "one state for each lane");
        let step = states.first().map_or(0, |s| s.step);
        assert!(
            states.iter().all(|s| s.step == step),
            "lane states from one step"
        );
        let needs = plan(&epoch, &index, lanes.clone(), states);
        let (budget_kib, needed) = (budget / 1024, peak(&needs));
        if needed > budget_kib {
            let needed = needed * 1024;
            return Err(Error::Budget { needed, budget });
        }
        let held = Arc::new(AtomicUsize::new(0));
        let (arrivals, driver) = prefetch(&needs, reader, index, budget_kib, held.clone());
        let mut loader = Loader {
            epoch: epoch.clone(),
            lanes: Vec::new(),
            evict: evictions(&needs),
            needs,
            next: 0,
            buffer: Buffer::default(),
            arrivals,
            held,
            stopped: false,
            _driver: driver,
        };
        // Resumed lanes replay their leftover now, so their row groups for
        // the resume step must be in memory first.
        loader.receive(step).await?;
        let buffer = &loader.buffer;
        loader.lanes = lanes
            .zip(states)
            .map(|(l, &s)| epoch.lane(l, s, buffer))
            .collect();
        Ok(loader)
    }

    /// The next step: one sequence of L tokens for each lane, in lane order,
    /// written into `out`. Returns the real-token count of each sequence, or
    /// `None` after the last step of the epoch.
    pub async fn next(&mut self, out: &mut [u32]) -> Result<Option<Vec<u32>>, Error> {
        let Some(step) = self.lanes.first().map(|l| l.state().step) else {
            return Ok(None);
        };
        if step == self.epoch.steps() {
            return Ok(None);
        }
        if self.stopped {
            return Err(Error::Stopped);
        }
        if let Err(e) = self.receive(step).await {
            self.stopped = true;
            return Err(e);
        }
        let seq_len = self.epoch.packing().seq_len as usize;
        assert_eq!(
            out.len(),
            self.lanes.len() * seq_len,
            "out must hold one sequence per lane"
        );
        let buffer = &self.buffer;
        let real = (self.lanes.iter_mut().zip(out.chunks_mut(seq_len)))
            .map(|(lane, seq)| lane.fill(buffer, seq))
            .collect();
        for g in self.evict.remove(&step).unwrap_or_default() {
            if let Some((tokens, _)) = self.buffer.0.remove(&g) {
                self.held.fetch_sub(tokens.bytes(), Ordering::Relaxed);
            }
        }
        Ok(Some(real))
    }

    /// The state of each lane, for a checkpoint. Prefetched row groups are
    /// not part of it: a resume reads them again.
    pub fn states(&self) -> Vec<LaneState> {
        self.lanes.iter().map(Lane::state).collect()
    }

    /// The decoded bytes in memory now: the row groups in use and those
    /// prefetched ahead of use. The budget caps this.
    pub fn resident_bytes(&self) -> usize {
        self.held.load(Ordering::Relaxed)
    }

    /// Waits until every row group needed at or before `step` is in memory.
    async fn receive(&mut self, step: u64) -> Result<(), Error> {
        while self.needs.get(self.next).is_some_and(|n| n.first <= step) {
            let want = self.needs[self.next].row_group;
            let (row_group, mut task) = self
                .arrivals
                .recv()
                .await
                .expect("the driver sends every need");
            debug_assert_eq!(row_group, want, "arrivals come in plan order");
            let read = (&mut task.0).await.map_err(Error::Join)?;
            let (tokens, permit) = read.map_err(Error::Read)?;
            self.buffer.0.insert(row_group, (tokens, permit));
            self.next += 1;
        }
        Ok(())
    }
}

/// The row groups to evict after each step: those whose last step it is.
fn evictions(needs: &[Need]) -> BTreeMap<u64, Vec<u32>> {
    let mut evict: BTreeMap<u64, Vec<u32>> = BTreeMap::new();
    for n in needs {
        evict.entry(n.last).or_default().push(n.row_group);
    }
    evict
}

/// Starts the prefetch. One driver takes the budget permits of each row
/// group in plan order, then a read slot, then starts its read. Permits are taken one row
/// group at a time: tokio's semaphore is first in, first out, so a later
/// row group cannot take permits that an earlier one waits for.
fn prefetch(
    needs: &[Need],
    reader: Arc<Reader>,
    index: Arc<Index>,
    budget_kib: u64,
    held: Arc<AtomicUsize>,
) -> (UnboundedReceiver<(u32, Abort<Read>)>, Abort<()>) {
    let (tx, rx) = unbounded_channel();
    let semaphore = Arc::new(Semaphore::new(budget_kib as usize));
    let slots = Arc::new(Semaphore::new(CONCURRENT_READS));
    let needs: Vec<(u32, u32)> = needs.iter().map(|n| (n.row_group, n.kib as u32)).collect();
    let driver = tokio::spawn(async move {
        for (row_group, kib) in needs {
            let Ok(permit) = semaphore.clone().acquire_many_owned(kib).await else {
                return;
            };
            let Ok(slot) = slots.clone().acquire_owned().await else {
                return;
            };
            let (reader, index, held) = (reader.clone(), index.clone(), held.clone());
            let read = tokio::spawn(async move {
                let tokens = reader.read(&index, row_group).await;
                drop(slot);
                let tokens = tokens?;
                held.fetch_add(tokens.bytes(), Ordering::Relaxed);
                Ok((tokens, permit))
            });
            if tx.send((row_group, Abort(read))).is_err() {
                return;
            }
        }
    });
    (rx, Abort(driver))
}

#[cfg(test)]
mod tests;
