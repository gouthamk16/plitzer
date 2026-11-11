use std::fmt;
use std::ops::Range;
use std::sync::Arc;

use tokio::sync::mpsc::UnboundedReceiver;
use tokio::sync::watch;
use tokio::task::{JoinError, JoinHandle};

use crate::index::Index;
use crate::pack::{Lane, Need, Orders, Position};
use crate::read::{self, Reader};

mod buffer;
mod prefetch;
mod sequence;
#[cfg(test)]
mod tests;

use buffer::Buffer;
use prefetch::{Arrival, Prefetch, Signals};
use sequence::{Key, Sequence, window_kib};

#[derive(Debug)]
pub enum Error {
    /// The budget is below one window of the GPU's lanes, which a run holds
    /// at once.
    Budget {
        needed: u64,
        budget: u64,
    },
    Read(read::Error),
    /// A read task panicked.
    Join(JoinError),
    /// An earlier read failed. `positions()` still holds the position before
    /// the failed step, so build a new loader from it.
    Stopped,
    /// A lane position that no run makes, such as from a hand-made checkpoint.
    State(String),
    /// A lane that never finds a token, so it can never fill a sequence.
    Empty(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Budget { needed, budget } => write!(
                f,
                "the prefetch budget is {budget} bytes, but one window of the lanes needs {needed}"
            ),
            Error::Read(e) => write!(f, "{e}"),
            Error::Join(e) => write!(f, "a read task failed: {e}"),
            Error::Stopped => write!(f, "an earlier read failed; resume from positions()"),
            Error::State(why) | Error::Empty(why) => write!(f, "{why}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Read(e) => Some(e),
            Error::Join(e) => Some(e),
            _ => None,
        }
    }
}

/// A task that is stopped when its handle is dropped, so a dropped loader
/// leaves no reads running.
pub(crate) struct Abort<T>(pub(crate) JoinHandle<T>);

impl<T> Drop for Abort<T> {
    fn drop(&mut self) {
        self.0.abort();
    }
}

/// The lanes of one GPU, fed by a prefetch of their row groups. Each step
/// gives one sequence of exactly L tokens for each lane, for as long as the
/// caller asks: a lane's stream goes on into the next epoch.
pub struct Loader {
    orders: Arc<Orders>,
    lanes: Vec<Lane>,
    buffer: Buffer,
    arrivals: UnboundedReceiver<Arrival>,
    /// The read taken from `arrivals` and not yet finished. It lives here,
    /// not in `receive`, so a dropped `next` future loses no read.
    pending: Option<Arrival>,
    demand: watch::Sender<Key>,
    slowest: watch::Sender<(u64, usize)>,
    /// Set when a read failed: `next` stops, and the lanes keep the
    /// position before the failed step.
    stopped: bool,
    _driver: Abort<()>,
}

impl Loader {
    /// The loader for lanes `lanes`, each at its position in `at`. `budget`
    /// caps the decoded bytes that the prefetch reads ahead; a read that a
    /// step waits for is made past it. The reads run as tokio tasks, and on
    /// a multi-thread runtime they go on while the caller trains.
    pub fn new(
        orders: Arc<Orders>,
        reader: Arc<Reader>,
        index: Arc<Index>,
        lanes: Range<u32>,
        at: &[Position],
        budget: u64,
    ) -> Result<Self, Error> {
        assert_eq!(at.len(), lanes.len(), "one position for each lane");
        let made: Result<Vec<Lane>, String> = (lanes.clone().zip(at))
            .map(|(l, &p)| Lane::new(&orders, l, p))
            .collect();
        let made = made.map_err(Error::State)?;
        let starts: Vec<(u64, usize)> = made.iter().map(Lane::slice).collect();
        let start = starts.iter().min().copied().unwrap_or((0, 0));
        let needed = window_kib(&orders, &index, lanes.clone(), start.0);
        if needed > budget / 1024 {
            let needed = needed * 1024;
            return Err(Error::Budget { needed, budget });
        }
        let (demand, wanted) = watch::channel((0, 0, 0));
        let (slowest, behind) = watch::channel(start);
        let prefetch = Prefetch {
            reader,
            index,
            budget_kib: budget / 1024,
            starts,
            windows: orders.epoch(start.0).windows().len() as u64,
        };
        let seq = Sequence::new(&orders, lanes, start);
        let signals = Signals {
            demand: wanted,
            slowest: behind,
        };
        let (arrivals, driver) = prefetch.start(seq, signals);
        Ok(Loader {
            orders,
            lanes: made,
            buffer: Buffer::default(),
            arrivals,
            pending: None,
            demand,
            slowest,
            stopped: false,
            _driver: driver,
        })
    }

    /// The next step: one sequence of L tokens for each lane, in lane order,
    /// written into `out`. Cancel safe: the lanes move only once every row
    /// group the step needs is in memory, after the last await.
    pub async fn next(&mut self, out: &mut [u32]) -> Result<(), Error> {
        if self.stopped {
            return Err(Error::Stopped);
        }
        let seq_len = self.orders.seq_len as usize;
        assert_eq!(
            out.len(),
            self.lanes.len() * seq_len,
            "out must hold one sequence per lane"
        );
        while let Some(key) = self.missing()? {
            if let Err(e) = self.receive(key).await {
                self.stopped = true;
                return Err(e);
            }
        }
        let before: Vec<(u64, usize)> = self.lanes.iter().map(Lane::slice).collect();
        for (lane, seq) in self.lanes.iter_mut().zip(out.chunks_mut(seq_len)) {
            lane.fill(&self.buffer, seq);
        }
        // Only after every lane's fill: `missing` found each row group in
        // memory for the whole step.
        for (i, before) in before.into_iter().enumerate() {
            self.buffer.release(&self.orders, &self.lanes[i], i, before);
        }
        let slowest = self
            .lanes
            .iter()
            .map(Lane::slice)
            .min()
            .expect("a GPU has lanes");
        self.slowest
            .send_if_modified(|s| (*s != slowest).then(|| *s = slowest).is_some());
        Ok(())
    }

    /// Each lane's position, for a checkpoint. Prefetched row groups are not
    /// part of it: a resume reads them again.
    pub fn positions(&self) -> Vec<Position> {
        self.lanes.iter().map(Lane::position).collect()
    }

    /// The decoded bytes in memory now, of the row groups that lanes take
    /// from or will take from soon.
    pub fn resident_bytes(&self) -> usize {
        self.buffer.bytes()
    }

    /// The key of the first row group that the next step needs and that is
    /// not in memory, or `None` when the step can run.
    fn missing(&self) -> Result<Option<Key>, Error> {
        for lane in &self.lanes {
            match lane.need(&self.buffer) {
                Need::Ready => {}
                Need::Missing { slice, row_group } => return Ok(Some(self.key(slice, row_group))),
                Need::Bad(why) => return Err(Error::State(why)),
                Need::Empty(why) => return Err(Error::Empty(why)),
            }
        }
        Ok(None)
    }

    fn key(&self, (epoch, window): (u64, usize), row_group: u32) -> Key {
        let e = self.orders.epoch(epoch);
        let groups = e.windows()[window].groups();
        let i = groups
            .binary_search(&row_group)
            .expect("a lane's row group is in its window");
        (epoch, window, i)
    }

    /// Takes arrivals in order until the one at `key` is in memory. Cancel
    /// safe: `recv` is, and the read in progress stays in `pending`.
    async fn receive(&mut self, key: Key) -> Result<(), Error> {
        self.demand
            .send_if_modified(|d| (*d < key).then(|| *d = key).is_some());
        loop {
            if self.pending.is_none() {
                let arrival = self.arrivals.recv().await;
                self.pending = Some(arrival.expect("the driver sends reads for ever"));
            }
            let arrival = self.pending.as_mut().expect("set above");
            let read = (&mut arrival.read.0).await;
            let arrival = self.pending.take().expect("set above");
            let (tokens, permit) = read.map_err(Error::Join)?.map_err(Error::Read)?;
            (self.buffer).hold(&self.orders, &self.lanes, &arrival, tokens, permit);
            if arrival.key >= key {
                return Ok(());
            }
        }
    }
}
