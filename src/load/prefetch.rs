//! The prefetch: one driver task starts the reads of the order, within the
//! byte budget. A read that the loader waits for is admitted past the
//! budget: lanes can be a window apart, and a run must never wait for itself.

use std::sync::Arc;

use tokio::sync::mpsc::{UnboundedReceiver, unbounded_channel};
use tokio::sync::{OwnedSemaphorePermit, Semaphore, watch};

use super::Abort;
use super::sequence::{Key, Sequence, kib};
use crate::index::Index;
use crate::read::{self, Reader, RowGroupTokens};

/// Reads in flight at once. The budget alone would allow thousands of
/// concurrent requests at start; the index caps footer reads the same way.
const CONCURRENT_READS: usize = 16;

/// Windows the prefetch reads past the slowest lane's window. Lanes stand
/// within about a window of each other, and the stream never ends, so a
/// large budget would otherwise read whole epochs ahead.
const AHEAD: u64 = 2;

/// A row group's share of the budget. One admitted past the budget added
/// its permits, and forgets them when dropped, so the budget comes back.
pub(super) struct Permit {
    permit: Option<OwnedSemaphorePermit>,
    over: bool,
}

impl Drop for Permit {
    fn drop(&mut self) {
        if let (true, Some(p)) = (self.over, self.permit.take()) {
            p.forget();
        }
    }
}

pub(super) type Read = Result<(RowGroupTokens, Permit), read::Error>;

/// One read of the order.
pub(super) struct Arrival {
    pub key: Key,
    pub row_group: u32,
    pub read: Abort<Read>,
}

pub(super) struct Prefetch {
    pub reader: Arc<Reader>,
    pub index: Arc<Index>,
    pub budget_kib: u64,
    /// Where each of the GPU's lanes starts: the row groups of earlier
    /// windows that no lane still takes from are not read.
    pub starts: Vec<(u64, usize)>,
    /// The windows of each epoch, the same in every epoch.
    pub windows: u64,
}

/// What the loader tells the driver.
pub(super) struct Signals {
    /// The key a step waits for: every read up to it starts at once.
    pub demand: watch::Receiver<Key>,
    /// The window of the slowest lane.
    pub slowest: watch::Receiver<(u64, usize)>,
}

impl Prefetch {
    /// Starts the driver over `seq`.
    pub fn start(
        self,
        mut seq: Sequence,
        mut signals: Signals,
    ) -> (UnboundedReceiver<Arrival>, Abort<()>) {
        let (tx, rx) = unbounded_channel();
        let driver = tokio::spawn(async move {
            let budget = Arc::new(Semaphore::new(self.budget_kib as usize));
            let slots = Arc::new(Semaphore::new(CONCURRENT_READS));
            loop {
                let w = seq.next_window();
                let now = (w.epoch, w.window);
                if !self.near(now, w.key(w.used().start), &mut signals).await {
                    return;
                }
                for i in w
                    .used()
                    .filter(|&i| w.users(i).any(|l| self.starts[l] <= now))
                {
                    let key = (w.key(i), w.groups[i]);
                    let read = self.read(key, &budget, &slots, &mut signals.demand);
                    let Some(arrival) = read.await else { return };
                    if tx.send(arrival).is_err() {
                        return;
                    }
                }
            }
        });
        (rx, Abort(driver))
    }

    /// Starts the read of row group `g` at `key` once the budget and a read
    /// slot allow it. `None` once the loader is gone.
    async fn read(
        &self,
        (key, g): (Key, u32),
        budget: &Arc<Semaphore>,
        slots: &Arc<Semaphore>,
        demand: &mut watch::Receiver<Key>,
    ) -> Option<Arrival> {
        let permit = admit(budget, permits(kib(&self.index, g)), key, demand).await?;
        let slot = slots.clone().acquire_owned().await.ok()?;
        let (reader, index) = (self.reader.clone(), self.index.clone());
        let read = tokio::spawn(async move {
            let tokens = reader.read(&index, g).await;
            drop(slot);
            Ok((tokens?, permit))
        });
        Some(Arrival {
            key,
            row_group: g,
            read: Abort(read),
        })
    }

    /// Waits until window `now` is at most `AHEAD` windows past the slowest
    /// lane, or a step waits for its first read at `first`. False once the
    /// loader is gone.
    async fn near(&self, now: (u64, usize), first: Key, signals: &mut Signals) -> bool {
        // Saturating: a hand-made checkpoint can hold any epoch.
        let ordinal =
            |(e, w): (u64, usize)| e.saturating_mul(self.windows).saturating_add(w as u64);
        loop {
            let slowest = *signals.slowest.borrow_and_update();
            if ordinal(now) <= ordinal(slowest).saturating_add(AHEAD)
                || *signals.demand.borrow_and_update() >= first
            {
                return true;
            }
            tokio::select! {
                changed = signals.slowest.changed() => if changed.is_err() { return false },
                changed = signals.demand.changed() => if changed.is_err() { return false },
            }
        }
    }
}

/// `kib` permits of the budget for the read at `key`: from the budget, or
/// added past it once the loader waits for `key` or a later read. `None`
/// once the loader is gone.
async fn admit(
    budget: &Arc<Semaphore>,
    kib: u32,
    key: Key,
    demand: &mut watch::Receiver<Key>,
) -> Option<Permit> {
    loop {
        if *demand.borrow_and_update() >= key {
            // The driver is the only one that takes permits, so these are its.
            budget.add_permits(kib as usize);
            let permit = budget.clone().acquire_many_owned(kib).await.ok()?;
            return Some(Permit {
                permit: Some(permit),
                over: true,
            });
        }
        tokio::select! {
            permit = budget.clone().acquire_many_owned(kib) => {
                return Some(Permit { permit: Some(permit.ok()?), over: false });
            }
            changed = demand.changed() => changed.ok()?,
        }
    }
}

/// `kib` as semaphore permits, which tokio counts in `u32`. Only a row group
/// past 4 TiB saturates.
fn permits(kib: u64) -> u32 {
    u32::try_from(kib).unwrap_or(u32::MAX)
}
