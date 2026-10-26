//! Resumed lanes: each lane's leftover is replayed one chunk at a time,
//! within the budget, before the prefetch starts. A lane copies the tokens
//! it takes, so each chunk's row groups are freed as soon as it is replayed.

use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;

use futures::future::try_join_all;
use tokio::sync::Semaphore;

use super::plan::Chunk;
use super::{CONCURRENT_READS, Error};
use crate::index::Index;
use crate::order::Doc;
use crate::pack::{Epoch, Lane, LaneState, Replay, Tokens};
use crate::read::{Reader, RowGroupTokens};

/// The row groups of one chunk.
struct Part(HashMap<u32, RowGroupTokens>);

impl Tokens for Part {
    fn tokens(&self, doc: Doc) -> &[u32] {
        self.0[&doc.row_group].row(doc.row as usize)
    }
}

/// Lanes `lanes`, each rebuilt from its state in `states` with the leftover
/// replayed from `chunks`. The chunks of all lanes together hold at most
/// `budget_kib` at once; lanes replay at the same time, each in stream order.
pub(super) async fn lanes(
    epoch: &Arc<Epoch>,
    reader: &Reader,
    index: &Index,
    lanes: Range<u32>,
    states: &[LaneState],
    chunks: Vec<Chunk>,
    budget_kib: u64,
) -> Result<Vec<Lane>, Error> {
    let number = epoch.order().epoch;
    let at = |l: u32| move |why| Error::State(format!("lane {l} of epoch {number}: {why}"));
    let mut replays = (lanes.clone().zip(states))
        .map(|(l, &s)| epoch.replay(l, s).map_err(at(l)))
        .collect::<Result<Vec<_>, _>>()?;
    let mut by_lane: Vec<Vec<Chunk>> = replays.iter().map(|_| Vec::new()).collect();
    for chunk in chunks {
        by_lane[chunk.lane].push(chunk);
    }
    let limits = Limits {
        budget: Semaphore::new(budget_kib as usize),
        slots: Semaphore::new(CONCURRENT_READS),
    };
    let replay = (replays.iter_mut().zip(by_lane))
        .map(|(lane, chunks)| replay(lane, chunks, reader, index, &limits));
    try_join_all(replay).await?;
    (lanes.zip(replays))
        .map(|(l, r)| r.finish().map_err(at(l)))
        .collect()
}

/// The budget, in KiB, and the read slots that all lanes share.
struct Limits {
    budget: Semaphore,
    slots: Semaphore,
}

/// Replays `chunks`, in order, into `lane`. Each chunk holds its row groups'
/// KiB of the budget until its documents are taken.
async fn replay(
    lane: &mut Replay,
    chunks: Vec<Chunk>,
    reader: &Reader,
    index: &Index,
    limits: &Limits,
) -> Result<(), Error> {
    for chunk in chunks {
        let _permit = limits.budget.acquire_many(chunk.kib as u32).await;
        let read = |g| async move {
            let _slot = limits.slots.acquire().await;
            reader.read(index, g).await.map(|tokens| (g, tokens))
        };
        let part = try_join_all(chunk.groups.iter().map(|&g| read(g))).await;
        lane.take(
            &Part(part.map_err(Error::Read)?.into_iter().collect()),
            chunk.docs,
        );
    }
    Ok(())
}
