//! Resumed lanes: each lane's leftover is replayed one chunk at a time,
//! within the budget, before the prefetch starts. A lane copies the tokens
//! it takes, so each chunk's row groups are freed as soon as it is replayed.

use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;

use futures::future::try_join_all;
use tokio::sync::Semaphore;

use super::plan::Chunk;
use super::{Abort, CONCURRENT_READS, Error};
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

/// What every lane's replay shares: the reader, the budget in KiB and the
/// read slots.
struct Shared {
    reader: Arc<Reader>,
    index: Arc<Index>,
    budget: Semaphore,
    slots: Arc<Semaphore>,
}

/// Lanes `lanes`, each rebuilt from its state in `states` with the leftover
/// replayed from `chunks`. The chunks of all lanes together hold at most
/// `budget_kib` at once; lanes replay at the same time, each in stream order.
pub(super) async fn lanes(
    epoch: &Arc<Epoch>,
    reader: Arc<Reader>,
    index: Arc<Index>,
    lanes: Range<u32>,
    states: &[LaneState],
    chunks: Vec<Chunk>,
    budget_kib: u64,
) -> Result<Vec<Lane>, Error> {
    let number = epoch.order().epoch;
    let names: Vec<String> = (lanes.clone())
        .map(|l| format!("lane {l} of epoch {number}"))
        .collect();
    let mut replays = (lanes.zip(states).zip(&names))
        .map(|((l, &s), name)| epoch.replay(l, s).map_err(|why| state(name, why)))
        .collect::<Result<Vec<_>, _>>()?;
    let mut by_lane: Vec<Vec<Chunk>> = replays.iter().map(|_| Vec::new()).collect();
    for chunk in chunks {
        by_lane[chunk.lane].push(chunk);
    }
    let shared = Shared {
        reader,
        index,
        budget: Semaphore::new(budget_kib as usize),
        slots: Arc::new(Semaphore::new(CONCURRENT_READS)),
    };
    let replay = (replays.iter_mut().zip(by_lane).zip(&names))
        .map(|((lane, chunks), name)| replay(lane, name, chunks, &shared));
    try_join_all(replay).await?;
    (replays.into_iter().zip(&names))
        .map(|(r, name)| r.finish().map_err(|why| state(name, why)))
        .collect()
}

/// Replays `chunks`, in order, into `lane`. Each chunk holds its row groups'
/// KiB of the budget until its documents are taken. The reads are tasks, so
/// their decode runs on the runtime's workers, and `Abort` stops them if
/// the replay is dropped.
async fn replay(
    lane: &mut Replay,
    name: &str,
    chunks: Vec<Chunk>,
    shared: &Shared,
) -> Result<(), Error> {
    for chunk in chunks {
        let _permit = shared.budget.acquire_many(chunk.kib as u32).await;
        let mut reads = Vec::new();
        for &g in &chunk.groups {
            let slot = shared.slots.clone().acquire_owned().await;
            let (reader, index) = (shared.reader.clone(), shared.index.clone());
            reads.push(Abort(tokio::spawn(async move {
                let tokens = reader.read(&index, g).await;
                drop(slot);
                tokens.map(|tokens| (g, tokens))
            })));
        }
        let mut part = HashMap::new();
        for mut read in reads {
            let (g, tokens) = (&mut read.0)
                .await
                .map_err(Error::Join)?
                .map_err(Error::Read)?;
            part.insert(g, tokens);
        }
        let take = lane.take(&Part(part), chunk.docs);
        take.map_err(|why| state(name, why))?;
    }
    Ok(())
}

/// A state that no run makes, named by its lane.
fn state(name: &str, why: String) -> Error {
    Error::State(format!("{name}: {why}"))
}
