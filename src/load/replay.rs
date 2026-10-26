//! Resumed lanes: each lane's leftover is replayed one part at a time,
//! within the budget, before the prefetch starts. A part is the leftover's
//! documents in one window. A lane copies the tokens it takes, so a part's
//! row groups are freed as soon as it is replayed.

use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;

use futures::future::try_join_all;
use tokio::sync::{Semaphore, SemaphorePermit};

use super::plan::kib;
use super::{Abort, CONCURRENT_READS, Error, permits};
use crate::index::Index;
use crate::order::Doc;
use crate::pack::{Epoch, Lane, LaneState, Replay, Tokens, at_lane};
use crate::read::{Reader, RowGroupTokens};

/// The row groups of one part.
struct Part(HashMap<u32, RowGroupTokens>);

impl Tokens for Part {
    fn tokens(&self, doc: Doc) -> &[u32] {
        self.0[&doc.row_group].row(doc.row as usize)
    }
}

/// What every lane's replay shares: the reader, the budget and the read
/// slots.
struct Shared {
    reader: Arc<Reader>,
    index: Arc<Index>,
    budget: u64,
    permits: Semaphore,
    slots: Arc<Semaphore>,
}

/// Lanes `lanes`, each rebuilt from its state in `states` with its leftover
/// replayed. The parts of all lanes together hold at most `budget` bytes at
/// once; lanes replay at the same time, each in stream order.
///
/// A part is in one lane slice of one window, and a run with no stop holds
/// that whole slice while the lane takes from it, so each part fits any
/// budget that run accepts. A larger part gives `Error::Budget`.
pub(super) async fn lanes(
    epoch: &Arc<Epoch>,
    reader: Arc<Reader>,
    index: Arc<Index>,
    lanes: Range<u32>,
    states: &[LaneState],
    budget: u64,
) -> Result<Vec<Lane>, Error> {
    let number = epoch.order().epoch;
    let state = |l: u32| move |why: String| Error::State(at_lane(l, number, &why));
    let mut replays = (lanes.clone().zip(states))
        .map(|(l, &s)| epoch.replay(l, s).map_err(state(l)))
        .collect::<Result<Vec<_>, _>>()?;
    let shared = Shared {
        reader,
        index,
        budget,
        permits: Semaphore::new((budget / 1024) as usize),
        slots: Arc::new(Semaphore::new(CONCURRENT_READS)),
    };
    let replay = (replays.iter_mut().zip(lanes.clone()))
        .map(|(lane, l)| replay_lane(lane, l, number, &shared));
    try_join_all(replay).await?;
    (replays.into_iter().zip(lanes))
        .map(|(r, l)| r.finish().map_err(state(l)))
        .collect()
}

/// Replays `lane`, lane `l` of epoch `number`, part by part. Each part holds
/// its row groups' KiB of the budget until its documents are taken. The
/// reads are tasks, so their decode runs on the runtime's workers, and
/// `Abort` stops them if the replay is dropped.
async fn replay_lane(lane: &mut Replay, l: u32, number: u64, shared: &Shared) -> Result<(), Error> {
    loop {
        let docs = lane.part();
        if docs.is_empty() {
            return Ok(());
        }
        let n = docs.len() as u64;
        let mut groups: Vec<u32> = docs.iter().map(|d| d.row_group).collect();
        groups.sort_unstable();
        groups.dedup();
        // The permits live until the part's tokens are taken and dropped.
        let (tokens, _held) = read(&groups, shared).await?;
        let take = lane.take(&tokens, n);
        take.map_err(|why| Error::State(at_lane(l, number, &why)))?;
    }
}

/// Row groups `groups`, read at once, and the permits of the budget that
/// they take.
async fn read<'a>(
    groups: &[u32],
    shared: &'a Shared,
) -> Result<(Part, SemaphorePermit<'a>), Error> {
    let kib: u64 = groups.iter().map(|&g| kib(&shared.index, g)).sum();
    if kib > shared.budget / 1024 {
        let (needed, budget) = (kib * 1024, shared.budget);
        return Err(Error::Budget { needed, budget });
    }
    let held = shared.permits.acquire_many(permits(kib)).await;
    let held = held.expect("the replay never closes its budget");
    let mut reads = Vec::new();
    for &g in groups {
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
    Ok((Part(part), held))
}
