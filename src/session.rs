//! One GPU's view of a run: its lanes across epochs, with checkpoints.

use std::fmt;
use std::ops::Range;
use std::sync::Arc;

use object_store::ObjectStore;
use object_store::path::Path;

use crate::checkpoint::{self, Checkpoint, Config};
use crate::index::{self, Index};
use crate::load::{self, Loader};
use crate::order::{Order, gpu_lanes};
use crate::pack::{Epoch, LaneState};
use crate::read::{self, Reader};

#[derive(Debug)]
pub enum Error {
    /// A setting that can never work, such as a world that does not divide B.
    Config(String),
    Index(index::Error),
    Read(read::Error),
    Load(load::Error),
    Checkpoint(checkpoint::Error),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Config(why) => write!(f, "{why}"),
            Error::Index(e) => write!(f, "{e}"),
            Error::Read(e) => write!(f, "{e}"),
            Error::Load(e) => write!(f, "{e}"),
            Error::Checkpoint(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Config(_) => None,
            Error::Index(e) => Some(e),
            Error::Read(e) => Some(e),
            Error::Load(e) => Some(e),
            Error::Checkpoint(e) => Some(e),
        }
    }
}

/// The lanes of GPU `rank` of `world`, from one epoch to the next.
pub struct Session {
    config: Config,
    index: Arc<Index>,
    fingerprint: u64,
    reader: Arc<Reader>,
    lanes: Range<u32>,
    budget: u64,
    epoch: u64,
    /// The lane states the next loader starts from.
    start: Vec<LaneState>,
    /// The loader of the epoch, made at its first step.
    loader: Option<Loader>,
}

impl Session {
    /// Opens the dataset at `prefix`, with the index cached at `cache`, and
    /// stands before the first step of epoch 0. `budget` is the loader's.
    pub async fn open(
        store: Arc<dyn ObjectStore>,
        prefix: Path,
        cache: &std::path::Path,
        config: Config,
        rank: u32,
        world: u32,
        budget: u64,
    ) -> Result<Session, Error> {
        validate(&config, rank, world)?;
        let index = Index::open(&*store, &prefix, &config.column, cache)
            .await
            .map_err(Error::Index)?;
        let reader = Reader::new(store, prefix, &index).map_err(Error::Read)?;
        let lanes = gpu_lanes(config.lanes, world, rank);
        Ok(Session {
            config,
            fingerprint: index.fingerprint(),
            index: Arc::new(index),
            reader: Arc::new(reader),
            start: vec![LaneState::default(); lanes.len()],
            lanes,
            budget,
            epoch: 0,
            loader: None,
        })
    }

    /// Stands before the first step of epoch `epoch` if it is later than the
    /// session's epoch, and does nothing otherwise. Trainers call this at the
    /// start of each epoch, also the one they resumed into, and a checkpoint
    /// after an epoch's last step is already at the next epoch. Going back
    /// would repeat data with no error; open a new session to do that.
    pub fn set_epoch(&mut self, epoch: u64) {
        if epoch > self.epoch {
            self.start_epoch(epoch);
        }
    }

    fn start_epoch(&mut self, epoch: u64) {
        self.epoch = epoch;
        self.start = vec![LaneState::default(); self.lanes.len()];
        self.loader = None;
    }

    /// Stands where `checkpoint` was taken. It must come from the same
    /// config and dataset, and hold this GPU's lanes: a global checkpoint
    /// works on any number of GPUs. It makes the loader at once, so it
    /// succeeds only if the run can go on from there; on an error the
    /// session stays where it was, and `checkpoint()` still gives that.
    pub async fn resume(&mut self, checkpoint: &Checkpoint) -> Result<(), Error> {
        let states = (checkpoint.check(&self.config, self.fingerprint))
            .and_then(|()| checkpoint.states(self.lanes.clone()))
            .map_err(Error::Checkpoint)?;
        let plan = Arc::new(self.epoch_plan(checkpoint.epoch));
        self.in_range(&plan, &states).map_err(Error::Checkpoint)?;
        // Drop the old loader first: its row groups are freed before the new
        // one fills its budget, and only reads still in flight outlive it,
        // until their next await. The session keeps its position, so an
        // error or a dropped future leaves it where it was.
        self.park();
        let loader = self.load(plan, &states).await?;
        (self.epoch, self.start, self.loader) = (checkpoint.epoch, states, Some(loader));
        Ok(())
    }

    /// Drops the loader and keeps its position in `epoch` and `start`. An
    /// ended loader stays: it holds no row groups, and it still owes the
    /// `None` that ends its epoch.
    fn park(&mut self) {
        if let Some(loader) = self.loader.take_if(|l| !l.ended()) {
            self.start = loader.states();
        }
    }

    /// The next step: one sequence of L tokens for each of this GPU's lanes,
    /// in lane order, written into `out`, and the real-token count of each.
    /// After the last step of an epoch it returns `None` and stands before
    /// the first step of the next. After an error, the next call retries
    /// the same step.
    pub async fn next(&mut self, out: &mut [u32]) -> Result<Option<Vec<u32>>, Error> {
        let loader = match &mut self.loader {
            Some(loader) => loader,
            None => {
                let plan = Arc::new(self.epoch_plan(self.epoch));
                self.loader.insert(self.load(plan, &self.start).await?)
            }
        };
        match loader.next(out).await {
            Ok(Some(real)) => Ok(Some(real)),
            Ok(None) => {
                self.start_epoch(self.epoch.saturating_add(1));
                Ok(None)
            }
            Err(e) => {
                self.start = loader.states();
                self.loader = None;
                Err(Error::Load(e))
            }
        }
    }

    /// This GPU's part of a checkpoint. `Checkpoint::merge` joins the parts
    /// of all GPUs into one that resumes on any number of GPUs. After the
    /// last step of an epoch it is the start of the next epoch, also before
    /// the call of `next` that returns `None`, so the parts of all GPUs agree.
    pub fn checkpoint(&self) -> Checkpoint {
        let (epoch, lanes) = match &self.loader {
            Some(l) if l.ended() => (
                self.epoch.saturating_add(1),
                vec![LaneState::default(); self.lanes.len()],
            ),
            Some(l) => (self.epoch, l.states()),
            None => (self.epoch, self.start.clone()),
        };
        Checkpoint {
            config: self.config.clone(),
            fingerprint: self.fingerprint,
            epoch,
            first_lane: self.lanes.start,
            lanes,
        }
    }

    /// True after the last step of the epoch, until the call of `next` that
    /// returns `None` moves the session to the next epoch.
    pub fn ended(&self) -> bool {
        self.loader.as_ref().is_some_and(Loader::ended)
    }

    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    pub fn lanes(&self) -> Range<u32> {
        self.lanes.clone()
    }

    /// Whether each of `states` lies inside the epoch of `plan`: no later
    /// than its last step, with its leftover not past the documents it has
    /// taken. A checkpoint from plitzer always passes. The states that this
    /// cannot refuse need the tokens; `load` refuses them.
    fn in_range(&self, plan: &Epoch, states: &[LaneState]) -> Result<(), checkpoint::Error> {
        let epoch = plan.order().epoch;
        for (lane, &state) in self.lanes.clone().zip(states) {
            let at = |why| format!("lane {lane} of epoch {epoch}: {why}");
            plan.check(lane, state)
                .map_err(|why| checkpoint::Error::Range(at(why)))?;
        }
        Ok(())
    }

    fn epoch_plan(&self, epoch: u64) -> Epoch {
        let c = &self.config;
        let order = Order {
            seed: c.seed,
            epoch,
            lanes: c.lanes,
            window: c.window,
            rows: &self.index.rows(),
        };
        Epoch::new(order, c.packing)
    }

    async fn load(&self, epoch: Arc<Epoch>, start: &[LaneState]) -> Result<Loader, Error> {
        let (reader, index) = (self.reader.clone(), self.index.clone());
        let loader = Loader::new(epoch, reader, index, self.lanes(), start, self.budget);
        loader.await.map_err(|e| match e {
            // Only a resumed checkpoint gives a state that no run makes.
            load::Error::State(why) => Error::Checkpoint(checkpoint::Error::Range(why)),
            e => Error::Load(e),
        })
    }
}

/// The most lanes: far past any real global batch. Each GPU keeps a state
/// for each of its lanes, so a larger value could only exhaust memory.
const MAX_LANES: u32 = 1 << 24;

/// The most tokens in one step of one GPU, 4 GiB of output, and in the
/// leftovers of one GPU: far past any real batch, and more could only
/// exhaust memory.
const MAX_STEP_TOKENS: u64 = 1 << 30;

/// Refuses the settings that the order and the packer would panic on, and
/// sizes that could only exhaust memory.
fn validate(c: &Config, rank: u32, world: u32) -> Result<(), Error> {
    let why = if c.lanes == 0 || c.window == 0 {
        "lanes and window must be positive".to_string()
    } else if c.packing.seq_len == 0 || c.packing.docs == 0 {
        "seq_len and docs must be positive".to_string()
    } else if world == 0 || !c.lanes.is_multiple_of(world) {
        format!("world {world} must divide {} lanes", c.lanes)
    } else if rank >= world {
        format!("rank {rank} out of range for world {world}")
    } else if c.lanes > MAX_LANES {
        format!("lanes {} is more than {MAX_LANES}", c.lanes)
    } else if u64::from(c.lanes / world) * u64::from(c.packing.seq_len) > MAX_STEP_TOKENS {
        format!("one step of one GPU holds more than {MAX_STEP_TOKENS} tokens")
    } else if u64::from(c.lanes / world) * u64::from(c.packing.cap) > MAX_STEP_TOKENS {
        format!("the leftovers of one GPU can hold more than {MAX_STEP_TOKENS} tokens")
    } else {
        return Ok(());
    };
    Err(Error::Config(why))
}

#[cfg(test)]
mod tests;
