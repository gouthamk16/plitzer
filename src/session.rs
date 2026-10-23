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
    /// works on any number of GPUs.
    pub fn resume(&mut self, checkpoint: &Checkpoint) -> Result<(), Error> {
        let states = (checkpoint.check(&self.config, self.fingerprint))
            .and_then(|()| checkpoint.states(self.lanes.clone()))
            .map_err(Error::Checkpoint)?;
        self.epoch = checkpoint.epoch;
        self.start = states;
        self.loader = None;
        Ok(())
    }

    /// The next step: one sequence of L tokens for each of this GPU's lanes,
    /// in lane order, written into `out`, and the real-token count of each.
    /// After the last step of an epoch it returns `None` and stands before
    /// the first step of the next. After an error, the next call retries
    /// the same step.
    pub async fn next(&mut self, out: &mut [u32]) -> Result<Option<Vec<u32>>, Error> {
        let loader = match &mut self.loader {
            Some(loader) => loader,
            None => self.loader.insert(self.load().await?),
        };
        match loader.next(out).await {
            Ok(Some(real)) => Ok(Some(real)),
            Ok(None) => {
                self.start_epoch(self.epoch + 1);
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
            Some(l) if l.ended() => (self.epoch + 1, vec![LaneState::default(); self.lanes.len()]),
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

    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    pub fn lanes(&self) -> Range<u32> {
        self.lanes.clone()
    }

    async fn load(&self) -> Result<Loader, Error> {
        let c = &self.config;
        let order = Order {
            seed: c.seed,
            epoch: self.epoch,
            lanes: c.lanes,
            window: c.window,
            rows: &self.index.rows(),
        };
        let epoch = Arc::new(Epoch::new(order, c.packing));
        let (reader, index) = (self.reader.clone(), self.index.clone());
        Loader::new(epoch, reader, index, self.lanes(), &self.start, self.budget)
            .await
            .map_err(Error::Load)
    }
}

/// Refuses the settings that the order and the packer would panic on.
fn validate(c: &Config, rank: u32, world: u32) -> Result<(), Error> {
    let why = if c.lanes == 0 || c.window == 0 {
        "lanes and window must be positive".to_string()
    } else if c.packing.seq_len == 0 || c.packing.docs == 0 {
        "seq_len and docs must be positive".to_string()
    } else if world == 0 || !c.lanes.is_multiple_of(world) {
        format!("world {world} must divide {} lanes", c.lanes)
    } else if rank >= world {
        format!("rank {rank} out of range for world {world}")
    } else {
        return Ok(());
    };
    Err(Error::Config(why))
}

#[cfg(test)]
mod tests;
