//! One GPU's view of a run: its lanes across epochs, with checkpoints.

use std::fmt;
use std::ops::Range;
use std::sync::Arc;

use object_store::ObjectStore;
use object_store::path::Path;

use crate::checkpoint::{self, Checkpoint, Config};
use crate::index::{self, Index};
use crate::load::{self, Loader};
use crate::order::gpu_lanes;
use crate::pack::{Orders, Position};
use crate::read::{self, Reader};

#[derive(Debug)]
pub enum Error {
    /// A setting that can never work, such as a world that does not divide B.
    Config(String),
    /// A dataset with no tokens, which can never fill a sequence.
    Empty,
    Index(index::Error),
    Read(read::Error),
    Load(load::Error),
    Checkpoint(checkpoint::Error),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Config(why) => write!(f, "{why}"),
            Error::Empty => write!(f, "the dataset has no tokens"),
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
            Error::Config(_) | Error::Empty => None,
            Error::Index(e) => Some(e),
            Error::Read(e) => Some(e),
            Error::Load(e) => Some(e),
            Error::Checkpoint(e) => Some(e),
        }
    }
}

/// The lanes of GPU `rank` of `world`, from one epoch to the next.
///
/// An epoch is S = ⌈T/(B·L)⌉ steps, for the dataset's T tokens. A lane's
/// stream runs on across epochs, so at an epoch's end a lane can be a few
/// documents into the next epoch or still in the last one: nothing is
/// dropped and nothing is padded.
pub struct Session {
    config: Config,
    fingerprint: u64,
    reader: Arc<Reader>,
    index: Arc<Index>,
    orders: Arc<Orders>,
    lanes: Range<u32>,
    budget: u64,
    /// S: the steps of each epoch.
    steps: u64,
    epoch: u64,
    step: u64,
    /// Where the lanes stand when there is no loader.
    start: Vec<Position>,
    /// The loader, made at the first step and kept across epochs.
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
        let tokens: u64 = (0..index.rows().len() as u32)
            .map(|g| u64::from(index.row_group(g).1.values))
            .sum();
        if tokens == 0 {
            return Err(Error::Empty);
        }
        let per_step = u64::from(config.lanes) * u64::from(config.seq_len);
        let c = &config;
        let orders = Orders::new(c.seed, c.lanes, c.window, c.seq_len, &index.rows());
        let lanes = gpu_lanes(config.lanes, world, rank);
        Ok(Session {
            fingerprint: index.fingerprint(),
            reader: Arc::new(reader),
            index: Arc::new(index),
            orders,
            start: vec![Position::default(); lanes.len()],
            lanes,
            budget,
            steps: tokens.div_ceil(per_step),
            epoch: 0,
            step: 0,
            loader: None,
            config,
        })
    }

    /// Stands before the first step of epoch `epoch` if it is later than the
    /// session's epoch, and does nothing otherwise. Trainers call this at the
    /// start of each epoch, also the one they resumed into, and a checkpoint
    /// after an epoch's last step is already at the next epoch. Going back
    /// would repeat data with no error; open a new session to do that.
    ///
    /// Right after the last step of an epoch, the next epoch is where the
    /// session goes anyway, and the lanes run on into it. Any other later
    /// epoch is a jump: the lanes start at its first document.
    pub fn set_epoch(&mut self, epoch: u64) {
        if self.ended() && epoch == self.epoch + 1 {
            (self.epoch, self.step) = (epoch, 0);
        } else if epoch > self.epoch {
            let start = Position {
                epoch,
                doc: 0,
                offset: 0,
            };
            (self.epoch, self.step) = (epoch, 0);
            self.start = vec![start; self.lanes.len()];
            self.loader = None;
        }
    }

    /// Stands where `checkpoint` was taken. It must come from the same
    /// config and dataset, and hold this GPU's lanes: a global checkpoint
    /// works on any number of GPUs. It makes the loader at once, so it
    /// succeeds only if the run can go on from there; on an error the
    /// session stays where it was, and `checkpoint()` still gives that.
    pub async fn resume(&mut self, checkpoint: &Checkpoint) -> Result<(), Error> {
        let positions = (checkpoint.check(&self.config, self.fingerprint))
            .and_then(|()| checkpoint.positions(self.lanes.clone()))
            .map_err(Error::Checkpoint)?;
        if checkpoint.step >= self.steps {
            let why = format!(
                "step {} is past the epoch's {} steps",
                checkpoint.step, self.steps
            );
            return Err(Error::Checkpoint(checkpoint::Error::Range(why)));
        }
        // Drop the old loader first, so its row groups are freed before the
        // new one reads. The session keeps its position meanwhile, so an
        // error leaves it where it was.
        self.park();
        let loader = self.load(&positions).map_err(|e| match e {
            // Only a resumed checkpoint gives a position that no run makes.
            load::Error::State(why) => Error::Checkpoint(checkpoint::Error::Range(why)),
            e => Error::Load(e),
        })?;
        (self.epoch, self.step) = (checkpoint.epoch, checkpoint.step);
        (self.start, self.loader) = (positions, Some(loader));
        Ok(())
    }

    /// The next step: one sequence of L tokens for each of this GPU's lanes,
    /// in lane order, written into `out`. True for a step; false after the
    /// epoch's last step, and the session then stands before the first step
    /// of the next. After an error, the next call retries the same step.
    pub async fn next(&mut self, out: &mut [u32]) -> Result<bool, Error> {
        if self.ended() {
            (self.epoch, self.step) = (self.epoch + 1, 0);
            return Ok(false);
        }
        let loader = match &mut self.loader {
            Some(loader) => loader,
            None => {
                let loader = self.load(&self.start).map_err(Error::Load)?;
                self.loader.insert(loader)
            }
        };
        if let Err(e) = loader.next(out).await {
            self.park();
            return Err(Error::Load(e));
        }
        self.step += 1;
        Ok(true)
    }

    /// This GPU's part of a checkpoint. `Checkpoint::merge` joins the parts
    /// of all GPUs into one that resumes on any number of GPUs. After the
    /// last step of an epoch it is the start of the next epoch, also before
    /// the call of `next` that returns false, so the parts of all GPUs agree.
    pub fn checkpoint(&self) -> Checkpoint {
        let (epoch, step) = match self.ended() {
            true => (self.epoch + 1, 0),
            false => (self.epoch, self.step),
        };
        Checkpoint {
            config: self.config.clone(),
            fingerprint: self.fingerprint,
            epoch,
            step,
            first_lane: self.lanes.start,
            lanes: self.positions(),
        }
    }

    /// True after the last step of the epoch, until the call of `next` that
    /// returns false moves the session to the next epoch.
    pub fn ended(&self) -> bool {
        self.step == self.steps
    }

    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    /// S: the steps of each epoch.
    pub fn steps(&self) -> u64 {
        self.steps
    }

    pub fn lanes(&self) -> Range<u32> {
        self.lanes.clone()
    }

    /// The bytes of the token chunks this session fetched from storage since
    /// `open`, in every epoch.
    pub fn bytes_read(&self) -> u64 {
        self.reader.bytes_read()
    }

    /// The counter of `bytes_read`, to read without the session.
    pub fn bytes_counter(&self) -> Arc<std::sync::atomic::AtomicU64> {
        self.reader.bytes_counter()
    }

    fn positions(&self) -> Vec<Position> {
        match &self.loader {
            Some(loader) => loader.positions(),
            None => self.start.clone(),
        }
    }

    /// Drops the loader and keeps its lanes' positions in `start`.
    fn park(&mut self) {
        if let Some(loader) = self.loader.take() {
            self.start = loader.positions();
        }
    }

    fn load(&self, at: &[Position]) -> Result<Loader, load::Error> {
        let (orders, reader, index) =
            (self.orders.clone(), self.reader.clone(), self.index.clone());
        Loader::new(orders, reader, index, self.lanes(), at, self.budget)
    }
}

/// The most lanes: far past any real global batch. Each GPU keeps a state
/// for each of its lanes, so a larger value could only exhaust memory.
const MAX_LANES: u32 = 1 << 24;

/// The most tokens in one step of one GPU, 4 GiB of output: far past any
/// real batch, and more could only exhaust memory.
const MAX_STEP_TOKENS: u64 = 1 << 30;

/// Refuses the settings that the order and the packer would panic on, and
/// sizes that could only exhaust memory.
fn validate(c: &Config, rank: u32, world: u32) -> Result<(), Error> {
    let why = if c.lanes == 0 || c.window == 0 {
        "lanes and window must be positive".to_string()
    } else if c.seq_len == 0 {
        "seq_len must be positive".to_string()
    } else if world == 0 || !c.lanes.is_multiple_of(world) {
        format!("world {world} must divide {} lanes", c.lanes)
    } else if rank >= world {
        format!("rank {rank} out of range for world {world}")
    } else if c.lanes > MAX_LANES {
        format!("lanes {} is more than {MAX_LANES}", c.lanes)
    } else if u64::from(c.lanes / world) * u64::from(c.seq_len) > MAX_STEP_TOKENS {
        format!("one step of one GPU holds more than {MAX_STEP_TOKENS} tokens")
    } else {
        return Ok(());
    };
    Err(Error::Config(why))
}

#[cfg(test)]
mod tests;
