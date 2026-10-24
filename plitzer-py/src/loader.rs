//! `Loader`: one GPU's `Session`, driven from Python.

use std::collections::HashMap;
use std::mem::ManuallyDrop;
use std::path::PathBuf;
use std::sync::{Mutex, MutexGuard, TryLockError};

use plitzer::checkpoint::{Checkpoint, Config};
use plitzer::pack::Packing;
use plitzer::session::Session;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyRange};

use crate::array::Batch;
use crate::errors::{checkpoint_error, session_error};
use crate::runtime::block_on;
use crate::settings::{int, locate};

/// The lanes of GPU `rank` of `world`, from one epoch to the next. Iterate
/// it for the rest of the current epoch; the next iteration gives the next.
#[pyclass(module = "plitzer._plitzer", frozen)]
pub struct Loader {
    /// Held for the whole of a step, also while it waits without the GIL.
    state: Mutex<State>,
    /// The checkpoint before the next step. `checkpoint`, `epoch` and `lanes`
    /// read it without `state`, so a signal handler or another thread can
    /// save a checkpoint while a step waits on a read.
    snapshot: Mutex<Checkpoint>,
    seq_len: usize,
    /// The process that made the loader. See `check_process` and `Drop`.
    pid: u32,
}

struct State {
    session: ManuallyDrop<Session>,
    /// Set when the epoch ended, so `__next__` keeps raising `StopIteration`
    /// as Python iterators do, until `__iter__` starts the next epoch.
    exhausted: bool,
}

impl Loader {
    /// The session, if no other call is using it. A step holds it while it
    /// waits, so a second caller gets an error here, never a deadlock.
    fn state(&self) -> PyResult<MutexGuard<'_, State>> {
        self.check_process()?;
        match self.state.try_lock() {
            Ok(state) => Ok(state),
            // A panic in an earlier call is already a Python exception.
            Err(TryLockError::Poisoned(e)) => Ok(e.into_inner()),
            Err(TryLockError::WouldBlock) => Err(PyRuntimeError::new_err(
                "another call is using this Loader; read checkpoint() and epoch at any time",
            )),
        }
    }

    fn save(&self, state: &State) {
        *self.snapshot() = state.session.checkpoint();
    }

    fn snapshot(&self) -> MutexGuard<'_, Checkpoint> {
        self.snapshot.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Refuses a loader copied into a forked child. Its session holds tasks
    /// of the parent's runtime, which no thread in the child runs, so a step
    /// would wait forever and a new epoch would drop tasks it cannot stop.
    fn check_process(&self) -> PyResult<()> {
        if self.pid == std::process::id() {
            return Ok(());
        }
        Err(PyRuntimeError::new_err(
            "this Loader was made in the parent of a forked process; make a \
             new Loader in this process, and use DataLoader(num_workers=0)",
        ))
    }
}

impl Drop for Loader {
    fn drop(&mut self) {
        // A forked child has none of the runtime's threads, and a lock they
        // held at the fork stays held. Dropping the session aborts runtime
        // tasks, which could wait on such a lock, so the child leaks it.
        if self.pid == std::process::id() {
            let state = self.state.get_mut().unwrap_or_else(|e| e.into_inner());
            // SAFETY: the session is not used after this.
            unsafe { ManuallyDrop::drop(&mut state.session) }
        }
    }
}

#[pymethods]
impl Loader {
    #[new]
    #[pyo3(signature = (
        location, *, seed, lanes, window, seq_len, docs, cap, pad, budget,
        rank=0, world=1, column="tokens".to_string(), cache=None, storage_options=None,
    ))]
    // Python callers pass these as keywords, so a struct would only add a
    // step. Integers come as i128, so a negative one is a `ConfigError`.
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        location: &str,
        seed: i128,
        lanes: i128,
        window: i128,
        seq_len: i128,
        docs: i128,
        cap: i128,
        pad: i128,
        budget: i128,
        rank: i128,
        world: i128,
        column: String,
        cache: Option<PathBuf>,
        storage_options: Option<HashMap<String, String>>,
    ) -> PyResult<Self> {
        let (store, prefix, cache) = locate(location, storage_options, &column, cache)?;
        let packing = Packing {
            seq_len: int("seq_len", seq_len)?,
            docs: int("docs", docs)?,
            cap: int("cap", cap)?,
            pad: int("pad", pad)?,
        };
        let config = Config {
            column,
            seed: int("seed", seed)?,
            lanes: int("lanes", lanes)?,
            window: int("window", window)?,
            packing,
        };
        let (rank, world) = (int("rank", rank)?, int("world", world)?);
        let open = Session::open(
            store,
            prefix,
            &cache,
            config,
            rank,
            world,
            int("budget", budget)?,
        );
        let session = block_on(py, open)?.map_err(session_error)?;
        let state = State {
            session: ManuallyDrop::new(session),
            exhausted: false,
        };
        Ok(Loader {
            snapshot: Mutex::new(state.session.checkpoint()),
            state: Mutex::new(state),
            seq_len: packing.seq_len as usize,
            pid: std::process::id(),
        })
    }

    /// The rest of the current epoch. A loader whose last step is taken, but
    /// which has not raised `StopIteration` yet, goes to the next epoch.
    fn __iter__(slf: PyRef<'_, Self>) -> PyResult<PyRef<'_, Self>> {
        {
            let mut state = slf.state()?;
            state.exhausted = false;
            if state.session.ended() {
                let next = state.session.epoch() + 1;
                state.session.set_epoch(next);
            }
            slf.save(&state);
        }
        Ok(slf)
    }

    /// The next step, or `StopIteration` after the last step of the epoch.
    fn __next__(&self, py: Python<'_>) -> PyResult<Option<Batch>> {
        let mut state = self.state()?;
        if state.exhausted {
            return Ok(None);
        }
        let mut tokens = vec![0; state.session.lanes().len() * self.seq_len];
        let real = block_on(py, state.session.next(&mut tokens))?;
        // Also after an error: the session then stands before the failed step.
        self.save(&state);
        let real = real.map_err(session_error)?;
        state.exhausted = real.is_none();
        Ok(real.map(|lengths| Batch::new(tokens, lengths, self.seq_len)))
    }

    /// Goes to the start of `epoch` if it is later than the current one. An
    /// ended iterator stays ended when the call changes nothing.
    fn set_epoch(&self, epoch: u64) -> PyResult<()> {
        let mut state = self.state()?;
        if epoch > state.session.epoch() {
            state.exhausted = false;
            state.session.set_epoch(epoch);
            self.save(&state);
        }
        Ok(())
    }

    /// This GPU's part of a checkpoint, as bytes: the state before the next
    /// step. It works during a step too, such as from a signal handler.
    /// `merge` joins the parts of all GPUs into one that resumes on any
    /// number of GPUs.
    fn checkpoint<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.snapshot().to_bytes())
    }

    fn resume(&self, checkpoint: &[u8]) -> PyResult<()> {
        let mut state = self.state()?;
        let checkpoint = Checkpoint::from_bytes(checkpoint).map_err(checkpoint_error)?;
        state.session.resume(&checkpoint).map_err(session_error)?;
        state.exhausted = false;
        self.save(&state);
        Ok(())
    }

    /// The epoch of the next step. After the last step of an epoch it is
    /// the next epoch, as in `checkpoint()`.
    #[getter]
    fn epoch(&self) -> u64 {
        self.snapshot().epoch
    }

    #[getter]
    fn lanes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyRange>> {
        let snapshot = self.snapshot();
        let start = snapshot.first_lane as isize;
        PyRange::new(py, start, start + snapshot.lanes.len() as isize)
    }
}
