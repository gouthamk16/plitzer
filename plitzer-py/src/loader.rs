//! `Loader`: one GPU's `Session`, driven from Python.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Mutex, MutexGuard, TryLockError};

use plitzer::checkpoint::Checkpoint;
use plitzer::session::Session;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyRange};

use crate::array::Batch;
use crate::errors::{checkpoint_error, session_error};
use crate::runtime::block_on;
use crate::settings::{Ints, config, int, locate};

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
    /// Taken only by `Drop`, in a forked child.
    session: Option<Session>,
    /// Set when the epoch ended, so `__next__` keeps raising `StopIteration`
    /// as Python iterators do, until `__iter__` starts the next epoch.
    exhausted: bool,
}

impl State {
    fn session(&mut self) -> &mut Session {
        self.session.as_mut().expect("only Drop takes the session")
    }

    fn checkpoint(&self) -> Checkpoint {
        self.session
            .as_ref()
            .expect("only Drop takes the session")
            .checkpoint()
    }
}

impl Loader {
    fn from_session(session: Session) -> Self {
        let seq_len = session.checkpoint().config.packing.seq_len as usize;
        let state = State {
            session: Some(session),
            exhausted: false,
        };
        Loader {
            snapshot: Mutex::new(state.checkpoint()),
            state: Mutex::new(state),
            seq_len,
            pid: std::process::id(),
        }
    }

    /// The session, if no other call is using it. A step holds it while it
    /// waits, so a second caller gets an error here, never a deadlock.
    fn state(&self) -> PyResult<MutexGuard<'_, State>> {
        self.check_process()?;
        match self.state.try_lock() {
            Ok(state) => Ok(state),
            // A panic left the lanes part way through a step: refuse them,
            // and keep the snapshot, which is still the last good state.
            Err(TryLockError::Poisoned(_)) => Err(PyRuntimeError::new_err(
                "an earlier call failed inside a step; make a new Loader and resume it from checkpoint()",
            )),
            Err(TryLockError::WouldBlock) => Err(PyRuntimeError::new_err(
                "another call is using this Loader; read checkpoint() and epoch at any time",
            )),
        }
    }

    fn save(&self, state: &State) {
        let checkpoint = state.checkpoint();
        *self.snapshot.lock().unwrap_or_else(|e| e.into_inner()) = checkpoint;
    }

    /// The snapshot, refused in a forked child: a parent thread may have
    /// held its lock at the fork, and no thread in the child releases it.
    fn snapshot(&self) -> PyResult<MutexGuard<'_, Checkpoint>> {
        self.check_process()?;
        Ok(self.snapshot.lock().unwrap_or_else(|e| e.into_inner()))
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
        if self.pid != std::process::id() {
            let state = self.state.get_mut().unwrap_or_else(|e| e.into_inner());
            std::mem::forget(state.session.take());
        }
    }
}

#[pymethods]
impl Loader {
    #[new]
    #[pyo3(signature = (
        location, *, seed, lanes, window, seq_len, docs, cap, pad, budget,
        rank=None, world=None, column="tokens".to_string(), cache=None, storage_options=None,
    ))]
    // Python callers pass these as keywords, so a struct would only add a
    // step. Integers come as Python objects, so any value that does not fit,
    // also one past i128, is a `ConfigError`.
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        location: &str,
        seed: &Bound<'_, PyAny>,
        lanes: &Bound<'_, PyAny>,
        window: &Bound<'_, PyAny>,
        seq_len: &Bound<'_, PyAny>,
        docs: &Bound<'_, PyAny>,
        cap: &Bound<'_, PyAny>,
        pad: &Bound<'_, PyAny>,
        budget: &Bound<'_, PyAny>,
        rank: Option<&Bound<'_, PyAny>>,
        world: Option<&Bound<'_, PyAny>>,
        column: String,
        cache: Option<PathBuf>,
        storage_options: Option<HashMap<String, String>>,
    ) -> PyResult<Self> {
        let (store, prefix, cache) = locate(location, storage_options, &column, cache)?;
        let ints = Ints {
            seed,
            lanes,
            window,
            seq_len,
            docs,
            cap,
            pad,
        };
        let config = config(column, ints)?;
        let rank = rank.map_or(Ok(0), |r| int("rank", r))?;
        let world = world.map_or(Ok(1), |w| int("world", w))?;
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
        Ok(Loader::from_session(session))
    }

    /// The rest of the current epoch. A loader whose last step is taken, but
    /// which has not raised `StopIteration` yet, goes to the next epoch.
    fn __iter__(slf: PyRef<'_, Self>) -> PyResult<PyRef<'_, Self>> {
        {
            let mut state = slf.state()?;
            state.exhausted = false;
            if state.session().ended() {
                let next = state.session().epoch().saturating_add(1);
                state.session().set_epoch(next);
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
        let mut tokens = vec![0; state.session().lanes().len() * self.seq_len];
        let real = block_on(py, state.session().next(&mut tokens))?;
        // Also after an error: the session then stands before the failed step.
        self.save(&state);
        let real = real.map_err(session_error)?;
        state.exhausted = real.is_none();
        let batch = real.map(|lengths| Batch::new(py, tokens, lengths, self.seq_len));
        batch.transpose()
    }

    /// Goes to the start of `epoch` if it is later than the current one. An
    /// ended iterator stays ended when the call changes nothing.
    fn set_epoch(&self, epoch: &Bound<'_, PyAny>) -> PyResult<()> {
        let epoch: u64 = int("epoch", epoch)?;
        let mut state = self.state()?;
        let at = state.session().epoch();
        state.session().set_epoch(epoch);
        // `epoch == at` after `StopIteration`: the session already stands at
        // the start of `epoch`, so the iterator opens for it.
        if epoch >= at {
            state.exhausted = false;
        }
        self.save(&state);
        Ok(())
    }

    /// This GPU's part of a checkpoint, as bytes: the state before the next
    /// step. It works during a step too, such as from a signal handler.
    /// `merge` joins the parts of all GPUs into one that resumes on any
    /// number of GPUs.
    fn checkpoint<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        // The snapshot lock is never held while Python runs: Python can
        // release the GIL, and `save` takes the lock while holding it.
        let bytes = self.snapshot()?.to_bytes();
        Ok(PyBytes::new(py, &bytes))
    }

    /// Stands where `checkpoint` was taken, and reads what the next step
    /// needs. On an error nothing changes, also not `checkpoint()`.
    fn resume(&self, py: Python<'_>, checkpoint: &[u8]) -> PyResult<()> {
        let mut state = self.state()?;
        let checkpoint = Checkpoint::from_bytes(checkpoint).map_err(checkpoint_error)?;
        block_on(py, state.session().resume(&checkpoint))?.map_err(session_error)?;
        state.exhausted = false;
        self.save(&state);
        Ok(())
    }

    /// The epoch of the next step. After the last step of an epoch it is
    /// the next epoch, as in `checkpoint()`.
    #[getter]
    fn epoch(&self) -> PyResult<u64> {
        Ok(self.snapshot()?.epoch)
    }

    #[getter]
    fn lanes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyRange>> {
        let (start, len) = {
            let snapshot = self.snapshot()?;
            (snapshot.first_lane as isize, snapshot.lanes.len() as isize)
        };
        PyRange::new(py, start, start + len)
    }
}
