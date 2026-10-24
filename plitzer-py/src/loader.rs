//! `Loader`: one GPU's `Session`, driven from Python.

use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::mem::ManuallyDrop;
use std::path::PathBuf;
use std::sync::Arc;

use object_store::ObjectStore;
use object_store::path::Path;

use plitzer::checkpoint::{Checkpoint, Config};
use plitzer::pack::Packing;
use plitzer::session::Session;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyRange};

use crate::array::Batch;
use crate::errors::{checkpoint_error, session_error, store_error};
use crate::runtime::block_on;

/// The lanes of GPU `rank` of `world`, from one epoch to the next. Iterate
/// it for the rest of the current epoch; the next iteration gives the next.
#[pyclass(module = "plitzer._plitzer")]
pub struct Loader {
    session: ManuallyDrop<Session>,
    seq_len: usize,
    /// Set when the epoch ended, so `__next__` keeps raising `StopIteration`
    /// as Python iterators do, until `__iter__` starts the next epoch.
    exhausted: bool,
    /// The process that made the loader. See `Drop`.
    pid: u32,
}

impl Loader {
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
            // SAFETY: the session is not used after this.
            unsafe { ManuallyDrop::drop(&mut self.session) }
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
    // Python callers pass these as keywords, so a struct would only add a step.
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        location: &str,
        seed: u64,
        lanes: u32,
        window: u32,
        seq_len: u32,
        docs: u32,
        cap: u32,
        pad: u32,
        budget: u64,
        rank: u32,
        world: u32,
        column: String,
        cache: Option<PathBuf>,
        storage_options: Option<HashMap<String, String>>,
    ) -> PyResult<Self> {
        let (store, prefix, cache) = locate(location, storage_options, &column, cache)?;
        let packing = Packing {
            seq_len,
            docs,
            cap,
            pad,
        };
        let config = Config {
            column,
            seed,
            lanes,
            window,
            packing,
        };
        let open = Session::open(store, prefix, &cache, config, rank, world, budget);
        let session = block_on(py, open)?.map_err(session_error)?;
        Ok(Loader {
            session: ManuallyDrop::new(session),
            seq_len: seq_len as usize,
            exhausted: false,
            pid: std::process::id(),
        })
    }

    /// The rest of the current epoch. A loader whose last step is taken, but
    /// which has not raised `StopIteration` yet, goes to the next epoch.
    fn __iter__(mut slf: PyRefMut<'_, Self>) -> PyResult<PyRefMut<'_, Self>> {
        slf.check_process()?;
        slf.exhausted = false;
        if slf.session.ended() {
            let next = slf.session.epoch() + 1;
            slf.session.set_epoch(next);
        }
        Ok(slf)
    }

    /// The next step, or `StopIteration` after the last step of the epoch.
    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<Batch>> {
        self.check_process()?;
        if self.exhausted {
            return Ok(None);
        }
        let mut tokens = vec![0; self.session.lanes().len() * self.seq_len];
        let real = block_on(py, self.session.next(&mut tokens))?.map_err(session_error)?;
        self.exhausted = real.is_none();
        Ok(real.map(|lengths| Batch::new(tokens, lengths, self.seq_len)))
    }

    /// Goes to the start of `epoch` if it is later than the current one. An
    /// ended iterator stays ended when the call changes nothing.
    fn set_epoch(&mut self, epoch: u64) -> PyResult<()> {
        self.check_process()?;
        if epoch > self.session.epoch() {
            self.exhausted = false;
            self.session.set_epoch(epoch);
        }
        Ok(())
    }

    /// This GPU's part of a checkpoint, as bytes. `merge` joins the parts of
    /// all GPUs into one that resumes on any number of GPUs.
    fn checkpoint<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.session.checkpoint().to_bytes())
    }

    fn resume(&mut self, checkpoint: &[u8]) -> PyResult<()> {
        self.check_process()?;
        let checkpoint = Checkpoint::from_bytes(checkpoint).map_err(checkpoint_error)?;
        self.session.resume(&checkpoint).map_err(session_error)?;
        self.exhausted = false;
        Ok(())
    }

    #[getter]
    fn epoch(&self) -> u64 {
        self.session.epoch()
    }

    #[getter]
    fn lanes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyRange>> {
        let lanes = self.session.lanes();
        PyRange::new(py, lanes.start as isize, lanes.end as isize)
    }
}

/// The store and prefix of `location`, and the index cache: `cache`, else
/// the default.
fn locate(
    location: &str,
    storage_options: Option<HashMap<String, String>>,
    column: &str,
    cache: Option<PathBuf>,
) -> PyResult<(Arc<dyn ObjectStore>, Path, PathBuf)> {
    let mut options: Vec<_> = storage_options.unwrap_or_default().into_iter().collect();
    options.sort();
    let at = plitzer::store::open(location, &options).map_err(store_error)?;
    // The dataset's id, not the spelling: `data` from two folders, or
    // `s3://b/p/` and `s3://b/p`, must not share or split an index.
    let cache = cache.unwrap_or_else(|| default_cache((&at.id, column)));
    Ok((at.store, at.prefix, cache))
}

/// `$PLITZER_CACHE_DIR`, else the user cache folder, then one file per
/// `key`. `DefaultHasher` can change with Rust; that only rebuilds the index
/// once.
fn default_cache(key: impl Hash) -> PathBuf {
    // An empty variable counts as unset, as the XDG spec says.
    let var = |name: &str| {
        let value = std::env::var_os(name).filter(|v| !v.is_empty());
        value.map(PathBuf::from)
    };
    let dir = var("PLITZER_CACHE_DIR").unwrap_or_else(|| {
        let user = var("XDG_CACHE_HOME")
            .or_else(|| var("LOCALAPPDATA"))
            .or_else(|| var("HOME").map(|h| h.join(".cache")))
            .unwrap_or_else(std::env::temp_dir);
        user.join("plitzer")
    });
    let mut hasher = DefaultHasher::new();
    key.hash(&mut hasher);
    dir.join(format!("{:016x}.idx", hasher.finish()))
}
