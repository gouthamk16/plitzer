//! `Loader`: one GPU's `Session`, driven from Python.

use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::path::PathBuf;

use plitzer::checkpoint::{Checkpoint, Config};
use plitzer::pack::Packing;
use plitzer::session::Session;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyRange};

use crate::array::Batch;
use crate::errors::{checkpoint_error, session_error, store_error};
use crate::runtime::block_on;

/// The lanes of GPU `rank` of `world`, from one epoch to the next. Iterate
/// it for the rest of the current epoch; the next iteration gives the next.
#[pyclass(module = "plitzer._plitzer")]
pub struct Loader {
    session: Session,
    seq_len: usize,
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
        let mut options: Vec<_> = storage_options.unwrap_or_default().into_iter().collect();
        options.sort();
        let (store, prefix) = plitzer::store::open(location, &options).map_err(store_error)?;
        // The resolved store, not the spelling: `data` from two folders, or
        // `s3://b/p/` and `s3://b/p`, must not share or split an index.
        let resolved = format!("{store}/{prefix}");
        let cache = cache.unwrap_or_else(|| default_cache(&resolved, &column));
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
            session,
            seq_len: seq_len as usize,
        })
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// The next step, or `StopIteration` after the last step of the epoch.
    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<Batch>> {
        let mut tokens = vec![0; self.session.lanes().len() * self.seq_len];
        let real = block_on(py, self.session.next(&mut tokens))?.map_err(session_error)?;
        Ok(real.map(|lengths| Batch::new(tokens, lengths, self.seq_len)))
    }

    fn set_epoch(&mut self, epoch: u64) {
        self.session.set_epoch(epoch);
    }

    /// This GPU's part of a checkpoint, as bytes. `merge` joins the parts of
    /// all GPUs into one that resumes on any number of GPUs.
    fn checkpoint<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.session.checkpoint().to_bytes())
    }

    fn resume(&mut self, checkpoint: &[u8]) -> PyResult<()> {
        let checkpoint = Checkpoint::from_bytes(checkpoint).map_err(checkpoint_error)?;
        self.session.resume(&checkpoint).map_err(session_error)
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

/// `$PLITZER_CACHE_DIR`, else the user cache folder, then one file per
/// resolved location and column. `DefaultHasher` can change with Rust; that only
/// rebuilds the index once.
fn default_cache(location: &str, column: &str) -> PathBuf {
    let var = |name: &str| std::env::var_os(name).map(PathBuf::from);
    let dir = var("PLITZER_CACHE_DIR").unwrap_or_else(|| {
        let user = var("XDG_CACHE_HOME")
            .or_else(|| var("LOCALAPPDATA"))
            .or_else(|| var("HOME").map(|h| h.join(".cache")))
            .unwrap_or_else(std::env::temp_dir);
        user.join("plitzer")
    });
    let mut hasher = DefaultHasher::new();
    (location, column).hash(&mut hasher);
    dir.join(format!("{:016x}.idx", hasher.finish()))
}
