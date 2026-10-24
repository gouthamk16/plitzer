//! The one tokio runtime of the process, and how Python waits on it.

use std::sync::Mutex;
use std::time::Duration;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use tokio::runtime::Runtime;

/// The runtime of this process, with the PID it belongs to.
static RUNTIME: Mutex<Option<(u32, &'static Runtime)>> = Mutex::new(None);

/// Worker threads: reads and their decodes, for one GPU's lanes.
const WORKERS: usize = 8;

/// Threads for blocking calls, such as local file reads: twice the reads a
/// loader keeps in flight.
const BLOCKING: usize = 32;

/// How long Python waits without a signal check. Ctrl-C takes at most this.
const SLICE: Duration = Duration::from_millis(100);

/// The runtime of this process. A forked child gets a new one: it has none
/// of the parent runtime's threads. The parent's runtime is leaked, never
/// dropped, since its drop would wait for those threads.
fn runtime() -> PyResult<&'static Runtime> {
    let mut runtime = RUNTIME.lock().unwrap_or_else(|e| e.into_inner());
    let pid = std::process::id();
    match *runtime {
        Some((owner, runtime)) if owner == pid => Ok(runtime),
        _ => {
            // Each GPU's process has its own runtime, and a loader has at most
            // 16 reads in flight: one worker per CPU would start hundreds of
            // threads per process on a large node.
            let cpus = std::thread::available_parallelism().map_or(4, |n| n.get());
            let new = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(cpus.min(WORKERS))
                .max_blocking_threads(BLOCKING)
                .enable_all()
                .thread_name("plitzer")
                .build()
                .map_err(|e| {
                    PyRuntimeError::new_err(format!("cannot start plitzer threads: {e}"))
                })?;
            let new: &'static Runtime = Box::leak(Box::new(new));
            *runtime = Some((pid, new));
            Ok(new)
        }
    }
}

/// Runs `fut` to its end without the GIL. Between slices of `SLICE` it
/// checks for signals, so Ctrl-C raises `KeyboardInterrupt` and drops `fut`.
pub fn block_on<F: Future + Send>(py: Python<'_>, fut: F) -> PyResult<F::Output>
where
    F::Output: Send,
{
    // Tokio refuses, with a panic, to block a thread that runs a runtime,
    // such as a callback from another Rust extension.
    if tokio::runtime::Handle::try_current().is_ok() {
        return Err(PyRuntimeError::new_err(
            "plitzer cannot wait on a thread that runs a tokio runtime; call it from another thread",
        ));
    }
    let runtime = runtime()?;
    let mut fut = std::pin::pin!(fut);
    loop {
        // `timeout` makes its timer at once, so it must be made in the runtime.
        let slice = py
            .detach(|| runtime.block_on(async { tokio::time::timeout(SLICE, fut.as_mut()).await }));
        match slice {
            Ok(out) => return Ok(out),
            Err(_) => py.check_signals()?,
        }
    }
}
