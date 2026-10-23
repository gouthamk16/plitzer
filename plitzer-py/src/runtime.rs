//! The one tokio runtime of the process, and how Python waits on it.

use std::sync::OnceLock;
use std::time::Duration;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use tokio::runtime::Runtime;

/// The runtime and the PID of the process that made it.
static RUNTIME: OnceLock<(Runtime, u32)> = OnceLock::new();

/// How long Python waits without a signal check. Ctrl-C takes at most this.
const SLICE: Duration = Duration::from_millis(100);

/// The runtime, refused in a forked child: the child has none of the
/// runtime's worker threads, so a wait in it would never end.
fn runtime() -> PyResult<&'static Runtime> {
    let (runtime, pid) = RUNTIME.get_or_init(|| {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .thread_name("plitzer")
            .build()
            .expect("cannot start the plitzer runtime threads");
        (runtime, std::process::id())
    });
    if *pid != std::process::id() {
        return Err(PyRuntimeError::new_err(
            "plitzer cannot run in a forked process; make the Loader in the \
             process that uses it, and use DataLoader(num_workers=0)",
        ));
    }
    Ok(runtime)
}

/// Runs `fut` to its end without the GIL. Between slices of `SLICE` it
/// checks for signals, so Ctrl-C raises `KeyboardInterrupt` and drops `fut`.
pub fn block_on<F: Future + Send>(py: Python<'_>, fut: F) -> PyResult<F::Output>
where
    F::Output: Send,
{
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
