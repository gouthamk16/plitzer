//! The `plitzer._plitzer` extension module.

mod array;
mod dlpack;
mod errors;
mod loader;
mod runtime;

use plitzer::checkpoint::Checkpoint;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

/// One global checkpoint from the parts of all GPUs, in any order.
#[pyfunction]
fn merge<'py>(py: Python<'py>, parts: Vec<Vec<u8>>) -> PyResult<Bound<'py, PyBytes>> {
    let parts = (parts.iter().map(|p| Checkpoint::from_bytes(p)))
        .collect::<Result<Vec<_>, _>>()
        .map_err(errors::checkpoint_error)?;
    let merged = Checkpoint::merge(parts).map_err(errors::checkpoint_error)?;
    Ok(PyBytes::new(py, &merged.to_bytes()))
}

#[pymodule]
mod _plitzer {
    #[pymodule_export]
    use super::array::{Array, Batch};
    #[pymodule_export]
    use super::errors::{CheckpointError, ConfigError, DataError, PlitzerError};
    #[pymodule_export]
    use super::loader::Loader;
    #[pymodule_export]
    use super::merge;
}
