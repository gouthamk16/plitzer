//! The Python exceptions, and which Rust error raises which.

use plitzer::{checkpoint, load, session, store};
use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;

create_exception!(
    plitzer,
    PlitzerError,
    PyException,
    "The base of every plitzer error."
);
create_exception!(
    plitzer,
    ConfigError,
    PlitzerError,
    "A setting that cannot work, such as a world that does not divide lanes or a budget below one step."
);
create_exception!(
    plitzer,
    CheckpointError,
    PlitzerError,
    "A checkpoint that is damaged, or from another setting or dataset."
);
create_exception!(
    plitzer,
    DataError,
    PlitzerError,
    "The data cannot be listed, read or decoded."
);

pub fn session_error(e: session::Error) -> PyErr {
    let message = e.to_string();
    match e {
        session::Error::Config(_) | session::Error::Load(load::Error::Budget { .. }) => {
            ConfigError::new_err(message)
        }
        session::Error::Checkpoint(_) => CheckpointError::new_err(message),
        session::Error::Index(_) | session::Error::Read(_) | session::Error::Load(_) => {
            DataError::new_err(message)
        }
    }
}

pub fn checkpoint_error(e: checkpoint::Error) -> PyErr {
    CheckpointError::new_err(e.to_string())
}

pub fn store_error(e: store::Error) -> PyErr {
    match e {
        store::Error::Store(_) => DataError::new_err(e.to_string()),
        store::Error::Scheme(_) | store::Error::Option(_) => ConfigError::new_err(e.to_string()),
    }
}
