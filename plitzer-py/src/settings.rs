//! The `Loader` settings from Python: integers, the location and the cache.

use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::path::PathBuf;
use std::sync::Arc;

use object_store::ObjectStore;
use object_store::path::Path;
use pyo3::prelude::*;

use crate::errors::{ConfigError, store_error};

/// `value` as setting `name`, or `ConfigError` if it does not fit, such as
/// a negative `world`.
pub fn int<T: TryFrom<i128>>(name: &str, value: i128) -> PyResult<T> {
    T::try_from(value)
        .map_err(|_| ConfigError::new_err(format!("{name} = {value} is out of range")))
}

/// The store and prefix of `location`, and the index cache: `cache`, else
/// the default.
pub fn locate(
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
    let cache = match cache {
        Some(cache) => cache,
        None => default_cache((&at.id, column)).ok_or_else(|| {
            ConfigError::new_err(
                "no cache folder: set HOME, XDG_CACHE_HOME or PLITZER_CACHE_DIR, or pass cache=",
            )
        })?,
    };
    Ok((at.store, at.prefix, cache))
}

/// `$PLITZER_CACHE_DIR`, else the user cache folder, then one file per
/// `key`. `DefaultHasher` can change with Rust; that only rebuilds the index
/// once. With no user folder, there is no default: a shared temp folder
/// would let other users break or poison the index.
fn default_cache(key: impl Hash) -> Option<PathBuf> {
    // An empty variable counts as unset, as the XDG spec says.
    let var = |name: &str| {
        let value = std::env::var_os(name).filter(|v| !v.is_empty());
        value.map(PathBuf::from)
    };
    let user = || {
        let dir = var("XDG_CACHE_HOME")
            .or_else(|| var("LOCALAPPDATA"))
            .or_else(|| var("HOME").map(|h| h.join(".cache")));
        dir.map(|d| d.join("plitzer"))
    };
    let dir = var("PLITZER_CACHE_DIR").or_else(user)?;
    let mut hasher = DefaultHasher::new();
    key.hash(&mut hasher);
    Some(dir.join(format!("{:016x}.idx", hasher.finish())))
}
