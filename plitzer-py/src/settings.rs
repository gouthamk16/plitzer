//! The `Loader` settings from Python: integers, the location and the cache.

use std::hash::{DefaultHasher, Hash, Hasher};
use std::path::PathBuf;
use std::sync::Arc;

use object_store::ObjectStore;
use object_store::path::Path;
use plitzer::checkpoint::Config;
use plitzer::pack::Packing;
use pyo3::conversion::FromPyObjectOwned;
use pyo3::exceptions::PyOverflowError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyMapping};
use url::Url;

use crate::errors::{ConfigError, store_error};

/// `value` as setting `name`, or `ConfigError`: for a value that is not an
/// integer, such as `4.0` or `True`, or one that does not fit, such as a
/// negative `world` or `2**200`. An integer is anything with `__index__`,
/// such as `np.int64`, except a `bool`.
pub fn int<'py, T: FromPyObjectOwned<'py>>(name: &str, value: &Bound<'py, PyAny>) -> PyResult<T> {
    // Python looks up `__index__` on the type, so the class `np.int64`,
    // which has it as an attribute, is not an integer.
    if value.is_instance_of::<PyBool>() || !value.get_type().hasattr("__index__")? {
        let kind = value.get_type().name()?;
        return Err(ConfigError::new_err(format!(
            "{name} must be an int, not {kind}"
        )));
    }
    // Only an overflow means the value does not fit; any other error comes
    // from the object's own `__index__`, so it goes to the caller as it is.
    value.extract::<T>().map_err(|e| {
        let e: PyErr = e.into();
        if e.is_instance_of::<PyOverflowError>(value.py()) {
            ConfigError::new_err(format!("{name} = {value} is out of range"))
        } else {
            e
        }
    })
}

/// Where a `Loader` runs, and its byte budget.
pub struct Placement {
    pub rank: u32,
    pub world: u32,
    pub budget: u64,
}

/// `rank`, `world` and `budget`, with the defaults of a single GPU. A
/// `world` above 1 needs a `rank`: with rank 0 on each GPU, all GPUs would
/// train on the same lanes with no error.
pub fn placement(
    rank: Option<&Bound<'_, PyAny>>,
    world: Option<&Bound<'_, PyAny>>,
    budget: &Bound<'_, PyAny>,
) -> PyResult<Placement> {
    let world = world.map_or(Ok(1), |w| int("world", w))?;
    let rank = match rank {
        Some(rank) => int("rank", rank)?,
        None if world <= 1 => 0,
        None => {
            return Err(ConfigError::new_err(format!(
                "world = {world} needs a rank"
            )));
        }
    };
    let budget = int("budget", budget)?;
    Ok(Placement {
        rank,
        world,
        budget,
    })
}

/// The integer settings of a `Loader`, as Python gave them. Named fields,
/// so a call cannot swap two of them.
pub struct Ints<'a, 'py> {
    pub seed: &'a Bound<'py, PyAny>,
    pub lanes: &'a Bound<'py, PyAny>,
    pub window: &'a Bound<'py, PyAny>,
    pub seq_len: &'a Bound<'py, PyAny>,
    pub docs: &'a Bound<'py, PyAny>,
    pub cap: &'a Bound<'py, PyAny>,
    pub pad: &'a Bound<'py, PyAny>,
}

/// The config from Python integers, each checked to fit.
pub fn config(column: String, i: Ints<'_, '_>) -> PyResult<Config> {
    let packing = Packing {
        seq_len: int("seq_len", i.seq_len)?,
        docs: int("docs", i.docs)?,
        cap: int("cap", i.cap)?,
        pad: pad(i.pad)?,
    };
    Ok(Config {
        column,
        seed: int("seed", i.seed)?,
        lanes: int("lanes", i.lanes)?,
        window: int("window", i.window)?,
        packing,
    })
}

/// The pad token as a `u32`. Batches reach Python as int32, so a negative
/// pad, such as -1, means the token with the same bits.
fn pad(value: &Bound<'_, PyAny>) -> PyResult<u32> {
    match value.extract::<i32>() {
        Ok(negative) if negative < 0 => Ok(negative as u32),
        _ => int("pad", value),
    }
}

/// The store and prefix of `location`, and the index cache: `cache`, else
/// the default.
pub fn locate(
    location: PathBuf,
    storage_options: Option<Bound<'_, PyMapping>>,
    column: &str,
    cache: Option<PathBuf>,
) -> PyResult<(Arc<dyn ObjectStore>, Path, PathBuf)> {
    let options = options(storage_options)?;
    let at = plitzer::store::open(&text(location)?, &options).map_err(store_error)?;
    // The dataset's id, not the spelling: `data` from two folders, or
    // `s3://b/p/` and `s3://b/p`, must not share or split an index.
    let cache = match cache {
        Some(cache) => cache,
        None => default_cache((&at.id, column)).ok_or_else(|| {
            ConfigError::new_err(
                "no cache folder: set HOME, XDG_CACHE_HOME or PLITZER_CACHE_DIR, or pass cache= with the path of the index file",
            )
        })?,
    };
    Ok((at.store, at.prefix, cache))
}

/// `location` as text for `store::open`. A local path that is not UTF-8
/// becomes a `file://` URL, which percent-encodes its bytes.
fn text(location: PathBuf) -> PyResult<String> {
    if let Some(text) = location.to_str() {
        return Ok(text.to_owned());
    }
    let url = std::path::absolute(&location).ok().map(Url::from_file_path);
    let url = url.and_then(Result::ok);
    url.map(String::from)
        .ok_or_else(|| ConfigError::new_err(format!("{location:?} is not a usable path")))
}

/// The storage options in a fixed order, each value as text. A `bool`, as
/// in `allow_http=True`, becomes `true` or `false`.
fn options(given: Option<Bound<'_, PyMapping>>) -> PyResult<Vec<(String, String)>> {
    let mut options = Vec::new();
    let items = given.map(|m| m.items()).transpose()?;
    for item in items.iter().flatten() {
        let (key, value): (Bound<'_, PyAny>, Bound<'_, PyAny>) = item.extract()?;
        let key: String = (key.extract()).map_err(|_| {
            ConfigError::new_err(format!("storage option {key} must have a str name"))
        })?;
        // Also `np.bool_`, which PyO3 extracts as a bool.
        let text = match value.extract::<bool>() {
            Ok(flag) => flag.to_string(),
            _ => value.extract::<String>().map_err(|_| {
                let kind = value
                    .get_type()
                    .name()
                    .map_or("?".into(), |n| n.to_string());
                ConfigError::new_err(format!("storage option {key:?} must be a str, not {kind}"))
            })?,
        };
        options.push((key, text));
    }
    options.sort();
    Ok(options)
}

/// `$PLITZER_CACHE_DIR`, else the user cache folder, then one file per
/// `key`. `DefaultHasher` can change with Rust; that only rebuilds the index
/// once. With no user folder, there is no default: a shared temp folder
/// would let other users break or poison the index.
fn default_cache(key: impl Hash) -> Option<PathBuf> {
    // An empty or relative variable counts as unset, as the XDG spec says:
    // a relative folder would differ for each working directory.
    let var = |name: &str| {
        let value = std::env::var_os(name).map(PathBuf::from);
        value.filter(|v| v.is_absolute())
    };
    let user = || {
        let dir = var("XDG_CACHE_HOME")
            .or_else(|| var("LOCALAPPDATA"))
            .or_else(|| var("HOME").map(|h| h.join(".cache")));
        dir.map(|d| d.join("plitzer"))
    };
    // plitzer's own variable is used as given, also when it is relative.
    let own = std::env::var_os("PLITZER_CACHE_DIR").filter(|v| !v.is_empty());
    let dir = own.map(PathBuf::from).or_else(user)?;
    let mut hasher = DefaultHasher::new();
    key.hash(&mut hasher);
    Some(dir.join(format!("{:016x}.idx", hasher.finish())))
}
