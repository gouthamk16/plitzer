//! The store and prefix of a dataset location, such as a local folder or
//! `s3://bucket/prefix`.

use std::fmt;
use std::sync::Arc;

use object_store::ObjectStore;
#[cfg(feature = "s3")]
use object_store::aws::{AmazonS3Builder, AmazonS3ConfigKey};
use object_store::local::LocalFileSystem;
use object_store::path::Path;
use url::Url;

#[derive(Debug)]
pub enum Error {
    /// The store could not be made, such as for a folder that does not exist.
    Store(object_store::Error),
    /// A URL scheme that this build cannot open.
    Scheme(String),
    /// A storage option that the store does not know, a bad S3 setting, or a
    /// bad location.
    Option(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Store(e) => write!(f, "{e}"),
            Error::Scheme(s) => write!(f, "plitzer cannot open {s:?} locations"),
            Error::Option(why) => write!(f, "{why}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Store(e) => Some(e),
            Error::Scheme(_) | Error::Option(_) => None,
        }
    }
}

/// An opened dataset location.
pub struct Location {
    pub store: Arc<dyn ObjectStore>,
    pub prefix: Path,
    /// The same text for each spelling of one dataset, such as `data/` and
    /// `file:///abs/data`, and different texts for different datasets. It
    /// holds no credentials.
    pub id: String,
}

/// The store and the prefix in it for `location`: a local folder, a
/// `file://` URL, or an `s3://bucket/prefix` URL with the `s3` feature. S3
/// reads `AWS_*` variables from the environment, and `options` (such as
/// `endpoint` or `region`) override them.
pub fn open(location: &str, options: &[(String, String)]) -> Result<Location, Error> {
    // Only `scheme://` is a URL, so `C:\data` and `runs:2024/data` are paths.
    if !location.contains("://") {
        return local(std::path::Path::new(location), options);
    }
    let url = Url::parse(location).map_err(|e| Error::Option(format!("{location:?}: {e}")))?;
    // `?` and `#` may be part of a real folder or key name; dropping what
    // follows them would open another dataset.
    if url.query().is_some() || url.fragment().is_some() {
        let why = format!("{location:?} has a `?` or `#`; percent-encode it as %3F or %23");
        return Err(Error::Option(why));
    }
    match url.scheme() {
        "file" => {
            let path = (url.to_file_path())
                .map_err(|()| Error::Option(format!("{location:?} is not a local path")))?;
            local(&path, options)
        }
        #[cfg(feature = "s3")]
        "s3" => s3(&url, options),
        scheme => Err(Error::Scheme(scheme.to_string())),
    }
}

fn local(dir: &std::path::Path, options: &[(String, String)]) -> Result<Location, Error> {
    if let Some((key, _)) = options.first() {
        let why = format!("a local folder takes no storage options, but got {key:?}");
        return Err(Error::Option(why));
    }
    let store = LocalFileSystem::new_with_prefix(dir).map_err(Error::Store)?;
    // The store opened the folder, so it exists; links are resolved.
    let id = std::fs::canonicalize(dir).unwrap_or_else(|_| dir.to_path_buf());
    // `Debug` keeps bytes that are not UTF-8 apart; `Display` would merge
    // two such folder names into one id.
    let id = format!("{id:?}");
    let (store, prefix) = (Arc::new(store), Path::from(""));
    Ok(Location { store, prefix, id })
}

#[cfg(feature = "s3")]
fn s3(url: &Url, options: &[(String, String)]) -> Result<Location, Error> {
    let bucket = url.host_str().unwrap_or_default();
    let mut builder = AmazonS3Builder::from_env().with_url(format!("s3://{bucket}"));
    for (key, value) in options {
        let key: AmazonS3ConfigKey = (key.to_ascii_lowercase().parse())
            .map_err(|_| Error::Option(format!("unknown S3 storage option {key:?}")))?;
        builder = builder.with_config(key, value);
    }
    // The endpoint, from the options or the environment, tells two servers
    // with the same bucket name apart. Keys and tokens stay out of the id.
    let endpoint = builder.get_config_value(&AmazonS3ConfigKey::Endpoint);
    // The builder fails only on settings, such as a missing bucket.
    let store = (builder.build()).map_err(|e| Error::Option(format!("bad S3 settings: {e}")))?;
    let prefix = Path::from_url_path(url.path())
        .map_err(|e| Error::Option(format!("bad prefix in {url}: {e}")))?;
    let id = format!("s3://{bucket}/{prefix} at {}", endpoint.unwrap_or_default());
    Ok(Location {
        store: Arc::new(store),
        prefix,
        id,
    })
}

#[cfg(test)]
mod tests {
    use object_store::ObjectStoreExt;

    use super::*;

    #[test]
    fn opens_local_folders() {
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(dir.path().join("a.parquet"), b"x").unwrap();
        let url = Url::from_directory_path(dir.path()).unwrap();
        let plain = dir.path().to_str().unwrap();
        let mut ids = Vec::new();
        for location in [
            plain,
            &format!("{plain}{}", std::path::MAIN_SEPARATOR),
            url.as_str(),
        ] {
            let at = open(location, &[]).unwrap();
            assert_eq!(at.prefix, Path::from(""));
            let path = Path::from("a.parquet");
            let meta = futures::executor::block_on(at.store.head(&path)).unwrap();
            assert_eq!(meta.size, 1, "{location}");
            ids.push(at.id);
        }
        assert!(ids.iter().all(|id| *id == ids[0]), "{ids:?}");
        let other = tempfile::TempDir::new().unwrap();
        assert_ne!(
            open(other.path().to_str().unwrap(), &[]).unwrap().id,
            ids[0]
        );
    }

    // Only `scheme://` makes a URL; `:` is legal in a POSIX folder name.
    #[cfg(unix)]
    #[test]
    fn a_colon_in_a_path_is_not_a_scheme() {
        let dir = tempfile::TempDir::new().unwrap();
        let odd = dir.path().join("runs:2024");
        std::fs::create_dir(&odd).unwrap();
        assert!(open(odd.to_str().unwrap(), &[]).is_ok());
    }

    #[test]
    fn refuses_what_it_cannot_open() {
        let dir = tempfile::TempDir::new().unwrap();
        let missing = dir.path().join("missing");
        let missing = missing.to_str().unwrap();
        assert!(matches!(open(missing, &[]), Err(Error::Store(_))));
        assert!(matches!(open("gs://bucket/x", &[]), Err(Error::Scheme(_))));
        for odd in ["file:///data/runs#2", "file:///data?x=1", "s3://b/runs#2"] {
            assert!(matches!(open(odd, &[]), Err(Error::Option(_))), "{odd}");
        }
        let options = [("region".to_string(), "x".to_string())];
        let here = dir.path().to_str().unwrap();
        assert!(matches!(open(here, &options), Err(Error::Option(_))));
    }

    #[cfg(feature = "s3")]
    #[test]
    fn refuses_unknown_s3_options() {
        let options = [("no_such_option".to_string(), "x".to_string())];
        assert!(matches!(open("s3://b/p", &options), Err(Error::Option(_))));
        assert!(
            matches!(open("s3:///p", &[]), Err(Error::Option(_))),
            "no bucket"
        );
    }

    // The id names the endpoint, so two servers do not share a cache, and
    // never the keys, so new keys do not force a new index.
    #[cfg(feature = "s3")]
    #[test]
    fn s3_id_has_the_endpoint_and_no_keys() {
        let id = |endpoint: &str, secret: &str| {
            let options: Vec<(String, String)> = [
                ("endpoint", endpoint),
                ("region", "us-east-1"),
                ("access_key_id", "AKIDEXAMPLE"),
                ("secret_access_key", secret),
            ]
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .into();
            open("s3://b/p", &options).unwrap().id
        };
        let a = id("http://one:7070", "s3cret-1");
        assert_eq!(a, id("http://one:7070", "s3cret-2"));
        assert_ne!(a, id("http://two:7070", "s3cret-1"));
        assert!(!a.contains("s3cret") && !a.contains("AKIDEXAMPLE"), "{a}");
    }
}
