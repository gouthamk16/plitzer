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

/// The store and the prefix in it for `location`: a local folder, a
/// `file://` URL, or an `s3://bucket/prefix` URL with the `s3` feature. S3
/// reads `AWS_*` variables from the environment, and `options` (such as
/// `endpoint` or `region`) override them.
pub fn open(
    location: &str,
    options: &[(String, String)],
) -> Result<(Arc<dyn ObjectStore>, Path), Error> {
    // A Windows path such as `C:\data` parses as a URL with scheme `c`.
    let url = Url::parse(location).ok().filter(|u| u.scheme().len() > 1);
    let Some(url) = url else {
        return local(std::path::Path::new(location), options);
    };
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

fn local(
    dir: &std::path::Path,
    options: &[(String, String)],
) -> Result<(Arc<dyn ObjectStore>, Path), Error> {
    if let Some((key, _)) = options.first() {
        let why = format!("a local folder takes no storage options, but got {key:?}");
        return Err(Error::Option(why));
    }
    let store = LocalFileSystem::new_with_prefix(dir).map_err(Error::Store)?;
    Ok((Arc::new(store), Path::from("")))
}

#[cfg(feature = "s3")]
fn s3(url: &Url, options: &[(String, String)]) -> Result<(Arc<dyn ObjectStore>, Path), Error> {
    let bucket = url.host_str().unwrap_or_default();
    let mut builder = AmazonS3Builder::from_env().with_url(format!("s3://{bucket}"));
    for (key, value) in options {
        let key: AmazonS3ConfigKey = (key.to_ascii_lowercase().parse())
            .map_err(|_| Error::Option(format!("unknown S3 storage option {key:?}")))?;
        builder = builder.with_config(key, value);
    }
    // The builder fails only on settings, such as a missing bucket.
    let store = (builder.build()).map_err(|e| Error::Option(format!("bad S3 settings: {e}")))?;
    let prefix = Path::from_url_path(url.path())
        .map_err(|e| Error::Option(format!("bad prefix in {url}: {e}")))?;
    Ok((Arc::new(store), prefix))
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
        for location in [dir.path().to_str().unwrap(), url.as_str()] {
            let (store, prefix) = open(location, &[]).unwrap();
            assert_eq!(prefix, Path::from(""));
            let path = Path::from("a.parquet");
            let meta = futures::executor::block_on(store.head(&path)).unwrap();
            assert_eq!(meta.size, 1, "{location}");
        }
    }

    #[test]
    fn refuses_what_it_cannot_open() {
        let dir = tempfile::TempDir::new().unwrap();
        let missing = dir.path().join("missing");
        let missing = missing.to_str().unwrap();
        assert!(matches!(open(missing, &[]), Err(Error::Store(_))));
        assert!(matches!(open("gs://bucket/x", &[]), Err(Error::Scheme(_))));
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
}
