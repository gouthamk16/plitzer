//! The store and prefix of a dataset location, such as a local folder or
//! `s3://bucket/prefix`.

use std::fmt;
use std::sync::Arc;

use object_store::ObjectStore;
use object_store::local::LocalFileSystem;
use object_store::path::Path;
use url::Url;

#[cfg(feature = "s3")]
mod s3;

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
    // The text can hold a password, so the error does not repeat it.
    let url = Url::parse(location).map_err(|e| Error::Option(format!("a bad URL: {e}")))?;
    // `?` and `#` may be part of a real folder or key name; dropping what
    // follows them would open another dataset.
    if url.query().is_some() || url.fragment().is_some() {
        let why = format!(
            "{} has a `?` or `#`; percent-encode it as %3F or %23",
            redact(&url)
        );
        return Err(Error::Option(why));
    }
    match url.scheme() {
        "file" => {
            let path = (url.to_file_path())
                .map_err(|()| Error::Option(format!("{} is not a local path", redact(&url))))?;
            local(&path, options)
        }
        #[cfg(feature = "s3")]
        "s3" => s3::open(
            object_store::aws::AmazonS3Builder::from_env(),
            &url,
            options,
        ),
        scheme => Err(Error::Scheme(scheme.to_string())),
    }
}

/// `url` for an error message: without the user, the password, the query
/// and the fragment, which can hold keys or tokens.
fn redact(url: &Url) -> String {
    let mut url = url.clone();
    // These fail only for a URL that cannot have a user, which has none.
    let _ = (url.set_username(""), url.set_password(None));
    url.set_query(None);
    url.set_fragment(None);
    url.to_string()
}

fn local(dir: &std::path::Path, options: &[(String, String)]) -> Result<Location, Error> {
    if let Some((key, _)) = options.first() {
        let why = format!("a local folder takes no storage options, but got {key:?}");
        return Err(Error::Option(why));
    }
    let store = LocalFileSystem::new_with_prefix(dir).map_err(Error::Store)?;
    if !dir.is_dir() {
        let why = format!("{dir:?} is not a folder; give the folder of the dataset");
        return Err(Error::Option(why));
    }
    // The store opened the folder, so it exists; links are resolved.
    let id = std::fs::canonicalize(dir).unwrap_or_else(|_| dir.to_path_buf());
    // `Debug` keeps bytes that are not UTF-8 apart; `Display` would merge
    // two such folder names into one id.
    let id = format!("{id:?}");
    let (store, prefix) = (Arc::new(store), Path::from(""));
    Ok(Location { store, prefix, id })
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
        let file = dir.path().join("a.parquet");
        std::fs::write(&file, b"x").unwrap();
        let file = open(file.to_str().unwrap(), &[]);
        assert!(
            matches!(file, Err(Error::Option(_))),
            "a file, not a folder"
        );
        assert!(matches!(open("gs://bucket/x", &[]), Err(Error::Scheme(_))));
        for odd in ["file:///data/runs#2", "file:///data?x=1", "s3://b/runs#2"] {
            assert!(matches!(open(odd, &[]), Err(Error::Option(_))), "{odd}");
        }
        // An error never repeats a password or a query.
        for secret in [
            "file:///data?token=s3cret",
            "file://u:s3cret@host/data#x",
            "s3://u:s3cret@[/p",
        ] {
            match open(secret, &[]) {
                Err(Error::Option(why)) => assert!(!why.contains("s3cret"), "{why}"),
                other => panic!("{secret}: {:?}", other.err()),
            }
        }
        let options = [("region".to_string(), "x".to_string())];
        let here = dir.path().to_str().unwrap();
        assert!(matches!(open(here, &options), Err(Error::Option(_))));
    }
}
