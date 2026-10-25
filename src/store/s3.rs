//! S3 locations: the builder from the environment, then the options.

use std::ffi::OsString;
use std::sync::Arc;

use object_store::aws::{AmazonS3Builder, AmazonS3ConfigKey};
use object_store::path::Path;
use url::Url;

use super::{Error, Location, redact};

/// The builder with the `AWS_*` variables, as `AmazonS3Builder::from_env`
/// makes it, but an empty variable counts as unset, as in the AWS CLI. A
/// compose file often sets `AWS_ENDPOINT_URL=` for real AWS.
pub(super) fn env() -> AmazonS3Builder {
    from_vars(std::env::vars_os())
}

fn from_vars(vars: impl Iterator<Item = (OsString, OsString)>) -> AmazonS3Builder {
    let mut builder = AmazonS3Builder::new();
    for (key, value) in vars {
        let (Some(key), Some(value)) = (key.to_str(), value.to_str()) else {
            continue;
        };
        if !key.starts_with("AWS_") || value.is_empty() {
            continue;
        }
        if let Ok(key) = key.to_ascii_lowercase().parse::<AmazonS3ConfigKey>() {
            builder = builder.with_config(key, value);
        }
    }
    builder
}

/// `env` is the builder with the environment's settings, which `options`
/// override.
pub(super) fn open(
    env: AmazonS3Builder,
    url: &Url,
    options: &[(String, String)],
) -> Result<Location, Error> {
    // Only the bucket is read from the host; a port or user would be lost.
    if url.port().is_some() || !url.username().is_empty() || url.password().is_some() {
        let why = format!(
            "{} has a port or user; pass them as storage options",
            redact(url)
        );
        return Err(Error::Option(why));
    }
    let bucket = url.host_str().unwrap_or_default();
    let builder = with_options(env.with_url(format!("s3://{bucket}")), options)?;
    // The endpoint, from the options or the environment, tells two servers
    // with the same bucket name apart. The builder prefers the S3-only
    // endpoint, so the id does too. Keys and tokens stay out of the id.
    let endpoint = (builder.get_config_value(&AmazonS3ConfigKey::S3Endpoint))
        .or_else(|| builder.get_config_value(&AmazonS3ConfigKey::Endpoint));
    check(&builder, endpoint.as_deref())?;
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

/// Refuses the settings that object_store panics on at the first request,
/// and an endpoint with a user or password, which its errors would repeat.
/// The values can come from the options or the environment.
fn check(builder: &AmazonS3Builder, endpoint: Option<&str>) -> Result<(), Error> {
    if let Some(endpoint) = endpoint {
        // `Url::parse` drops spaces at the ends, but the request would not.
        let odd = endpoint
            .chars()
            .any(|c| c.is_whitespace() || c.is_control());
        let url = Url::parse(endpoint).ok().filter(|u| u.has_host() && !odd);
        let why = match url {
            Some(u) if !matches!(u.scheme(), "http" | "https") => "must be http:// or https://",
            Some(u) if !u.username().is_empty() || u.password().is_some() => {
                "must not hold a user or password"
            }
            Some(_) => "",
            None => "must be a URL such as http://host:9000, with no spaces",
        };
        if !why.is_empty() {
            return Err(Error::Option(format!("the S3 endpoint {why}")));
        }
    }
    // object_store puts these in request headers and unwraps the result.
    let headers = [
        AmazonS3ConfigKey::AccessKeyId,
        AmazonS3ConfigKey::SecretAccessKey,
        AmazonS3ConfigKey::Token,
        AmazonS3ConfigKey::Region,
    ];
    for key in headers {
        if (builder.get_config_value(&key)).is_some_and(|v| v.chars().any(char::is_control)) {
            let why = format!(
                "{} holds a control character, such as a newline",
                key.as_ref()
            );
            return Err(Error::Option(why));
        }
    }
    Ok(())
}

/// `builder` with each of `options`; an unknown one is an error.
fn with_options(
    mut builder: AmazonS3Builder,
    options: &[(String, String)],
) -> Result<AmazonS3Builder, Error> {
    let mut keys = Vec::new();
    for (key, value) in options {
        let key: AmazonS3ConfigKey = (key.to_ascii_lowercase().parse())
            .map_err(|_| Error::Option(format!("unknown S3 storage option {key:?}")))?;
        builder = builder.with_config(key, value);
        keys.push(key);
    }
    // The builder prefers the S3-only endpoint, which can come from
    // `AWS_ENDPOINT_URL_S3`; an `endpoint` option must still win over it.
    let (general, s3_only) = (AmazonS3ConfigKey::Endpoint, AmazonS3ConfigKey::S3Endpoint);
    if keys.contains(&general) && !keys.contains(&s3_only) {
        let endpoint = builder.get_config_value(&general).unwrap_or_default();
        builder = builder.with_config(s3_only, endpoint);
    }
    Ok(builder)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store;

    #[test]
    fn refuses_unknown_s3_options() {
        let options = [("no_such_option".to_string(), "x".to_string())];
        assert!(matches!(
            store::open("s3://b/p", &options),
            Err(Error::Option(_))
        ));
        assert!(
            matches!(store::open("s3:///p", &[]), Err(Error::Option(_))),
            "no bucket"
        );
    }

    // The id names the endpoint, so two servers do not share a cache, and
    // never the keys, so new keys do not force a new index.
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
            store::open("s3://b/p", &options).unwrap().id
        };
        let a = id("http://one:7070", "s3cret-1");
        assert_eq!(a, id("http://one:7070", "s3cret-2"));
        assert_ne!(a, id("http://two:7070", "s3cret-1"));
        assert!(!a.contains("s3cret") && !a.contains("AKIDEXAMPLE"), "{a}");
        // The S3-only endpoint wins in the builder, so it names the id too.
        let s3_only = [(
            "aws_endpoint_url_s3".to_string(),
            "http://three:7070".to_string(),
        )];
        assert!(
            store::open("s3://b/p", &s3_only)
                .unwrap()
                .id
                .contains("three")
        );
        // An `endpoint` option wins over `AWS_ENDPOINT_URL_S3`.
        let env = AmazonS3Builder::new().with_config(AmazonS3ConfigKey::S3Endpoint, "http://env");
        let url = Url::parse("s3://b/p").unwrap();
        let option = [("endpoint".to_string(), "http://four:7070".to_string())];
        let at = open(env, &url, &option).unwrap();
        assert!(at.id.ends_with("at http://four:7070"), "{}", at.id);
    }

    // object_store panics on these at the first request; plitzer refuses
    // them at `open`, also from the environment, and repeats no secret.
    #[test]
    fn refuses_settings_that_would_panic() {
        let url = Url::parse("s3://b/p").unwrap();
        let refused = |env: AmazonS3Builder, options: &[(&str, &str)]| {
            let options: Vec<_> = (options.iter())
                .map(|&(k, v)| (k.to_string(), v.to_string()))
                .collect();
            match open(env, &url, &options) {
                Err(Error::Option(why)) => !why.contains("s3cret"),
                _ => false,
            }
        };
        let bad = [
            "localhost:9000",
            "http://host:9000 ",
            "",
            "ftp://host",
            "http://u:s3cret@host:9000",
        ];
        for endpoint in bad {
            assert!(
                refused(AmazonS3Builder::new(), &[("endpoint", endpoint)]),
                "{endpoint:?}"
            );
        }
        let env = AmazonS3Builder::new().with_config(AmazonS3ConfigKey::Endpoint, "localhost:9000");
        assert!(refused(env, &[]), "an endpoint from the environment");
        let token = [("endpoint", "http://host:9000"), ("token", "s3cret\n")];
        assert!(
            refused(AmazonS3Builder::new(), &token),
            "a newline in a token"
        );
        let region = [("endpoint", "http://host:9000"), ("region", "us-east-1\n")];
        assert!(
            refused(AmazonS3Builder::new(), &region),
            "a newline in a region"
        );
        for odd in ["s3://b:9000/p", "s3://user@b/p", "s3://user:s3cret@b/p"] {
            match store::open(odd, &[]) {
                Err(Error::Option(why)) => assert!(!why.contains("s3cret"), "{why}"),
                _ => panic!("{odd}"),
            }
        }
    }

    // An empty `AWS_ENDPOINT_URL=`, as a compose file sets for real AWS, is
    // unset, so the builder's default endpoint is used.
    #[test]
    fn an_empty_variable_is_unset() {
        let vars = [("AWS_ENDPOINT_URL", ""), ("AWS_REGION", "eu-west-1")];
        let vars = vars.map(|(k, v)| (OsString::from(k), OsString::from(v)));
        let env = from_vars(vars.into_iter());
        assert_eq!(env.get_config_value(&AmazonS3ConfigKey::Endpoint), None);
        let region = env.get_config_value(&AmazonS3ConfigKey::Region);
        assert_eq!(region.as_deref(), Some("eu-west-1"));
        assert!(open(env, &Url::parse("s3://b/p").unwrap(), &[]).is_ok());
    }
}
