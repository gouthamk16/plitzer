use proptest::prelude::*;

use super::*;
use crate::index::{Codec, File, Index, RowGroup};

const SCHEMA: &str = "message schema { required int32 tokens; }";

pub(crate) fn file(path: &str, rows: &[u32]) -> File {
    let group = |(i, &rows)| RowGroup {
        rows,
        start: 4 + 100 * i as u64,
        len: 100,
        codec: Codec::Snappy,
    };
    File {
        path: path.into(),
        size: 1000,
        e_tag: Some("t".into()),
        row_groups: rows.iter().enumerate().map(group).collect(),
    }
}

pub(crate) fn index(files: Vec<File>) -> Index {
    Index::new(
        "tokens".into(),
        "message schema { required int32 tokens; }".into(),
        files,
    )
    .unwrap()
}

#[test]
fn fingerprint_golden() {
    let golden = index(vec![file("a/x.parquet", &[2]), file("b.parquet", &[3, 1])]);
    assert_eq!(golden.fingerprint(), 8_288_418_835_622_463_668);
    // The cache bytes, with ETags, are pinned too.
    let cache = fnv1a(&encode(&golden, true));
    assert_eq!(cache, 1_150_502_440_377_594_925);

    let mut other = golden.clone();
    other.files[0].e_tag = None;
    assert_eq!(other.fingerprint(), golden.fingerprint());
    other.files[1].row_groups[1].rows = 2;
    assert_ne!(other.fingerprint(), golden.fingerprint());
}

#[test]
fn bad_schema_does_not_decode() {
    let mut bad = index(vec![file("a", &[1])]);
    bad.schema = "message schema { not a schema".into();
    assert_eq!(decode(&encode(&bad, true)), None);
}

fn codec() -> impl Strategy<Value = Codec> {
    (0u8..7).prop_map(|b| Codec::from_byte(b).unwrap())
}

fn files() -> impl Strategy<Value = Vec<File>> {
    let group = (any::<u32>(), any::<u64>(), any::<u32>(), codec());
    let group = group.prop_map(|(rows, start, len, codec)| RowGroup {
        rows,
        start,
        len,
        codec,
    });
    let f = (
        ".{0,8}",
        any::<u64>(),
        proptest::option::of(".{0,8}"),
        prop::collection::vec(group, 0..5),
    );
    prop::collection::vec(
        f.prop_map(|(path, size, e_tag, row_groups)| File {
            path,
            size,
            e_tag,
            row_groups,
        }),
        0..5,
    )
}

proptest! {
    #[test]
    fn sidecar_roundtrip(files in files(), column in ".{0,8}") {
        let index = Index::new(column, SCHEMA.into(), files).unwrap();
        prop_assert_eq!(decode(&encode(&index, true)), Some(index));
    }

    #[test]
    fn decode_never_panics(files in files(), cut in any::<prop::sample::Index>(), junk in prop::collection::vec(any::<u8>(), 0..64)) {
        let bytes = encode(&index(files), true);
        let cut = cut.index(bytes.len());
        prop_assert_eq!(decode(&bytes[..cut]), None);
        decode(&junk);
    }
}
