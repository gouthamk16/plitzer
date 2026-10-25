//! Columns, tokens and row counts that the reader refuses.

use super::*;

#[tokio::test]
async fn bad_columns_are_errors() {
    let nested = {
        let inner = GenericListBuilder::<i32, _>::new(Int32Builder::new());
        let mut b = GenericListBuilder::<i32, _>::new(inner);
        b.values().values().append_value(1);
        b.values().append(true);
        b.append(true);
        Arc::new(b.finish()) as ArrayRef
    };
    let dates = {
        let mut b = GenericListBuilder::<i32, _>::new(Date32Builder::new());
        b.values().append_value(1);
        b.append(true);
        Arc::new(b.finish()) as ArrayRef
    };
    // The printer does not quote names, so this schema would not parse back.
    let spaced = {
        let item = arrow_schema::Field::new("my item", arrow_schema::DataType::Int32, true);
        let mut b = GenericListBuilder::<i32, _>::new(Int32Builder::new()).with_field(item);
        b.values().append_value(1);
        b.append(true);
        Arc::new(b.finish()) as ArrayRef
    };
    // Nanosecond timestamps have a logical type but no converted type.
    let nanos = {
        let mut b = GenericListBuilder::<i32, _>::new(TimestampNanosecondBuilder::new());
        b.values().append_value(1);
        b.append(true);
        Arc::new(b.finish()) as ArrayRef
    };
    let text: ArrayRef = Arc::new(arrow_array::StringArray::from(vec!["a"]));
    for tokens in [nested, text, dates, nanos, spaced] {
        let d = data();
        let p = props(Compression::SNAPPY, false);
        write(&d.path("x.parquet"), tokens, p);
        assert!(matches!(d.index().await, Err(IndexError::Column { .. })));
    }
}

#[tokio::test]
async fn missing_or_differing_columns_are_errors() {
    let d = data();
    let p = props(Compression::SNAPPY, false);
    write(
        &d.path("a.parquet"),
        token_column(Kind::ListI32, 3, false),
        p.clone(),
    );
    write(
        &d.path("b.parquet"),
        token_column(Kind::ListI64, 3, false),
        p,
    );
    match d.index().await {
        Err(IndexError::Column { path, .. }) => assert_eq!(path, "b.parquet"),
        other => panic!("expected a column error, got {other:?}"),
    }
    let missing = Index::build(&*d.store, &d.prefix, "input_ids").await;
    assert!(matches!(missing, Err(IndexError::Column { .. })));
}

// The schema printer does not quote names, so the index stores the column
// under a fixed name. A name with a space must still read.
#[tokio::test]
async fn bad_tokens_are_errors() {
    for tokens in [[Some(3), Some(-1)], [Some(3), None]] {
        let d = data();
        let mut b = GenericListBuilder::<i32, _>::new(Int32Builder::new());
        b.values().extend(tokens);
        b.append(true);
        let p = props(Compression::SNAPPY, false);
        write(&d.path("x.parquet"), Arc::new(b.finish()), p);
        let index = d.index().await.unwrap();
        let reader = Reader::new(d.store.clone(), d.prefix.clone(), &index).unwrap();
        let got = reader.read(&index, 0).await;
        assert!(
            matches!(got, Err(Error::BadToken { .. })),
            "{tokens:?}: {got:?}"
        );
    }
}

// A footer that counts more rows than its chunk holds is an error, not a
// panic when a lane later asks for a missing row.
#[tokio::test]
async fn row_count_must_match_the_footer() {
    let d = data();
    let p = props(Compression::SNAPPY, false);
    write(
        &d.path("x.parquet"),
        token_column(Kind::ListI32, 3, false),
        p,
    );
    let mut index = d.index().await.unwrap();
    index.files_mut()[0].row_groups[0].rows = 5;
    let reader = Reader::new(d.store.clone(), d.prefix.clone(), &index).unwrap();
    let got = reader.read(&index, 0).await;
    assert!(matches!(got, Err(Error::Decode { .. })), "{got:?}");
}
