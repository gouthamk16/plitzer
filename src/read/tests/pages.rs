//! The reader against random token lists, over many small pages.

use proptest::prelude::*;

use super::*;

/// Rows of up to 40 tokens below 70 000, so the dictionary has a real
/// width; `None` is a null row, which reads as empty.
fn rows() -> impl Strategy<Value = Vec<Option<Vec<u32>>>> {
    let row = prop::option::weighted(0.9, prop::collection::vec(0u32..70_000, 0..40));
    prop::collection::vec(row, 1..300)
}

fn column(rows: &[Option<Vec<u32>>], unsigned: bool) -> ArrayRef {
    if unsigned {
        let mut b = GenericListBuilder::<i32, _>::new(UInt32Builder::new());
        for r in rows {
            if let Some(r) = r {
                b.values().append_slice(r);
            }
            b.append(r.is_some());
        }
        Arc::new(b.finish())
    } else {
        let mut b = GenericListBuilder::<i32, _>::new(Int32Builder::new());
        for r in rows {
            if let Some(r) = r {
                b.values().extend(r.iter().map(|&t| Some(t as i32)));
            }
            b.append(r.is_some());
        }
        Arc::new(b.finish())
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(24))]
    #[test]
    fn reads_what_was_written(rows in rows(), dictionary: bool, unsigned: bool, page in 1usize..64) {
        let rt = tokio::runtime::Builder::new_current_thread().build().unwrap();
        let d = data();
        // Small pages, so each chunk has many, and levels cross page edges.
        let props = WriterProperties::builder()
            .set_dictionary_enabled(dictionary)
            .set_data_page_row_count_limit(page)
            .set_write_batch_size(page)
            .set_max_row_group_row_count(Some(100))
            .build();
        write(&d.path("x.parquet"), column(&rows, unsigned), props);
        let index = rt.block_on(d.index()).unwrap();
        let got = rt.block_on(d.read_all(&index));
        let want: Vec<Vec<u32>> = rows.into_iter().map(Option::unwrap_or_default).collect();
        prop_assert_eq!(got, want);
    }
}
