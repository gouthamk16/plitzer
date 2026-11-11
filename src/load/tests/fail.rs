use super::*;

/// Fills the chunk of `row_group` in its file with bytes that do not decode.
fn corrupt(d: &Data, row_group: u32) {
    let (file, rg) = d.index.row_group(row_group);
    let path = d.dir.path().join(&file.path);
    let mut bytes = std::fs::read(&path).unwrap();
    let chunk = rg.start as usize..rg.start as usize + rg.len as usize;
    bytes[chunk].fill(0xAB);
    std::fs::write(&path, bytes).unwrap();
}

// A read that fails returns its error once. The lanes have not moved, so
// `positions()` is a valid resume point, and later calls report `Stopped`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn failed_read_stops_the_loader() {
    let d = data(17).await;
    // A row group of epoch 0's second window: read after the first steps.
    let later = d.orders.epoch(0).windows()[1].groups()[0];
    corrupt(&d, later);
    let fresh = vec![Position::default(); LANES as usize];
    let (o, r, i) = (d.orders.clone(), d.reader.clone(), d.index.clone());
    let mut loader = Loader::new(o, r, i, 0..LANES, &fresh, PLENTY).unwrap();
    let mut out = vec![0; (LANES * SEQ_LEN) as usize];
    let mut before = loader.positions();
    let err = loop {
        match loader.next(&mut out).await {
            Ok(()) => before = loader.positions(),
            Err(e) => break e,
        }
    };
    assert!(matches!(err, Error::Read(_)), "{err:?}");
    assert_eq!(loader.positions(), before);
    assert!(before.iter().all(|p| p.epoch == 0), "failed in epoch 0");
    assert!(matches!(loader.next(&mut out).await, Err(Error::Stopped)));
}

// A position past its document is found at the first step, with the tokens.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_bad_position_is_an_error() {
    let d = data(17).await;
    let mut at = vec![Position::default(); LANES as usize];
    at[1].offset = 100_000;
    let (o, r, i) = (d.orders.clone(), d.reader.clone(), d.index.clone());
    let mut loader = Loader::new(o, r, i, 0..LANES, &at, PLENTY).unwrap();
    let mut out = vec![0; (LANES * SEQ_LEN) as usize];
    let got = loader.next(&mut out).await;
    assert!(
        matches!(&got, Err(Error::State(why)) if why.contains("lane 1")),
        "{got:?}"
    );
}
