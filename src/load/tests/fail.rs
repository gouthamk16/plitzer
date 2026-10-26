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

// A read that fails mid-epoch returns its error once. The lanes have not
// moved, so `states()` is a valid resume point, and later calls report
// `Stopped` instead of reading a row group that never arrived.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn failed_read_stops_the_loader() {
    let d = data(17).await;
    let fresh = vec![LaneState::default(); LANES as usize];
    let needs = plan(&d.epoch, &d.index, 0..LANES, &fresh);
    let bad = needs
        .iter()
        .find(|n| n.first >= 2)
        .expect("a row group needed later");
    corrupt(&d, bad.row_group);

    let mut loader = Loader::new(
        d.epoch.clone(),
        d.reader.clone(),
        d.index.clone(),
        0..LANES,
        &fresh,
        PLENTY,
    )
    .await
    .unwrap();
    let mut out = vec![0; (LANES * PACKING.seq_len) as usize];
    let mut before = loader.states();
    let err = loop {
        match loader.next(&mut out).await {
            Ok(Some(_)) => before = loader.states(),
            Ok(None) => panic!("the corrupt row group was never read"),
            Err(e) => break e,
        }
    };
    assert!(matches!(err, Error::Read(_)), "{err:?}");
    assert_eq!(loader.states(), before);
    assert!(matches!(loader.next(&mut out).await, Err(Error::Stopped)));
}

// `new` reads the first step's row groups, also with no leftover to replay,
// so a resume into data that cannot be read fails at once, not at `next`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn new_reads_the_first_step() {
    let d = data(17).await;
    let fresh = vec![LaneState::default(); LANES as usize];
    let needs = plan(&d.epoch, &d.index, 0..LANES, &fresh);
    let first = needs
        .iter()
        .find(|n| n.first == 0)
        .expect("a row group at step 0");
    corrupt(&d, first.row_group);
    let got = Loader::new(
        d.epoch.clone(),
        d.reader.clone(),
        d.index.clone(),
        0..LANES,
        &fresh,
        PLENTY,
    )
    .await;
    assert!(matches!(got, Err(Error::Read(_))), "{:?}", got.err());
}
