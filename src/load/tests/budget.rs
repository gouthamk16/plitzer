use super::*;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn budget_is_checked_and_kept() {
    let d = data(7).await;
    let want = direct(&d, &all(&d).await);
    let fresh = vec![LaneState::default(); LANES as usize];
    let needed = peak(&plan(&d.epoch, &d.index, 0..LANES, &fresh)) * 1024;
    let below = Loader::new(
        d.epoch.clone(),
        d.reader.clone(),
        d.index.clone(),
        0..LANES,
        &fresh,
        needed - 1024,
    )
    .await;
    assert!(
        matches!(below, Err(Error::Budget { .. })),
        "{:?}",
        below.err()
    );
    // The exact peak must finish, not deadlock. `run` checks the resident bytes.
    let got = tokio::time::timeout(
        Duration::from_secs(20),
        run(&d, 1, &fresh, u64::MAX, needed),
    )
    .await;
    assert_eq!(got.expect("no deadlock at the peak budget").0, want);
}

// With room for everything, the prefetch reads every row group ahead of
