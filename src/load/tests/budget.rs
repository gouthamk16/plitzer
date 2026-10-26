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

// A resume at any step, and the retry after a failed read (a new loader
// from `states()`), runs with the budget of a fresh run of the epoch.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn resume_needs_no_more_than_a_fresh_run() {
    for seed in [5, 7] {
        let d = data(seed).await;
        let want = direct(&d, &all(&d).await);
        let fresh = vec![LaneState::default(); LANES as usize];
        for world in [1, 2] {
            let mut budget = 0;
            for rank in 0..world {
                let lanes = gpu_lanes(LANES, world, rank);
                let own = &fresh[lanes.start as usize..lanes.end as usize];
                budget = budget.max(peak(&plan(&d.epoch, &d.index, lanes, own)) * 1024);
            }
            for k in 0..want.len() as u64 {
                let (head, states) = run(&d, world, &fresh, k, budget).await;
                let tail = run(&d, world, &states, u64::MAX, budget);
                let tail = tokio::time::timeout(Duration::from_secs(20), tail).await;
                let (tail, _) = tail.expect("no deadlock at the fresh budget");
                let joined: Vec<_> = head.into_iter().chain(tail).collect();
                assert_eq!(joined, want, "seed {seed}, world {world}, resume at {k}");
            }
        }
    }
}

// In a flush step the plan from the cursor is empty, but the leftover still
// needs its row groups. A budget below a part of the replay is refused,
// never a replay that waits for more permits than the budget has.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn budget_counts_the_replay() {
    let d = data(5).await;
    let fresh = vec![LaneState::default(); LANES as usize];
    let (_, mut states) = run(&d, 1, &fresh, 0, PLENTY).await;
    let mut steps = 0;
    let found = loop {
        let needs = resume(&d.epoch, &d.index, 0..LANES, &states).unwrap();
        let cursors = cursors(&d.epoch, 0..LANES, &states).unwrap();
        let leftover = states.iter().zip(&cursors).any(|(s, &c)| s.doc < c);
        if needs.is_empty() && leftover {
            break states;
        }
        steps += 1;
        assert!(steps < d.epoch.steps(), "no flush step with a leftover");
        states = run(&d, 1, &fresh, steps, PLENTY).await.1;
    };
    let loader = Loader::new(
        d.epoch.clone(),
        d.reader.clone(),
        d.index.clone(),
        0..LANES,
        &found,
        1024,
    );
    let got = tokio::time::timeout(Duration::from_secs(10), loader).await;
    let got = got.expect("refused, not waiting for permits");
    assert!(matches!(got, Err(Error::Budget { .. })), "{:?}", got.err());
}
