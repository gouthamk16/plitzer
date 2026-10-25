use std::time::Duration;

use futures::FutureExt;
use object_store::throttle::{ThrottleConfig, ThrottledStore};

use super::*;

// A dropped `next` future, as from a timeout or Ctrl-C in Python, loses
// no read: polling each call only once until it is ready gives the same
// batches.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn next_is_cancel_safe() {
    let d = data();
    let want = reference(&d).await;
    // Each read takes 5 ms, so the first call, which must read files, is
    // never ready at its first poll. `load::tests::next_is_cancel_safe`
    // checks drops during a read.
    let throttle = ThrottleConfig {
        wait_get_per_call: Duration::from_millis(5),
        ..ThrottleConfig::default()
    };
    let local = LocalFileSystem::new_with_prefix(d.files.path()).unwrap();
    let store = Arc::new(ThrottledStore::new(local, throttle));
    let cache = d.cache.path().join("index");
    let mut one = Session::open(store, Path::from(""), &cache, config(), 0, 1, PLENTY)
        .await
        .unwrap();
    // Poll each call once; if it is not ready, drop it and do the step again.
    let mut seq = vec![0; (4 * config().packing.seq_len) as usize];
    let mut got = Vec::new();
    let mut cancelled = 0;
    while got.len() < want.len() {
        let real = match one.next(&mut seq).now_or_never() {
            Some(real) => real,
            None => {
                cancelled += 1;
                one.next(&mut seq).await
            }
        };
        got.push(real.unwrap().map(|_| seq.clone()));
    }
    assert!(cancelled > 0, "no call was cancelled");
    assert_eq!(got, want);
}

// A `resume` dropped while it reads leaves the session where it was: its
// checkpoint and its next batches are those of a run with no resume.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn resume_is_cancel_safe() {
    let d = data();
    let want = reference(&d).await;
    let throttle = ThrottleConfig {
        wait_get_per_call: Duration::from_millis(20),
        ..ThrottleConfig::default()
    };
    let local = LocalFileSystem::new_with_prefix(d.files.path()).unwrap();
    let store = Arc::new(ThrottledStore::new(local, throttle));
    let cache = d.cache.path().join("index");
    let mut one = Session::open(store, Path::from(""), &cache, config(), 0, 1, PLENTY)
        .await
        .unwrap();
    let mut seq = vec![0; (4 * config().packing.seq_len) as usize];
    let mut got = Vec::new();
    for _ in 0..3 {
        got.push(one.next(&mut seq).await.unwrap().map(|_| seq.clone()));
    }
    let before = one.checkpoint();
    let mut elsewhere = before.clone();
    elsewhere.epoch = 1;
    elsewhere.lanes.fill(LaneState::default());
    // Its reads take 20 ms, so one poll never finishes the resume.
    assert!(one.resume(&elsewhere).now_or_never().is_none());
    assert_eq!(one.checkpoint(), before);
    while got.len() < want.len() {
        got.push(one.next(&mut seq).await.unwrap().map(|_| seq.clone()));
    }
    assert_eq!(got, want);
}
