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
    let mut seq = vec![0; (4 * config().seq_len) as usize];
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
        got.push(real.unwrap().then(|| seq.clone()));
    }
    assert!(cancelled > 0, "no call was cancelled");
    assert_eq!(got, want);
}

// After a resume, the first step reads; dropping it at its first poll loses
// nothing, and the run goes on as one with no stop.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn first_step_after_a_resume_is_cancel_safe() {
    let d = data();
    let want = reference(&d).await;
    let throttle = ThrottleConfig {
        wait_get_per_call: Duration::from_millis(20),
        ..ThrottleConfig::default()
    };
    let mut two = open(&d, config(), 0, 1).await.unwrap();
    let mut seq = vec![0; (4 * config().seq_len) as usize];
    let mut got = Vec::new();
    for _ in 0..5 {
        two.next(&mut seq).await.unwrap();
        got.push(Some(seq.clone()));
    }
    let saved = two.checkpoint();
    let local = LocalFileSystem::new_with_prefix(d.files.path()).unwrap();
    let store = Arc::new(ThrottledStore::new(local, throttle));
    let cache = d.cache.path().join("index");
    let mut one = Session::open(store, Path::from(""), &cache, config(), 0, 1, PLENTY)
        .await
        .unwrap();
    one.resume(&saved).await.unwrap();
    assert!(
        one.next(&mut seq).now_or_never().is_none(),
        "its reads take 20 ms"
    );
    assert_eq!(one.checkpoint(), saved);
    while got.len() < want.len() {
        got.push(one.next(&mut seq).await.unwrap().then(|| seq.clone()));
    }
    assert_eq!(got, want);
}
