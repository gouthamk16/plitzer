use std::time::Duration;

use futures::FutureExt;
use object_store::throttle::{ThrottleConfig, ThrottledStore};

use super::*;

// A `next` future dropped while a read is in flight loses no read. Reads
// take 20 ms, so a step's first poll finds its row groups' reads running.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn next_is_cancel_safe() {
    let d = data(19).await;
    let want = direct(&d, &all(&d).await);
    let throttle = ThrottleConfig {
        wait_get_per_call: Duration::from_millis(20),
        ..ThrottleConfig::default()
    };
    let local = LocalFileSystem::new_with_prefix(d.dir.path()).unwrap();
    let store = Arc::new(ThrottledStore::new(local, throttle));
    let reader = Arc::new(Reader::new(store, Path::from(""), &d.index).unwrap());
    let fresh = vec![LaneState::default(); LANES as usize];
    let lanes = 0..LANES;
    let mut loader = Loader::new(
        d.epoch.clone(),
        reader,
        d.index.clone(),
        lanes,
        &fresh,
        PLENTY,
    )
    .await
    .unwrap();
    let mut out = vec![0; (LANES * PACKING.seq_len) as usize];
    let (mut got, mut dropped_mid_read) = (Vec::new(), 0);
    loop {
        let real = match loader.next(&mut out).now_or_never() {
            Some(real) => real,
            None => {
                dropped_mid_read += usize::from(loader.pending.is_some());
                loader.next(&mut out).await
            }
        };
        if real.unwrap().is_none() {
            break;
        }
        got.push(out.clone());
    }
    assert!(dropped_mid_read > 0, "no future was dropped during a read");
    assert_eq!(got, want);
}
