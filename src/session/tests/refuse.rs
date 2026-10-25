use super::*;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn resume_refuses_other_runs() {
    let d = data();
    let mut one = sessions(&d, 1).await;
    run(&mut one, 2).await;
    let saved = one[0].checkpoint();
    let field = |r: Result<(), Error>| match r {
        Err(Error::Checkpoint(checkpoint::Error::Mismatch { field, .. })) => field,
        other => panic!("{other:?}"),
    };
    let mut c = config();
    c.packing.seq_len = 16;
    let mut other = open(&d, c, 0, 1).await.unwrap();
    assert_eq!(field(other.resume(&saved).await), "seq_len");
    let mut c = config();
    c.seed = 12;
    let mut other = open(&d, c, 0, 1).await.unwrap();
    assert_eq!(field(other.resume(&saved).await), "seed");
    write_file(d.files.path(), 4);
    let mut other = open(&d, config(), 0, 1).await.unwrap();
    assert_eq!(field(other.resume(&saved).await), "fingerprint");
}

// Checksummed bytes can still be made by hand. States that no run makes are
// refused at `resume`, before the packer would panic on them.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn resume_refuses_states_outside_the_epoch() {
    let d = data();
    let mut one = sessions(&d, 1).await;
    run(&mut one, 2).await;
    let good = one[0].checkpoint();
    let range =
        |r: Result<(), Error>| matches!(r, Err(Error::Checkpoint(checkpoint::Error::Range(_))));
    let mut bad = good.clone();
    bad.lanes.iter_mut().for_each(|s| s.step = 1_000_000);
    assert!(range(one[0].resume(&bad).await), "step past the epoch");
    let mut bad = good.clone();
    bad.lanes[0].doc = 1_000_000;
    assert!(range(one[0].resume(&bad).await), "leftover past the cursor");
    // A leftover that starts earlier than the real one passes the cheap
    // check, but can exceed the cap; `resume` finds it when it makes the
    // loader, as an error, not a panic, and the session stays at `good`.
    let mut over_cap = false;
    for doc in 0..good.lanes[0].doc {
        let mut bad = good.clone();
        bad.lanes[0] = LaneState {
            doc,
            offset: 0,
            ..bad.lanes[0]
        };
        match one[0].resume(&bad).await {
            Err(Error::Checkpoint(checkpoint::Error::Range(why))) => {
                assert!(why.starts_with("lane 0 of epoch 0: "), "{why}");
                over_cap |= why.contains("more than the cap");
            }
            other => panic!("doc {doc}: {other:?}"),
        }
        assert_eq!(
            one[0].checkpoint(),
            good,
            "a refused resume changes nothing"
        );
    }
    assert!(over_cap, "no earlier leftover start exceeded the cap");
    // Also a session with no loader yet keeps its position.
    let mut fresh = sessions(&d, 1).await;
    let before = fresh[0].checkpoint();
    let mut bad = good.clone();
    bad.lanes[0] = LaneState {
        doc: 0,
        offset: 0,
        ..bad.lanes[0]
    };
    assert!(range(fresh[0].resume(&bad).await));
    assert_eq!(fresh[0].checkpoint(), before);
    one[0].resume(&good).await.unwrap();
}

// A refused resume after the last step keeps the `None` that ends the epoch,
// so this rank does not run into the next epoch ahead of the others.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn refused_resume_keeps_the_epoch_end() {
    let d = data();
    let end = reference(&d)
        .await
        .iter()
        .position(Option::is_none)
        .unwrap();
    let mut one = sessions(&d, 1).await;
    run(&mut one, 2).await;
    let mut bad = one[0].checkpoint();
    bad.lanes[0] = LaneState {
        doc: 0,
        offset: 0,
        ..bad.lanes[0]
    };
    run(&mut one, end - 2).await;
    assert!(one[0].ended());
    let before = one[0].checkpoint();
    let refused = one[0].resume(&bad).await;
    assert!(matches!(refused, Err(Error::Checkpoint(_))), "{refused:?}");
    assert!(one[0].ended() && one[0].checkpoint() == before);
    assert_eq!(run(&mut one, 1).await, [None]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn resume_needs_this_ranks_lanes() {
    let d = data();
    let mut two = sessions(&d, 2).await;
    run(&mut two, 2).await;
    let part = two[1].checkpoint();
    let mut again = sessions(&d, 2).await;
    again[1].resume(&part).await.unwrap();
    let refused = again[0].resume(&part).await;
    assert!(matches!(
        refused,
        Err(Error::Checkpoint(checkpoint::Error::Parts(_)))
    ));
}

#[tokio::test]
async fn open_refuses_bad_settings() {
    let d = data();
    let refused = |r: Result<Session, Error>| matches!(r, Err(Error::Config(_)));
    assert!(
        refused(open(&d, config(), 0, 3).await),
        "3 does not divide 4"
    );
    assert!(refused(open(&d, config(), 2, 2).await), "rank past world");
    assert!(refused(open(&d, config(), 0, 0).await), "no GPUs");
    let mut c = config();
    c.window = 0;
    assert!(refused(open(&d, c, 0, 1).await));
    let mut c = config();
    c.packing.docs = 0;
    assert!(refused(open(&d, c, 0, 1).await));
    let mut c = config();
    c.lanes = 1 << 25;
    assert!(refused(open(&d, c, 0, 1).await), "too many lanes");
    let mut c = config();
    c.packing.seq_len = u32::MAX;
    assert!(refused(open(&d, c, 0, 1).await), "a step too large");
    let mut c = config();
    c.packing.cap = u32::MAX;
    assert!(refused(open(&d, c, 0, 1).await), "leftovers too large");
}
