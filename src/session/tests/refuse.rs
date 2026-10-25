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
    assert_eq!(field(other.resume(&saved)), "seq_len");
    let mut c = config();
    c.seed = 12;
    let mut other = open(&d, c, 0, 1).await.unwrap();
    assert_eq!(field(other.resume(&saved)), "seed");
    write_file(d.files.path(), 4);
    let mut other = open(&d, config(), 0, 1).await.unwrap();
    assert_eq!(field(other.resume(&saved)), "fingerprint");
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
    assert!(range(one[0].resume(&bad)), "step past the epoch");
    let mut bad = good.clone();
    bad.lanes[0].doc = 1_000_000;
    assert!(range(one[0].resume(&bad)), "leftover past the cursor");
    let mut bad = good.clone();
    bad.epoch = MAX_EPOCH + 1;
    assert!(range(one[0].resume(&bad)), "epoch too large");
    // The checkpoint after the last step of `MAX_EPOCH - 1` stands here.
    let mut last = good.clone();
    last.epoch = MAX_EPOCH;
    last.lanes.fill(LaneState::default());
    one[0].resume(&last).unwrap();
    let mut bad = good.clone();
    bad.epoch = MAX_EPOCH + 1;
    assert!(range(one[0].resume(&bad)), "epoch too large");
    assert!(matches!(one[0].set_epoch(MAX_EPOCH), Err(Error::Config(_))));
    one[0].resume(&good).unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn resume_needs_this_ranks_lanes() {
    let d = data();
    let mut two = sessions(&d, 2).await;
    run(&mut two, 2).await;
    let part = two[1].checkpoint();
    let mut again = sessions(&d, 2).await;
    again[1].resume(&part).unwrap();
    let refused = again[0].resume(&part);
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
