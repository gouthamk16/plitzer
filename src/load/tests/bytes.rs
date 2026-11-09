use super::*;

/// The chunk bytes of row groups `groups`.
fn chunk_bytes(index: &Index, groups: impl IntoIterator<Item = u32>) -> u64 {
    let len = |g| u64::from(index.row_group(g).1.len);
    groups.into_iter().map(len).sum()
}

// Each GPU fetches each row group of its plan once, and nothing else: one
// GPU reads exactly the dataset, and the edges shared by GPUs are the only
// extra reads.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn each_gpu_reads_its_plan_once() {
    for seed in [1, 2, 3] {
        let d = data(seed).await;
        let dataset = chunk_bytes(&d.index, 0..d.index.rows().len() as u32);
        let fresh = vec![LaneState::default(); LANES as usize];
        for world in [1, 2, 4] {
            let mut total = 0;
            for rank in 0..world {
                let lanes = gpu_lanes(LANES, world, rank);
                let store = Arc::new(LocalFileSystem::new_with_prefix(d.dir.path()).unwrap());
                let reader = Arc::new(Reader::new(store, Path::from(""), &d.index).unwrap());
                let (epoch, index) = (d.epoch.clone(), d.index.clone());
                let own = &fresh[lanes.start as usize..lanes.end as usize];
                let mut loader =
                    Loader::new(epoch, reader.clone(), index, lanes.clone(), own, PLENTY)
                        .await
                        .unwrap();
                let mut out = vec![0; lanes.len() * PACKING.seq_len as usize];
                while loader.next(&mut out).await.unwrap().is_some() {}
                let planned = plan(&d.epoch, &d.index, lanes, own).into_iter();
                let want = chunk_bytes(&d.index, planned.map(|n| n.row_group));
                assert_eq!(
                    reader.bytes_read(),
                    want,
                    "seed {seed}, rank {rank} of {world}"
                );
                total += want;
            }
            assert!(total >= dataset, "seed {seed}, world {world}");
            if world == 1 {
                assert_eq!(total, dataset, "seed {seed}");
            }
        }
    }
}
