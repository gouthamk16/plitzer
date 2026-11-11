use super::*;

// A GPU reads each row group of its windows once per window: what it reads
// is at least every window before its slowest lane, and at most every
// window up to its fastest lane and the prefetch's reach past it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn each_window_is_read_once() {
    for (seed, world) in [(1, 1), (2, 1), (3, 2)] {
        let d = data(seed).await;
        let lanes = gpu_lanes(LANES, world, 0);
        let store = Arc::new(LocalFileSystem::new_with_prefix(d.dir.path()).unwrap());
        let reader = Arc::new(Reader::new(store, Path::from(""), &d.index).unwrap());
        let fresh = vec![Position::default(); lanes.len()];
        let (o, i) = (d.orders.clone(), d.index.clone());
        let mut loader = Loader::new(o, reader.clone(), i, lanes.clone(), &fresh, PLENTY).unwrap();
        let mut out = vec![0; lanes.len() * SEQ_LEN as usize];
        for _ in 0..steps(&d) {
            loader.next(&mut out).await.unwrap();
        }
        // The prefetch is still reading, so settle it first.
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        let slices: Vec<_> = loader.lanes.iter().map(Lane::slice).collect();
        let (slowest, fastest) = (*slices.iter().min().unwrap(), *slices.iter().max().unwrap());
        let bytes = |ws: Vec<WindowGroups>| -> u64 {
            let chunk =
                |w: &WindowGroups, i: usize| u64::from(d.index.row_group(w.groups[i]).1.len);
            ws.iter()
                .flat_map(|w| w.used().map(move |i| chunk(w, i)))
                .sum()
        };
        let passed = windows(&d, lanes.clone(), (0, 0), slowest);
        let low = bytes(passed) - bytes(windows(&d, lanes.clone(), slowest, slowest));
        let high = bytes(windows(&d, lanes.clone(), (0, 0), after(&d, fastest, 2)));
        let got = reader.bytes_read();
        assert!(
            low <= got && got <= high,
            "seed {seed}: {low} <= {got} <= {high}"
        );
    }
}
