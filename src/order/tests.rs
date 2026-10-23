use super::*;
use proptest::prelude::*;

/// The given row groups' documents in storage order.
fn storage(rows: &[u32], groups: impl IntoIterator<Item = u32>) -> Vec<Doc> {
    groups
        .into_iter()
        .flat_map(|g| (0..rows[g as usize]).map(move |row| Doc { row_group: g, row }))
        .collect()
}

/// Every lane's documents, window by window.
fn all_lanes(o: &Order) -> Vec<Vec<Vec<Doc>>> {
    let lanes = |w: &Window| (0..o.lanes).map(|l| o.lane(w, l)).collect();
    o.windows().iter().map(lanes).collect()
}

// Small row groups and lane counts close to them, so empty row groups,
// empty lanes (N < B) and short last windows all come up often.
fn inputs() -> impl Strategy<Value = (Vec<u32>, u32, u32, u64)> {
    (
        prop::collection::vec(0u32..20, 0..30),
        1u32..12,
        1u32..6,
        any::<u64>(),
    )
}

proptest! {
    #[test]
    fn exactly_once((rows, lanes, window, seed) in inputs()) {
        let o = Order { seed, epoch: 3, lanes, window, rows: &rows };
        let mut seen: Vec<Doc> = all_lanes(&o).into_iter().flatten().flatten().collect();
        seen.sort_unstable();
        prop_assert_eq!(seen, storage(&rows, 0..rows.len() as u32));
    }

    #[test]
    fn windows_partition((rows, lanes, window, seed) in inputs()) {
        let o = Order { seed, epoch: 3, lanes, window, rows: &rows };
        let windows = o.windows();
        let groups: Vec<&[u32]> = windows.iter().map(Window::groups).collect();
        for (i, g) in groups.iter().enumerate() {
            prop_assert!(g.is_sorted());
            if i + 1 < groups.len() {
                prop_assert_eq!(g.len(), window as usize);
            }
        }
        let mut all = groups.concat();
        all.sort_unstable();
        prop_assert_eq!(all, (0..rows.len() as u32).collect::<Vec<_>>());
    }

    // Sorted lane slices laid end to end in lane order must rebuild the
    // window's storage order: each slice is one contiguous run.
    #[test]
    fn slices_contiguous((rows, lanes, window, seed) in inputs()) {
        let o = Order { seed, epoch: 3, lanes, window, rows: &rows };
        for (w, lanes) in o.windows().iter().zip(all_lanes(&o)) {
            let sizes: Vec<usize> = lanes.iter().map(Vec::len).collect();
            let spread = sizes.iter().max().unwrap() - sizes.iter().min().unwrap();
            prop_assert!(spread <= 1, "sizes {:?}", sizes);
            let mut joined = Vec::new();
            for mut lane in lanes {
                lane.sort_unstable();
                joined.extend(lane);
            }
            prop_assert_eq!(joined, storage(&rows, w.groups().iter().copied()));
        }
    }

    #[test]
    fn lane_len_matches((rows, lanes, window, seed) in inputs()) {
        let o = Order { seed, epoch: 3, lanes, window, rows: &rows };
        for w in o.windows() {
            for l in 0..lanes {
                prop_assert_eq!(o.lane_len(&w, l), o.lane(&w, l).len() as u64);
            }
        }
    }

    #[test]
    fn lane_groups_match((rows, lanes, window, seed) in inputs()) {
        let o = Order { seed, epoch: 3, lanes, window, rows: &rows };
        for w in o.windows() {
            for (l, range) in (0..lanes).zip(o.lane_groups(&w, 0..lanes)) {
                let mut touched: Vec<u32> = o.lane(&w, l).iter().map(|d| d.row_group).collect();
                touched.dedup();
                touched.sort_unstable();
                touched.dedup();
                // Zero-row groups inside the range are touched by no document.
                let range_groups: Vec<u32> = w.groups()[range]
                    .iter()
                    .copied()
                    .filter(|&g| rows[g as usize] > 0)
                    .collect();
                prop_assert_eq!(touched, range_groups);
            }
        }
    }

    #[test]
    fn gpu_lanes_partition(per in 1u32..50, world in 1u32..20) {
        let lanes = per * world;
        let joined: Vec<u32> = (0..world).flat_map(|r| gpu_lanes(lanes, world, r)).collect();
        prop_assert_eq!(joined, (0..lanes).collect::<Vec<_>>());
    }
}

#[test]
#[should_panic(expected = "must divide")]
fn gpu_lanes_rejects_uneven_world() {
    gpu_lanes(10, 4, 0);
}

#[test]
fn golden() {
    let rows = [3, 1, 4, 1, 5, 9, 2, 6];
    let o = Order {
        seed: 7,
        epoch: 0,
        lanes: 2,
        window: 3,
        rows: &rows,
    };
    let pairs = |docs: &[Doc]| {
        docs.iter()
            .map(|d| (d.row_group, d.row))
            .collect::<Vec<_>>()
    };
    let got: Vec<Vec<Vec<(u32, u32)>>> = all_lanes(&o)
        .iter()
        .map(|w| w.iter().map(|l| pairs(l)).collect())
        .collect();
    let groups: Vec<Vec<u32>> = o.windows().iter().map(|w| w.groups().to_vec()).collect();
    assert_eq!(groups, [[1, 5, 6].as_slice(), &[2, 4, 7], &[0, 3]]);
    #[rustfmt::skip]
    assert_eq!(got, [
        [&[(5, 1), (1, 0), (5, 0), (5, 4), (5, 2), (5, 3)][..],
         &[(6, 1), (6, 0), (5, 8), (5, 5), (5, 6), (5, 7)]],
        [&[(4, 0), (2, 0), (2, 3), (2, 1), (2, 2), (4, 2), (4, 1)],
         &[(7, 3), (7, 5), (4, 3), (7, 2), (4, 4), (7, 0), (7, 4), (7, 1)]],
        [&[(0, 0), (0, 1)],
         &[(3, 0), (0, 2)]],
    ]);

    // Many windows and lanes, so a key built from the wrong window or
    // lane number changes the hash.
    let rows: Vec<u32> = (0..500).map(|i| i % 37).collect();
    let o = Order {
        seed: 7,
        epoch: 0,
        lanes: 8,
        window: 16,
        rows: &rows,
    };
    let hash = all_lanes(&o).iter().flatten().flatten().fold(0u64, |h, d| {
        let x = u64::from(d.row_group) << 32 | u64::from(d.row);
        h.wrapping_mul(1_000_003).wrapping_add(x)
    });
    assert_eq!(hash, 2_028_779_180_813_407_981);
}

#[test]
fn epochs_differ() {
    let rows = [1; 100];
    let at = |epoch| Order {
        seed: 7,
        epoch,
        lanes: 1,
        window: 10,
        rows: &rows,
    };
    assert_ne!(at(0).windows(), at(1).windows());

    let rows = [100];
    let at = |epoch| Order {
        seed: 7,
        epoch,
        lanes: 1,
        window: 1,
        rows: &rows,
    };
    let lane = |epoch| {
        let o = at(epoch);
        o.lane(&o.windows()[0], 0)
    };
    assert_ne!(lane(0), lane(1));
}
