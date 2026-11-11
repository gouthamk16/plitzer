use super::*;
use proptest::prelude::*;

/// Every lane's parts, window by window.
fn all_lanes(o: &Order) -> Vec<Vec<Vec<Part>>> {
    let lanes = |w: &Window| (0..o.lanes).map(|l| o.lane(w, l)).collect();
    o.windows().iter().map(lanes).collect()
}

/// Parts laid end to end, with the neighbours in one row group joined.
fn joined(parts: impl IntoIterator<Item = Part>) -> Vec<Part> {
    let mut out: Vec<Part> = Vec::new();
    for p in parts {
        match out.last_mut() {
            Some(last) if last.row_group == p.row_group && last.slots.end == p.slots.start => {
                last.slots.end = p.slots.end;
            }
            _ => out.push(p),
        }
    }
    out
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
    // The lanes' slices, laid end to end in lane order, are the window's
    // slots in storage order: each slot is in exactly one slice, and each
    // slice is one contiguous run.
    #[test]
    fn exactly_once((slots, lanes, window, seed) in inputs()) {
        let o = Order { seed, epoch: 3, lanes, window, slots: &slots };
        for (w, lanes) in o.windows().iter().zip(all_lanes(&o)) {
            let all: Vec<Part> = w.groups().iter()
                .filter(|&&g| slots[g as usize] > 0)
                .map(|&g| Part { row_group: g, slots: 0..u64::from(slots[g as usize]) })
                .collect();
            prop_assert_eq!(joined(lanes.into_iter().flatten()), all);
        }
    }

    #[test]
    fn windows_partition((slots, lanes, window, seed) in inputs()) {
        let o = Order { seed, epoch: 3, lanes, window, slots: &slots };
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
        prop_assert_eq!(all, (0..slots.len() as u32).collect::<Vec<_>>());
    }

    // Each lane gets the same share of a window's slots, to within one.
    #[test]
    fn slices_balanced((slots, lanes, window, seed) in inputs()) {
        let o = Order { seed, epoch: 3, lanes, window, slots: &slots };
        for w in o.windows() {
            let sizes: Vec<u64> = (0..lanes).map(|l| o.lane_len(&w, l)).collect();
            let spread = sizes.iter().max().unwrap() - sizes.iter().min().unwrap();
            prop_assert!(spread <= 1, "sizes {:?}", sizes);
        }
    }

    #[test]
    fn lane_len_matches((slots, lanes, window, seed) in inputs()) {
        let o = Order { seed, epoch: 3, lanes, window, slots: &slots };
        for w in o.windows() {
            for l in 0..lanes {
                let parts: u64 = o.lane(&w, l).iter().map(|p| p.slots.end - p.slots.start).sum();
                prop_assert_eq!(o.lane_len(&w, l), parts);
            }
        }
    }

    #[test]
    fn lane_groups_match((slots, lanes, window, seed) in inputs()) {
        let o = Order { seed, epoch: 3, lanes, window, slots: &slots };
        for w in o.windows() {
            for (l, range) in (0..lanes).zip(o.lane_groups(&w, 0..lanes)) {
                let touched: Vec<u32> = o.lane(&w, l).iter().map(|p| p.row_group).collect();
                // Row groups with no slots inside the range are in no part.
                let range_groups: Vec<u32> = w.groups()[range]
                    .iter()
                    .copied()
                    .filter(|&g| slots[g as usize] > 0)
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
    let slots = [3, 1, 4, 1, 5, 9, 2, 6];
    let o = Order {
        seed: 7,
        epoch: 0,
        lanes: 2,
        window: 3,
        slots: &slots,
    };
    let triples = |parts: &[Part]| -> Vec<(u32, u64, u64)> {
        parts
            .iter()
            .map(|p| (p.row_group, p.slots.start, p.slots.end))
            .collect()
    };
    let got: Vec<Vec<Vec<(u32, u64, u64)>>> = all_lanes(&o)
        .iter()
        .map(|w| w.iter().map(|l| triples(l)).collect())
        .collect();
    let groups: Vec<Vec<u32>> = o.windows().iter().map(|w| w.groups().to_vec()).collect();
    assert_eq!(groups, [[1, 5, 6].as_slice(), &[2, 4, 7], &[0, 3]]);
    #[rustfmt::skip]
    assert_eq!(got, [
        [&[(1, 0, 1), (5, 0, 5)][..], &[(5, 5, 9), (6, 0, 2)]],
        [&[(2, 0, 4), (4, 0, 3)], &[(4, 3, 5), (7, 0, 6)]],
        [&[(0, 0, 2)], &[(0, 2, 3), (3, 0, 1)]],
    ]);
    let mut items: Vec<u32> = (0..10).collect();
    o.shuffle_lane(&o.windows()[1], 1, &mut items);
    assert_eq!(items, [2, 0, 3, 1, 9, 8, 5, 4, 6, 7]);

    // Many windows and lanes, so a key built from the wrong window or
    // lane number changes the hash.
    let slots: Vec<u32> = (0..500).map(|i| i % 37 * 11).collect();
    let o = Order {
        seed: 7,
        epoch: 0,
        lanes: 8,
        window: 16,
        slots: &slots,
    };
    let mut hash = 0u64;
    for w in o.windows() {
        for l in 0..8 {
            let mut items: Vec<u32> = (0..20).collect();
            o.shuffle_lane(&w, l, &mut items);
            let parts = o.lane(&w, l).into_iter();
            let xs =
                parts.map(|p| u64::from(p.row_group) << 40 | p.slots.start << 20 | p.slots.end);
            for x in xs.chain(items.into_iter().map(u64::from)) {
                hash = hash.wrapping_mul(1_000_003).wrapping_add(x);
            }
        }
    }
    assert_eq!(hash, 17_673_357_937_393_329_819);
}

#[test]
fn epochs_differ() {
    let slots = [1; 100];
    let at = |epoch| Order {
        seed: 7,
        epoch,
        lanes: 1,
        window: 10,
        slots: &slots,
    };
    assert_ne!(at(0).windows(), at(1).windows());

    let slots = [100];
    let lane = |epoch| {
        let o = Order {
            seed: 7,
            epoch,
            lanes: 1,
            window: 1,
            slots: &slots,
        };
        let mut items: Vec<u32> = (0..100).collect();
        o.shuffle_lane(&o.windows()[0], 0, &mut items);
        items
    };
    assert_ne!(lane(0), lane(1));
}
