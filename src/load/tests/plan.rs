use std::collections::HashSet;

use super::*;

// Each GPU plans only the row groups of its own lanes' slices. Neighbours
// share at most the row groups at the edges of their slices.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn reads_only_own_row_groups() {
    let d = data(11).await;
    let total = d.index.rows().iter().filter(|&&r| r > 0).count();
    for world in [2, 4] {
        let mut planned = 0;
        let mut union = HashSet::new();
        for rank in 0..world {
            let lanes = gpu_lanes(LANES, world, rank);
            let states = vec![LaneState::default(); lanes.len()];
            let needs = plan(&d.epoch, &d.index, lanes, &states);
            planned += needs.len();
            union.extend(needs.iter().map(|n| n.row_group));
        }
        let edges = (world as usize - 1) * d.epoch.windows().len();
        assert!(union.len() >= total, "world {world}");
        assert!(
            planned <= total + edges,
            "world {world}: {planned} > {total} + {edges}"
        );
    }
}

// Every document a lane takes at step s is in a row group whose planned
// range holds that step, also after a resume at step k, where the plan
// starts at the cursor.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn plan_matches_lane_streams() {
    for seed in [13, 14] {
        let d = data(seed).await;
        let all = all(&d).await;
        let k = d.epoch.steps() / 2;
        let mut lanes: Vec<Lane> = (0..LANES)
            .map(|l| d.epoch.lane(l, LaneState::default(), &all).unwrap())
            .collect();
        let mut seq = vec![0; PACKING.seq_len as usize];
        for _ in 0..k {
            lanes.iter_mut().for_each(|l| {
                l.fill(&all, &mut seq);
            });
        }
        let states: Vec<LaneState> = lanes.iter().map(Lane::state).collect();
        for world in [1, 2] {
            for rank in 0..world {
                let range = gpu_lanes(LANES, world, rank);
                let own = &states[range.start as usize..range.end as usize];
                let cursors = cursors(&d.epoch, range.clone(), own).unwrap();
                let ahead = at_cursors(own, &cursors);
                let needs = plan(&d.epoch, &d.index, range.clone(), &ahead);
                let leftover = own.iter().zip(&cursors).any(|(s, &c)| s.doc < c);
                assert!(leftover, "some lane has a leftover to replay");
                let at = |step: u64, rg: u32| {
                    needs
                        .iter()
                        .any(|n| n.row_group == rg && n.first <= step && step <= n.last)
                };
                let mut lanes: Vec<Lane> = (range.zip(own))
                    .map(|(l, &s)| d.epoch.lane(l, s, &all).unwrap())
                    .collect();
                all.log.borrow_mut().clear();
                for step in k..d.epoch.steps() {
                    for l in &mut lanes {
                        l.fill(&all, &mut seq);
                    }
                    for rg in all.log.take() {
                        assert!(at(step, rg), "row group {rg} not planned at step {step}");
                    }
                }
            }
        }
    }
}
