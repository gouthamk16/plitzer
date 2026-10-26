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
// range holds that step. On resume at step k, the plan starts at the cursor,
// and each replayed document is in one of its lane's chunks.
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
                let cursors: Vec<u64> = (range.clone().zip(own))
                    .map(|(l, &s)| d.epoch.check(l, s).unwrap())
                    .collect();
                let ahead: Vec<LaneState> = (own.iter().zip(&cursors))
                    .map(|(s, &doc)| LaneState {
                        doc,
                        offset: 0,
                        ..*s
                    })
                    .collect();
                let needs = plan(&d.epoch, &d.index, range.clone(), &ahead);
                let chunks = chunks(&d.epoch, &d.index, range.clone(), own, &cursors);
                assert!(!chunks.is_empty(), "some lane has a leftover to replay");
                let at = |step: u64, rg: u32| {
                    needs
                        .iter()
                        .any(|n| n.row_group == rg && n.first <= step && step <= n.last)
                };
                all.log.borrow_mut().clear();
                let mut lanes = Vec::new();
                for (i, (l, &s)) in range.zip(own).enumerate() {
                    lanes.push(d.epoch.lane(l, s, &all).unwrap());
                    let mut replayed = chunks.iter().filter(|c| c.lane == i);
                    for rg in all.log.take() {
                        assert!(
                            replayed.clone().any(|c| c.groups.contains(&rg)),
                            "replayed row group {rg} of lane {l} in no chunk"
                        );
                    }
                    let docs: u64 = replayed.by_ref().map(|c| c.docs).sum();
                    assert_eq!(
                        docs,
                        cursors[i] - s.doc,
                        "lane {l}: chunks cover the leftover"
                    );
                }
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
