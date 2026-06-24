use vox_sim::deterministic::{DeterministicRng, PlayerAction, SimulationRecorder};

#[test]
fn same_seed_same_output() {
    let mut rng1 = DeterministicRng::new(12345);
    let mut rng2 = DeterministicRng::new(12345);

    let vals1: Vec<u64> = (0..100).map(|_| rng1.next_u64()).collect();
    let vals2: Vec<u64> = (0..100).map(|_| rng2.next_u64()).collect();

    assert_eq!(vals1, vals2);
    assert_eq!(rng1.draws(), 100);
    assert_eq!(rng2.draws(), 100);
}

#[test]
fn different_seeds_differ() {
    let mut rng1 = DeterministicRng::new(1);
    let mut rng2 = DeterministicRng::new(2);

    let v1 = rng1.next_u64();
    let v2 = rng2.next_u64();
    assert_ne!(v1, v2);
}

#[test]
fn reset_reproduces_sequence() {
    let mut rng = DeterministicRng::new(42);
    let first_run: Vec<f64> = (0..50).map(|_| rng.next_f64()).collect();

    rng.reset();
    assert_eq!(rng.draws(), 0);

    let second_run: Vec<f64> = (0..50).map(|_| rng.next_f64()).collect();
    assert_eq!(first_run, second_run);
}

#[test]
fn recorder_saves_and_loads() {
    let dir = std::env::temp_dir().join("vox_sim_test_recording.json");

    let mut recorder = SimulationRecorder::new();
    let actions = vec![PlayerAction {
        action_type: "build".into(),
        payload: vec![1, 2, 3],
    }];
    recorder.record_tick(0, actions.clone(), 999);
    recorder.record_tick(1, vec![], 1000);

    recorder.save_recording(&dir).expect("save should succeed");
    let loaded = SimulationRecorder::load_recording(&dir).expect("load should succeed");

    assert_eq!(loaded.tick_count(), 2);
    let (replayed_actions, seed) = loaded.replay_tick(0).unwrap();
    assert_eq!(replayed_actions, actions);
    assert_eq!(seed, 999);

    std::fs::remove_file(&dir).ok();
}

#[test]
fn replay_matches_original() {
    let mut recorder = SimulationRecorder::new();

    // Simulate 10 ticks, recording RNG outputs
    let base_seed = 7777u64;
    let mut original_values = Vec::new();

    for tick in 0..10u64 {
        let seed = base_seed.wrapping_add(tick);
        let action = PlayerAction {
            action_type: format!("action_{tick}"),
            payload: vec![tick as u8],
        };
        recorder.record_tick(tick, vec![action], seed);

        let mut rng = DeterministicRng::new(seed);
        original_values.push(rng.next_u64());
    }

    // Replay and verify identical RNG output
    for tick in 0..10u64 {
        let (actions, seed) = recorder.replay_tick(tick).unwrap();
        assert_eq!(actions.len(), 1);
        assert_eq!(actions[0].action_type, format!("action_{tick}"));

        let mut rng = DeterministicRng::new(seed);
        assert_eq!(rng.next_u64(), original_values[tick as usize]);
    }
}

#[test]
fn draw_counter_tracks_all_methods() {
    let mut rng = DeterministicRng::new(1);
    rng.next_u64();
    rng.next_f64();
    rng.next_range(0, 100);
    assert_eq!(rng.draws(), 3);
}

#[test]
fn recording_serializes_ticks_in_ascending_order() {
    // Record ticks in deliberately scrambled order; the saved action-log JSON MUST
    // list them in ascending tick order so the replay file is byte-deterministic
    // across runs/machines (BTreeMap guarantee). A HashMap-backed `records` would
    // serialize them in arbitrary process-stable order — different file bytes for
    // the same recording, breaking shareable/diffable replays.
    let path = std::env::temp_dir().join("vox_sim_recorder_order_test.json");
    let mut recorder = SimulationRecorder::new();
    for &t in &[5u64, 2, 8, 1] {
        recorder.record_tick(t, vec![], t * 10);
    }
    recorder.save_recording(&path).expect("save should succeed");
    let json = std::fs::read_to_string(&path).expect("read back the recording");
    std::fs::remove_file(&path).ok();

    // Tick numbers serialize as JSON object keys "1","2","5","8".
    let p1 = json.find("\"1\":").expect("tick 1 key present");
    let p2 = json.find("\"2\":").expect("tick 2 key present");
    let p5 = json.find("\"5\":").expect("tick 5 key present");
    let p8 = json.find("\"8\":").expect("tick 8 key present");
    assert!(
        p1 < p2 && p2 < p5 && p5 < p8,
        "recording must serialize ticks ascending for a byte-deterministic replay \
         file; got positions 1@{p1} 2@{p2} 5@{p5} 8@{p8} — did `records` revert to HashMap?"
    );
}
