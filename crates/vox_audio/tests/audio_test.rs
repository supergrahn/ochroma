use glam::Vec3;
use vox_audio::{AudioEngine, AudioSource};

#[test]
fn close_source_louder_than_far() {
    let mut engine = AudioEngine::new(64);
    engine.set_listener(Vec3::ZERO);
    let close = AudioSource { id: 1, position: Vec3::new(1.0, 0.0, 0.0), volume: 1.0, looping: false, clip: "test".into() };
    let far = AudioSource { id: 2, position: Vec3::new(100.0, 0.0, 0.0), volume: 1.0, looping: false, clip: "test".into() };
    assert!(engine.effective_volume(&close) > engine.effective_volume(&far));
}

#[test]
fn max_sources_evicts_quietest() {
    let mut engine = AudioEngine::new(2);
    engine.set_listener(Vec3::ZERO);
    engine.play(AudioSource { id: 1, position: Vec3::new(1.0, 0.0, 0.0), volume: 1.0, looping: false, clip: "a".into() });
    engine.play(AudioSource { id: 2, position: Vec3::new(2.0, 0.0, 0.0), volume: 1.0, looping: false, clip: "b".into() });
    engine.play(AudioSource { id: 3, position: Vec3::new(100.0, 0.0, 0.0), volume: 0.1, looping: false, clip: "c".into() });
    engine.tick(0.016);
    assert_eq!(engine.active_count(), 2);
}

#[test]
fn priority_sort_loudest_first() {
    let mut engine = AudioEngine::new(64);
    engine.set_listener(Vec3::ZERO);
    engine.play(AudioSource { id: 0, position: Vec3::new(50.0, 0.0, 0.0), volume: 1.0, looping: false, clip: "far".into() });
    engine.play(AudioSource { id: 0, position: Vec3::new(1.0, 0.0, 0.0), volume: 1.0, looping: false, clip: "close".into() });
    let sorted = engine.active_sources_by_priority();
    assert_eq!(sorted[0].clip, "close");
}
