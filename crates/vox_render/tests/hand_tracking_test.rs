use glam::Vec3;
use vox_render::hand_tracking::*;

fn make_pinch_hand() -> HandState {
    let mut hand = HandState::default();
    // Move thumb and index tips very close together.
    hand.finger_tips[THUMB] = Vec3::new(0.0, 0.0, -0.05);
    hand.finger_tips[INDEX] = Vec3::new(0.005, 0.0, -0.05);
    hand
}

fn make_grab_hand() -> HandState {
    let mut hand = HandState::default();
    // Curl all fingers close to their bases (near the palm).
    for i in 0..5 {
        hand.finger_tips[i] = hand.finger_bases[i] + Vec3::new(0.0, 0.0, -0.01);
    }
    hand
}

fn make_point_hand() -> HandState {
    let mut hand = HandState::default();
    // Index fully extended.
    hand.finger_tips[INDEX] = hand.finger_bases[INDEX] + Vec3::new(0.0, 0.0, -0.10);
    // Others curled.
    for &f in &[MIDDLE, RING, PINKY] {
        hand.finger_tips[f] = hand.finger_bases[f] + Vec3::new(0.0, 0.0, -0.01);
    }
    // Thumb curled too.
    hand.finger_tips[THUMB] = hand.finger_bases[THUMB] + Vec3::new(0.0, 0.0, -0.01);
    hand
}

#[test]
fn detect_pinch() {
    let recognizer = GestureRecognizer::new();
    let hand = make_pinch_hand();
    assert_eq!(recognizer.recognize(&hand), Gesture::Pinch);
}

#[test]
fn detect_grab() {
    let recognizer = GestureRecognizer::new();
    let hand = make_grab_hand();
    assert_eq!(recognizer.recognize(&hand), Gesture::Grab);
}

#[test]
fn detect_point() {
    let recognizer = GestureRecognizer::new();
    let hand = make_point_hand();
    assert_eq!(recognizer.recognize(&hand), Gesture::Point);
}

#[test]
fn detect_none_default() {
    let recognizer = GestureRecognizer::new();
    let hand = HandState::default();
    assert_eq!(recognizer.recognize(&hand), Gesture::None);
}

#[test]
fn pinch_threshold_boundary() {
    let recognizer = GestureRecognizer::new();
    let mut hand = HandState::default();
    // Place thumb and index exactly at threshold — should NOT pinch.
    let threshold = recognizer.thresholds.pinch_distance;
    hand.finger_tips[THUMB] = Vec3::ZERO;
    hand.finger_tips[INDEX] = Vec3::new(threshold + 0.001, 0.0, 0.0);
    assert_ne!(recognizer.recognize(&hand), Gesture::Pinch);

    // Just under threshold — should pinch.
    hand.finger_tips[INDEX] = Vec3::new(threshold - 0.005, 0.0, 0.0);
    assert_eq!(recognizer.recognize(&hand), Gesture::Pinch);
}

#[test]
fn interaction_mapping() {
    let mgr = InteractionManager::new();

    assert_eq!(mgr.update(&make_pinch_hand()), InteractionAction::Select);
    assert_eq!(mgr.update(&make_grab_hand()), InteractionAction::Move);
    assert_eq!(mgr.update(&make_point_hand()), InteractionAction::Place);
    assert_eq!(mgr.update(&HandState::default()), InteractionAction::None);
}
