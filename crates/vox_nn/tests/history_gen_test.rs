use vox_nn::history_gen::*;

#[test]
fn generate_history_produces_city() {
    let history = generate_history(42);
    assert!(!history.city_name.is_empty());
    assert!(history.founding_year >= 800 && history.founding_year <= 1800);
    assert!(!history.eras.is_empty());
    assert!(!history.landmarks.is_empty());
    assert!(!history.district_names.is_empty());
}

#[test]
fn deterministic_history() {
    let a = generate_history(42);
    let b = generate_history(42);
    assert_eq!(a.city_name, b.city_name);
    assert_eq!(a.founding_year, b.founding_year);
    assert_eq!(a.eras.len(), b.eras.len());
}

#[test]
fn different_seeds_different_cities() {
    let a = generate_history(1);
    let b = generate_history(2);
    assert_ne!(a.city_name, b.city_name);
}

#[test]
fn citizen_names_generated() {
    let (first, last) = generate_citizen_name(42);
    assert!(!first.is_empty());
    assert!(!last.is_empty());
}

#[test]
fn eras_are_chronological() {
    let history = generate_history(99);
    for window in history.eras.windows(2) {
        assert!(
            window[0].end_year <= window[1].end_year,
            "Eras should be chronological"
        );
    }
}
