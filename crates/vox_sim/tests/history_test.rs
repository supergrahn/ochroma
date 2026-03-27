use vox_sim::history::*;

#[test]
fn time_series_records_and_retrieves() {
    let mut ts = TimeSeries::new("test", 100);
    ts.record(0.0, 10.0);
    ts.record(1.0, 20.0);
    ts.record(2.0, 30.0);
    assert_eq!(ts.latest(), Some(30.0));
    assert_eq!(ts.len(), 3);
}

#[test]
fn time_series_caps_at_max_points() {
    let mut ts = TimeSeries::new("test", 5);
    for i in 0..10 {
        ts.record(i as f64, i as f64);
    }
    assert_eq!(ts.len(), 5);
    assert_eq!(ts.latest(), Some(9.0));
}

#[test]
fn time_series_stats() {
    let mut ts = TimeSeries::new("test", 100);
    ts.record(0.0, 10.0);
    ts.record(1.0, 20.0);
    ts.record(2.0, 30.0);
    assert_eq!(ts.min(), 10.0);
    assert_eq!(ts.max(), 30.0);
    assert!((ts.average() - 20.0).abs() < 0.01);
}

#[test]
fn game_history_records_all_metrics() {
    let mut history = GameHistory::new(1000);
    history.record_tick(0.0, 100, 50000.0, 1000.0, 800.0, 0.7, 0.1, 0.05, 0.3);
    history.record_tick(1.0, 110, 50200.0, 1100.0, 850.0, 0.72, 0.09, 0.04, 0.35);
    assert_eq!(history.population.len(), 2);
    assert_eq!(history.population.latest(), Some(110.0));
}
