use vox_sim::seasons::Season;

#[test]
fn season_from_day() {
    assert_eq!(Season::from_day(0), Season::Spring);
    assert_eq!(Season::from_day(100), Season::Summer);
    assert_eq!(Season::from_day(200), Season::Autumn);
    assert_eq!(Season::from_day(300), Season::Winter);
}

#[test]
fn winter_has_high_heating() {
    assert!(Season::Winter.heating_cost_multiplier() > Season::Summer.heating_cost_multiplier());
}

#[test]
fn winter_has_no_crops() {
    assert_eq!(Season::Winter.crop_growth_rate(), 0.0);
    assert!(Season::Spring.crop_growth_rate() > 0.0);
}
