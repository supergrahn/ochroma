//! Tests for [`vox_aether::sky_model`] — solar/lunar positions and lighting.
use vox_aether::sky_model::{Observer, Sky, composite_lit, solar_dir};

fn obs(lat: f32, lon: f32, day: u32, hour: f32) -> Observer {
    Observer {
        latitude_deg: lat,
        longitude_deg: lon,
        day_of_year: day,
        hour,
    }
}

#[test]
fn the_sun_is_up_by_day_and_the_moon_takes_over_at_night() {
    let noon = Sky::at_time(0.5, 0.0, 0.0);
    let midnight = Sky::at_time(0.0, 0.0, 0.0);
    assert!(noon.sun_dir().y > 0.5, "sun high at noon");
    assert!(noon.light().is_day && noon.light().intensity > 0.5);
    assert!(!midnight.light().is_day, "moon at midnight");
    assert!(
        midnight.light().intensity < noon.light().intensity,
        "moon dimmer than sun"
    );
    assert!(
        midnight.light().intensity > 0.0,
        "the moon still gives a little light"
    );
}

#[test]
fn clouds_and_fog_dim_the_light() {
    let clear = Sky::at_time(0.5, 0.0, 0.0);
    let cloudy = Sky::at_time(0.5, 0.8, 0.0);
    let foggy = Sky::at_time(0.5, 0.0, 0.8);
    assert!(
        cloudy.light().intensity < clear.light().intensity,
        "clouds dim the sun"
    );
    assert!(
        foggy.light().intensity < clear.light().intensity,
        "fog dims the sun"
    );
    let both = Sky::at_time(0.5, 0.8, 0.8);
    assert!(both.light().intensity < cloudy.light().intensity);
    assert!(both.light().intensity < foggy.light().intensity);
}

#[test]
fn the_sun_crosses_the_sky_east_to_west() {
    let sunrise = Sky::at_time(0.25, 0.0, 0.0).sun_dir();
    let sunset = Sky::at_time(0.75, 0.0, 0.0).sun_dir();
    assert!(sunrise.x > 0.5, "sunrise in the east (+X)");
    assert!(sunset.x < -0.5, "sunset in the west (-X)");
    assert!(
        sunrise.y.abs() < 0.2 && sunset.y.abs() < 0.2,
        "both near the horizon"
    );
}

#[test]
fn night_geometry_is_dimmer_than_day_geometry() {
    let geom = [[200u8, 200, 200, 255]];
    let bg = [[10u8, 10, 20, 255]];
    let (day, _) = composite_lit(&geom, &bg, Sky::at_time(0.5, 0.0, 0.0));
    let (night, _) = composite_lit(&geom, &bg, Sky::at_time(0.0, 0.0, 0.0));
    let lum = |p: [u8; 4]| p[0] as u32 + p[1] as u32 + p[2] as u32;
    assert!(
        lum(night[0]) < lum(day[0]),
        "the same wall is darker at night"
    );
}

#[test]
fn the_real_sun_is_overhead_at_the_equator_at_an_equinox_noon() {
    let s = solar_dir(obs(0.0, 0.0, 80, 12.0));
    assert!(
        s.y > 0.98,
        "equator equinox noon: sun overhead, alt sin={}",
        s.y
    );
    let mid = solar_dir(obs(0.0, 0.0, 80, 0.0));
    assert!(
        mid.y < -0.9,
        "equator midnight: sun below horizon, {}",
        mid.y
    );
}

#[test]
fn summer_noon_sun_is_higher_than_winter_noon_in_the_north() {
    let summer = solar_dir(obs(59.9, 10.75, 172, 12.0)).y;
    let winter = solar_dir(obs(59.9, 10.75, 355, 12.0)).y;
    assert!(
        summer > winter,
        "summer noon higher than winter: {summer} vs {winter}"
    );
    assert!(summer > 0.7, "high summer sun in Oslo");
    assert!(
        winter > 0.0 && winter < 0.25,
        "low winter sun in Oslo, just above the horizon"
    );
}

#[test]
fn the_sun_rises_in_the_eastern_sky() {
    let s = solar_dir(obs(45.0, 0.0, 80, 8.0));
    assert!(s.y > 0.0, "sun is up by 8am near equinox");
    assert!(s.x > 0.0, "morning sun is in the east (+X)");
}

#[test]
fn a_real_polar_summer_keeps_the_sun_up_at_midnight() {
    let s = solar_dir(obs(70.0, 25.0, 172, 0.0));
    assert!(
        s.y > 0.0,
        "midnight sun above the Arctic Circle, alt sin={}",
        s.y
    );
}
