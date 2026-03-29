# Lighting Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `SkyModel` to `vox_render` that computes sun direction from time-of-day and a Preetham sky dome that returns horizon/zenith/sun colors for use in the renderer's background clear color.

**Architecture:** The existing `crates/vox_render/src/lighting.rs` already contains `SunModel` (sun direction, intensity, daytime check), `sky_color()` (simplified Preetham), `PointLight`, and `LightManager`. This plan extends the module with: (1) a free-function `sun_direction(hour, latitude_deg)` simplified variant (declination=0), (2) a `SkyColors` struct with zenith/horizon/sun RGB, (3) a proper `preetham_sky(sun_dir)` function with turbidity=2.5, (4) a `SkyModel` resource with time advancement and sky color output, and (5) a `SkyPlugin` with `sky_update_system`.

**Tech Stack:** `bevy_ecs = "0.16"`, `bevy_app = "0.16"`, `glam::Vec3`, self-contained math (no external atmosphere crates)

---

## Key Files (read before editing)

- `crates/vox_render/src/lighting.rs` — existing `SunModel`, `sky_color`, `PointLight`, `LightManager`
- `crates/vox_render/src/lib.rs` — already has `pub mod lighting;`
- `crates/vox_render/Cargo.toml` — already has `bevy_ecs`, `bevy_app`, `glam`

## File Structure

**Modify:**
- `crates/vox_render/src/lighting.rs` — add `sun_direction`, `SkyColors`, `preetham_sky`, `SkyModel`, `SkyPlugin`, `sky_update_system`

**No new files required.**

---

### Task 1: `sun_direction` free function and `SkyColors` struct

**Files:**
- Modify: `crates/vox_render/src/lighting.rs`

Add a simplified `sun_direction(hour, latitude_deg)` that assumes declination=0 (equinox) and computes elevation + azimuth from hour angle, returning a normalized Vec3. Also add the `SkyColors` struct.

- [ ] **Step 1: Write failing tests** — add a `#[cfg(test)] mod tests` block at the bottom of `lighting.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    #[test]
    fn sun_direction_noon_is_high() {
        let dir = sun_direction(12.0, 45.0);
        assert!(dir.y > 0.5, "sun at noon should be high; got y={}", dir.y);
        assert!((dir.length() - 1.0).abs() < 1e-4, "direction should be normalized");
    }

    #[test]
    fn sun_direction_midnight_is_below_horizon() {
        let dir = sun_direction(0.0, 45.0);
        assert!(dir.y < 0.0, "sun at midnight should be below horizon; got y={}", dir.y);
    }

    #[test]
    fn sun_direction_6am_is_near_horizon() {
        let dir = sun_direction(6.0, 0.0);
        // At equator, equinox, 6am -> sun near horizon
        assert!(dir.y.abs() < 0.3, "sun at 6am equator should be near horizon; got y={}", dir.y);
    }

    #[test]
    fn sky_colors_values_in_range() {
        let dir = sun_direction(12.0, 45.0);
        let colors = preetham_sky(dir);
        for c in &colors.zenith {
            assert!(*c >= 0.0 && *c <= 1.0, "zenith component {} out of [0,1]", c);
        }
        for c in &colors.horizon {
            assert!(*c >= 0.0 && *c <= 1.0, "horizon component {} out of [0,1]", c);
        }
        for c in &colors.sun {
            assert!(*c >= 0.0 && *c <= 1.0, "sun component {} out of [0,1]", c);
        }
    }

    #[test]
    fn sky_colors_sunset_has_warm_horizon() {
        let dir = sun_direction(18.5, 45.0);
        let colors = preetham_sky(dir);
        // At sunset the horizon should be warmer (more red than blue)
        assert!(
            colors.horizon[0] > colors.horizon[2],
            "sunset horizon should be redder than blue: r={} b={}",
            colors.horizon[0], colors.horizon[2]
        );
    }
}
```

- [ ] **Step 2: Confirm they fail**

```bash
cargo test -p vox_render --lib -- lighting::tests::sun_direction_noon_is_high 2>&1 | tail -5
```

Expected: FAIL — `sun_direction` (free function) not defined.

- [ ] **Step 3: Implement** — add after the existing `LightManager` impl block, before `#[cfg(test)]`:

```rust
// ── Simplified Sun Direction ──────────────────────────────────────────────

/// Compute sun direction from hour (0-24) and latitude (degrees).
///
/// Uses a simplified model with declination = 0 (equinox approximation).
/// Returns a normalized Vec3 where +Y is up, +X is east, +Z is south.
pub fn sun_direction(hour: f32, latitude_deg: f32) -> Vec3 {
    let lat = latitude_deg.to_radians();
    // Hour angle: 0 at noon, negative morning, positive afternoon
    let hour_angle = (hour - 12.0) * 15.0_f32.to_radians();
    // Declination = 0 (equinox)
    let sin_alt = lat.sin() * 0.0 + lat.cos() * 1.0 * hour_angle.cos();
    let altitude = sin_alt.asin();
    let cos_az = (0.0 - lat.sin() * sin_alt) / (lat.cos() * altitude.cos() + 1e-10);
    let azimuth = if hour_angle.sin() > 0.0 {
        std::f32::consts::PI - cos_az.clamp(-1.0, 1.0).acos()
    } else {
        std::f32::consts::PI + cos_az.clamp(-1.0, 1.0).acos()
    };

    Vec3::new(
        -azimuth.sin() * altitude.cos(),
        altitude.sin(),
        -azimuth.cos() * altitude.cos(),
    )
    .normalize()
}

// ── Sky Colors ────────────────────────────────────────────────────────────

/// Zenith, horizon, and sun disk colors from a Preetham-inspired sky model.
#[derive(Debug, Clone, Copy)]
pub struct SkyColors {
    /// RGB color at the zenith (straight up).
    pub zenith: [f32; 3],
    /// RGB color at the horizon.
    pub horizon: [f32; 3],
    /// RGB color of the sun disk.
    pub sun: [f32; 3],
}

/// Compute sky colors using a simplified Preetham model with turbidity = 2.5.
///
/// The sun altitude drives the overall color tone:
/// - High sun: blue zenith, pale horizon, white-yellow sun
/// - Low sun (sunset): deep blue zenith, orange/red horizon, orange sun
/// - Below horizon: dark zenith, dark horizon, no sun
///
/// All returned values are clamped to [0, 1].
pub fn preetham_sky(sun_dir: Vec3) -> SkyColors {
    let turbidity: f32 = 2.5;
    let sun_alt = sun_dir.y.max(0.0); // altitude above horizon [0, 1]
    let sun_below = sun_dir.y < 0.0;

    if sun_below {
        // Night sky
        let night = 0.02;
        return SkyColors {
            zenith: [night * 0.5, night * 0.5, night * 0.8],
            horizon: [night, night, night * 1.2],
            sun: [0.0, 0.0, 0.0],
        };
    }

    // Zenith luminance (Preetham Eq. 7 simplified)
    let chi = (4.0 / 9.0 - turbidity / 120.0)
        * (std::f32::consts::PI - 2.0 * sun_alt.acos()).max(0.0);
    let zenith_y = ((4.0453 * turbidity - 4.971) * chi.tan()
        - 0.2155 * turbidity + 2.4192)
        .max(0.0)
        / 20.0; // scale to [0, ~1]

    // Zenith chromaticity (blue sky, less blue with turbidity)
    let zenith_r = (0.15 + 0.05 * (turbidity - 2.0)).clamp(0.0, 1.0) * zenith_y;
    let zenith_g = (0.2 + 0.1 * sun_alt).clamp(0.0, 1.0) * zenith_y;
    let zenith_b = (0.45 + 0.3 * sun_alt - 0.05 * turbidity).clamp(0.0, 1.0) * zenith_y;

    // Horizon: warmer, brighter, especially at low sun angles
    let sunset_factor = 1.0 - sun_alt; // 0 at zenith sun, 1 at horizon sun
    let horizon_r = (0.7 * sunset_factor + 0.3 * sun_alt).clamp(0.0, 1.0);
    let horizon_g = (0.35 * sunset_factor + 0.3 * sun_alt).clamp(0.0, 1.0);
    let horizon_b = (0.15 * sunset_factor + 0.4 * sun_alt).clamp(0.0, 1.0);

    // Sun disk color
    let sun_r = (1.0 - 0.3 * sun_alt).clamp(0.0, 1.0);
    let sun_g = (0.85 - 0.3 * sunset_factor).clamp(0.0, 1.0);
    let sun_b = (0.6 * sun_alt).clamp(0.0, 1.0);

    SkyColors {
        zenith: [zenith_r.clamp(0.0, 1.0), zenith_g.clamp(0.0, 1.0), zenith_b.clamp(0.0, 1.0)],
        horizon: [horizon_r, horizon_g, horizon_b],
        sun: [sun_r, sun_g, sun_b],
    }
}
```

- [ ] **Step 4: Run tests**

```bash
cargo test -p vox_render --lib -- lighting::tests 2>&1 | tail -10
```

Expected: all 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add crates/vox_render/src/lighting.rs
git commit -m "feat(lighting): sun_direction free function + SkyColors + preetham_sky model"
```

---

### Task 2: `SkyModel` resource with time advancement

**Files:**
- Modify: `crates/vox_render/src/lighting.rs`

Add a `SkyModel` struct that holds `hour` and `latitude_deg`, with an `update(dt)` method that advances time and a `sky_colors()` method.

- [ ] **Step 1: Write failing tests** — add inside `mod tests`:

```rust
    #[test]
    fn sky_model_advances_time() {
        let mut model = SkyModel::new(12.0, 45.0);
        model.update(3600.0); // 1 hour in seconds
        assert!((model.hour - 13.0).abs() < 0.01, "hour should advance by 1; got {}", model.hour);
    }

    #[test]
    fn sky_model_wraps_at_24() {
        let mut model = SkyModel::new(23.5, 45.0);
        model.update(3600.0); // 1 hour
        assert!(model.hour < 24.0, "hour should wrap; got {}", model.hour);
        assert!((model.hour - 0.5).abs() < 0.01);
    }

    #[test]
    fn sky_model_sky_colors_returns_valid() {
        let model = SkyModel::new(12.0, 45.0);
        let colors = model.sky_colors();
        assert!(colors.zenith[2] > 0.0, "noon zenith should have some blue");
    }
```

- [ ] **Step 2: Confirm they fail**

```bash
cargo test -p vox_render --lib -- lighting::tests::sky_model_advances_time 2>&1 | tail -5
```

Expected: FAIL — `SkyModel` not defined (the existing `SunModel` is a different struct).

- [ ] **Step 3: Implement** — add after `preetham_sky`:

```rust
// ── Sky Model (time-driven) ───────────────────────────────────────────────

/// Time-of-day driven sky model that provides continuously updating sky colors.
///
/// Call `update(dt_seconds)` each frame to advance the clock.
/// Call `sky_colors()` to get the current zenith/horizon/sun colors.
pub struct SkyModel {
    /// Current hour of day (0.0 .. 24.0).
    pub hour: f32,
    /// Observer latitude in degrees.
    pub latitude_deg: f32,
    /// Time scale multiplier (1.0 = real-time, 60.0 = 1 min per second).
    pub time_scale: f32,
}

impl SkyModel {
    pub fn new(hour: f32, latitude_deg: f32) -> Self {
        Self {
            hour,
            latitude_deg,
            time_scale: 1.0,
        }
    }

    /// Advance time by `dt` seconds (real-time, before time_scale).
    pub fn update(&mut self, dt: f32) {
        self.hour += (dt * self.time_scale) / 3600.0;
        self.hour %= 24.0;
        if self.hour < 0.0 {
            self.hour += 24.0;
        }
    }

    /// Get the current sun direction.
    pub fn sun_dir(&self) -> Vec3 {
        sun_direction(self.hour, self.latitude_deg)
    }

    /// Get the current sky colors.
    pub fn sky_colors(&self) -> SkyColors {
        preetham_sky(self.sun_dir())
    }
}
```

- [ ] **Step 4: Run tests**

```bash
cargo test -p vox_render --lib -- lighting::tests 2>&1 | tail -10
```

Expected: all 8 tests pass.

- [ ] **Step 5: Commit**

```bash
git add crates/vox_render/src/lighting.rs
git commit -m "feat(lighting): SkyModel with time advancement and sky_colors output"
```

---

### Task 3: `SkyPlugin` with ECS resource and system

**Files:**
- Modify: `crates/vox_render/src/lighting.rs`

Expose `SkyModel` as a bevy_ecs `Resource` and add a `sky_update_system` + `SkyPlugin`.

- [ ] **Step 1: Write failing tests** — add inside `mod tests`:

```rust
    #[test]
    fn sky_plugin_builds_without_panic() {
        use bevy_app::App;
        let mut app = App::new();
        app.add_plugins(SkyPlugin::new(12.0, 45.0));
    }

    #[test]
    fn sky_update_system_advances_time() {
        use bevy_ecs::schedule::Schedule;
        use bevy_ecs::world::World;

        let mut world = World::new();
        world.insert_resource(SkyModelResource(SkyModel::new(12.0, 45.0)));
        world.insert_resource(SkyDeltaTime(1.0)); // 1 second

        let mut schedule = Schedule::default();
        schedule.add_systems(sky_update_system);
        schedule.run(&mut world);

        let res = world.resource::<SkyModelResource>();
        let expected = 12.0 + 1.0 / 3600.0;
        assert!(
            (res.0.hour - expected).abs() < 1e-5,
            "hour should advance by 1s; got {}",
            res.0.hour
        );
    }
```

- [ ] **Step 2: Confirm they fail**

```bash
cargo test -p vox_render --lib -- lighting::tests::sky_plugin_builds 2>&1 | tail -5
```

Expected: FAIL — `SkyPlugin` not defined.

- [ ] **Step 3: Implement** — add after `SkyModel` impl, with necessary imports at the top of the file:

```rust
use bevy_ecs::prelude::*;

/// bevy_ecs Resource wrapping `SkyModel`.
#[derive(Resource)]
pub struct SkyModelResource(pub SkyModel);

/// Delta time resource for `sky_update_system` (seconds).
#[derive(Resource, Debug, Clone, Copy)]
pub struct SkyDeltaTime(pub f32);

impl Default for SkyDeltaTime {
    fn default() -> Self {
        Self(1.0 / 60.0)
    }
}

/// System that advances `SkyModelResource` by `SkyDeltaTime` each frame.
pub fn sky_update_system(
    dt: Res<SkyDeltaTime>,
    mut sky: ResMut<SkyModelResource>,
) {
    sky.0.update(dt.0);
}

/// Bevy plugin that inserts `SkyModelResource` and registers `sky_update_system`.
pub struct SkyPlugin {
    pub initial_hour: f32,
    pub latitude_deg: f32,
}

impl SkyPlugin {
    pub fn new(initial_hour: f32, latitude_deg: f32) -> Self {
        Self { initial_hour, latitude_deg }
    }
}

impl bevy_app::Plugin for SkyPlugin {
    fn build(&self, app: &mut bevy_app::App) {
        app.insert_resource(SkyModelResource(SkyModel::new(self.initial_hour, self.latitude_deg)));
        app.insert_resource(SkyDeltaTime::default());
        app.add_systems(bevy_app::Update, sky_update_system);
    }
}
```

- [ ] **Step 4: Run tests**

```bash
cargo test -p vox_render --lib -- lighting::tests 2>&1 | tail -10
```

Expected: all 10 tests pass.

- [ ] **Step 5: Commit**

```bash
git add crates/vox_render/src/lighting.rs
git commit -m "feat(lighting): SkyPlugin + sky_update_system ECS integration for time-of-day sky"
```

---

## Self-Review Checklist

- [x] **Spec coverage:** All 3 tasks covered (sun_direction + SkyColors, SkyModel, SkyPlugin)
- [x] **No placeholders:** All code blocks are complete with real math and types
- [x] **Type consistency:** `Vec3` from glam throughout, `f32` angles in radians internally
- [x] **TDD:** Tests written before implementation for every task
- [x] **Self-contained math:** No external atmosphere crates; Preetham coefficients inline
- [x] **Engine generality:** No game-specific concepts; pure rendering infrastructure
- [x] **Existing patterns:** Follows `render_ecs.rs` Plugin pattern with Resource + system
