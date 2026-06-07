//! AI-creates-code v1: a VETTED deterministic template library that turns an
//! [`crate::shell::intent::IntentAction::GenerateScript`] into a real, compilable
//! Rhai script on disk.
//!
//! This is the FIRST half of "Ask Ochroma generates, not just navigates": a
//! sentence → a typed template + clamped params → a hot-reloadable `.rhai` file
//! the Content browser picks up and the running game can swap in live. v1 is
//! fully deterministic (no model); the LLM upgrade reuses every piece here — it
//! only changes WHERE the (`template`, `params`) pair comes from, never the
//! template bodies, the clamps, or the compile-gate.
//!
//! Conventions are lifted from `assets/scripts/walking_sim.rhai` (the established
//! hot-reload script shape): each tunable lives as a LITERAL inside its own
//! zero-arg accessor `fn` (Rhai script-level `const`s are not visible inside `fn`
//! bodies the host calls directly), and a per-frame accessor takes `(t, phase)`
//! and returns a single number. The host calls these by name every frame.
//!
//! THREE invariants, each load-bearing and tested:
//!  1. Every numeric param is CLAMPED to a documented range before substitution —
//!     hostile values (`speed = 1e9`) can never generate a pathological literal
//!     (the wave-8/12 lesson: every numeric input path gets clamped).
//!  2. Every generated source is COMPILED by a real `rhai::Engine` before it is
//!     offered. A template that fails to compile is a bug caught at generation
//!     time — [`generate`] returns `Err`, never a broken script.
//!  3. Generation is DETERMINISTIC: the same (template, params) always yields
//!     byte-identical source.

use std::fmt;

/// Which vetted template to instantiate. Friendly, stable identifiers the parser
/// and (later) the LLM both emit. Adding a variant forces the exhaustive matches
/// in [`ScriptTemplate::all`], [`ScriptTemplate::id`], and
/// [`ScriptTemplate::from_id`] to be updated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScriptTemplate {
    /// Rotate an entity about an axis at a tunable speed.
    Spin,
    /// Vertical oscillation (bob) with a tunable amplitude + period.
    Bob,
    /// Oscillate an emitter's intensity between a min and max over a period.
    PulseLight,
}

impl ScriptTemplate {
    /// Every template (exhaustive — adding a variant breaks this until updated).
    pub fn all() -> [ScriptTemplate; 3] {
        [ScriptTemplate::Spin, ScriptTemplate::Bob, ScriptTemplate::PulseLight]
    }

    /// The stable string id used in receipts, file-name stems, and the LLM/parser
    /// contract.
    pub fn id(self) -> &'static str {
        match self {
            ScriptTemplate::Spin => "spin",
            ScriptTemplate::Bob => "bob",
            ScriptTemplate::PulseLight => "pulse_light",
        }
    }

    /// Parse a template id (the inverse of [`Self::id`]). `None` for an unknown id
    /// so a model/parser can never name a phantom template.
    pub fn from_id(id: &str) -> Option<ScriptTemplate> {
        match id {
            "spin" => Some(ScriptTemplate::Spin),
            "bob" => Some(ScriptTemplate::Bob),
            "pulse_light" => Some(ScriptTemplate::PulseLight),
            _ => None,
        }
    }
}

impl fmt::Display for ScriptTemplate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.id())
    }
}

/// One numeric parameter slot's documented, inclusive clamp range. The clamp is
/// the single authority on what literals can reach a generated script — a hostile
/// value is folded back to `min`/`max`, so the worst a caller can do is pin a
/// param to a documented extreme.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Range {
    pub min: f32,
    pub max: f32,
    pub default: f32,
}

impl Range {
    const fn new(min: f32, max: f32, default: f32) -> Self {
        Range { min, max, default }
    }

    /// Clamp a value into `[min, max]`. NaN folds to `default` (a NaN literal in a
    /// script would be both pathological and non-deterministic to format), and
    /// `+/-inf` fold to the bound via `clamp`.
    pub fn clamp(&self, v: f32) -> f32 {
        if v.is_nan() {
            self.default
        } else {
            v.clamp(self.min, self.max)
        }
    }
}

/// The documented clamp ranges for every template slot. Public so tests assert
/// the exact "documented max" a hostile value folds to.
pub mod ranges {
    use super::Range;
    // spin
    pub const SPIN_SPEED: Range = Range::new(0.0, 16.0, 0.4);
    pub const SPIN_AXIS: Range = Range::new(0.0, 2.0, 1.0); // 0=X 1=Y 2=Z (rounded)
    // bob
    pub const BOB_AMPLITUDE: Range = Range::new(0.0, 8.0, 0.3);
    pub const BOB_PERIOD: Range = Range::new(0.1, 60.0, 3.0);
    // pulse_light
    pub const PULSE_MIN: Range = Range::new(0.0, 100.0, 0.2);
    pub const PULSE_MAX: Range = Range::new(0.0, 100.0, 1.0);
    pub const PULSE_PERIOD: Range = Range::new(0.1, 60.0, 2.0);
}

/// The typed, clamped parameters for one template instantiation. Each constructor
/// CLAMPS on the way in, so a `Params` value can only ever hold documented-range
/// numbers — there is no path to construct an out-of-range param.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Params {
    Spin { speed: f32, axis: f32 },
    Bob { amplitude: f32, period: f32 },
    PulseLight { min: f32, max: f32, period: f32 },
}

impl Params {
    /// Clamp + build spin params. `axis` is rounded to the nearest of 0/1/2.
    pub fn spin(speed: f32, axis: f32) -> Params {
        Params::Spin {
            speed: ranges::SPIN_SPEED.clamp(speed),
            axis: ranges::SPIN_AXIS.clamp(axis).round(),
        }
    }

    /// Clamp + build bob params.
    pub fn bob(amplitude: f32, period: f32) -> Params {
        Params::Bob {
            amplitude: ranges::BOB_AMPLITUDE.clamp(amplitude),
            period: ranges::BOB_PERIOD.clamp(period),
        }
    }

    /// Clamp + build pulse-light params. `max` is additionally floored to `min`
    /// after clamping so the generated `max >= min` always holds (an inverted
    /// range would still compile, but the documented contract is min..=max).
    pub fn pulse_light(min: f32, max: f32, period: f32) -> Params {
        let cmin = ranges::PULSE_MIN.clamp(min);
        let cmax = ranges::PULSE_MAX.clamp(max).max(cmin);
        Params::PulseLight {
            min: cmin,
            max: cmax,
            period: ranges::PULSE_PERIOD.clamp(period),
        }
    }

    /// The template this `Params` instantiates.
    pub fn template(&self) -> ScriptTemplate {
        match self {
            Params::Spin { .. } => ScriptTemplate::Spin,
            Params::Bob { .. } => ScriptTemplate::Bob,
            Params::PulseLight { .. } => ScriptTemplate::PulseLight,
        }
    }
}

/// A successfully generated, compile-verified script ready to be written to disk.
#[derive(Debug, Clone, PartialEq)]
pub struct GeneratedScript {
    /// The file stem (no extension, no directory), e.g. `windmill_spin`.
    pub name: String,
    /// The full Rhai source (already proven to compile).
    pub source: String,
    /// Which template produced it.
    pub template: ScriptTemplate,
}

/// Why a generation attempt failed. A `CompileFailed` here is a TEMPLATE BUG (the
/// clamped params can never legitimately produce uncompilable source) — it exists
/// so the compile-gate is enforced, not silently skipped.
#[derive(Debug, Clone, PartialEq)]
pub enum GenError {
    /// The template/params disagreed (e.g. `Params::Bob` for `ScriptTemplate::Spin`).
    Mismatch,
    /// The generated source failed to compile in a real `rhai::Engine`.
    CompileFailed(String),
    /// The requested name had no usable characters (all stripped by sanitizing).
    EmptyName,
}

impl fmt::Display for GenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GenError::Mismatch => write!(f, "template/params mismatch"),
            GenError::CompileFailed(e) => write!(f, "generated script failed to compile: {e}"),
            GenError::EmptyName => write!(f, "script name was empty after sanitizing"),
        }
    }
}

/// Format an `f32` as a Rhai float literal that ALWAYS parses as a float (so the
/// engine never treats `1` as an int where a float is expected) and is
/// deterministic. Two decimal places mirrors the receipt formatting elsewhere in
/// the shell; trailing `.0` is kept (e.g. `4.00`).
fn lit(v: f32) -> String {
    // Clamp ranges keep this finite; defend anyway so a NaN/inf can never reach a
    // literal (it would not parse / would be non-deterministic).
    let v = if v.is_finite() { v } else { 0.0 };
    format!("{v:.4}")
}

/// Generate a compile-verified [`GeneratedScript`] for `template` with the given
/// `name` and clamped `params`.
///
/// `name` is sanitized to a safe file stem (lowercase ascii alnum + `_`); the
/// `params` must match `template` or [`GenError::Mismatch`] is returned. The
/// produced source is COMPILED by a real `rhai::Engine` before it is returned —
/// the compile-gate (invariant 2) is the reason this returns `Result`.
pub fn generate(
    template: ScriptTemplate,
    name: &str,
    params: Params,
) -> Result<GeneratedScript, GenError> {
    if params.template() != template {
        return Err(GenError::Mismatch);
    }
    let stem = sanitize_stem(name);
    if stem.is_empty() {
        return Err(GenError::EmptyName);
    }

    let source = match params {
        Params::Spin { speed, axis } => spin_source(speed, axis),
        Params::Bob { amplitude, period } => bob_source(amplitude, period),
        Params::PulseLight { min, max, period } => pulse_light_source(min, max, period),
    };

    // Invariant 2: compile the generated source in a REAL engine. A failure here
    // is a template bug — surface it rather than ever offering a broken script.
    let engine = rhai::Engine::new();
    if let Err(e) = engine.compile(&source) {
        return Err(GenError::CompileFailed(e.to_string()));
    }

    Ok(GeneratedScript {
        name: stem,
        source,
        template,
    })
}

/// Lowercase the name and keep only `[a-z0-9_]`, collapsing every other run into a
/// single `_`, then trim leading/trailing `_`. Mirrors a conservative asset-naming
/// discipline so a generated file name can never escape the scripts directory or
/// carry shell-hostile characters.
fn sanitize_stem(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    let mut prev_us = false;
    for ch in name.trim().to_lowercase().chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch);
            prev_us = false;
        } else if !prev_us {
            out.push('_');
            prev_us = true;
        }
    }
    out.trim_matches('_').to_string()
}

// ============================================================================
// The vetted template bodies. Each follows walking_sim.rhai's shape: tunables as
// literals inside zero-arg accessors, a per-frame `(t, phase)` accessor returning
// one number. The literals are the CLAMPED params, formatted via `lit`.
// ============================================================================

/// `spin`: rotate an entity. `spin_speed()` is the revolutions/second the host
/// multiplies into a rotation; `spin_axis()` selects 0=X / 1=Y / 2=Z. `spin_angle`
/// is the per-frame accessor the host calls with `(t, phase)`.
fn spin_source(speed: f32, axis: f32) -> String {
    let tau = std::f32::consts::TAU;
    format!(
        "// GENERATED by Ochroma (template: spin) — hot-reloadable.\n\
         // Rotates an entity about an axis. Edit the literals and the running\n\
         // game re-spins on the fly (the host polls this file ~2x/second).\n\
         //\n\
         // The host calls spin_angle(t, phase) every frame for the rotation in\n\
         // radians, and spin_axis() for which axis to spin about.\n\
         \n\
         // Revolutions per second (clamped {min}..={max}).\n\
         fn spin_speed() {{\n    {speed}\n}}\n\
         \n\
         // Axis to spin about: 0 = X, 1 = Y, 2 = Z.\n\
         fn spin_axis() {{\n    {axis}\n}}\n\
         \n\
         // Rotation angle in radians at time t (seconds) with a per-entity phase.\n\
         fn spin_angle(t, phase) {{\n    (t * spin_speed() + phase) * {tau}\n}}\n",
        min = lit(ranges::SPIN_SPEED.min),
        max = lit(ranges::SPIN_SPEED.max),
        speed = lit(speed),
        axis = lit(axis),
        tau = lit(tau),
    )
}

/// `bob`: vertical oscillation. `bob_amplitude()` in metres, `bob_period()` in
/// seconds. `bob_offset(t, phase)` is the per-frame vertical offset the host adds
/// to an entity's Y. Mirrors walking_sim's `bob_amplitude`/`orb_bob` convention.
fn bob_source(amplitude: f32, period: f32) -> String {
    let tau = std::f32::consts::TAU;
    format!(
        "// GENERATED by Ochroma (template: bob) — hot-reloadable.\n\
         // Bobs an entity up and down. Edit the literals and the running game\n\
         // re-bobs on the fly (the host polls this file ~2x/second).\n\
         //\n\
         // The host calls bob_offset(t, phase) every frame for the vertical\n\
         // offset (metres) to add to the entity's height.\n\
         \n\
         // How high it bobs, in metres (clamped {min}..={max}).\n\
         fn bob_amplitude() {{\n    {amp}\n}}\n\
         \n\
         // Seconds for one full up-and-down cycle.\n\
         fn bob_period() {{\n    {period}\n}}\n\
         \n\
         // Vertical bob offset at time t (seconds) with a per-entity phase.\n\
         fn bob_offset(t, phase) {{\n    \
         (t / bob_period() * {tau} + phase).sin() * bob_amplitude()\n}}\n",
        min = lit(ranges::BOB_AMPLITUDE.min),
        max = lit(ranges::BOB_AMPLITUDE.max),
        amp = lit(amplitude),
        period = lit(period),
        tau = lit(tau),
    )
}

/// `pulse_light`: oscillate an emitter's intensity between `pulse_min()` and
/// `pulse_max()` over `pulse_period()` seconds. `pulse_intensity(t, phase)` is the
/// per-frame intensity the host applies to a light/emitter.
fn pulse_light_source(min: f32, max: f32, period: f32) -> String {
    let tau = std::f32::consts::TAU;
    format!(
        "// GENERATED by Ochroma (template: pulse_light) — hot-reloadable.\n\
         // Pulses an emitter's intensity between a min and a max. Edit the\n\
         // literals and the running game re-pulses on the fly (the host polls\n\
         // this file ~2x/second).\n\
         //\n\
         // The host calls pulse_intensity(t, phase) every frame for the emitter\n\
         // intensity to apply.\n\
         \n\
         // Dimmest intensity (clamped {rmin}..={rmax}).\n\
         fn pulse_min() {{\n    {min}\n}}\n\
         \n\
         // Brightest intensity (clamped {rmin}..={rmax}, never below pulse_min).\n\
         fn pulse_max() {{\n    {max}\n}}\n\
         \n\
         // Seconds for one full dim-to-bright-to-dim cycle.\n\
         fn pulse_period() {{\n    {period}\n}}\n\
         \n\
         // Emitter intensity at time t (seconds) with a per-emitter phase.\n\
         fn pulse_intensity(t, phase) {{\n    \
         let s = ((t / pulse_period() * {tau} + phase).sin() + 1.0) * 0.5;\n    \
         pulse_min() + (pulse_max() - pulse_min()) * s\n}}\n",
        rmin = lit(ranges::PULSE_MIN.min),
        rmax = lit(ranges::PULSE_MIN.max),
        min = lit(min),
        max = lit(max),
        period = lit(period),
        tau = lit(tau),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Compile a source in a real engine; panics with the rhai error on failure.
    fn assert_compiles(src: &str) {
        let engine = rhai::Engine::new();
        engine
            .compile(src)
            .unwrap_or_else(|e| panic!("generated source must compile: {e}\n--- source ---\n{src}"));
    }

    #[test]
    fn spin_generates_compiles_and_substitutes() {
        let g = generate(ScriptTemplate::Spin, "windmill_spin", Params::spin(2.0, 1.0))
            .expect("spin must generate");
        assert_eq!(g.template, ScriptTemplate::Spin);
        // A known function name proves the template is not empty.
        assert!(g.source.contains("fn spin_angle("), "must define spin_angle");
        assert!(g.source.contains("fn spin_speed()"), "must define spin_speed");
        // The substituted literal appears verbatim (lit() formats to 4 decimals).
        assert!(g.source.contains("2.0000"), "the speed literal must be substituted: {}", g.source);
        // It really compiles.
        assert_compiles(&g.source);
    }

    #[test]
    fn bob_generates_compiles_and_substitutes() {
        let g = generate(ScriptTemplate::Bob, "orb_bob", Params::bob(1.25, 4.0))
            .expect("bob must generate");
        assert!(g.source.contains("fn bob_offset("), "must define bob_offset");
        assert!(g.source.contains("fn bob_amplitude()"), "must define bob_amplitude");
        assert!(g.source.contains("1.2500"), "amplitude literal substituted: {}", g.source);
        assert!(g.source.contains("4.0000"), "period literal substituted: {}", g.source);
        assert_compiles(&g.source);
    }

    #[test]
    fn pulse_light_generates_compiles_and_substitutes() {
        let g = generate(
            ScriptTemplate::PulseLight,
            "lamp_pulse",
            Params::pulse_light(0.5, 3.0, 2.5),
        )
        .expect("pulse_light must generate");
        assert!(g.source.contains("fn pulse_intensity("), "must define pulse_intensity");
        assert!(g.source.contains("fn pulse_min()"), "must define pulse_min");
        assert!(g.source.contains("fn pulse_max()"), "must define pulse_max");
        assert!(g.source.contains("0.5000"), "min literal substituted: {}", g.source);
        assert!(g.source.contains("3.0000"), "max literal substituted: {}", g.source);
        assert_compiles(&g.source);
    }

    #[test]
    fn hostile_speed_clamps_to_documented_max() {
        // speed = 1e9 must fold to the documented SPIN_SPEED.max (16.0).
        let g = generate(ScriptTemplate::Spin, "x", Params::spin(1e9, 1.0)).unwrap();
        let max_lit = lit(ranges::SPIN_SPEED.max); // "16.0000"
        assert!(
            g.source.contains(&format!("fn spin_speed() {{\n    {max_lit}\n}}")),
            "hostile speed must clamp to the documented max {max_lit}: {}",
            g.source
        );
        // And the pathological value never reaches the literal.
        assert!(!g.source.contains("1000000000"), "raw hostile value must not appear");
        assert_compiles(&g.source);
    }

    #[test]
    fn hostile_negative_and_nan_clamp() {
        // Negative amplitude folds to 0.0 (BOB_AMPLITUDE.min); NaN period → default.
        let g = generate(ScriptTemplate::Bob, "x", Params::bob(-100.0, f32::NAN)).unwrap();
        assert!(g.source.contains("fn bob_amplitude() {\n    0.0000\n}"), "neg amp clamps to 0: {}", g.source);
        let def = lit(ranges::BOB_PERIOD.default); // "3.0000"
        assert!(g.source.contains(&format!("fn bob_period() {{\n    {def}\n}}")), "NaN period → default: {}", g.source);
        assert_compiles(&g.source);
    }

    #[test]
    fn pulse_max_never_below_min() {
        // Caller inverts the range: max(0.1) < min(5.0). Generated max is floored to min.
        let g = generate(ScriptTemplate::PulseLight, "x", Params::pulse_light(5.0, 0.1, 2.0)).unwrap();
        match Params::pulse_light(5.0, 0.1, 2.0) {
            Params::PulseLight { min, max, .. } => assert!(max >= min, "max must be floored to min"),
            _ => unreachable!(),
        }
        assert_compiles(&g.source);
    }

    #[test]
    fn mismatched_params_error() {
        // Bob params for a Spin template is a hard error, never a broken script.
        let err = generate(ScriptTemplate::Spin, "x", Params::bob(1.0, 1.0)).unwrap_err();
        assert_eq!(err, GenError::Mismatch);
    }

    #[test]
    fn empty_name_errors() {
        let err = generate(ScriptTemplate::Spin, "@@@", Params::spin(1.0, 1.0)).unwrap_err();
        assert_eq!(err, GenError::EmptyName);
    }

    #[test]
    fn name_is_sanitized_to_safe_stem() {
        let g = generate(ScriptTemplate::Spin, "  My Windmill!! Spin  ", Params::spin(1.0, 1.0)).unwrap();
        assert_eq!(g.name, "my_windmill_spin", "name must sanitize to a safe stem");
    }

    #[test]
    fn generation_is_deterministic() {
        let a = generate(ScriptTemplate::Spin, "w", Params::spin(2.0, 1.0)).unwrap();
        let b = generate(ScriptTemplate::Spin, "w", Params::spin(2.0, 1.0)).unwrap();
        assert_eq!(a.source, b.source, "same inputs must yield byte-identical source");
    }

    #[test]
    fn template_id_roundtrips() {
        for t in ScriptTemplate::all() {
            assert_eq!(ScriptTemplate::from_id(t.id()), Some(t), "{t} must round-trip");
        }
        assert_eq!(ScriptTemplate::from_id("nope"), None);
    }
}
