// walking_sim game script — hot-reloadable.
//
// This file drives visible game behaviour. Edit it WHILE the game is running and
// the orb bob/pulse and windmill speed change on the fly (the host polls this
// file's mtime ~2x/second and recompiles + swaps it without losing game state).
// A syntax error here does NOT crash the game: the host keeps the last good
// version running and surfaces the error in the HUD.
//
// The host calls the accessor functions below every frame and expects a single
// number back. Tune the literals inside each accessor and watch the game change.
//
// NOTE: Rhai script-level `const` values are NOT visible inside `fn` bodies when
// the host calls a function directly, so each tunable lives as a literal inside
// its own accessor (the single source of truth the host reads).

// --- Tunables (edit these literals) ----------------------------------------

// How high the orbs bob, in metres. Try 0.30 (calm) up to 1.20 (bouncy).
fn bob_amplitude() {
    0.30
}

// How fast the orbs bob, in radians/second.
fn bob_speed() {
    2.0
}

// How much the orbs pulse (scale wobble) as a fraction of base size.
fn pulse_depth() {
    0.20
}

// Windmill spin speed multiplier (1.0 = the base 0.4 rev/s).
fn windmill_speed() {
    1.0
}

// --- Derived behaviour (host may call these for per-orb motion) -------------

// Vertical bob offset for an orb at the given time + per-orb phase.
fn orb_bob(t, phase) {
    (t * bob_speed() + phase).sin() * bob_amplitude()
}

// Scale pulse factor (around 1.0) for an orb at the given time + phase.
fn orb_pulse(t, phase) {
    1.0 + (t * 3.0 + phase).sin() * pulse_depth()
}
