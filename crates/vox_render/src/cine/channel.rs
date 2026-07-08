//! Cubic-Hermite scalar animation channel — the atomic unit of the cinematic
//! camera system.
//!
//! A [`Channel`] is a sorted list of [`Key`]s over an integer-frame clock.
//! Sampling between two keys uses a true cubic-Hermite basis (not a lerp), so
//! `Interp::Cubic` with zero tangents produces the smooth `3t² - 2t³` ease and
//! non-zero tangents shape the curve. `Interp::Auto` derives Catmull-Rom
//! tangents with no-overshoot clamping at local extrema.
//!
//! Everything here is pure: no I/O, no GPU, `Send + Sync`, never panics, and
//! sampling clamps to the end keys.

use serde::{Deserialize, Serialize};

/// Frame rate as an exact rational (e.g. `60/1`, `24000/1001`).
///
/// Kept rational so frame↔seconds conversion is exact and reproducible — no
/// wall-clock, no float drift.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrameRate {
    pub num: u32,
    pub den: u32,
}

impl FrameRate {
    pub const fn new(num: u32, den: u32) -> Self {
        Self { num, den }
    }

    /// Seconds per frame.
    pub fn seconds_per_frame(&self) -> f64 {
        self.den as f64 / self.num as f64
    }

    /// Frames per second as an `f64`.
    pub fn fps(&self) -> f64 {
        self.num as f64 / self.den as f64
    }

    /// Convert a (possibly fractional) frame index to seconds.
    pub fn frame_to_seconds(&self, frame: f64) -> f64 {
        frame * self.seconds_per_frame()
    }
}

impl Default for FrameRate {
    fn default() -> Self {
        Self { num: 60, den: 1 }
    }
}

/// Per-key interpolation mode toward the *next* key.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Interp {
    /// Hold the key's value until the next key (step).
    Constant,
    /// Straight linear interpolation to the next key.
    Linear,
    /// Cubic-Hermite with explicit tangents (units: value per frame).
    Cubic { tan_in: f32, tan_out: f32 },
    /// Cubic-Hermite with auto Catmull-Rom tangents (no-overshoot clamped).
    Auto,
}

/// A single keyframe: an integer frame, a value, and the interpolation mode
/// used on the segment leaving this key.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Key<T> {
    pub frame: i32,
    pub value: T,
    pub interp: Interp,
}

/// A sorted set of scalar keyframes, sampled with cubic-Hermite basis.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Channel<T> {
    keys: Vec<Key<T>>,
}

impl<T> Channel<T> {
    pub fn new() -> Self {
        Self { keys: Vec::new() }
    }

    /// Number of keys.
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// Read-only access to the sorted key list.
    pub fn keys(&self) -> &[Key<T>] {
        &self.keys
    }
}

impl Channel<f32> {
    /// Insert a key, keeping the list sorted by frame. If a key already exists
    /// at that frame it is replaced.
    pub fn insert(&mut self, key: Key<f32>) {
        match self.keys.binary_search_by_key(&key.frame, |k| k.frame) {
            Ok(i) => self.keys[i] = key,
            Err(i) => self.keys.insert(i, key),
        }
    }

    /// Construct directly from a slice of `(frame, value)` pairs using `Auto`
    /// interpolation. Convenience for serde/round-trip channels.
    pub fn from_auto_pairs(pairs: &[(i32, f32)]) -> Self {
        let mut ch = Self::new();
        for &(frame, value) in pairs {
            ch.insert(Key {
                frame,
                value,
                interp: Interp::Auto,
            });
        }
        ch
    }

    /// Auto / Catmull-Rom tangent for the key at index `i` (value units per
    /// frame), with monotone (no-overshoot) clamping at local extrema.
    fn auto_tangent(&self, i: usize) -> f32 {
        let n = self.keys.len();
        if n < 2 {
            return 0.0;
        }
        let cur = &self.keys[i];
        if i == 0 {
            // One-sided forward difference at the start.
            let nxt = &self.keys[1];
            let dt = (nxt.frame - cur.frame) as f32;
            return if dt != 0.0 {
                (nxt.value - cur.value) / dt
            } else {
                0.0
            };
        }
        if i == n - 1 {
            // One-sided backward difference at the end.
            let prv = &self.keys[i - 1];
            let dt = (cur.frame - prv.frame) as f32;
            return if dt != 0.0 {
                (cur.value - prv.value) / dt
            } else {
                0.0
            };
        }
        let prv = &self.keys[i - 1];
        let nxt = &self.keys[i + 1];
        let dt = (nxt.frame - prv.frame) as f32;
        let m = if dt != 0.0 {
            (nxt.value - prv.value) / dt
        } else {
            0.0
        };
        // No-overshoot clamp: if the key is a local extremum, flatten the
        // tangent so the curve does not overshoot past the value.
        let d_prev = cur.value - prv.value;
        let d_next = nxt.value - cur.value;
        if d_prev * d_next <= 0.0 { 0.0 } else { m }
    }

    /// Sample the channel at a (possibly fractional) frame. Clamps to the first
    /// and last key values outside the keyed range. Pure, never panics.
    pub fn sample(&self, frame: f64) -> f32 {
        let n = self.keys.len();
        if n == 0 {
            return 0.0;
        }
        if n == 1 {
            return self.keys[0].value;
        }
        let f = frame as f32;
        // Clamp outside the keyed range.
        if f <= self.keys[0].frame as f32 {
            return self.keys[0].value;
        }
        if f >= self.keys[n - 1].frame as f32 {
            return self.keys[n - 1].value;
        }
        // Find the segment [i, i+1] containing `f`.
        let mut i = 0;
        while i + 1 < n && (self.keys[i + 1].frame as f32) <= f {
            i += 1;
        }
        let k0 = &self.keys[i];
        let k1 = &self.keys[i + 1];
        let f0 = k0.frame as f32;
        let f1 = k1.frame as f32;
        let span = f1 - f0;
        if span <= 0.0 {
            return k0.value;
        }
        let t = (f - f0) / span; // normalized [0,1] within the segment

        match k0.interp {
            Interp::Constant => k0.value,
            Interp::Linear => k0.value + (k1.value - k0.value) * t,
            Interp::Cubic { tan_out, .. } => {
                // tan_out is the leaving tangent of k0; the arriving tangent of
                // k1 is its tan_in.
                let m0 = tan_out * span; // scale slope (per-frame) into segment param
                let m1 = match k1.interp {
                    Interp::Cubic { tan_in, .. } => tan_in * span,
                    Interp::Auto => self.auto_tangent(i + 1) * span,
                    _ => 0.0,
                };
                hermite(k0.value, k1.value, m0, m1, t)
            }
            Interp::Auto => {
                let m0 = self.auto_tangent(i) * span;
                let m1 = self.auto_tangent(i + 1) * span;
                hermite(k0.value, k1.value, m0, m1, t)
            }
        }
    }
}

/// Cubic-Hermite basis interpolation.
///
/// `p0`,`p1` endpoint values; `m0`,`m1` tangents already scaled into the
/// segment's parameter space; `t` in `[0,1]`.
fn hermite(p0: f32, p1: f32, m0: f32, m1: f32, t: f32) -> f32 {
    let t2 = t * t;
    let t3 = t2 * t;
    let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
    let h10 = t3 - 2.0 * t2 + t;
    let h01 = -2.0 * t3 + 3.0 * t2;
    let h11 = t3 - t2;
    h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hermite_two_keys_exact() {
        let mut ch = Channel::<f32>::new();
        ch.insert(Key {
            frame: 0,
            value: 10.0,
            interp: Interp::Cubic {
                tan_in: 0.0,
                tan_out: 0.0,
            },
        });
        ch.insert(Key {
            frame: 30,
            value: 50.0,
            interp: Interp::Cubic {
                tan_in: 0.0,
                tan_out: 0.0,
            },
        });
        // Hermite with zero tangents == smooth (3t^2-2t^3) ease; at t=0.5: 10 + 40*0.5 = 30.0
        println!("sample(15) = {:.4}", ch.sample(15.0));
        assert!(
            (ch.sample(15.0) - 30.0).abs() < 1e-4,
            "got {}",
            ch.sample(15.0)
        );
        // at frame 7.5 (t=0.25): basis h01=0.15625 -> value 10 + 40*0.15625 = 16.25
        println!("sample(7.5) = {}", ch.sample(7.5));
        assert!(
            (ch.sample(7.5) - 16.25).abs() < 1e-4,
            "got {}",
            ch.sample(7.5)
        );
        // clamp past the last key
        assert!((ch.sample(99.0) - 50.0).abs() < 1e-6);
    }
}
