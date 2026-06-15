//! SQUAD/SLERP free-look rotation track for cinematic cameras.
//!
//! A [`QuatTrack`] interpolates a sequence of orientation keyframes so that the
//! camera's **angular velocity is continuous across interior keys** — a plain
//! per-segment SLERP produces velocity *kinks* at every key (the derivative is
//! discontinuous), which reads as a visible "snap" when panning through a key.
//!
//! The fix is SQUAD (Spherical and Quadrangle interpolation, Shoemake): each
//! segment is a quadratic-blend of four quaternions — the two segment endpoints
//! plus two *intermediate* control quaternions `s_i`, `s_{i+1}` chosen so the
//! tangents match across keys:
//!
//! ```text
//! s_i = q_i · exp( -¼ ( log(q_i⁻¹·q_{i+1}) + log(q_i⁻¹·q_{i-1}) ) )
//! squad(q_i, q_{i+1}, s_i, s_{i+1}, t)
//!     = slerp( slerp(q_i, q_{i+1}, t), slerp(s_i, s_{i+1}, t), 2t(1-t) )
//! ```
//!
//! Inner SLERPs reuse [`crate::animation::blend_tree::shortest_arc_slerp`]
//! (which flips the sign of `b` when `a·b < 0`, guaranteeing the short path).
//! When two quats are nearly identical (`cosΩ > 0.9995`) we fall back to a
//! normalized lerp to avoid division blow-up.
//!
//! Pure: no I/O, no GPU, `Send + Sync`, never panics, clamps past the end keys.

use crate::animation::blend_tree::shortest_arc_slerp;
use glam::Quat;

/// Threshold above which `slerp` is numerically unstable and we use `nlerp`.
const NLERP_COS_THRESHOLD: f32 = 0.9995;

/// A keyframed quaternion orientation track with SQUAD interpolation, giving
/// continuous angular velocity across interior keyframes.
///
/// Keys are stored sorted by ascending frame. `sample(frame)` clamps to the
/// endpoints and never panics.
#[derive(Debug, Clone)]
pub struct QuatTrack {
    /// `(frame, orientation)` keys, sorted by ascending frame, hemisphere-aligned.
    keys: Vec<(i32, Quat)>,
}

impl QuatTrack {
    /// Build a track from `(frame, quat)` keys. Keys are sorted by frame,
    /// normalized, and hemisphere-aligned (each successive key is negated when
    /// it lies on the opposite hemisphere of its predecessor, `q·q_prev < 0`),
    /// so all interpolation walks the short arc consistently.
    pub fn new(keys: Vec<(i32, Quat)>) -> Self {
        let mut keys = keys;
        keys.sort_by_key(|(f, _)| *f);
        // Normalize + hemisphere-align so dot products between neighbors are ≥ 0.
        for k in keys.iter_mut() {
            k.1 = k.1.normalize();
        }
        for i in 1..keys.len() {
            if keys[i].1.dot(keys[i - 1].1) < 0.0 {
                keys[i].1 = -keys[i].1;
            }
        }
        Self { keys }
    }

    /// Number of keys.
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    /// True when the track has no keys.
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// Sample the orientation at a (possibly fractional) frame. Clamps to the
    /// first/last key outside `[first_frame, last_frame]`. Returns
    /// [`Quat::IDENTITY`] for an empty track.
    pub fn sample(&self, frame: f64) -> Quat {
        let n = self.keys.len();
        if n == 0 {
            return Quat::IDENTITY;
        }
        if n == 1 {
            return self.keys[0].1;
        }
        let first = self.keys[0].0 as f64;
        let last = self.keys[n - 1].0 as f64;
        if frame <= first {
            return self.keys[0].1;
        }
        if frame >= last {
            return self.keys[n - 1].1;
        }
        // Find the segment [i, i+1] containing `frame`.
        let mut i = 0usize;
        while i + 1 < n && (self.keys[i + 1].0 as f64) <= frame {
            i += 1;
        }
        let (f0, q0) = self.keys[i];
        let (f1, q1) = self.keys[i + 1];
        let span = (f1 - f0) as f64;
        let t = if span > 0.0 {
            ((frame - f0 as f64) / span) as f32
        } else {
            0.0
        };

        // Intermediate control quats for the two ends of this segment.
        let s0 = self.intermediate(i);
        let s1 = self.intermediate(i + 1);

        squad(q0, q1, s0, s1, t)
    }

    /// Compute the SQUAD intermediate (control) quaternion `s_i` for key `i`.
    ///
    /// `s_i = q_i · exp( -¼ ( log(q_i⁻¹·q_{i+1}) + log(q_i⁻¹·q_{i-1}) ) )`.
    ///
    /// At the endpoints (no neighbor on one side) we mirror the single available
    /// neighbor so the boundary segment still has a well-defined tangent.
    fn intermediate(&self, i: usize) -> Quat {
        let n = self.keys.len();
        let qi = self.keys[i].1;
        let qi_inv = qi.inverse();
        let prev = if i > 0 { self.keys[i - 1].1 } else { qi };
        let next = if i + 1 < n { self.keys[i + 1].1 } else { qi };

        let log_next = quat_log(hemisphere(qi_inv * next, qi_inv));
        let log_prev = quat_log(hemisphere(qi_inv * prev, qi_inv));
        let inner = (log_next + log_prev) * -0.25;
        (qi * quat_exp(inner)).normalize()
    }
}

/// SQUAD blend of two segment endpoints with their intermediate control quats.
fn squad(q0: Quat, q1: Quat, s0: Quat, s1: Quat, t: f32) -> Quat {
    let a = nlerp_or_slerp(q0, q1, t);
    let b = nlerp_or_slerp(s0, s1, t);
    nlerp_or_slerp(a, b, 2.0 * t * (1.0 - t)).normalize()
}

/// Short-arc SLERP, with a normalized-lerp fallback when the endpoints are
/// nearly identical (numerically unstable region for SLERP).
fn nlerp_or_slerp(a: Quat, b: Quat, t: f32) -> Quat {
    let a = a.normalize();
    let mut b = b.normalize();
    if a.dot(b) < 0.0 {
        b = -b;
    }
    if a.dot(b) > NLERP_COS_THRESHOLD {
        // nlerp: cheap, stable, and the path is indistinguishable at this angle.
        (a + (b - a) * t).normalize()
    } else {
        shortest_arc_slerp(a, b, t)
    }
}

/// Align `q` to the same hemisphere as `ref_q` (negate if their dot is negative),
/// so the logarithm picks the short rotation.
fn hemisphere(q: Quat, ref_q: Quat) -> Quat {
    if q.dot(ref_q) < 0.0 {
        -q
    } else {
        q
    }
}

/// Quaternion logarithm of a (assumed unit) quaternion: `log(q) = [θ·v̂, 0]`
/// where `θ = atan2(|v|, w)` and `v̂ = v/|v|`. The result is a pure quaternion
/// (zero scalar part) encoding `½·angle·axis`.
fn quat_log(q: Quat) -> Quat {
    let v = glam::vec3(q.x, q.y, q.z);
    let v_len = v.length();
    if v_len < 1e-8 {
        // Near-identity rotation: log ≈ 0.
        return Quat::from_xyzw(0.0, 0.0, 0.0, 0.0);
    }
    let theta = v_len.atan2(q.w);
    let scaled = v * (theta / v_len);
    Quat::from_xyzw(scaled.x, scaled.y, scaled.z, 0.0)
}

/// Quaternion exponential of a pure quaternion `q = [v, 0]`:
/// `exp(q) = [sin|v|·v̂, cos|v|]`.
fn quat_exp(q: Quat) -> Quat {
    let v = glam::vec3(q.x, q.y, q.z);
    let v_len = v.length();
    if v_len < 1e-8 {
        return Quat::IDENTITY;
    }
    let s = v_len.sin() / v_len;
    Quat::from_xyzw(v.x * s, v.y * s, v.z * s, v_len.cos()).normalize()
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Quat;

    #[test]
    fn squad_angular_velocity_continuous() {
        let track = QuatTrack::new(vec![
            (0, Quat::IDENTITY),
            (30, Quat::from_rotation_y(0.5)),
            (60, Quat::from_rotation_y(0.5) * Quat::from_rotation_x(0.4)), // turn axis at the interior key
        ]);
        let eps = 0.25;
        let omega = |f: f64| {
            let a = track.sample(f - eps);
            let b = track.sample(f + eps);
            (a.inverse() * b).to_axis_angle().1 as f64 / (2.0 * eps) // angle/Δt
        };
        let jump = (omega(30.0 - 1.0) - omega(30.0 + 1.0)).abs();
        println!("angular velocity discontinuity {}", jump);
        assert!(jump < 1e-3, "angular velocity discontinuity {}", jump);
    }
}
