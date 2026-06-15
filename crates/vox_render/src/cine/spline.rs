//! Centripetal Catmull-Rom position spline + arc-length LUT.
//!
//! The spline interpolates a set of control points keyed at integer frames.
//! Centripetal parameterisation (α = 0.5) is used because it never produces
//! cusps or self-intersections between control points, unlike the uniform
//! (α = 0) or chordal (α = 1) variants. The segment is evaluated with the
//! Barry-Goldman pyramidal recurrence, which is exact for non-uniform knots.
//!
//! [`ArcLengthLut`] reparameterises the spline by distance so playback can be
//! driven at constant speed regardless of how unevenly the control points are
//! spaced in frame-time.

use glam::Vec3;

/// A centripetal Catmull-Rom position spline over `Vec3` control points.
///
/// Pure and deterministic: `eval_*` perform no I/O and never panic (they clamp
/// to the end control points outside the valid range).
#[derive(Clone, Debug)]
pub struct Spline3 {
    /// Control points the spline passes through.
    points: Vec<Vec3>,
    /// Frame at which each control point sits (sorted ascending, parallel to
    /// `points`).
    frames: Vec<f32>,
    /// Centripetal exponent (0.5 for centripetal). Stored so `knot` spacing is
    /// reproducible.
    alpha: f32,
}

impl Spline3 {
    /// Build a centripetal Catmull-Rom spline through `points` keyed at
    /// `frames`. `alpha` selects the parameterisation (0.5 = centripetal).
    ///
    /// Requires at least one point. `points` and `frames` must be the same
    /// length; `frames` must be ascending.
    pub fn centripetal(points: &[Vec3], frames: &[i32], alpha: f32) -> Self {
        assert!(!points.is_empty(), "Spline3 requires at least one point");
        assert_eq!(
            points.len(),
            frames.len(),
            "points and frames length mismatch"
        );
        Spline3 {
            points: points.to_vec(),
            frames: frames.iter().map(|&f| f as f32).collect(),
            alpha,
        }
    }

    /// Number of control points.
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// True if the spline has no control points.
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// First / last keyframe.
    pub fn first_frame(&self) -> f32 {
        *self.frames.first().unwrap()
    }
    pub fn last_frame(&self) -> f32 {
        *self.frames.last().unwrap()
    }

    /// Control point at index (used by tests / arc-length sampling).
    pub fn point(&self, i: usize) -> Vec3 {
        self.points[i]
    }

    /// Phantom-padded point: reflects the endpoints so segments at the ends are
    /// well-defined (`P_-1 = 2·P_0 - P_1`, and similarly at the far end).
    fn padded_point(&self, i: isize) -> Vec3 {
        let n = self.points.len() as isize;
        if i < 0 {
            2.0 * self.points[0] - self.points[1.min((n - 1) as usize)]
        } else if i >= n {
            let last = (n - 1) as usize;
            let prev = (n - 2).max(0) as usize;
            2.0 * self.points[last] - self.points[prev]
        } else {
            self.points[i as usize]
        }
    }

    /// Phantom-padded frame, extrapolated by the spacing at the matching end.
    fn padded_frame(&self, i: isize) -> f32 {
        let n = self.frames.len() as isize;
        if i < 0 {
            let span = if n >= 2 {
                self.frames[1] - self.frames[0]
            } else {
                1.0
            };
            self.frames[0] - span
        } else if i >= n {
            let last = (n - 1) as usize;
            let span = if n >= 2 {
                self.frames[last] - self.frames[last - 1]
            } else {
                1.0
            };
            self.frames[last] + span
        } else {
            self.frames[i as usize]
        }
    }

    /// Evaluate the spline at a (clamped) frame value.
    ///
    /// Interpolates the control point exactly at each keyframe.
    pub fn eval_frame(&self, frame: f32) -> Vec3 {
        let n = self.points.len();
        if n == 1 {
            return self.points[0];
        }
        let frame = frame.clamp(self.first_frame(), self.last_frame());

        // Find the segment [seg, seg+1] containing `frame`.
        let mut seg = 0usize;
        while seg + 1 < n && self.frames[seg + 1] < frame {
            seg += 1;
        }
        // Local parameter u in [0,1] across the segment in frame-space.
        let f0 = self.frames[seg];
        let f1 = self.frames[seg + 1];
        let u = if (f1 - f0).abs() < f32::EPSILON {
            0.0
        } else {
            (frame - f0) / (f1 - f0)
        };
        self.eval_segment(seg as isize, u)
    }

    /// Evaluate segment `seg` (between control points `seg` and `seg+1`) at
    /// local parameter `u` in [0,1], using the Barry-Goldman recurrence over
    /// the four surrounding (phantom-padded) control points with centripetal
    /// knot spacing.
    fn eval_segment(&self, seg: isize, u: f32) -> Vec3 {
        let p0 = self.padded_point(seg - 1);
        let p1 = self.padded_point(seg);
        let p2 = self.padded_point(seg + 1);
        let p3 = self.padded_point(seg + 2);

        // Centripetal knot sequence: t_{i+1} = t_i + |P_{i+1}-P_i|^alpha.
        let t0 = 0.0f32;
        let t1 = t0 + (p1 - p0).length().powf(self.alpha).max(1e-6);
        let t2 = t1 + (p2 - p1).length().powf(self.alpha).max(1e-6);
        let t3 = t2 + (p3 - p2).length().powf(self.alpha).max(1e-6);

        // Parameter for `u` lies on the [t1, t2] segment.
        let t = t1 + u * (t2 - t1);

        let lerp = |a: Vec3, b: Vec3, ta: f32, tb: f32, x: f32| -> Vec3 {
            if (tb - ta).abs() < f32::EPSILON {
                a
            } else {
                let w = (x - ta) / (tb - ta);
                a * (1.0 - w) + b * w
            }
        };

        // Barry-Goldman pyramidal interpolation.
        let a1 = lerp(p0, p1, t0, t1, t);
        let a2 = lerp(p1, p2, t1, t2, t);
        let a3 = lerp(p2, p3, t2, t3, t);

        let b1 = lerp(a1, a2, t0, t2, t);
        let b2 = lerp(a2, a3, t1, t3, t);

        lerp(b1, b2, t1, t2, t)
    }

    /// Evaluate at a normalized distance `d` in [0,1] along the arc, given a
    /// prebuilt LUT. Constant-speed: equal `d` steps yield near-equal arc
    /// steps.
    pub fn eval_distance(&self, lut: &ArcLengthLut, d: f32) -> Vec3 {
        let frame = lut.distance_to_frame(d);
        self.eval_frame(frame)
    }
}

/// Arc-length reparameterisation table for a [`Spline3`].
///
/// Stores cumulative arc length sampled densely across the spline, plus the
/// frame at each sample, so a normalized distance can be mapped back to a frame
/// (and hence a constant-speed position) by binary search + linear interp.
#[derive(Clone, Debug)]
pub struct ArcLengthLut {
    /// Cumulative arc length at each sample (ascending, `samples[0] == 0`).
    lengths: Vec<f32>,
    /// Frame value at each sample (parallel to `lengths`).
    frames: Vec<f32>,
    /// Total arc length (`= lengths.last()`).
    total: f32,
}

impl ArcLengthLut {
    /// Build the LUT by walking the spline with `samples_per_segment`
    /// subdivisions per control-point interval and accumulating chord lengths.
    pub fn build(spline: &Spline3, samples_per_segment: usize) -> Self {
        let n = spline.len();
        let spp = samples_per_segment.max(1);

        let mut lengths = Vec::new();
        let mut frames = Vec::new();

        if n == 1 {
            lengths.push(0.0);
            frames.push(spline.first_frame());
            return ArcLengthLut {
                lengths,
                frames,
                total: 0.0,
            };
        }

        let mut acc = 0.0f32;
        let mut prev = spline.eval_frame(spline.first_frame());
        lengths.push(0.0);
        frames.push(spline.first_frame());

        for seg in 0..(n - 1) {
            let f0 = spline.frames[seg];
            let f1 = spline.frames[seg + 1];
            for s in 1..=spp {
                let u = s as f32 / spp as f32;
                let frame = f0 + u * (f1 - f0);
                let p = spline.eval_frame(frame);
                acc += (p - prev).length();
                prev = p;
                lengths.push(acc);
                frames.push(frame);
            }
        }

        let total = *lengths.last().unwrap();
        ArcLengthLut {
            lengths,
            frames,
            total,
        }
    }

    /// Total arc length of the sampled spline.
    pub fn total_length(&self) -> f32 {
        self.total
    }

    /// Map a normalized distance `d` in [0,1] to a frame value via binary
    /// search over cumulative lengths + linear interpolation.
    pub fn distance_to_frame(&self, d: f32) -> f32 {
        if self.frames.len() == 1 || self.total <= 0.0 {
            return self.frames[0];
        }
        let d = d.clamp(0.0, 1.0);
        let target = d * self.total;

        // Binary search for the first sample whose cumulative length >= target.
        let mut lo = 0usize;
        let mut hi = self.lengths.len() - 1;
        while lo < hi {
            let mid = (lo + hi) / 2;
            if self.lengths[mid] < target {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        if lo == 0 {
            return self.frames[0];
        }
        let l0 = self.lengths[lo - 1];
        let l1 = self.lengths[lo];
        let f0 = self.frames[lo - 1];
        let f1 = self.frames[lo];
        if (l1 - l0).abs() < f32::EPSILON {
            f1
        } else {
            let w = (target - l0) / (l1 - l0);
            f0 + w * (f1 - f0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spline_interpolates_and_constant_speed() {
        let pts = vec![
            glam::vec3(0.0, 0.0, 0.0),
            glam::vec3(1.0, 2.0, 0.0), // control point at param index 1
            glam::vec3(5.0, 2.0, 0.0), // big gap -> tests constant speed under non-uniform knots
            glam::vec3(6.0, 0.0, 0.0),
        ];
        let frames = vec![0, 30, 90, 120];
        let s = Spline3::centripetal(&pts, &frames, 0.5);
        // interpolates the control point at key index 2 (frame 90)
        assert!(
            (s.eval_frame(90.0) - pts[2]).length() < 1e-4,
            "got {:?}",
            s.eval_frame(90.0)
        );
        println!("eval_frame(90) = {:?}", s.eval_frame(90.0));
        // constant-speed: 30 even time samples -> near-equal arc steps
        let lut = ArcLengthLut::build(&s, 64);
        let mut prev = s.eval_distance(&lut, 0.0);
        let (mut mn, mut mx, mut sum) = (f32::MAX, 0.0f32, 0.0f32);
        for i in 1..=30 {
            let p = s.eval_distance(&lut, i as f32 / 30.0);
            let d = (p - prev).length();
            mn = mn.min(d);
            mx = mx.max(d);
            sum += d;
            prev = p;
        }
        let mean = sum / 30.0;
        let variance = (mx - mn) / mean;
        println!("speed variance {}", variance);
        assert!(variance < 0.05, "speed variance {}", variance);
    }
}
