//! Spline system for roads and rivers.
//! Catmull-Rom splines drive road/river geometry and PCG placement along paths.

use glam::Vec3;

/// A Catmull-Rom spline through a set of control points.
#[derive(Debug, Clone)]
pub struct CatmullRomSpline {
    pub control_points: Vec<Vec3>,
    pub tension: f32,   // 0.5 = standard Catmull-Rom
    pub closed: bool,
}

impl CatmullRomSpline {
    pub fn new(control_points: Vec<Vec3>) -> Self {
        Self { control_points, tension: 0.5, closed: false }
    }

    /// Sample the spline at parameter `t` ∈ [0.0, N-1] where N = num segments.
    pub fn sample(&self, t: f32) -> Vec3 {
        let n = self.control_points.len();
        if n == 0 { return Vec3::ZERO; }
        if n == 1 { return self.control_points[0]; }

        let t = t.clamp(0.0, (n - 1) as f32);
        let seg = t.floor() as usize;
        let local_t = t.fract();

        let i1 = seg.min(n - 1);
        let i2 = (seg + 1).min(n - 1);
        let i0 = if seg == 0 { 0 } else { seg - 1 };
        let i3 = (seg + 2).min(n - 1);

        let p0 = self.control_points[i0];
        let p1 = self.control_points[i1];
        let p2 = self.control_points[i2];
        let p3 = self.control_points[i3];

        // Catmull-Rom coefficients
        let t2 = local_t * local_t;
        let t3 = t2 * local_t;
        let alpha = self.tension;

        let m1 = alpha * (p2 - p0);
        let m2 = alpha * (p3 - p1);

        let a = 2.0 * p1 - 2.0 * p2 + m1 + m2;
        let b = -3.0 * p1 + 3.0 * p2 - 2.0 * m1 - m2;
        let c = m1;
        let d = p1;

        a * t3 + b * t2 + c * local_t + d
    }

    /// Sample the spline tangent (direction) at parameter t.
    pub fn tangent(&self, t: f32) -> Vec3 {
        let eps = 0.001;
        let a = self.sample(t - eps);
        let b = self.sample(t + eps);
        (b - a).normalize_or_zero()
    }

    /// Compute the approximate arc length via sampling.
    pub fn arc_length(&self, samples: u32) -> f32 {
        let n = self.control_points.len();
        if n < 2 { return 0.0; }
        let total_t = (n - 1) as f32;
        let step = total_t / samples as f32;
        let mut length = 0.0;
        let mut prev = self.sample(0.0);
        for i in 1..=samples {
            let next = self.sample(i as f32 * step);
            length += (next - prev).length();
            prev = next;
        }
        length
    }

    /// Sample evenly-spaced points along the spline.
    pub fn sample_even(&self, count: usize) -> Vec<Vec3> {
        let n = self.control_points.len();
        if n < 2 || count == 0 { return Vec::new(); }
        let total_t = (n - 1) as f32;
        (0..count)
            .map(|i| self.sample(i as f32 / (count - 1).max(1) as f32 * total_t))
            .collect()
    }
}

/// A road spline with width and material properties.
#[derive(Debug, Clone)]
pub struct RoadSpline {
    pub spline: CatmullRomSpline,
    pub width: f32,     // road width in metres
    pub material_name: String,
}

impl RoadSpline {
    pub fn new(control_points: Vec<Vec3>, width: f32) -> Self {
        Self { spline: CatmullRomSpline::new(control_points), width, material_name: "road_asphalt".into() }
    }

    /// Sample points along the left and right road edges.
    pub fn sample_edges(&self, count: usize) -> (Vec<Vec3>, Vec<Vec3>) {
        let center_points = self.spline.sample_even(count);
        let n = self.spline.control_points.len();
        let total_t = (n - 1).max(1) as f32;
        let left: Vec<Vec3> = center_points.iter().enumerate().map(|(i, &c)| {
            let t = i as f32 / count.max(1) as f32 * total_t;
            let tangent = self.spline.tangent(t);
            let normal = Vec3::Y.cross(tangent).normalize_or_zero();
            c + normal * self.width * 0.5
        }).collect();
        let right: Vec<Vec3> = center_points.iter().enumerate().map(|(i, &c)| {
            let t = i as f32 / count.max(1) as f32 * total_t;
            let tangent = self.spline.tangent(t);
            let normal = tangent.cross(Vec3::Y).normalize_or_zero();
            c + normal * self.width * 0.5
        }).collect();
        (left, right)
    }
}

/// A river spline with depth and flow rate.
#[derive(Debug, Clone)]
pub struct RiverSpline {
    pub spline: CatmullRomSpline,
    pub width: f32,
    pub depth: f32,
    pub flow_rate: f32,  // metres/second
}

impl RiverSpline {
    pub fn new(control_points: Vec<Vec3>, width: f32) -> Self {
        Self { spline: CatmullRomSpline::new(control_points), width, depth: 2.0, flow_rate: 1.0 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn four_point_spline() -> CatmullRomSpline {
        CatmullRomSpline::new(vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(10.0, 0.0, 0.0),
            Vec3::new(20.0, 5.0, 0.0),
            Vec3::new(30.0, 0.0, 0.0),
        ])
    }

    #[test]
    fn catmull_rom_endpoint() {
        let spline = four_point_spline();
        let start = spline.sample(0.0);
        assert!((start - Vec3::new(0.0, 0.0, 0.0)).length() < 1e-4);
    }

    #[test]
    fn catmull_rom_arc_length_positive() {
        let spline = four_point_spline();
        let len = spline.arc_length(100);
        assert!(len > 0.0);
    }

    #[test]
    fn road_edges_count() {
        let road = RoadSpline::new(vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(10.0, 0.0, 0.0),
            Vec3::new(20.0, 0.0, 0.0),
            Vec3::new(30.0, 0.0, 0.0),
        ], 4.0);
        let (left, right) = road.sample_edges(10);
        assert_eq!(left.len(), 10);
        assert_eq!(right.len(), 10);
    }

    #[test]
    fn catmull_rom_sample_midpoint_is_between_endpoints() {
        let spline = four_point_spline();
        let n = spline.control_points.len();
        let mid = spline.sample((n - 1) as f32 * 0.5);
        let start = spline.control_points[0];
        let end = *spline.control_points.last().unwrap();
        // Midpoint x should be between start and end x
        assert!(mid.x > start.x && mid.x < end.x,
            "midpoint.x={} should be between {} and {}", mid.x, start.x, end.x);
    }
}
