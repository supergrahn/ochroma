use glam::Vec3;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RoadType {
    DirtPath,
    LocalStreet,
    Avenue,
    Highway,
    RailTrack,
}

impl RoadType {
    pub fn lanes(&self) -> u32 {
        match self { Self::DirtPath => 1, Self::LocalStreet => 2, Self::Avenue => 4, Self::Highway => 6, Self::RailTrack => 1 }
    }
    pub fn width(&self) -> f32 {
        match self { Self::DirtPath => 3.0, Self::LocalStreet => 8.0, Self::Avenue => 16.0, Self::Highway => 24.0, Self::RailTrack => 4.0 }
    }
    pub fn speed_limit_kmh(&self) -> f32 {
        match self { Self::DirtPath => 20.0, Self::LocalStreet => 50.0, Self::Avenue => 60.0, Self::Highway => 120.0, Self::RailTrack => 80.0 }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoadSegment {
    pub id: u32,
    pub road_type: RoadType,
    pub start: Vec3,
    pub end: Vec3,
    pub control_point: Option<Vec3>, // For Bezier curves
}

impl RoadSegment {
    /// Sample a point along the road at parameter t (0.0 to 1.0).
    pub fn sample(&self, t: f32) -> Vec3 {
        match self.control_point {
            Some(cp) => {
                // Quadratic Bezier: (1-t)²P0 + 2(1-t)tP1 + t²P2
                let u = 1.0 - t;
                self.start * (u * u) + cp * (2.0 * u * t) + self.end * (t * t)
            }
            None => self.start.lerp(self.end, t),
        }
    }

    /// Approximate length by sampling N points.
    pub fn length(&self) -> f32 {
        let n = 20;
        let mut total = 0.0;
        for i in 0..n {
            let t0 = i as f32 / n as f32;
            let t1 = (i + 1) as f32 / n as f32;
            total += self.sample(t0).distance(self.sample(t1));
        }
        total
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Intersection {
    pub id: u32,
    pub position: Vec3,
    pub connected_segments: Vec<u32>,
}

pub struct RoadNetwork {
    pub segments: Vec<RoadSegment>,
    pub intersections: Vec<Intersection>,
    next_seg_id: u32,
    next_int_id: u32,
}

impl RoadNetwork {
    pub fn new() -> Self {
        Self { segments: Vec::new(), intersections: Vec::new(), next_seg_id: 0, next_int_id: 0 }
    }

    pub fn add_straight(&mut self, road_type: RoadType, start: Vec3, end: Vec3) -> u32 {
        let id = self.next_seg_id;
        self.next_seg_id += 1;
        self.segments.push(RoadSegment { id, road_type, start, end, control_point: None });
        self.auto_intersect(id);
        id
    }

    pub fn add_curve(&mut self, road_type: RoadType, start: Vec3, control: Vec3, end: Vec3) -> u32 {
        let id = self.next_seg_id;
        self.next_seg_id += 1;
        self.segments.push(RoadSegment { id, road_type, start, end, control_point: Some(control) });
        self.auto_intersect(id);
        id
    }

    /// Check if a new segment's endpoints are near existing segments and create intersections.
    fn auto_intersect(&mut self, new_seg_id: u32) {
        let seg = &self.segments[self.segments.iter().position(|s| s.id == new_seg_id).unwrap()];
        let start = seg.start;
        let end = seg.end;
        let threshold = 2.0; // metres

        let mut intersect_starts: Vec<(Vec3, Vec<u32>)> = Vec::new();
        let mut intersect_ends: Vec<(Vec3, Vec<u32>)> = Vec::new();

        for existing in &self.segments {
            if existing.id == new_seg_id { continue; }
            if existing.start.distance(start) < threshold || existing.end.distance(start) < threshold {
                intersect_starts.push((start, vec![new_seg_id, existing.id]));
            }
            if existing.start.distance(end) < threshold || existing.end.distance(end) < threshold {
                intersect_ends.push((end, vec![new_seg_id, existing.id]));
            }
        }

        for (pos, segs) in intersect_starts.into_iter().chain(intersect_ends.into_iter()) {
            self.create_intersection_at(pos, &segs);
        }
    }

    fn create_intersection_at(&mut self, position: Vec3, segments: &[u32]) {
        // Avoid duplicate intersections
        if self.intersections.iter().any(|i| i.position.distance(position) < 1.0) {
            return;
        }
        let id = self.next_int_id;
        self.next_int_id += 1;
        self.intersections.push(Intersection {
            id, position, connected_segments: segments.to_vec(),
        });
    }

    pub fn segment_count(&self) -> usize { self.segments.len() }
    pub fn intersection_count(&self) -> usize { self.intersections.len() }
    pub fn total_length(&self) -> f32 { self.segments.iter().map(|s| s.length()).sum() }
}

impl Default for RoadNetwork {
    fn default() -> Self {
        Self::new()
    }
}
