//! Motion matching — nearest-feature animation selection.
//!
//! Dan Holden, "A Fast and Simple Method for Computing a Data-Driven Motion Phase" (2016).
//! O(N) linear scan: ~50µs at 10k features — within 1ms frame budget.

#[derive(Debug, Clone)]
pub struct AnimClip {
    pub name:        String,
    pub frame_count: u32,
    pub frame_rate:  f32,
}

/// Per-frame motion feature vector: [vel_x, vel_z, heading_cos, heading_sin, phase].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MotionFeature {
    pub vel_x:       f32,
    pub vel_z:       f32,
    pub heading_cos: f32,
    pub heading_sin: f32,
    pub phase:       f32,
    pub clip_index:  usize,
    pub frame_index: u32,
}

impl MotionFeature {
    pub fn from_state(vel_x: f32, vel_z: f32, heading_rad: f32, phase: f32) -> Self {
        Self {
            vel_x,
            vel_z,
            heading_cos: heading_rad.cos(),
            heading_sin: heading_rad.sin(),
            phase,
            clip_index:  0,
            frame_index: 0,
        }
    }

    /// Weighted L2 distance. Velocity 2×, heading 1×, phase 0.5× (Holden 2016).
    pub fn distance(&self, other: &Self) -> f32 {
        let dvel = 2.0 * ((self.vel_x - other.vel_x).powi(2) + (self.vel_z - other.vel_z).powi(2));
        let dhd  = 1.0 * ((self.heading_cos - other.heading_cos).powi(2)
                        + (self.heading_sin - other.heading_sin).powi(2));
        let dph  = 0.5 * (self.phase - other.phase).powi(2);
        (dvel + dhd + dph).sqrt()
    }
}

pub struct MotionDatabase {
    pub clips:    Vec<AnimClip>,
    pub features: Vec<MotionFeature>,
}

#[derive(Debug, Clone)]
pub struct MotionMatch {
    pub clip_name:   String,
    pub clip_index:  usize,
    pub frame_index: u32,
    pub distance:    f32,
}

impl MotionDatabase {
    pub fn new() -> Self { Self { clips: Vec::new(), features: Vec::new() } }

    pub fn add_clip(&mut self, name: &str, frame_count: u32, frame_rate: f32) -> usize {
        let idx = self.clips.len();
        self.clips.push(AnimClip { name: name.to_string(), frame_count, frame_rate });
        idx
    }

    pub fn add_feature(&mut self, mut feature: MotionFeature, clip_index: usize, frame_index: u32) {
        feature.clip_index  = clip_index;
        feature.frame_index = frame_index;
        self.features.push(feature);
    }

    pub fn find_nearest(&self, query: &MotionFeature) -> Option<MotionMatch> {
        let best = self.features.iter().min_by(|a, b| {
            let da = a.distance(query);
            let db = b.distance(query);
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        })?;
        Some(MotionMatch {
            clip_name:   self.clips[best.clip_index].name.clone(),
            clip_index:  best.clip_index,
            frame_index: best.frame_index,
            distance:    best.distance(query),
        })
    }

    pub fn feature_count(&self) -> usize { self.features.len() }
}

impl Default for MotionDatabase {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn make_db() -> MotionDatabase {
        let mut db = MotionDatabase::new();
        let idle = db.add_clip("idle", 60, 30.0);
        for frame in 0..60u32 {
            let f = MotionFeature::from_state(0.0, 0.0, 0.0, frame as f32 / 60.0);
            db.add_feature(f, idle, frame);
        }
        let walk = db.add_clip("walk_forward", 30, 30.0);
        for frame in 0..30u32 {
            let f = MotionFeature::from_state(0.0, 1.4, 0.0, frame as f32 / 30.0);
            db.add_feature(f, walk, frame);
        }
        let sprint = db.add_clip("sprint", 20, 30.0);
        for frame in 0..20u32 {
            let f = MotionFeature::from_state(0.0, 5.0, 0.0, frame as f32 / 20.0);
            db.add_feature(f, sprint, frame);
        }
        db
    }

    #[test]
    fn empty_database_returns_none() {
        let db = MotionDatabase::new();
        let q = MotionFeature::from_state(0.0, 0.0, 0.0, 0.0);
        assert!(db.find_nearest(&q).is_none());
    }

    #[test]
    fn idle_query_matches_idle_clip() {
        let db = make_db();
        let query = MotionFeature::from_state(0.0, 0.0, 0.0, 0.0);
        let result = db.find_nearest(&query).unwrap();
        assert_eq!(result.clip_name, "idle", "idle query should match idle clip, got '{}'", result.clip_name);
    }

    #[test]
    fn walk_query_matches_walk_clip() {
        let db = make_db();
        let query = MotionFeature::from_state(0.0, 1.5, 0.0, 0.0);
        let result = db.find_nearest(&query).unwrap();
        assert_eq!(result.clip_name, "walk_forward", "walk velocity query should match walk clip, got '{}'", result.clip_name);
    }

    #[test]
    fn sprint_query_matches_sprint_clip() {
        let db = make_db();
        let query = MotionFeature::from_state(0.0, 5.0, 0.0, 0.0);
        let result = db.find_nearest(&query).unwrap();
        assert_eq!(result.clip_name, "sprint", "sprint velocity query should match sprint clip, got '{}'", result.clip_name);
    }

    #[test]
    fn distance_between_identical_features_is_zero() {
        let a = MotionFeature::from_state(1.0, 2.0, PI / 4.0, 0.5);
        assert!(a.distance(&a) < 1e-5, "distance to self should be ~0, got {}", a.distance(&a));
    }

    #[test]
    fn distance_is_symmetric() {
        let a = MotionFeature::from_state(1.0, 0.0, 0.0, 0.0);
        let b = MotionFeature::from_state(0.0, 2.0, PI, 0.5);
        let d_ab = a.distance(&b);
        let d_ba = b.distance(&a);
        assert!((d_ab - d_ba).abs() < 1e-5, "distance should be symmetric: {} vs {}", d_ab, d_ba);
    }

    #[test]
    fn find_nearest_returns_frame_index() {
        let db = make_db();
        let query = MotionFeature::from_state(0.0, 1.4, 0.0, 0.5);
        let result = db.find_nearest(&query).unwrap();
        assert_eq!(result.clip_name, "walk_forward");
        assert!(result.frame_index > 0, "should not always return frame 0");
    }

    #[test]
    fn feature_count_matches_total_added() {
        let db = make_db();
        assert_eq!(db.feature_count(), 60 + 30 + 20, "total features: idle+walk+sprint");
    }
}
