//! Environment Query System (EQS) — finds optimal positions in the game world
//! based on scoring criteria. Analogous to Unreal Engine's EQS.

use glam::Vec3;

/// A query that generates candidate positions and scores them.
pub struct EQSQuery {
    pub generator: QueryGenerator,
    pub tests: Vec<QueryTest>,
}

/// How to generate candidate points.
pub enum QueryGenerator {
    /// Points on a circle around a center.
    Circle { center: Vec3, radius: f32, count: u32 },
    /// Points on a grid.
    Grid { center: Vec3, half_extent: f32, spacing: f32 },
    /// Points along a line.
    Line { start: Vec3, end: Vec3, count: u32 },
}

/// Scoring and filtering tests applied to each candidate point.
pub enum QueryTest {
    /// Score by distance to a point. Closer = higher score.
    DistanceTo { point: Vec3, weight: f32 },
    /// Score by distance FROM a point. Further = higher score.
    DistanceFrom { point: Vec3, weight: f32 },
    /// Score by dot product with a direction (prefer points in a direction).
    DirectionPreference { origin: Vec3, direction: Vec3, weight: f32 },
    /// Score by height (prefer higher/lower ground).
    HeightPreference { prefer_high: bool, weight: f32 },
    /// Filter: only keep points within range.
    RangeFilter { center: Vec3, min_dist: f32, max_dist: f32 },
}

/// A scored candidate position.
#[derive(Debug, Clone, Copy)]
pub struct QueryResult {
    pub position: Vec3,
    pub score: f32,
}

impl QueryGenerator {
    /// Generate candidate positions.
    pub fn generate(&self) -> Vec<Vec3> {
        match self {
            QueryGenerator::Circle { center, radius, count } => {
                let mut points = Vec::with_capacity(*count as usize);
                for i in 0..*count {
                    let angle = (i as f32 / *count as f32) * std::f32::consts::TAU;
                    let x = center.x + radius * angle.cos();
                    let z = center.z + radius * angle.sin();
                    points.push(Vec3::new(x, center.y, z));
                }
                points
            }
            QueryGenerator::Grid { center, half_extent, spacing } => {
                let mut points = Vec::new();
                let start = -(*half_extent);
                let end = *half_extent;
                let mut x = start;
                while x <= end + f32::EPSILON {
                    let mut z = start;
                    while z <= end + f32::EPSILON {
                        points.push(Vec3::new(center.x + x, center.y, center.z + z));
                        z += spacing;
                    }
                    x += spacing;
                }
                points
            }
            QueryGenerator::Line { start, end, count } => {
                if *count <= 1 {
                    return vec![*start];
                }
                let mut points = Vec::with_capacity(*count as usize);
                for i in 0..*count {
                    let t = i as f32 / (*count - 1) as f32;
                    points.push(*start + (*end - *start) * t);
                }
                points
            }
        }
    }
}

impl QueryTest {
    /// Score a candidate position. Returns `None` if the point is filtered out.
    pub fn score(&self, position: Vec3) -> Option<f32> {
        match self {
            QueryTest::DistanceTo { point, weight } => {
                let dist = position.distance(*point);
                // Invert distance so closer = higher. Use 1/(1+d) for bounded scoring.
                Some(weight / (1.0 + dist))
            }
            QueryTest::DistanceFrom { point, weight } => {
                let dist = position.distance(*point);
                Some(weight * dist)
            }
            QueryTest::DirectionPreference { origin, direction, weight } => {
                let to_point = (position - *origin).normalize_or_zero();
                let dir = direction.normalize_or_zero();
                let dot = to_point.dot(dir).max(0.0);
                Some(weight * dot)
            }
            QueryTest::HeightPreference { prefer_high, weight } => {
                let h = position.y;
                let score = if *prefer_high { h } else { -h };
                Some(weight * score)
            }
            QueryTest::RangeFilter { center, min_dist, max_dist } => {
                let dist = position.distance(*center);
                if dist >= *min_dist && dist <= *max_dist {
                    Some(0.0) // pass filter, contributes no score
                } else {
                    None // filtered out
                }
            }
        }
    }
}

impl EQSQuery {
    /// Run the query: generate candidates, score them, return sorted results (best first).
    pub fn run(&self) -> Vec<QueryResult> {
        let candidates = self.generator.generate();
        let mut results = Vec::new();

        for pos in candidates {
            let mut total_score = 0.0f32;
            let mut filtered = false;
            for test in &self.tests {
                match test.score(pos) {
                    Some(s) => total_score += s,
                    None => {
                        filtered = true;
                        break;
                    }
                }
            }
            if !filtered {
                results.push(QueryResult { position: pos, score: total_score });
            }
        }

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn circle_generates_correct_count() {
        let generator = QueryGenerator::Circle {
            center: Vec3::ZERO,
            radius: 10.0,
            count: 8,
        };
        let points = generator.generate();
        assert_eq!(points.len(), 8);
        // All points should be roughly at distance 10 from center
        for p in &points {
            let dist = p.distance(Vec3::ZERO);
            assert!((dist - 10.0).abs() < 0.01, "point distance was {dist}");
        }
    }

    #[test]
    fn grid_generates_points() {
        let generator = QueryGenerator::Grid {
            center: Vec3::ZERO,
            half_extent: 2.0,
            spacing: 1.0,
        };
        let points = generator.generate();
        // -2, -1, 0, 1, 2 => 5 per axis => 25
        assert_eq!(points.len(), 25);
    }

    #[test]
    fn line_generates_correct_count() {
        let generator = QueryGenerator::Line {
            start: Vec3::ZERO,
            end: Vec3::new(10.0, 0.0, 0.0),
            count: 5,
        };
        let points = generator.generate();
        assert_eq!(points.len(), 5);
        assert!((points[0] - Vec3::ZERO).length() < 0.01);
        assert!((points[4] - Vec3::new(10.0, 0.0, 0.0)).length() < 0.01);
    }

    #[test]
    fn distance_to_scores_closer_higher() {
        let target = Vec3::new(5.0, 0.0, 0.0);
        let test = QueryTest::DistanceTo { point: target, weight: 1.0 };
        let close_score = test.score(Vec3::new(4.0, 0.0, 0.0)).unwrap();
        let far_score = test.score(Vec3::new(0.0, 0.0, 0.0)).unwrap();
        assert!(close_score > far_score, "closer should score higher: {close_score} vs {far_score}");
    }

    #[test]
    fn range_filter_removes_out_of_range() {
        let query = EQSQuery {
            generator: QueryGenerator::Circle {
                center: Vec3::ZERO,
                radius: 10.0,
                count: 16,
            },
            tests: vec![
                QueryTest::RangeFilter {
                    center: Vec3::ZERO,
                    min_dist: 5.0,
                    max_dist: 15.0,
                },
            ],
        };
        let results = query.run();
        // All circle points are at radius 10, within [5, 15], so all pass
        assert_eq!(results.len(), 16);

        // Now filter that excludes all
        let query2 = EQSQuery {
            generator: QueryGenerator::Circle {
                center: Vec3::ZERO,
                radius: 10.0,
                count: 16,
            },
            tests: vec![
                QueryTest::RangeFilter {
                    center: Vec3::ZERO,
                    min_dist: 20.0,
                    max_dist: 30.0,
                },
            ],
        };
        let results2 = query2.run();
        assert_eq!(results2.len(), 0);
    }

    #[test]
    fn best_result_is_closest_to_target() {
        let target = Vec3::new(10.0, 0.0, 0.0);
        let query = EQSQuery {
            generator: QueryGenerator::Circle {
                center: Vec3::ZERO,
                radius: 5.0,
                count: 32,
            },
            tests: vec![
                QueryTest::DistanceTo { point: target, weight: 1.0 },
            ],
        };
        let results = query.run();
        assert!(!results.is_empty());
        let best = results[0].position;
        // Best point should be the one closest to (10,0,0), which is at (5,0,0)
        assert!(best.x > 0.0, "best x should be positive: {best}");
        let best_dist = best.distance(target);
        for r in &results[1..] {
            assert!(r.position.distance(target) >= best_dist - 0.01);
        }
    }
}
