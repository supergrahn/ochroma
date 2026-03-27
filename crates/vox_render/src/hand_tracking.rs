use glam::Vec3;

/// Indices into `HandState::finger_tips`.
pub const THUMB: usize = 0;
pub const INDEX: usize = 1;
pub const MIDDLE: usize = 2;
pub const RING: usize = 3;
pub const PINKY: usize = 4;

/// Per-hand tracking state.
#[derive(Debug, Clone)]
pub struct HandState {
    /// Palm centre in world space.
    pub palm_position: Vec3,
    /// Fingertip positions: [thumb, index, middle, ring, pinky].
    pub finger_tips: [Vec3; 5],
    /// Finger knuckle (base) positions, same order.
    pub finger_bases: [Vec3; 5],
}

impl Default for HandState {
    fn default() -> Self {
        let palm = Vec3::ZERO;
        // Default: fingers splayed outward from palm at rest.
        let tips = [
            Vec3::new(-0.04, 0.02, -0.06),
            Vec3::new(-0.02, 0.0, -0.10),
            Vec3::new(0.0, 0.0, -0.10),
            Vec3::new(0.02, 0.0, -0.09),
            Vec3::new(0.04, 0.0, -0.08),
        ];
        let bases = [
            Vec3::new(-0.03, 0.01, -0.02),
            Vec3::new(-0.015, 0.0, -0.02),
            Vec3::new(0.0, 0.0, -0.02),
            Vec3::new(0.015, 0.0, -0.02),
            Vec3::new(0.03, 0.0, -0.02),
        ];
        Self {
            palm_position: palm,
            finger_tips: tips,
            finger_bases: bases,
        }
    }
}

impl HandState {
    /// Distance between two fingertips.
    pub fn tip_distance(&self, a: usize, b: usize) -> f32 {
        self.finger_tips[a].distance(self.finger_tips[b])
    }

    /// How curled a finger is: 0.0 = fully extended, 1.0 = fully curled to palm.
    /// Computed as ratio of (base-to-tip distance) versus a reference open length.
    pub fn curl(&self, finger: usize) -> f32 {
        let open_length = 0.08_f32; // approximate open finger length
        let current = self.finger_tips[finger].distance(self.finger_bases[finger]);
        (1.0 - (current / open_length)).clamp(0.0, 1.0)
    }
}

/// Recognized hand gestures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gesture {
    None,
    Pinch,
    Grab,
    Point,
}

/// Thresholds for gesture recognition.
#[derive(Debug, Clone)]
pub struct GestureThresholds {
    /// Max distance (m) between thumb and index for pinch.
    pub pinch_distance: f32,
    /// Min curl value to consider a finger "curled".
    pub curl_threshold: f32,
    /// Max curl value to consider a finger "extended".
    pub extend_threshold: f32,
}

impl Default for GestureThresholds {
    fn default() -> Self {
        Self {
            pinch_distance: 0.025,
            curl_threshold: 0.6,
            extend_threshold: 0.3,
        }
    }
}

/// Detects gestures from hand state.
pub struct GestureRecognizer {
    pub thresholds: GestureThresholds,
}

impl GestureRecognizer {
    pub fn new() -> Self {
        Self {
            thresholds: GestureThresholds::default(),
        }
    }

    pub fn with_thresholds(thresholds: GestureThresholds) -> Self {
        Self { thresholds }
    }

    /// Recognize the current gesture.
    pub fn recognize(&self, hand: &HandState) -> Gesture {
        let t = &self.thresholds;

        // Grab: all fingers curled (checked first — more specific than pinch).
        let all_curled = (0..5).all(|f| hand.curl(f) >= t.curl_threshold);
        if all_curled {
            return Gesture::Grab;
        }

        // Pinch: thumb and index tips close together.
        let pinch_dist = hand.tip_distance(THUMB, INDEX);
        if pinch_dist < t.pinch_distance {
            return Gesture::Pinch;
        }

        // Point: index extended, others curled.
        let index_extended = hand.curl(INDEX) < t.extend_threshold;
        let others_curled = [MIDDLE, RING, PINKY]
            .iter()
            .all(|&f| hand.curl(f) >= t.curl_threshold);
        if index_extended && others_curled {
            return Gesture::Point;
        }

        Gesture::None
    }
}

impl Default for GestureRecognizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Game action mapped from a gesture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InteractionAction {
    None,
    Select,
    Move,
    Place,
}

/// Maps gestures to game interaction actions.
pub struct InteractionManager {
    recognizer: GestureRecognizer,
}

impl InteractionManager {
    pub fn new() -> Self {
        Self {
            recognizer: GestureRecognizer::new(),
        }
    }

    pub fn with_recognizer(recognizer: GestureRecognizer) -> Self {
        Self { recognizer }
    }

    /// Determine the interaction action for the current hand state.
    pub fn update(&self, hand: &HandState) -> InteractionAction {
        match self.recognizer.recognize(hand) {
            Gesture::Pinch => InteractionAction::Select,
            Gesture::Grab => InteractionAction::Move,
            Gesture::Point => InteractionAction::Place,
            Gesture::None => InteractionAction::None,
        }
    }
}

impl Default for InteractionManager {
    fn default() -> Self {
        Self::new()
    }
}
