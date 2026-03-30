//! Spectral material fingerprinting.
//!
//! Identifies material classes from 8-band spectral signatures using
//! a compact descriptor + nearest-centroid classifier.
//! Used by AI perception (guards recognize materials by spectral "smell"),
//! physics (material-correct collision sounds), and PCG (material-consistent placement).

/// Recognized material classes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MaterialClass {
    Stone,
    Metal,
    Wood,
    Vegetation,
    Water,
    Fire,
    Skin,
    Fabric,
}

/// Compact handcrafted descriptor for an 8-band spectral signature.
#[derive(Debug, Clone, Copy)]
pub struct SpectralDescriptor {
    pub mean: f32,
    pub variance: f32,
    pub skew: f32,
    pub red_blue_ratio: f32,
    pub peak_band: u8,
}

impl SpectralDescriptor {
    pub fn from_spectral(s: &[f32; 8]) -> Self {
        let mean = s.iter().sum::<f32>() / 8.0;

        let variance = s.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / 8.0;

        let high_sum: f32 = s[4..8].iter().sum();
        let low_sum: f32 = s[0..4].iter().sum();
        let skew = high_sum / (low_sum + 0.001) - 1.0;

        let red_blue_ratio = (s[6] + s[7]) / (s[0] + s[1] + 0.001);

        let peak_band = s
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i as u8)
            .unwrap_or(0);

        Self {
            mean,
            variance,
            skew,
            red_blue_ratio,
            peak_band,
        }
    }
}

// Hardcoded material centroids (canonical spectral signatures)
pub const STONE_SPECTRAL: [f32; 8] = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]; // grey
pub const METAL_SPECTRAL: [f32; 8] = [0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6]; // bright grey
pub const WOOD_SPECTRAL: [f32; 8] = [0.1, 0.1, 0.15, 0.2, 0.4, 0.6, 0.7, 0.5]; // brown/orange
pub const VEGETATION_SPECTRAL: [f32; 8] = [0.05, 0.1, 0.5, 0.7, 0.5, 0.2, 0.1, 0.05]; // green peak
pub const WATER_SPECTRAL: [f32; 8] = [0.7, 0.6, 0.5, 0.3, 0.1, 0.05, 0.02, 0.01]; // blue
pub const FIRE_SPECTRAL: [f32; 8] = [0.0, 0.0, 0.05, 0.1, 0.4, 0.8, 1.0, 0.7]; // red-orange
pub const SKIN_SPECTRAL: [f32; 8] = [0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 0.6, 0.4]; // warm
pub const FABRIC_SPECTRAL: [f32; 8] = [0.3, 0.4, 0.4, 0.35, 0.3, 0.25, 0.2, 0.2]; // neutral

/// Compute Euclidean distance in 8D spectral space.
pub fn spectral_distance(a: &[f32; 8], b: &[f32; 8]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}

/// Classify a spectral signature to the nearest material centroid.
pub fn classify(spectral: &[f32; 8]) -> MaterialClass {
    let centroids: &[(MaterialClass, &[f32; 8])] = &[
        (MaterialClass::Stone, &STONE_SPECTRAL),
        (MaterialClass::Metal, &METAL_SPECTRAL),
        (MaterialClass::Wood, &WOOD_SPECTRAL),
        (MaterialClass::Vegetation, &VEGETATION_SPECTRAL),
        (MaterialClass::Water, &WATER_SPECTRAL),
        (MaterialClass::Fire, &FIRE_SPECTRAL),
        (MaterialClass::Skin, &SKIN_SPECTRAL),
        (MaterialClass::Fabric, &FABRIC_SPECTRAL),
    ];

    centroids
        .iter()
        .min_by(|(_, a), (_, b)| {
            spectral_distance(spectral, a)
                .partial_cmp(&spectral_distance(spectral, b))
                .unwrap()
        })
        .map(|(class, _)| *class)
        .unwrap()
}

/// Database of spectral fingerprints for custom and built-in materials.
pub struct SpectralFingerprintDb {
    entries: Vec<(MaterialClass, [f32; 8], String)>,
}

impl SpectralFingerprintDb {
    /// Create a new database pre-populated with all 8 built-in materials.
    pub fn new() -> Self {
        let mut db = Self {
            entries: Vec::new(),
        };
        db.add(MaterialClass::Stone, STONE_SPECTRAL, "Stone".to_string());
        db.add(MaterialClass::Metal, METAL_SPECTRAL, "Metal".to_string());
        db.add(MaterialClass::Wood, WOOD_SPECTRAL, "Wood".to_string());
        db.add(
            MaterialClass::Vegetation,
            VEGETATION_SPECTRAL,
            "Vegetation".to_string(),
        );
        db.add(MaterialClass::Water, WATER_SPECTRAL, "Water".to_string());
        db.add(MaterialClass::Fire, FIRE_SPECTRAL, "Fire".to_string());
        db.add(MaterialClass::Skin, SKIN_SPECTRAL, "Skin".to_string());
        db.add(
            MaterialClass::Fabric,
            FABRIC_SPECTRAL,
            "Fabric".to_string(),
        );
        db
    }

    /// Add a custom entry to the database.
    pub fn add(&mut self, class: MaterialClass, spectral: [f32; 8], name: String) {
        self.entries.push((class, spectral, name));
    }

    /// Return the nearest (class, distance) for a given spectral signature.
    pub fn nearest(&self, spectral: &[f32; 8]) -> Option<(MaterialClass, f32)> {
        self.entries
            .iter()
            .map(|(class, centroid, _)| {
                let dist = spectral_distance(spectral, centroid);
                (*class, dist)
            })
            .min_by(|(_, da), (_, db)| da.partial_cmp(db).unwrap())
    }
}

impl Default for SpectralFingerprintDb {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_fire_spectral() {
        assert_eq!(classify(&FIRE_SPECTRAL), MaterialClass::Fire);
    }

    #[test]
    fn classify_water_spectral() {
        assert_eq!(classify(&WATER_SPECTRAL), MaterialClass::Water);
    }

    #[test]
    fn classify_vegetation_spectral() {
        assert_eq!(classify(&VEGETATION_SPECTRAL), MaterialClass::Vegetation);
    }

    #[test]
    fn descriptor_peak_band_correct() {
        let desc = SpectralDescriptor::from_spectral(&FIRE_SPECTRAL);
        // FIRE_SPECTRAL: [0.0, 0.0, 0.05, 0.1, 0.4, 0.8, 1.0, 0.7] — max at index 6
        assert_eq!(desc.peak_band, 6);
    }

    #[test]
    fn spectral_distance_self_is_zero() {
        let x = [0.1_f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        assert_eq!(spectral_distance(&x, &x), 0.0);
    }

    #[test]
    fn fingerprint_db_nearest_matches_classify() {
        let db = SpectralFingerprintDb::new();
        let samples: &[(&[f32; 8], MaterialClass)] = &[
            (&STONE_SPECTRAL, MaterialClass::Stone),
            (&METAL_SPECTRAL, MaterialClass::Metal),
            (&WOOD_SPECTRAL, MaterialClass::Wood),
            (&VEGETATION_SPECTRAL, MaterialClass::Vegetation),
            (&WATER_SPECTRAL, MaterialClass::Water),
            (&FIRE_SPECTRAL, MaterialClass::Fire),
            (&SKIN_SPECTRAL, MaterialClass::Skin),
            (&FABRIC_SPECTRAL, MaterialClass::Fabric),
        ];
        for (spectral, expected_class) in samples {
            let classify_result = classify(spectral);
            let (db_class, _dist) = db.nearest(spectral).expect("db should not be empty");
            assert_eq!(classify_result, *expected_class, "classify mismatch for {:?}", expected_class);
            assert_eq!(db_class, *expected_class, "db.nearest mismatch for {:?}", expected_class);
        }
    }
}
