//! Biome-driven ambient soundscape.
//! Maps forge-terrain BiomeKind to spectral synthesis parameters.

/// Matches forge-terrain BiomeKind exactly (copied as plain enum — no dep on forge required).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BiomeKind {
    Alpine          = 0,
    Tundra          = 1,
    Forest          = 2,
    Grassland       = 3,
    Desert          = 4,
    Wetland         = 5,
    Coastal         = 6,
    SubalpineShrub  = 7,
    Savanna         = 8,
    Taiga           = 9,
    TropicalRainforest = 10,
}

impl BiomeKind {
    pub fn from_id(id: u8) -> Self {
        match id {
            0  => Self::Alpine,
            1  => Self::Tundra,
            2  => Self::Forest,
            3  => Self::Grassland,
            4  => Self::Desert,
            5  => Self::Wetland,
            6  => Self::Coastal,
            7  => Self::SubalpineShrub,
            8  => Self::Savanna,
            9  => Self::Taiga,
            10 => Self::TropicalRainforest,
            _  => Self::Grassland,
        }
    }
}

/// Ambient mix weights per biome: [wind, water, insects, ice]
#[derive(Debug, Clone)]
pub struct BiomeAmbientMix {
    pub wind:    f32,
    pub water:   f32,
    pub insects: f32,
    pub ice:     f32,
}

impl BiomeAmbientMix {
    pub fn for_biome(biome: BiomeKind) -> Self {
        match biome {
            BiomeKind::Alpine             => Self { wind: 0.8, water: 0.2, insects: 0.0, ice: 0.4 },
            BiomeKind::Tundra             => Self { wind: 0.9, water: 0.1, insects: 0.0, ice: 0.7 },
            BiomeKind::Forest             => Self { wind: 0.2, water: 0.1, insects: 0.7, ice: 0.0 },
            BiomeKind::Grassland          => Self { wind: 0.4, water: 0.0, insects: 0.5, ice: 0.0 },
            BiomeKind::Desert             => Self { wind: 0.6, water: 0.0, insects: 0.2, ice: 0.0 },
            BiomeKind::Wetland            => Self { wind: 0.1, water: 0.6, insects: 0.9, ice: 0.0 },
            BiomeKind::Coastal            => Self { wind: 0.5, water: 0.8, insects: 0.1, ice: 0.0 },
            BiomeKind::SubalpineShrub     => Self { wind: 0.6, water: 0.1, insects: 0.2, ice: 0.2 },
            BiomeKind::Savanna            => Self { wind: 0.5, water: 0.0, insects: 0.6, ice: 0.0 },
            BiomeKind::Taiga              => Self { wind: 0.3, water: 0.1, insects: 0.4, ice: 0.1 },
            BiomeKind::TropicalRainforest => Self { wind: 0.1, water: 0.3, insects: 1.0, ice: 0.0 },
        }
    }

    /// Blend from current mix toward target with temporal smoothing.
    pub fn blend_toward(&self, target: &Self, alpha: f32) -> Self {
        Self {
            wind:    self.wind    + (target.wind    - self.wind)    * alpha,
            water:   self.water   + (target.water   - self.water)   * alpha,
            insects: self.insects + (target.insects - self.insects) * alpha,
            ice:     self.ice     + (target.ice     - self.ice)     * alpha,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wetland_has_high_insects() {
        let mix = BiomeAmbientMix::for_biome(BiomeKind::Wetland);
        println!("wetland insects={}", mix.insects);
        assert!(mix.insects > 0.8, "wetland insects={}", mix.insects);
    }

    #[test]
    fn tundra_has_no_insects() {
        let mix = BiomeAmbientMix::for_biome(BiomeKind::Tundra);
        println!("tundra insects={}", mix.insects);
        assert_eq!(mix.insects, 0.0, "tundra should have no insects");
    }

    #[test]
    fn alpine_has_wind_and_ice() {
        let mix = BiomeAmbientMix::for_biome(BiomeKind::Alpine);
        assert!(mix.wind > 0.6 && mix.ice > 0.3, "alpine: wind={} ice={}", mix.wind, mix.ice);
    }

    #[test]
    fn blend_converges() {
        let a = BiomeAmbientMix::for_biome(BiomeKind::Desert);
        let b = BiomeAmbientMix::for_biome(BiomeKind::Forest);
        let blended = a.blend_toward(&b, 1.0);
        assert!((blended.insects - b.insects).abs() < 1e-5, "full blend should equal target");
    }

    #[test]
    fn biome_from_id_roundtrips() {
        for id in 0u8..7 {
            let biome = BiomeKind::from_id(id);
            assert_eq!(biome as u8, id);
        }
    }
}
