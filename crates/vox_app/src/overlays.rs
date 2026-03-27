use half::f16;
use vox_core::types::GaussianSplat;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OverlayType {
    None,
    Traffic,
    LandValue,
    ZoneColour,
    ServiceCoverage,
}

/// Generate overlay splats for a given type.
pub fn generate_overlay_splats(
    overlay: OverlayType,
    zone_plots: &[(f32, f32, &str)],  // (x, z, zone_type)
    service_positions: &[(f32, f32, f32)], // (x, z, radius)
    traffic_densities: &[(f32, f32, f32)], // (x, z, density_0_to_1)
) -> Vec<GaussianSplat> {
    match overlay {
        OverlayType::None => Vec::new(),
        OverlayType::ZoneColour => {
            zone_plots.iter().map(|(x, z, zt)| {
                let (r, g, b) = match *zt {
                    "residential" => (0.2, 0.4, 0.8), // blue
                    "commercial" => (0.8, 0.7, 0.2), // yellow
                    "industrial" => (0.6, 0.2, 0.6), // purple
                    _ => (0.5, 0.5, 0.5),
                };
                make_overlay_splat(*x, *z, r, g, b, 0.3)
            }).collect()
        }
        OverlayType::ServiceCoverage => {
            let mut splats = Vec::new();
            for (cx, cz, radius) in service_positions {
                let steps = (*radius / 5.0).ceil() as i32;
                for dx in -steps..=steps {
                    for dz in -steps..=steps {
                        let x = cx + dx as f32 * 5.0;
                        let z = cz + dz as f32 * 5.0;
                        let dist = ((x - cx).powi(2) + (z - cz).powi(2)).sqrt();
                        if dist <= *radius {
                            let alpha = 0.15 * (1.0 - dist / radius);
                            splats.push(make_overlay_splat(x, z, 0.2, 0.5, 0.9, alpha));
                        }
                    }
                }
            }
            splats
        }
        OverlayType::Traffic => {
            traffic_densities.iter().map(|(x, z, density)| {
                let r = *density;
                let g = 1.0 - density;
                make_overlay_splat(*x, *z, r, g, 0.0, 0.25)
            }).collect()
        }
        OverlayType::LandValue => {
            // Simple: closer to services = higher value
            zone_plots.iter().map(|(x, z, _)| {
                let min_dist = service_positions.iter()
                    .map(|(sx, sz, _)| ((x - sx).powi(2) + (z - sz).powi(2)).sqrt())
                    .fold(f32::MAX, f32::min);
                let value = (1.0 - (min_dist / 500.0).min(1.0)).max(0.0);
                make_overlay_splat(*x, *z, 0.1, value * 0.8, 0.1, 0.2)
            }).collect()
        }
    }
}

fn make_overlay_splat(x: f32, z: f32, r: f32, g: f32, b: f32, alpha: f32) -> GaussianSplat {
    // Encode RGB as approximate spectral values
    // R → 620nm, G → 540nm, B → 460nm
    let spectral: [u16; 8] = [
        f16::from_f32(b * 0.5).to_bits(), // 380nm
        f16::from_f32(b).to_bits(),        // 420nm
        f16::from_f32(b).to_bits(),        // 460nm
        f16::from_f32(g * 0.5).to_bits(),  // 500nm
        f16::from_f32(g).to_bits(),        // 540nm
        f16::from_f32(r * 0.5).to_bits(),  // 580nm
        f16::from_f32(r).to_bits(),        // 620nm
        f16::from_f32(r * 0.5).to_bits(),  // 660nm
    ];

    GaussianSplat {
        position: [x, 0.15, z], // above terrain
        scale: [2.5, 0.01, 2.5], // flat disc
        rotation: [0, 0, 0, 32767],
        opacity: (alpha * 255.0).min(255.0) as u8,
        _pad: [0; 3],
        spectral,
    }
}
