//! Integration tests for the `TerrainScene` foliage-scatter and texture-paint
//! pipeline stages added on top of the generate -> deform -> splat facade.
//!
//! Every assertion checks a real computed/geometric outcome:
//!   * foliage counts are bounded by the analytic density expectation and the
//!     slope filter is re-verified against the actual placements, and
//!   * painted material weights are read back at exact texels and compared to
//!     hand-computed brush falloff values.

use vox_terrain::foliage::FoliageRule;
use vox_terrain::scene::TerrainScene;

/// Build a 48x48 (horizontal) volume with a flat low ground plane and a solid
/// hill (sphere) poking through it. The hill gives the derived surface a real
/// spread of slopes: a near-flat top and surroundings plus genuinely steep
/// flanks, so a slope constraint actually excludes placements rather than being
/// a no-op.
fn hilly_scene() -> TerrainScene {
    let mut scene = TerrainScene::with_ground(48, 32, 48, 1.0, /*ground*/ -8.0, /*grass*/ 1, /*seed*/ 7);
    // Hill centred at the world origin; its top rises above the ground plane.
    scene.sculpt_fill_sphere([0.0, -2.0, 0.0], 10.0, /*dirt*/ 2);
    scene
}

#[test]
fn scatter_foliage_respects_density_and_slope() {
    let scene = hilly_scene();

    // The heightmap the scene derives drives both the analytic density
    // expectation and the slope re-check below — sample it once.
    let hm = scene.build_surface_heightmap();
    let area = hm.area();
    // 48 cells * 1.0 m cell size, squared.
    assert!(
        (area - 2304.0).abs() < 1e-3,
        "derived heightmap area must be 48m * 48m = 2304 m^2, got {area}"
    );

    let density = 5.0f32; // instances per 100 m^2
    let max_slope = 25.0f32; // degrees
    let rule = FoliageRule {
        name: "Shrub".into(),
        asset_path: "shrub.ply".into(),
        density,
        // Wide height band so HEIGHT never filters — this test isolates SLOPE.
        min_height: -1000.0,
        max_height: 1000.0,
        max_slope,
        min_scale: 0.5,
        max_scale: 1.5,
        random_rotation: true,
        // No clustering, so a placement's stored position is exactly the
        // sampled (x, z) the slope filter tested — re-checking slope at the
        // position is therefore a faithful verification of the filter.
        cluster_radius: 0.0,
    };

    // Analytic upper bound: with NO filtering the scatterer emits exactly
    // floor(area/100 * density) candidates.
    let candidates = (area / 100.0 * density) as usize;
    assert_eq!(candidates, 115, "sanity: expected 115 raw candidates");

    let instances = scene.scatter_foliage(std::slice::from_ref(&rule), /*seed*/ 42);

    // (1) DENSITY: the placed count must be within the expected range for this
    //     density. It can never exceed the raw candidate count, and because the
    //     steep flanks reject a real fraction it must be meaningfully below it
    //     yet still a large majority of candidates survive on the flat areas.
    let n = instances.len();
    assert!(
        n <= candidates,
        "placed foliage ({n}) cannot exceed raw candidates ({candidates})"
    );
    assert!(
        (70..candidates).contains(&n),
        "placed foliage ({n}) must land in [70, {candidates}) for density {density}: \
         a non-trivial slope-rejected fraction below the cap, but most flat ground kept"
    );

    // (2) SLOPE: NO instance may sit on ground steeper than the rule's limit.
    //     Re-evaluate the surface slope at every actual placement. Track the
    //     steepest accepted placement to prove the constraint is the binding
    //     one (it should approach, but not cross, the limit).
    let mut steepest_accepted = 0.0f32;
    for inst in &instances {
        let slope = hm.slope_at(inst.position[0], inst.position[2]);
        assert!(
            slope <= max_slope + 1e-3,
            "foliage placed on too-steep slope: {slope:.3} deg > limit {max_slope} deg \
             at {:?}",
            inst.position
        );
        steepest_accepted = steepest_accepted.max(slope);
    }
    // The terrain genuinely contains slopes above the limit (the hill flanks),
    // so the steepest *accepted* placement must come close to the limit —
    // confirming the filter let in everything up to the boundary and nothing
    // past it, rather than the terrain simply being flat everywhere.
    assert!(
        steepest_accepted > 10.0,
        "steepest accepted placement ({steepest_accepted:.3} deg) should be well above flat: \
         proves the slope filter is exercised, not vacuous"
    );

    // (3) DETERMINISM: same seed + surface -> identical placement count.
    let again = scene.scatter_foliage(std::slice::from_ref(&rule), 42);
    assert_eq!(
        again.len(),
        n,
        "scatter_foliage must be deterministic for a fixed seed and surface"
    );
}

#[test]
fn texture_paint_writes_expected_texel_weights() {
    // A flat scene is fine here — texture paint is independent of terrain shape.
    let mut scene = TerrainScene::with_ground(16, 16, 16, 1.0, 0.0, /*grass*/ 1, /*seed*/ 3);

    // 32x32 splat map with two material layers. Layer 0 (grass) becomes the
    // fully-weighted base everywhere; layer 1 (dirt) starts at weight 0.
    scene.init_splat_map(32, 32);
    let grass = scene.add_material_layer("grass", "mat_grass", [0.05, 0.05, 0.08, 0.12, 0.40, 0.25, 0.08, 0.05]);
    let dirt = scene.add_material_layer("dirt", "mat_dirt", [0.10, 0.12, 0.15, 0.20, 0.22, 0.20, 0.18, 0.15]);
    assert_eq!(grass, 0);
    assert_eq!(dirt, 1);

    let (cx, cz) = (16usize, 16usize);
    let radius = 4usize;

    // BEFORE: the base layer owns the centre texel entirely, dirt is absent.
    assert_eq!(
        scene.material_weight_at(cx, cz, grass),
        Some(1.0),
        "base grass layer must start at full weight at the centre texel"
    );
    assert_eq!(
        scene.material_weight_at(cx, cz, dirt),
        Some(0.0),
        "dirt layer must start at zero weight at the centre texel"
    );

    // PAINT dirt at the centre with full strength.
    let strength = 1.0f32;
    scene.paint_material(cx, cz, dirt, strength, radius);

    // AFTER: at the exact brush centre the falloff is 1.0 (dist 0), so dirt's
    // pre-normalisation weight is min(0.0 + strength*1.0, 1.0) = 1.0 and grass
    // stays at 1.0; per-texel normalisation then splits them 0.5 / 0.5.
    let w_dirt_centre = scene
        .material_weight_at(cx, cz, dirt)
        .expect("dirt weight must exist after paint");
    let w_grass_centre = scene
        .material_weight_at(cx, cz, grass)
        .expect("grass weight must exist after paint");
    assert!(
        (w_dirt_centre - 0.5).abs() < 1e-5,
        "dirt weight at brush centre must normalise to 0.5, got {w_dirt_centre}"
    );
    assert!(
        (w_grass_centre - 0.5).abs() < 1e-5,
        "grass weight at brush centre must normalise to 0.5, got {w_grass_centre}"
    );
    // Weights at the centre must sum to 1 after normalisation.
    assert!(
        ((w_dirt_centre + w_grass_centre) - 1.0).abs() < 1e-5,
        "centre texel weights must sum to 1, got {}",
        w_dirt_centre + w_grass_centre
    );

    // A texel partway out (2 of 4 radius) gets a smaller painted weight than
    // the centre because of the linear distance falloff: falloff = 1 - dist/r.
    // dist = 2, r = 4 -> falloff = 0.5 -> dirt pre-norm = 0.5, grass = 1.0 ->
    // normalised dirt = 0.5 / 1.5 = 1/3.
    let (mx, mz) = (cx + 2, cz);
    let w_dirt_mid = scene
        .material_weight_at(mx, mz, dirt)
        .expect("dirt weight must exist at mid texel");
    assert!(
        (w_dirt_mid - (1.0 / 3.0)).abs() < 1e-4,
        "dirt weight 2 texels from a radius-4 centre must normalise to 1/3, got {w_dirt_mid}"
    );
    // Falloff is monotonic: the mid texel must carry less dirt than the centre.
    assert!(
        w_dirt_mid < w_dirt_centre,
        "dirt weight must decrease with distance from brush centre: \
         centre {w_dirt_centre}, mid {w_dirt_mid}"
    );

    // A texel well outside the brush radius is untouched: still pure grass.
    let (ox, oz) = (cx + radius + 3, cz);
    assert_eq!(
        scene.material_weight_at(ox, oz, dirt),
        Some(0.0),
        "dirt weight outside the brush radius must remain 0"
    );
    assert_eq!(
        scene.material_weight_at(ox, oz, grass),
        Some(1.0),
        "grass weight outside the brush radius must remain 1"
    );

    // The blended spectral the renderer samples at the centre must be the
    // 50/50 mix of the two layers' SPDs (computed directly from the inputs).
    let spectral = scene
        .sample_material_spectral(cx, cz)
        .expect("spectral sample must exist when a splat map is present");
    let grass_spd = [0.05, 0.05, 0.08, 0.12, 0.40, 0.25, 0.08, 0.05];
    let dirt_spd = [0.10, 0.12, 0.15, 0.20, 0.22, 0.20, 0.18, 0.15];
    for c in 0..8 {
        let expected = grass_spd[c] * 0.5 + dirt_spd[c] * 0.5;
        assert!(
            (spectral[c] - expected).abs() < 1e-5,
            "blended spectral[{c}] must be the 50/50 mix {expected}, got {}",
            spectral[c]
        );
    }

    // Out-of-range layer index must report None (not panic, not a bogus value).
    assert_eq!(
        scene.material_weight_at(cx, cz, 99),
        None,
        "querying a non-existent layer must return None"
    );
}
