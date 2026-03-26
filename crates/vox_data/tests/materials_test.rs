use vox_core::spectral::{Illuminant, spectral_to_xyz, xyz_to_srgb};
use vox_data::materials::MaterialLibrary;

#[test]
fn library_has_base_materials() {
    let lib = MaterialLibrary::default();
    assert!(lib.get("brick_red").is_some());
    assert!(lib.get("concrete_raw").is_some());
    assert!(lib.get("glass_clear").is_some());
    assert!(lib.get("vegetation_leaf").is_some());
    assert!(lib.get("metal_steel").is_some());
    assert!(lib.get("asphalt_dry").is_some());
    assert!(lib.get("slate_grey").is_some());
    assert!(lib.get("water_still").is_some());
    assert!(lib.get("soil_dry").is_some());
    assert!(lib.get("wood_painted_green").is_some());
}

#[test]
fn brick_red_looks_reddish_under_d65() {
    let lib = MaterialLibrary::default();
    let brick = lib.get("brick_red").unwrap();
    let xyz = spectral_to_xyz(&brick.spd, &Illuminant::d65());
    let rgb = xyz_to_srgb(xyz);
    assert!(rgb[0] > rgb[1], "brick R={:.3} > G={:.3}", rgb[0], rgb[1]);
    assert!(rgb[0] > rgb[2], "brick R={:.3} > B={:.3}", rgb[0], rgb[2]);
}

#[test]
fn vegetation_looks_greenish_under_d65() {
    let lib = MaterialLibrary::default();
    let veg = lib.get("vegetation_leaf").unwrap();
    let xyz = spectral_to_xyz(&veg.spd, &Illuminant::d65());
    let rgb = xyz_to_srgb(xyz);
    assert!(rgb[1] > rgb[0], "G={:.3} > R={:.3}", rgb[1], rgb[0]);
}
