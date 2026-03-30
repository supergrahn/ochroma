use ochroma_engine::prelude::*;
use glam::Quat;

#[test]
fn engine_prelude_imports_core_types() {
    let splat = GaussianSplat::volume(
        [0.0, 0.0, 0.0],
        [0.1, 0.1, 0.1],
        Quat::IDENTITY,
        255,
        [15360; 16],
    );
    assert_eq!(splat.opacity(), 255);
}

#[test]
fn engine_prelude_imports_render() {
    let cam = CameraController::new(16.0 / 9.0);
    assert!(cam.view_proj().x_axis.length() > 0.0);
}

#[test]
fn engine_prelude_imports_data() {
    let lib = MaterialLibrary::default();
    assert!(lib.get("brick_red").is_some());
}

#[test]
fn engine_sdk_version() {
    assert_eq!(env!("CARGO_PKG_VERSION"), "0.1.0");
}
