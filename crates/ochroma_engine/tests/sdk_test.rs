use ochroma_engine::prelude::*;

#[test]
fn engine_prelude_imports_core_types() {
    let splat = GaussianSplat {
        position: [0.0, 0.0, 0.0],
        scale: [0.1, 0.1, 0.1],
        rotation: [0, 0, 0, 32767],
        opacity: 255,
        _pad: [0; 3],
        spectral: [15360; 8],
    };
    assert_eq!(splat.opacity, 255);
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
