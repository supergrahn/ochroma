use vox_app::content_browser::{classify, ContentType, ContentAction};
use std::path::Path;

#[test]
fn classify_ply_is_gaussian_splat() {
    assert_eq!(classify(Path::new("foo.ply")), ContentType::GaussianSplat);
}

#[test]
fn classify_glb_is_mesh() {
    assert_eq!(classify(Path::new("model.glb")), ContentType::Mesh);
}

#[test]
fn classify_vxm_is_ochroma_asset() {
    assert_eq!(classify(Path::new("asset.vxm")), ContentType::OchromaAsset);
}

#[test]
fn content_action_import_asset_variant_exists() {
    let action = ContentAction::ImportAsset(std::path::PathBuf::from("model.glb"));
    if let ContentAction::ImportAsset(p) = action {
        assert_eq!(p.extension().unwrap(), "glb");
    } else {
        panic!("wrong variant");
    }
}
