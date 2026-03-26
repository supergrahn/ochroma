use vox_core::types::{GaussianSplat, SplatInstance};
use glam::{Vec3, Quat};
use uuid::Uuid;

#[test]
fn gaussian_splat_size_is_52_bytes() {
    assert_eq!(std::mem::size_of::<GaussianSplat>(), 52);
}

#[test]
fn splat_instance_has_required_fields() {
    let inst = SplatInstance {
        asset_uuid: Uuid::new_v4(),
        position: Vec3::ZERO,
        rotation: Quat::IDENTITY,
        instance_id: 1,
    };
    assert_eq!(inst.instance_id, 1);
}
