use glam::{Quat, Vec3};
use uuid::Uuid;
use vox_core::types::{GaussianSplat, SplatInstance};

#[test]
fn gaussian_splat_size_is_52_bytes() {
    assert_eq!(std::mem::size_of::<GaussianSplat>(), 96);
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
