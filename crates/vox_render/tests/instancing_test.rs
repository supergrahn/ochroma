use vox_render::gpu::instancing::{InstanceManager, InstanceTransform};
use glam::{Vec3, Quat};
use uuid::Uuid;

#[test]
fn register_and_add_instances() {
    let mut mgr = InstanceManager::new();
    let uuid = Uuid::new_v4();
    mgr.register_asset(uuid, 1000);
    mgr.add_instance(uuid, Vec3::ZERO, Quat::IDENTITY, 1.0, 0, 0);
    mgr.add_instance(uuid, Vec3::new(10.0, 0.0, 0.0), Quat::IDENTITY, 1.0, 1, 0);
    assert_eq!(mgr.total_instances(), 2);
}

#[test]
fn total_splats_calculated() {
    let mut mgr = InstanceManager::new();
    let uuid = Uuid::new_v4();
    mgr.register_asset(uuid, 5000);
    for i in 0..100 {
        mgr.add_instance(uuid, Vec3::new(i as f32, 0.0, 0.0), Quat::IDENTITY, 1.0, i, 0);
    }
    assert_eq!(mgr.total_splats(), 500_000); // 100 instances x 5000 splats
}

#[test]
fn memory_savings_significant() {
    let mut mgr = InstanceManager::new();
    let uuid = Uuid::new_v4();
    mgr.register_asset(uuid, 100_000); // 100k splats per building
    for i in 0..1000 {
        mgr.add_instance(uuid, Vec3::new(i as f32 * 10.0, 0.0, 0.0), Quat::IDENTITY, 1.0, i, 0);
    }
    let ratio = mgr.memory_savings_ratio();
    assert!(ratio < 0.01, "Instancing 1000x should save >99% memory, ratio={}", ratio);
}

#[test]
fn clear_instances_resets() {
    let mut mgr = InstanceManager::new();
    let uuid = Uuid::new_v4();
    mgr.register_asset(uuid, 100);
    mgr.add_instance(uuid, Vec3::ZERO, Quat::IDENTITY, 1.0, 0, 0);
    assert_eq!(mgr.total_instances(), 1);
    mgr.clear_instances();
    assert_eq!(mgr.total_instances(), 0);
}

#[test]
fn instance_transform_size() {
    assert_eq!(std::mem::size_of::<InstanceTransform>(), 80);
}

#[test]
fn active_batches_filters_empty() {
    let mut mgr = InstanceManager::new();
    let uuid1 = Uuid::new_v4();
    let uuid2 = Uuid::new_v4();
    mgr.register_asset(uuid1, 100);
    mgr.register_asset(uuid2, 200);
    mgr.add_instance(uuid1, Vec3::ZERO, Quat::IDENTITY, 1.0, 0, 0);
    // uuid2 has no instances
    assert_eq!(mgr.active_batches().len(), 1);
}
