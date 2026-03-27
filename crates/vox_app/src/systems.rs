use bevy_ecs::prelude::*;
use glam::Vec3;
use vox_core::ecs::{SplatInstanceComponent, SplatAssetComponent, LodLevel};
use vox_core::types::GaussianSplat;
use vox_render::frustum::Frustum;
use vox_render::lod;

/// Resource: current camera state.
#[derive(Resource, Debug)]
pub struct CameraState {
    pub position: Vec3,
    pub view_proj: glam::Mat4,
}

/// Resource: visible splats after culling + LOD.
#[derive(Resource, Default)]
pub struct VisibleSplats {
    pub splats: Vec<GaussianSplat>,
}

/// Component: marks an instance as visible this frame.
#[derive(Component)]
pub struct Visible;

/// Frustum cull instances.
pub fn frustum_cull_system(
    mut commands: Commands,
    camera: Res<CameraState>,
    query: Query<(Entity, &SplatInstanceComponent)>,
) {
    let frustum = Frustum::from_view_proj(camera.view_proj);
    for (entity, instance) in query.iter() {
        let radius = instance.scale * 10.0;
        if frustum.contains_sphere(instance.position, radius) {
            commands.entity(entity).insert(Visible);
        } else {
            commands.entity(entity).remove::<Visible>();
        }
    }
}

/// Select LOD for visible instances.
pub fn lod_select_system(
    camera: Res<CameraState>,
    mut query: Query<&mut SplatInstanceComponent, With<Visible>>,
) {
    for mut instance in query.iter_mut() {
        let distance = instance.position.distance(camera.position);
        instance.lod = match lod::select_lod(distance) {
            lod::LodLevel::Full => LodLevel::Full,
            lod::LodLevel::Reduced => LodLevel::Reduced,
        };
    }
}

/// Gather visible splats into VisibleSplats resource.
pub fn gather_splats_system(
    mut visible: ResMut<VisibleSplats>,
    instances: Query<&SplatInstanceComponent, With<Visible>>,
    assets: Query<&SplatAssetComponent>,
) {
    visible.splats.clear();
    for instance in instances.iter() {
        for asset in assets.iter() {
            if asset.uuid == instance.asset_uuid {
                let offset = instance.position;
                for splat in &asset.splats {
                    let mut ws = *splat;
                    ws.position[0] += offset.x;
                    ws.position[1] += offset.y;
                    ws.position[2] += offset.z;
                    visible.splats.push(ws);
                }
                break;
            }
        }
    }
}
