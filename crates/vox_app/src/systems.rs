use bevy_ecs::prelude::*;
use glam::Vec3;
use vox_core::ecs::{SplatInstanceComponent, SplatAssetComponent, LodLevel};
use vox_core::types::GaussianSplat;
use vox_render::frustum::Frustum;
use vox_render::lod;
#[cfg(feature = "spectra-native")]
use vox_render::spectra_render::native::SpectraBackendSystem;
#[cfg(feature = "spectra-native")]
use vox_render::SpectraCameraParams;

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
        let new_lod = match lod::select_lod(distance) {
            lod::LodLevel::Full => LodLevel::Full,
            lod::LodLevel::Reduced => LodLevel::Reduced,
        };
        if instance.lod != new_lod {
            instance.lod = new_lod;
        }
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
                    let p = ws.position();
                    ws.set_position([p[0] + offset.x, p[1] + offset.y, p[2] + offset.z]);
                    visible.splats.push(ws);
                }
                break;
            }
        }
    }
}

/// Drive the Spectra native GPU renderer one tick.
///
/// Uses Bevy's `Changed<SplatInstanceComponent>` to detect mutations —
/// fires whenever position, scale, spectral values, or any other field on
/// a `SplatInstanceComponent` changes in the ECS. `vox_render` has no Bevy
/// dep; change detection lives here.
///
/// Camera orientation is derived from `CameraState.view_proj` (inverse).
/// `fov_y`, `near`, and `far` use defaults until `CameraState` exposes them.
#[cfg(feature = "spectra-native")]
pub fn spectra_render_system(
    mut backend: ResMut<SpectraBackendSystem>,
    changed:     Query<(), Changed<SplatInstanceComponent>>,
    visible:     Res<VisibleSplats>,
    camera:      Res<CameraState>,
) {
    let scene_changed = !changed.is_empty();

    // Derive camera vectors from the inverse view-projection matrix.
    let inv = camera.view_proj.inverse();
    let fwd = (inv * glam::Vec4::new(0.0, 0.0, -1.0, 0.0)).truncate().normalize();
    let up  = (inv * glam::Vec4::new(0.0, 1.0,  0.0, 0.0)).truncate().normalize();
    let pos = camera.position;  // direct field — correct for any matrix type

    let cam = SpectraCameraParams {
        position: pos.into(),
        forward:  fwd.into(),
        up:       up.into(),
        // Defaults until CameraState exposes fov/near/far:
        fov_y: std::f32::consts::FRAC_PI_4,
        near:  0.1,
        far:   1000.0,
    };

    backend.tick(&visible.splats, cam, scene_changed);
}
