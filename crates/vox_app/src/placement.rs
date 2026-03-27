use bevy_ecs::prelude::*;
use glam::{Mat4, Quat, Vec3};
use vox_core::ecs::{LodLevel, SplatInstanceComponent};
use crate::ui::UiAction;

static NEXT_INSTANCE_ID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(2000);

/// Process pending UI actions against the ECS world.
pub fn process_actions(world: &mut World, actions: &[UiAction]) {
    for action in actions {
        match action {
            UiAction::PlaceAsset { asset_uuid, position } => {
                let id = NEXT_INSTANCE_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                world.spawn(SplatInstanceComponent {
                    asset_uuid: *asset_uuid,
                    position: *position,
                    rotation: Quat::IDENTITY,
                    scale: 1.0,
                    instance_id: id,
                    lod: LodLevel::Full,
                });
                println!("[ochroma] Placed asset at ({:.1}, {:.1}, {:.1})", position.x, position.y, position.z);
            }
            UiAction::SelectInstance { instance_id } => {
                println!("[ochroma] Selected instance {}", instance_id);
            }
            UiAction::Deselect => {}
            UiAction::PlaceService { service_type, position } => {
                println!("[ochroma] Place service '{}' at ({:.1}, {:.1}, {:.1})", service_type, position.x, position.y, position.z);
            }
            UiAction::ZoneArea { zone_type, position } => {
                println!("[ochroma] Zone '{}' at ({:.1}, {:.1}, {:.1})", zone_type, position.x, position.y, position.z);
            }
            UiAction::ChangeGameSpeed { speed } => {
                println!("[ochroma] Game speed changed to {}", speed);
            }
        }
    }
}

/// Simple ground-plane ray cast: given camera pos and ray direction, find where ray hits y=0.
pub fn ray_ground_intersection(origin: Vec3, direction: Vec3) -> Option<Vec3> {
    if direction.y.abs() < 1e-6 {
        return None; // Ray parallel to ground
    }
    let t = -origin.y / direction.y;
    if t < 0.0 {
        return None; // Behind camera
    }
    Some(origin + direction * t)
}

/// Compute ray origin and direction from screen coordinates and camera matrices.
pub fn screen_to_ray(
    screen_x: f32,
    screen_y: f32,
    width: u32,
    height: u32,
    inv_view_proj: Mat4,
) -> (Vec3, Vec3) {
    // Convert screen coords to NDC
    let ndc_x = (2.0 * screen_x / width as f32) - 1.0;
    let ndc_y = 1.0 - (2.0 * screen_y / height as f32); // flip Y

    let near_point = inv_view_proj.project_point3(Vec3::new(ndc_x, ndc_y, -1.0));
    let far_point = inv_view_proj.project_point3(Vec3::new(ndc_x, ndc_y, 1.0));

    let direction = (far_point - near_point).normalize();
    (near_point, direction)
}
