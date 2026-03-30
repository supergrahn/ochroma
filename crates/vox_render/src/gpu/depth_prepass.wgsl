// depth_prepass.wgsl — Per-splat center depth and screen-space velocity.
//
// One thread per splat.  Projects each splat to screen, writes:
//   - View-space depth (-clip.w) to depth_out at the projected pixel
//   - Screen-space velocity (current - previous frame pixel pos) to velocity_out
//
// Race condition note: multiple splats may map to the same pixel.  WGSL has no
// atomic min on f32, so the last write wins.  This is acceptable for this
// approximation-quality prepass.

// ---------------------------------------------------------------------------
// Structs
// ---------------------------------------------------------------------------

struct CameraUniform {
    view_proj:     mat4x4<f32>,
    view:          mat4x4<f32>,
    inv_view:      mat4x4<f32>,
    viewport_size: vec2<f32>,
    _pad:          vec2<f32>,
}

struct GpuSplatFull {
    position_depth: vec4<f32>,
    conic:          vec3<f32>,
    _pad0:          f32,
    opacity_color:  vec4<f32>,
    spectral:       array<f32, 8>,
}

struct DepthPrepassParams {
    width:       u32,
    height:      u32,
    splat_count: u32,
    _pad:        u32,
}

// ---------------------------------------------------------------------------
// Bindings
// ---------------------------------------------------------------------------

@group(0) @binding(0) var<uniform>        camera:       CameraUniform;
@group(0) @binding(1) var<uniform>        prev_camera:  CameraUniform;
@group(0) @binding(2) var<storage, read>  splats:       array<GpuSplatFull>;
@group(0) @binding(3) var                 depth_out:    texture_storage_2d<r32float, write>;
@group(0) @binding(4) var                 velocity_out: texture_storage_2d<rg32float, write>;
@group(0) @binding(5) var<uniform>        params:       DepthPrepassParams;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Project a world-space position to integer screen coordinates and clip.w.
/// Returns (screen_pixel_xy, clip_w).  `screen_pixel_xy` is in pixel space
/// with the origin at the top-left.
fn project(vp: mat4x4<f32>, world_pos: vec3<f32>, w: f32, h: f32) -> vec3<f32> {
    let clip   = vp * vec4<f32>(world_pos, 1.0);
    let ndc    = clip.xy / clip.w;
    let screen = (ndc * 0.5 + 0.5) * vec2<f32>(w, h);
    return vec3<f32>(screen.x, screen.y, clip.w);
}

// ---------------------------------------------------------------------------
// Compute entry point
// ---------------------------------------------------------------------------

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.splat_count {
        return;
    }

    let sp = splats[idx];
    let w  = f32(params.width);
    let h  = f32(params.height);

    // --- Current frame projection ---
    let curr = project(camera.view_proj, sp.position_depth.xyz, w, h);
    let curr_screen = curr.xy;
    let clip_w      = curr.z;

    // Discard splats behind the camera or outside NDC.
    if clip_w <= 0.0 {
        return;
    }

    let px = i32(curr_screen.x);
    let py = i32(curr_screen.y);
    if px < 0 || py < 0 || px >= i32(params.width) || py >= i32(params.height) {
        return;
    }

    let coord = vec2<i32>(px, py);

    // Write view-space depth: -clip.w (positive = in front of camera).
    textureStore(depth_out, coord, vec4<f32>(-clip_w, 0.0, 0.0, 0.0));

    // --- Previous frame projection ---
    let prev        = project(prev_camera.view_proj, sp.position_depth.xyz, w, h);
    let prev_screen = prev.xy;

    // Velocity in pixels (current - previous).
    let velocity = curr_screen - prev_screen;
    textureStore(velocity_out, coord, vec4<f32>(velocity.x, velocity.y, 0.0, 0.0));
}
