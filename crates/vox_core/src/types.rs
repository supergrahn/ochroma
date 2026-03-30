use bytemuck::{Pod, Zeroable};
use glam::{Quat, Vec3};
use half::f16;
use uuid::Uuid;

/// Hybrid Gaussian splat: 2DGS flat disk (kind=0) or 3DGS ellipsoid (kind=1).
///
/// GPU layout: 96 bytes, 16-byte aligned.
///
/// **Do not construct with struct literal syntax.** Use [`GaussianSplat::surface`]
/// or [`GaussianSplat::volume`] — this keeps construction isolated so that
/// layout changes only require updating those two functions.
///
/// **2DGS surface (kind=0):**
///   `tangent_u` / `tangent_v` are unit vectors defining the disk plane.
///   Normal = `cross(tangent_u, tangent_v)` — always exact.
///   `scale_u` / `scale_v` are disk semi-axes (radii). `scale_w` unused (0).
///   `rotation` unused — identity `[0, 0, 0, 32767]`.
///
/// **3DGS volume (kind=1):**
///   `tangent_u` / `tangent_v` unused (zero).
///   `scale_u` / `scale_v` / `scale_w` are ellipsoid half-axes.
///   `rotation` is quantized quaternion XYZW, each component / 32767.
///
/// **Spectral:** 16 bands, 380–755 nm at 25 nm steps (USGS wavelength grid).
///   Each value is f16 stored as u16 (use [`GaussianSplat::spectral_f32`] to read).
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct GaussianSplat {
    position:  [f32; 3],   // 12  world-space centroid
    kind:      u32,        //  4  0=surface(2DGS)  1=volume(3DGS)
    tangent_u: [f32; 3],   // 12  2DGS disk u-axis | 3DGS: zero
    scale_u:   f32,        //  4  2DGS u-radius    | 3DGS: x-scale
    tangent_v: [f32; 3],   // 12  2DGS disk v-axis | 3DGS: zero
    scale_v:   f32,        //  4  2DGS v-radius    | 3DGS: y-scale
    rotation:  [i16; 4],   //  8  2DGS: identity   | 3DGS: quat XYZW /32767
    scale_w:   f32,        //  4  2DGS: 0          | 3DGS: z-scale
    opacity:   u8,         //  1
    _pad:      [u8; 3],    //  3
    spectral:  [u16; 16],  // 32  f16 per band, 380–755 nm
}                          // = 96 bytes total

const _: () = assert!(std::mem::size_of::<GaussianSplat>() == 96);

impl GaussianSplat {
    /// Number of spectral bands: 380–755 nm at 25 nm steps (USGS wavelength grid).
    pub const BANDS: usize = 16;
}

/// A placed instance of an asset in the world.
#[derive(Debug, Clone)]
pub struct SplatInstance {
    pub asset_uuid: Uuid,
    pub position: Vec3,
    pub rotation: Quat,
    pub instance_id: u32,
}

// ---------------------------------------------------------------------------
// Constructors
// ---------------------------------------------------------------------------

impl GaussianSplat {
    /// Construct a 2DGS surface splat.
    ///
    /// `tangent_u` and `tangent_v` must be unit vectors spanning the disk plane.
    /// Normal = `cross(tangent_u, tangent_v)`.
    pub fn surface(
        position:  [f32; 3],
        tangent_u: [f32; 3],
        tangent_v: [f32; 3],
        scale_u:   f32,
        scale_v:   f32,
        opacity:   u8,
        spectral:  [u16; 16],
    ) -> Self {
        Self {
            position,
            kind: 0,
            tangent_u,
            scale_u,
            tangent_v,
            scale_v,
            rotation: [0, 0, 0, 32767],
            scale_w: 0.0,
            opacity,
            _pad: [0; 3],
            spectral,
        }
    }

    /// Construct a 3DGS volume splat.
    ///
    /// `scale` is `[x, y, z]` half-axes of the ellipsoid.
    /// `rotation` is quaternion XYZW.
    pub fn volume(
        position: [f32; 3],
        scale:    [f32; 3],
        rotation: Quat,
        opacity:  u8,
        spectral: [u16; 16],
    ) -> Self {
        Self {
            position,
            kind: 1,
            tangent_u: [0.0; 3],
            scale_u: scale[0],
            tangent_v: [0.0; 3],
            scale_v: scale[1],
            rotation: [
                (rotation.x * 32767.0) as i16,
                (rotation.y * 32767.0) as i16,
                (rotation.z * 32767.0) as i16,
                (rotation.w * 32767.0) as i16,
            ],
            scale_w: scale[2],
            opacity,
            _pad: [0; 3],
            spectral,
        }
    }
}

// ---------------------------------------------------------------------------
// Read accessors
// ---------------------------------------------------------------------------

impl GaussianSplat {
    pub fn position(&self) -> [f32; 3] { self.position }
    pub fn kind(&self) -> u32 { self.kind }
    pub fn is_surface(&self) -> bool { self.kind == 0 }
    pub fn is_volume(&self) -> bool { self.kind == 1 }
    pub fn tangent_u(&self) -> [f32; 3] { self.tangent_u }
    pub fn tangent_v(&self) -> [f32; 3] { self.tangent_v }
    pub fn scale_u(&self) -> f32 { self.scale_u }
    pub fn scale_v(&self) -> f32 { self.scale_v }
    pub fn scale_w(&self) -> f32 { self.scale_w }
    /// Half-axes as `[scale_u, scale_v, scale_w]`.
    pub fn scales(&self) -> [f32; 3] { [self.scale_u, self.scale_v, self.scale_w] }
    pub fn rotation_raw(&self) -> [i16; 4] { self.rotation }
    pub fn opacity(&self) -> u8 { self.opacity }
    pub fn spectral(&self) -> &[u16; 16] { &self.spectral }

    /// Decode one spectral band to f32.
    pub fn spectral_f32(&self, band: usize) -> f32 {
        f16::from_bits(self.spectral[band]).to_f32()
    }

    /// Decode rotation quaternion to `glam::Quat`.
    pub fn decoded_rotation(&self) -> Quat {
        Quat::from_xyzw(
            self.rotation[0] as f32 / 32767.0,
            self.rotation[1] as f32 / 32767.0,
            self.rotation[2] as f32 / 32767.0,
            self.rotation[3] as f32 / 32767.0,
        )
    }

    /// Exact surface normal for 2DGS splats; rotation z-axis approximation for 3DGS.
    pub fn normal(&self) -> [f32; 3] {
        if self.kind == 0 {
            let u = Vec3::from(self.tangent_u);
            let v = Vec3::from(self.tangent_v);
            let n = u.cross(v);
            let len = n.length();
            if len > 1e-8 { (n / len).into() } else { [0.0, 1.0, 0.0] }
        } else {
            self.decoded_rotation().mul_vec3(Vec3::Z).into()
        }
    }
}

// ---------------------------------------------------------------------------
// Mutable accessors
// ---------------------------------------------------------------------------

impl GaussianSplat {
    pub fn position_mut(&mut self) -> &mut [f32; 3] { &mut self.position }
    pub fn spectral_mut(&mut self) -> &mut [u16; 16] { &mut self.spectral }

    pub fn set_position(&mut self, pos: [f32; 3]) { self.position = pos; }
    pub fn set_opacity(&mut self, opacity: u8) { self.opacity = opacity; }

    pub fn set_scales(&mut self, u: f32, v: f32, w: f32) {
        self.scale_u = u;
        self.scale_v = v;
        self.scale_w = w;
    }

    pub fn set_tangents(&mut self, u: [f32; 3], v: [f32; 3]) {
        self.tangent_u = u;
        self.tangent_v = v;
    }

    /// Apply a `TransformComponent`-style transform (scale → rotate → translate)
    /// to this splat in place. For 2DGS, tangent axes are rotated; for both
    /// kinds, scales are multiplied by the geometric mean of the transform scale.
    pub fn apply_transform(&mut self, position: Vec3, rotation: Quat, scale: Vec3) {
        let scaled = Vec3::from(self.position) * scale;
        self.position = (rotation * scaled + position).into();

        let sf = (scale.x * scale.y * scale.z).cbrt();
        self.scale_u *= sf;
        self.scale_v *= sf;
        self.scale_w *= sf;

        if self.kind == 0 {
            self.tangent_u = (rotation * Vec3::from(self.tangent_u)).into();
            self.tangent_v = (rotation * Vec3::from(self.tangent_v)).into();
        }
    }
}
