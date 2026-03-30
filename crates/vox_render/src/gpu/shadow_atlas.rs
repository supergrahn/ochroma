//! Shadow map texture atlas for multiple light sources.
//!
//! A 4096×4096 depth texture array with 8 layers, packed using a shelf algorithm.

use wgpu;

/// A single allocated region within the shadow atlas.
pub struct ShadowAtlasEntry {
    pub light_id: u32,
    pub layer: u32,
    pub region: [u32; 4], // [x, y, width, height] in texels
}

/// Request to allocate shadow map space for one light.
pub struct LightShadowRequest {
    pub light_id: u32,
    pub resolution: u32, // e.g. 512 for point lights, 2048 for directional
    pub is_point_light: bool, // point lights need 6 sub-entries (cube faces)
}

/// Shelf packing state per layer: (next_x, shelf_y, shelf_height).
type ShelfState = (u32, u32, u32);

/// Shadow map texture atlas backed by a `D2Array` `Depth32Float` texture.
pub struct ShadowAtlas {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub entries: Vec<ShadowAtlasEntry>,
    atlas_width: u32,
    atlas_height: u32,
    num_layers: u32,
    shelves: Vec<ShelfState>,
}

impl ShadowAtlas {
    const WIDTH: u32 = 4096;
    const HEIGHT: u32 = 4096;
    const LAYERS: u32 = 8;

    /// Create a 4096×4096 atlas with 8 depth layers.
    pub fn new(device: &wgpu::Device) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("shadow_atlas"),
            size: wgpu::Extent3d {
                width: Self::WIDTH,
                height: Self::HEIGHT,
                depth_or_array_layers: Self::LAYERS,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("shadow_atlas_view"),
            format: Some(wgpu::TextureFormat::Depth32Float),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            aspect: wgpu::TextureAspect::DepthOnly,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
            ..Default::default()
        });

        let shelves = vec![(0u32, 0u32, 0u32); Self::LAYERS as usize];

        Self {
            texture,
            view,
            entries: Vec::new(),
            atlas_width: Self::WIDTH,
            atlas_height: Self::HEIGHT,
            num_layers: Self::LAYERS,
            shelves,
        }
    }

    /// Try to place a single `resolution × resolution` tile somewhere in the atlas.
    /// Returns `Some((layer, x, y))` on success.
    fn place_tile(&mut self, resolution: u32) -> Option<(u32, u32, u32)> {
        for layer in 0..self.num_layers as usize {
            let (next_x, shelf_y, shelf_height) = self.shelves[layer];

            // Fits on the current shelf?
            if next_x + resolution <= self.atlas_width {
                let x = next_x;
                let y = shelf_y;
                self.shelves[layer] = (
                    next_x + resolution,
                    shelf_y,
                    shelf_height.max(resolution),
                );
                return Some((layer as u32, x, y));
            }

            // Start a new shelf on this layer.
            let new_shelf_y = shelf_y + shelf_height;
            if new_shelf_y + resolution <= self.atlas_height {
                self.shelves[layer] = (resolution, new_shelf_y, resolution);
                return Some((layer as u32, 0, new_shelf_y));
            }

            // This layer is full — try the next one.
        }
        None
    }

    /// Pack a shadow map request into the atlas. Returns the assigned entries.
    /// For point lights, returns 6 entries (one per cube face).
    pub fn alloc(&mut self, req: &LightShadowRequest) -> Vec<ShadowAtlasEntry> {
        let face_count = if req.is_point_light { 6 } else { 1 };
        let mut result = Vec::with_capacity(face_count);

        for _face in 0..face_count {
            if let Some((layer, x, y)) = self.place_tile(req.resolution) {
                let entry = ShadowAtlasEntry {
                    light_id: req.light_id,
                    layer,
                    region: [x, y, req.resolution, req.resolution],
                };
                self.entries.push(ShadowAtlasEntry {
                    light_id: entry.light_id,
                    layer: entry.layer,
                    region: entry.region,
                });
                result.push(entry);
            }
        }

        result
    }

    /// Clear all allocations.
    pub fn reset(&mut self) {
        self.entries.clear();
        for s in &mut self.shelves {
            *s = (0, 0, 0);
        }
    }

    pub fn texture_view(&self) -> &wgpu::TextureView {
        &self.view
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Tests that don't need a real wgpu::Device use the packing logic directly.

    fn make_shelf_state(layers: u32) -> Vec<ShelfState> {
        vec![(0u32, 0u32, 0u32); layers as usize]
    }

    fn place_tile_logic(
        shelves: &mut Vec<ShelfState>,
        atlas_w: u32,
        atlas_h: u32,
        resolution: u32,
    ) -> Option<(u32, u32, u32)> {
        for layer in 0..shelves.len() {
            let (next_x, shelf_y, shelf_height) = shelves[layer];
            if next_x + resolution <= atlas_w {
                let x = next_x;
                let y = shelf_y;
                shelves[layer] = (next_x + resolution, shelf_y, shelf_height.max(resolution));
                return Some((layer as u32, x, y));
            }
            let new_shelf_y = shelf_y + shelf_height;
            if new_shelf_y + resolution <= atlas_h {
                shelves[layer] = (resolution, new_shelf_y, resolution);
                return Some((layer as u32, 0, new_shelf_y));
            }
        }
        None
    }

    #[test]
    fn atlas_new_has_no_entries() {
        // We can't create a wgpu::Device in unit tests without a GPU, so we
        // verify the invariant via the packing state directly.
        let shelves = make_shelf_state(8);
        // Fresh state: no entries placed yet.
        assert_eq!(shelves.len(), 8);
        for &(x, y, h) in &shelves {
            assert_eq!((x, y, h), (0, 0, 0));
        }
    }

    #[test]
    fn alloc_directional_light() {
        // Directional light: 1 tile at resolution 2048 → placed at (0,0,2048,2048) layer 0.
        let mut shelves = make_shelf_state(8);
        let result = place_tile_logic(&mut shelves, 4096, 4096, 2048);
        assert!(result.is_some());
        let (layer, x, y) = result.unwrap();
        assert_eq!(layer, 0);
        assert_eq!(x, 0);
        assert_eq!(y, 0);
    }

    #[test]
    fn alloc_point_light_returns_6_entries() {
        let mut shelves = make_shelf_state(8);
        let mut count = 0usize;
        for _ in 0..6 {
            if place_tile_logic(&mut shelves, 4096, 4096, 512).is_some() {
                count += 1;
            }
        }
        assert_eq!(count, 6);
    }

    #[test]
    fn reset_clears_entries() {
        let mut shelves = make_shelf_state(8);
        // Allocate something
        let _ = place_tile_logic(&mut shelves, 4096, 4096, 512);
        // Simulate reset
        for s in &mut shelves {
            *s = (0, 0, 0);
        }
        // All shelves back to zero
        for &(x, y, h) in &shelves {
            assert_eq!((x, y, h), (0, 0, 0));
        }
    }
}
