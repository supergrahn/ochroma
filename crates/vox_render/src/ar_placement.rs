use glam::Vec3;

/// Orientation of a detected AR surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SurfaceOrientation {
    Horizontal,
    Vertical,
}

/// A detected AR plane surface.
#[derive(Debug, Clone)]
pub struct ARSurface {
    pub id: u64,
    /// Centre of the detected plane in world space.
    pub position: Vec3,
    /// Surface normal (points away from the surface).
    pub normal: Vec3,
    /// Extent (half-size) in the plane's local X and Z axes.
    pub extent: [f32; 2],
    pub orientation: SurfaceOrientation,
}

impl ARSurface {
    pub fn new_horizontal(id: u64, position: Vec3, extent: [f32; 2]) -> Self {
        Self {
            id,
            position,
            normal: Vec3::Y,
            extent,
            orientation: SurfaceOrientation::Horizontal,
        }
    }

    pub fn new_vertical(id: u64, position: Vec3, normal: Vec3, extent: [f32; 2]) -> Self {
        Self {
            id,
            position,
            normal: normal.normalize(),
            extent,
            orientation: SurfaceOrientation::Vertical,
        }
    }

    /// Surface area in square metres.
    pub fn area(&self) -> f32 {
        self.extent[0] * 2.0 * self.extent[1] * 2.0
    }
}

/// Manages an AR session and its detected surfaces.
pub struct ARSession {
    surfaces: Vec<ARSurface>,
    next_id: u64,
    pub active: bool,
}

impl ARSession {
    pub fn new() -> Self {
        Self {
            surfaces: Vec::new(),
            next_id: 1,
            active: false,
        }
    }

    /// Start the AR session.
    pub fn start(&mut self) {
        self.active = true;
    }

    /// Stop the AR session and clear surfaces.
    pub fn stop(&mut self) {
        self.active = false;
        self.surfaces.clear();
    }

    /// Simulate detection of a new horizontal surface (e.g., table).
    pub fn detect_horizontal(&mut self, position: Vec3, extent: [f32; 2]) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.surfaces.push(ARSurface::new_horizontal(id, position, extent));
        id
    }

    /// Simulate detection of a new vertical surface (e.g., wall).
    pub fn detect_vertical(&mut self, position: Vec3, normal: Vec3, extent: [f32; 2]) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.surfaces
            .push(ARSurface::new_vertical(id, position, normal, extent));
        id
    }

    pub fn surface_count(&self) -> usize {
        self.surfaces.len()
    }

    pub fn get_surface(&self, id: u64) -> Option<&ARSurface> {
        self.surfaces.iter().find(|s| s.id == id)
    }

    /// Return all horizontal surfaces.
    pub fn horizontal_surfaces(&self) -> Vec<&ARSurface> {
        self.surfaces
            .iter()
            .filter(|s| s.orientation == SurfaceOrientation::Horizontal)
            .collect()
    }

    /// Return the largest horizontal surface (most likely a table/floor).
    pub fn largest_horizontal(&self) -> Option<&ARSurface> {
        self.horizontal_surfaces()
            .into_iter()
            .max_by(|a, b| a.area().partial_cmp(&b.area()).unwrap())
    }
}

impl Default for ARSession {
    fn default() -> Self {
        Self::new()
    }
}

/// Snap a world-space position onto a detected surface.
/// Projects `pos` onto the surface plane along the surface normal.
pub fn place_on_surface(surface: &ARSurface, pos: Vec3) -> Vec3 {
    // Project onto the plane: remove the component along the normal
    // relative to the surface position.
    let offset = pos - surface.position;
    let along_normal = offset.dot(surface.normal);
    pos - surface.normal * along_normal
}

/// Result of projecting a city onto a table surface.
#[derive(Debug, Clone)]
pub struct CityProjection {
    /// Centre of the projected city on the table.
    pub centre: Vec3,
    /// Uniform scale factor applied to the city.
    pub scale: f32,
    /// Offset to apply to world-space city coordinates.
    pub offset: Vec3,
}

/// Scale an entire city (given its AABB) to fit on a detected horizontal surface.
///
/// `city_min` / `city_max` define the city's axis-aligned bounding box in
/// world coordinates. The city is uniformly scaled and centred on the surface.
pub fn project_city_to_table(
    surface: &ARSurface,
    city_min: Vec3,
    city_max: Vec3,
) -> CityProjection {
    let city_size = city_max - city_min;
    let city_centre = (city_min + city_max) * 0.5;

    // Use the horizontal extents (X, Z) of the city.
    let city_w = city_size.x;
    let city_d = city_size.z;

    let table_w = surface.extent[0] * 2.0;
    let table_d = surface.extent[1] * 2.0;

    // Uniform scale: fit the largest city dimension into the table.
    let scale = if city_w > 0.0 && city_d > 0.0 {
        (table_w / city_w).min(table_d / city_d)
    } else if city_w > 0.0 {
        table_w / city_w
    } else if city_d > 0.0 {
        table_d / city_d
    } else {
        1.0
    };

    let offset = surface.position - city_centre * scale;

    CityProjection {
        centre: surface.position,
        scale,
        offset,
    }
}
