//! Live preview thumbnails for node-graph nodes (rank #10 adoption candidate:
//! UE Material Editor "Live Nodes" / Unity Shader Graph node previews).
//!
//! [`node_thumbnail`] renders a REAL miniature visualization of a [`PortData`]
//! value into a flat `Vec<[u8;4]>` (RGBA, row-major, `w*h` pixels). Every pixel
//! derives from the actual data — there are no icons or placeholders. The output
//! is engine-agnostic and pure CPU so the editor can cache one per node and the
//! `vox_ui` widget can blit it onto the node body.
//!
//! Design per `PortData` kind:
//! - `Terrain` — top-down grayscale height map (heights normalized min..max,
//!   nearest-cell sampled into the thumbnail grid).
//! - `BiomeMap` — top-down per-cell biome color map (each cell → its biome hue).
//! - `Splats` — top-down orthographic scatter of real splat positions, colored
//!   by spectral-dominant hue, density-accumulated. Sampling is capped at
//!   [`SPLAT_SAMPLE_CAP`] for huge sets.
//! - `Mesh` — filled top-down silhouette rasterized from real triangles (XZ
//!   projection), normalized to the mesh bounds.
//! - `LodMesh` — same as `Mesh`, using LOD level 0 (the highest-detail mesh).
//! - `SpectralField` — 16-band vertical bar mini-chart (bar height ∝ band value).
//! - `ScalarVec` / `SplatWeights` — sparkline of the real values.
//! - `Scalar` — a single horizontal bar whose fill ∝ the value (saturating).
//! - `Instances` — top-down point scatter of the real instance XZ positions.

use crate::node_graph::PortData;

/// Background pixel for thumbnails. The widget and tests treat any pixel equal to
/// this as "empty". Chosen distinct from the data colormaps so foreground always
/// differs from background.
pub const BG: [u8; 4] = [18, 20, 26, 255];

/// Hard cap on how many splats/instances are sampled into a scatter thumbnail.
/// Huge splat sets are subsampled with a fixed stride so generation stays O(cap)
/// rather than O(N). Documented bound referenced by the splat path.
pub const SPLAT_SAMPLE_CAP: usize = 16_384;

/// Render a real miniature visualization of `data` into a `w*h` RGBA buffer.
/// Single pass over the data where feasible. Returns exactly `w*h` pixels; an
/// empty/degenerate input yields an all-background buffer.
pub fn node_thumbnail(data: &PortData, w: usize, h: usize) -> Vec<[u8; 4]> {
    let mut buf = vec![BG; w.max(1) * h.max(1)];
    if w == 0 || h == 0 {
        return buf;
    }
    match data {
        PortData::Terrain(t) => render_terrain(&mut buf, w, h, &t.heights, t.resolution as usize),
        PortData::BiomeMap(b) => render_biome(&mut buf, w, h, b),
        PortData::Splats(s) => render_splats(&mut buf, w, h, s),
        PortData::Mesh(m) => render_mesh_positions(&mut buf, w, h, &m.positions, &m.indices),
        PortData::LodMesh(l) => {
            if let Some(m) = l.first() {
                render_mesh_positions(&mut buf, w, h, &m.positions, &m.indices);
            }
        }
        PortData::SpectralField(f) => render_bars(&mut buf, w, h, &f[..]),
        PortData::ScalarVec(v) => render_sparkline(&mut buf, w, h, v),
        PortData::SplatWeights(weights) => {
            // Flatten the [r,g,b,a]-style weight tuples into a single series.
            let flat: Vec<f32> = weights.iter().flat_map(|q| q.iter().copied()).collect();
            render_sparkline(&mut buf, w, h, &flat);
        }
        PortData::Scalar(v) => render_scalar(&mut buf, w, h, *v),
        PortData::Instances(p) => render_points(&mut buf, w, h, p),
    }
    buf
}

#[inline]
fn put(buf: &mut [[u8; 4]], w: usize, h: usize, x: usize, y: usize, c: [u8; 4]) {
    if x < w && y < h {
        buf[y * w + x] = c;
    }
}

/// Terrain: nearest-cell sample the heightfield grid into the thumbnail and map
/// the normalized height to a blue→green→white grayscale-ish ramp so a height
/// ramp produces a visible brightness gradient.
fn render_terrain(buf: &mut [[u8; 4]], w: usize, h: usize, heights: &[f32], res: usize) {
    if heights.is_empty() || res == 0 {
        return;
    }
    let (mut mn, mut mx) = (f32::INFINITY, f32::NEG_INFINITY);
    for &v in heights {
        mn = mn.min(v);
        mx = mx.max(v);
    }
    let range = (mx - mn).max(1e-6);
    let rows = heights.len() / res; // may be < res for ragged data
    let rows = rows.max(1);
    for ty in 0..h {
        // Map thumbnail row → terrain row.
        let sy = (ty * rows) / h;
        for tx in 0..w {
            let sx = (tx * res) / w;
            let idx = sy * res + sx;
            if idx >= heights.len() {
                continue;
            }
            let n = ((heights[idx] - mn) / range).clamp(0.0, 1.0);
            // Dark blue (low) → green (mid) → white (high): monotonic in luminance.
            let r = (n * n * 255.0) as u8;
            let g = (n * 220.0) as u8;
            let b = (40.0 + (1.0 - n) * 120.0 + n * 95.0) as u8;
            put(buf, w, h, tx, ty, [r, g, b, 255]);
        }
    }
}

/// A stable, visually distinct color per biome byte. Derived deterministically
/// so every cell value maps to one hue; no external table needed.
fn biome_color(byte: u8) -> [u8; 4] {
    // 11 known biomes (0..=10); anything else falls through to a hashed hue.
    const TABLE: [[u8; 3]; 11] = [
        [235, 240, 245], // Alpine — near white
        [170, 190, 205], // Tundra — pale blue-gray
        [30, 120, 45],   // Forest — deep green
        [120, 190, 70],  // Grassland — light green
        [220, 195, 110], // Desert — sand
        [60, 130, 140],  // Wetland — teal
        [90, 160, 200],  // Coastal — blue
        [120, 150, 90],  // SubalpineShrub — olive
        [200, 170, 80],  // Savanna — tan
        [40, 90, 70],    // Taiga — dark teal-green
        [20, 100, 55],   // TropicalRainforest — saturated green
    ];
    if (byte as usize) < TABLE.len() {
        let c = TABLE[byte as usize];
        [c[0], c[1], c[2], 255]
    } else {
        let r = byte.wrapping_mul(73).wrapping_add(40);
        let g = byte.wrapping_mul(151).wrapping_add(30);
        let b = byte.wrapping_mul(199).wrapping_add(60);
        [r, g, b, 255]
    }
}

/// BiomeMap: square-grid per-cell color map. The cell grid is assumed square
/// (`side = round(sqrt(len))`), matching the terrain→biome pipeline.
fn render_biome(buf: &mut [[u8; 4]], w: usize, h: usize, cells: &[u8]) {
    if cells.is_empty() {
        return;
    }
    let side = (cells.len() as f64).sqrt().round() as usize;
    let side = side.max(1);
    for ty in 0..h {
        let sy = (ty * side) / h;
        for tx in 0..w {
            let sx = (tx * side) / w;
            let idx = sy * side + sx;
            if idx >= cells.len() {
                continue;
            }
            put(buf, w, h, tx, ty, biome_color(cells[idx]));
        }
    }
}

/// Map a spectral band index (0..15, 380–755nm) to an approximate visible-light
/// RGB hue, so a splat's dominant band colors its scatter point realistically.
fn band_hue(band: usize) -> [f32; 3] {
    // 16 bands over 380..755nm. Blend a coarse violet→red ramp.
    let t = band as f32 / 15.0;
    // Piecewise wavelength→RGB-ish: violet, blue, cyan, green, yellow, red.
    if t < 0.2 {
        [0.4 + t, 0.0, 0.8]
    } else if t < 0.4 {
        [0.0, (t - 0.2) * 5.0, 1.0]
    } else if t < 0.6 {
        [0.0, 1.0, 1.0 - (t - 0.4) * 5.0]
    } else if t < 0.8 {
        [(t - 0.6) * 5.0, 1.0, 0.0]
    } else {
        [1.0, 1.0 - (t - 0.8) * 5.0, 0.0]
    }
}

/// Splats: top-down (XZ) orthographic scatter. Each sampled splat is binned to a
/// thumbnail pixel; color is its spectral-dominant band hue; overlapping splats
/// accumulate density (brighten). Sampling capped at [`SPLAT_SAMPLE_CAP`].
fn render_splats(buf: &mut [[u8; 4]], w: usize, h: usize, splats: &[vox_core::types::GaussianSplat]) {
    if splats.is_empty() {
        return;
    }
    // Bounds over X (→thumbnail x) and Z (→thumbnail y).
    let (mut minx, mut maxx) = (f32::INFINITY, f32::NEG_INFINITY);
    let (mut minz, mut maxz) = (f32::INFINITY, f32::NEG_INFINITY);
    for s in splats {
        let p = s.position();
        minx = minx.min(p[0]);
        maxx = maxx.max(p[0]);
        minz = minz.min(p[2]);
        maxz = maxz.max(p[2]);
    }
    let rx = (maxx - minx).max(1e-6);
    let rz = (maxz - minz).max(1e-6);

    // Density accumulator per pixel + the dominant-band hue (averaged).
    let n = w * h;
    let mut density = vec![0u32; n];
    let mut accum = vec![[0.0f32; 3]; n];

    // Fixed stride subsampling for huge sets → O(SPLAT_SAMPLE_CAP).
    let stride = (splats.len() / SPLAT_SAMPLE_CAP).max(1);
    for s in splats.iter().step_by(stride) {
        let p = s.position();
        let fx = (p[0] - minx) / rx;
        let fz = (p[2] - minz) / rz;
        let tx = ((fx * (w as f32 - 1.0)).round() as usize).min(w - 1);
        let ty = ((fz * (h as f32 - 1.0)).round() as usize).min(h - 1);
        // Dominant spectral band → hue.
        let mut best = 0usize;
        let mut best_v = f32::NEG_INFINITY;
        for b in 0..16 {
            let v = s.spectral_f32(b);
            if v > best_v {
                best_v = v;
                best = b;
            }
        }
        let hue = band_hue(best);
        let idx = ty * w + tx;
        density[idx] += 1;
        accum[idx][0] += hue[0];
        accum[idx][1] += hue[1];
        accum[idx][2] += hue[2];
    }

    let max_d = density.iter().copied().max().unwrap_or(1).max(1) as f32;
    for i in 0..n {
        let d = density[i];
        if d == 0 {
            continue;
        }
        // Average hue, scaled by a density-driven brightness (log-ish via sqrt).
        let inv = 1.0 / d as f32;
        let bright = (0.35 + 0.65 * (d as f32 / max_d).sqrt()).min(1.0);
        let r = (accum[i][0] * inv * bright * 255.0) as u8;
        let g = (accum[i][1] * inv * bright * 255.0) as u8;
        let b = (accum[i][2] * inv * bright * 255.0) as u8;
        buf[i] = [r.max(20), g.max(20), b.max(20), 255];
    }
}

/// Mesh: filled top-down (XZ) silhouette. Rasterizes each real triangle into the
/// normalized thumbnail via barycentric scanline fill. Color encodes the
/// triangle's normalized height (Y) so the silhouette reads as a shaded map.
fn render_mesh_positions(
    buf: &mut [[u8; 4]],
    w: usize,
    h: usize,
    positions: &[[f32; 3]],
    indices: &[[u32; 3]],
) {
    if positions.is_empty() || indices.is_empty() {
        return;
    }
    let (mut minx, mut maxx) = (f32::INFINITY, f32::NEG_INFINITY);
    let (mut minz, mut maxz) = (f32::INFINITY, f32::NEG_INFINITY);
    let (mut miny, mut maxy) = (f32::INFINITY, f32::NEG_INFINITY);
    for p in positions {
        minx = minx.min(p[0]);
        maxx = maxx.max(p[0]);
        minz = minz.min(p[2]);
        maxz = maxz.max(p[2]);
        miny = miny.min(p[1]);
        maxy = maxy.max(p[1]);
    }
    let rx = (maxx - minx).max(1e-6);
    let rz = (maxz - minz).max(1e-6);
    let ry = (maxy - miny).max(1e-6);

    let to_px = |p: &[f32; 3]| -> (f32, f32, f32) {
        let x = (p[0] - minx) / rx * (w as f32 - 1.0);
        let y = (p[2] - minz) / rz * (h as f32 - 1.0);
        let hy = (p[1] - miny) / ry; // normalized height for shading
        (x, y, hy)
    };

    for tri in indices {
        let (a, b, c) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        if a >= positions.len() || b >= positions.len() || c >= positions.len() {
            continue;
        }
        let pa = to_px(&positions[a]);
        let pb = to_px(&positions[b]);
        let pc = to_px(&positions[c]);
        fill_triangle(buf, w, h, pa, pb, pc);
    }
}

/// Scanline-fill a triangle given (x, y, shade) vertices. Shade (0..1) drives a
/// cool→warm fill so the silhouette is visible against the background.
fn fill_triangle(buf: &mut [[u8; 4]], w: usize, h: usize, a: (f32, f32, f32), b: (f32, f32, f32), c: (f32, f32, f32)) {
    let min_x = a.0.min(b.0).min(c.0).floor().max(0.0) as usize;
    let max_x = (a.0.max(b.0).max(c.0).ceil() as usize).min(w.saturating_sub(1));
    let min_y = a.1.min(b.1).min(c.1).floor().max(0.0) as usize;
    let max_y = (a.1.max(b.1).max(c.1).ceil() as usize).min(h.saturating_sub(1));

    let area = edge(a, b, c);
    if area.abs() < 1e-6 {
        return;
    }
    for y in min_y..=max_y {
        for x in min_x..=max_x {
            let p = (x as f32 + 0.5, y as f32 + 0.5, 0.0);
            let w0 = edge(b, c, p);
            let w1 = edge(c, a, p);
            let w2 = edge(a, b, p);
            // Inside if all the same sign as area.
            let inside = (w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0)
                || (w0 <= 0.0 && w1 <= 0.0 && w2 <= 0.0);
            if !inside {
                continue;
            }
            let l0 = w0 / area;
            let l1 = w1 / area;
            let l2 = w2 / area;
            let shade = (l0 * a.2 + l1 * b.2 + l2 * c.2).clamp(0.0, 1.0);
            let r = (60.0 + shade * 195.0) as u8;
            let g = (90.0 + shade * 140.0) as u8;
            let bch = (140.0 - shade * 90.0) as u8;
            put(buf, w, h, x, y, [r, g, bch, 255]);
        }
    }
}

#[inline]
fn edge(a: (f32, f32, f32), b: (f32, f32, f32), p: (f32, f32, f32)) -> f32 {
    (b.0 - a.0) * (p.1 - a.1) - (b.1 - a.1) * (p.0 - a.0)
}

/// SpectralField (or any series): vertical bar mini-chart. Bar height ∝ value
/// (normalized to the series max). Each bar is colored by its band hue so the
/// chart reads spectrally; taller/brighter bars sit where the value is larger.
fn render_bars(buf: &mut [[u8; 4]], w: usize, h: usize, vals: &[f32]) {
    if vals.is_empty() {
        return;
    }
    let mx = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max).max(1e-6);
    let n = vals.len();
    for tx in 0..w {
        let band = (tx * n) / w;
        let v = (vals[band] / mx).clamp(0.0, 1.0);
        let bar_top = ((1.0 - v) * (h as f32 - 1.0)).round() as usize;
        let hue = band_hue(band);
        // Brighter bars where value is larger.
        let bright = 0.4 + 0.6 * v;
        let col = [
            (hue[0] * bright * 255.0) as u8,
            (hue[1] * bright * 255.0) as u8,
            (hue[2] * bright * 255.0) as u8,
            255,
        ];
        for ty in bar_top..h {
            put(buf, w, h, tx, ty, col);
        }
    }
}

/// ScalarVec / SplatWeights: sparkline over the real values (min..max scaled),
/// drawn as a connected polyline so the shape of the data is visible.
fn render_sparkline(buf: &mut [[u8; 4]], w: usize, h: usize, vals: &[f32]) {
    if vals.is_empty() {
        return;
    }
    let mut mn = f32::INFINITY;
    let mut mx = f32::NEG_INFINITY;
    for &v in vals {
        mn = mn.min(v);
        mx = mx.max(v);
    }
    let range = (mx - mn).max(1e-6);
    let n = vals.len();
    let y_of = |v: f32| -> usize {
        let t = ((v - mn) / range).clamp(0.0, 1.0);
        (((1.0 - t) * (h as f32 - 1.0)).round() as usize).min(h - 1)
    };
    let mut prev: Option<(usize, usize)> = None;
    for tx in 0..w {
        // Sample the nearest value for this column.
        let i = if w <= 1 { 0 } else { (tx * (n - 1)) / (w - 1) };
        let y = y_of(vals[i.min(n - 1)]);
        let col = [90, 220, 160, 255];
        if let Some((px, py)) = prev {
            // Draw a vertical run connecting the previous sample to this one.
            let (lo, hi) = if py <= y { (py, y) } else { (y, py) };
            for yy in lo..=hi {
                put(buf, w, h, px, yy, col);
            }
        }
        put(buf, w, h, tx, y, col);
        prev = Some((tx, y));
    }
}

/// Scalar: a single horizontal fill bar. Fill fraction ∝ value, saturating so a
/// finite magnitude is always visible. Negative values fill from the right.
fn render_scalar(buf: &mut [[u8; 4]], w: usize, h: usize, v: f64) {
    // Saturating map: |v| compressed via v/(1+|v|) so any finite value shows.
    let frac = (v.abs() / (1.0 + v.abs())) as f32;
    let fill = ((frac * w as f32).round() as usize).min(w);
    let col = if v >= 0.0 {
        [80, 200, 120, 255]
    } else {
        [200, 110, 80, 255]
    };
    let y0 = h / 4;
    let y1 = (h * 3 / 4).max(y0 + 1);
    for ty in y0..y1.min(h) {
        for tx in 0..fill {
            let x = if v >= 0.0 { tx } else { w - 1 - tx };
            put(buf, w, h, x, ty, col);
        }
    }
}

/// Instances: top-down (XZ) point scatter of the real instance positions.
fn render_points(buf: &mut [[u8; 4]], w: usize, h: usize, pts: &[[f32; 3]]) {
    if pts.is_empty() {
        return;
    }
    let (mut minx, mut maxx) = (f32::INFINITY, f32::NEG_INFINITY);
    let (mut minz, mut maxz) = (f32::INFINITY, f32::NEG_INFINITY);
    for p in pts {
        minx = minx.min(p[0]);
        maxx = maxx.max(p[0]);
        minz = minz.min(p[2]);
        maxz = maxz.max(p[2]);
    }
    let rx = (maxx - minx).max(1e-6);
    let rz = (maxz - minz).max(1e-6);
    let stride = (pts.len() / SPLAT_SAMPLE_CAP).max(1);
    for p in pts.iter().step_by(stride) {
        let fx = (p[0] - minx) / rx;
        let fz = (p[2] - minz) / rz;
        let tx = ((fx * (w as f32 - 1.0)).round() as usize).min(w - 1);
        let ty = ((fz * (h as f32 - 1.0)).round() as usize).min(h - 1);
        put(buf, w, h, tx, ty, [180, 200, 240, 255]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node_graph::{EditorMesh, HeightfieldSpatial};
    use vox_core::types::GaussianSplat;

    fn non_bg(buf: &[[u8; 4]]) -> usize {
        buf.iter().filter(|&&p| p != BG).count()
    }

    /// Average luminance of a single thumbnail row.
    fn row_lum(buf: &[[u8; 4]], w: usize, y: usize) -> f64 {
        let mut s = 0.0;
        for x in 0..w {
            let p = buf[y * w + x];
            s += p[0] as f64 + p[1] as f64 + p[2] as f64;
        }
        s / w as f64
    }

    #[test]
    fn terrain_ramp_brightens_monotonically_down_rows() {
        // Heightfield ramp: height increases with row index → lower rows (higher y)
        // brighter. 16x16 grid.
        let res = 16usize;
        let mut heights = vec![0.0f32; res * res];
        for r in 0..res {
            for c in 0..res {
                heights[r * res + c] = r as f32; // ramp along rows
            }
        }
        let hf = HeightfieldSpatial { heights, resolution: res as u32, world_size: 100.0 };
        let (w, h) = (64usize, 40usize);
        let buf = node_thumbnail(&PortData::Terrain(hf), w, h);

        let top = row_lum(&buf, w, 2);
        let bottom = row_lum(&buf, w, h - 3);
        assert!(bottom > top, "ramp must brighten downward: top={top:.1} bottom={bottom:.1}");
        // Monotonic: each sampled row no dimmer than several rows above it.
        let q1 = row_lum(&buf, w, h / 4);
        let q3 = row_lum(&buf, w, 3 * h / 4);
        assert!(q3 > q1, "lower quartile row must be brighter: q1={q1:.1} q3={q3:.1}");
        // min < max sanity.
        assert!(top < bottom);
    }

    #[test]
    fn splats_clustered_in_corner_fill_that_quadrant_most() {
        // 200 splats clustered in the -X,-Z corner, plus 4 spread anchors so the
        // bounds span the whole [0,10] square (otherwise the cluster fills it all).
        let q = glam::Quat::IDENTITY;
        let mut splats = Vec::new();
        for i in 0..200 {
            let f = i as f32 / 200.0;
            splats.push(GaussianSplat::volume(
                [f * 0.5, 0.0, f * 0.5],
                [1.0, 1.0, 1.0],
                q,
                255,
                [0u16; 16],
            ));
        }
        // Bounds anchors at the other three corners + far corner.
        for c in [[10.0, 0.0, 0.0], [0.0, 0.0, 10.0], [10.0, 0.0, 10.0]] {
            splats.push(GaussianSplat::volume(c, [1.0, 1.0, 1.0], q, 255, [0u16; 16]));
        }
        let (w, h) = (64usize, 40usize);
        let buf = node_thumbnail(&PortData::Splats(splats), w, h);

        // Count non-background pixels per quadrant. The -X,-Z cluster maps to the
        // top-left quadrant (x small → tx small, z small → ty small).
        let mut q = [0usize; 4];
        for y in 0..h {
            for x in 0..w {
                if buf[y * w + x] == BG {
                    continue;
                }
                let qi = (if x >= w / 2 { 1 } else { 0 }) + (if y >= h / 2 { 2 } else { 0 });
                q[qi] += 1;
            }
        }
        // Top-left quadrant (index 0) must dominate.
        assert!(
            q[0] > q[1] && q[0] > q[2] && q[0] > q[3],
            "clustered quadrant must have most pixels: {q:?}"
        );
    }

    #[test]
    fn spectral_field_red_dominant_right_bars_taller_and_brighter() {
        // Red dominant = high-index bands (≈620–700nm sit in the upper bands).
        // Ramp 0..1 across the 16 bands so the right bars are tallest/brightest.
        let mut f = [0.0f32; 16];
        for (i, slot) in f.iter_mut().enumerate() {
            *slot = i as f32 / 15.0;
        }
        let (w, h) = (64usize, 40usize);
        let buf = node_thumbnail(&PortData::SpectralField(f), w, h);

        // Compare a left column vs a right column: count filled (non-bg) pixels
        // (bar height) and summed brightness.
        let col_stats = |x: usize| -> (usize, u64) {
            let mut filled = 0usize;
            let mut lum = 0u64;
            for y in 0..h {
                let p = buf[y * w + x];
                if p != BG {
                    filled += 1;
                    lum += p[0] as u64 + p[1] as u64 + p[2] as u64;
                }
            }
            (filled, lum)
        };
        let (lf, ll) = col_stats(2);
        let (rf, rl) = col_stats(w - 3);
        assert!(rf > lf, "right bar must be taller: left_filled={lf} right_filled={rf}");
        assert!(rl > ll, "right bar must be brighter: left_lum={ll} right_lum={rl}");
    }

    #[test]
    fn biome_map_distinct_cells_distinct_colors() {
        // 4x4 map with two different biome bytes → at least 2 distinct colors.
        let cells = vec![0u8, 0, 2, 2, 0, 0, 2, 2, 4, 4, 6, 6, 4, 4, 6, 6];
        let buf = node_thumbnail(&PortData::BiomeMap(cells), 32, 32);
        let mut colors = std::collections::HashSet::new();
        for p in &buf {
            colors.insert((p[0], p[1], p[2]));
        }
        assert!(colors.len() >= 4, "distinct biome bytes must yield distinct colors, got {}", colors.len());
        assert!(non_bg(&buf) > 0);
    }

    #[test]
    fn mesh_silhouette_fills_real_triangle() {
        // A single big triangle covering most of the XZ unit square.
        let m = EditorMesh {
            positions: vec![[0.0, 0.0, 0.0], [1.0, 0.5, 0.0], [0.0, 1.0, 1.0]],
            normals: vec![],
            indices: vec![[0, 1, 2]],
            material_id: 0,
        };
        let (w, h) = (64usize, 40usize);
        let buf = node_thumbnail(&PortData::Mesh(m), w, h);
        let filled = non_bg(&buf);
        // A triangle covering ~half the unit square should fill a meaningful chunk.
        assert!(filled > (w * h) / 8, "mesh silhouette too sparse: {filled} / {}", w * h);
    }

    #[test]
    fn scalar_bar_fill_scales_with_value() {
        let small = node_thumbnail(&PortData::Scalar(0.1), 64, 40);
        let large = node_thumbnail(&PortData::Scalar(100.0), 64, 40);
        assert!(non_bg(&large) > non_bg(&small), "larger scalar must fill more pixels");
        assert!(non_bg(&small) > 0, "even a small scalar must draw something");
    }

    #[test]
    fn scalarvec_sparkline_is_non_empty() {
        let v: Vec<f32> = (0..20).map(|i| (i as f32 * 0.5).sin()).collect();
        let buf = node_thumbnail(&PortData::ScalarVec(v), 64, 40);
        assert!(non_bg(&buf) > 0, "sparkline must draw pixels");
    }

    #[test]
    fn empty_data_is_all_background() {
        let buf = node_thumbnail(&PortData::Splats(vec![]), 64, 40);
        assert_eq!(non_bg(&buf), 0, "empty splats → all background");
        assert_eq!(buf.len(), 64 * 40);
    }
}
