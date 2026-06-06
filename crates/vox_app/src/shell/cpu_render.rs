//! A self-contained CPU rasterizer for egui `FullOutput`.
//!
//! The headless `shell_snapshot` bin (and the AA/no-bitmap tests) need to turn a
//! real egui frame into RGBA pixels WITHOUT a GPU or a window. egui's normal
//! display path is egui-wgpu; here we instead tessellate to triangle meshes and
//! rasterize them ourselves with barycentric UV interpolation into the font /
//! color texture atlases. This is the same path the design points at ("rasterize
//! the egui paint mesh yourself"). Proportional glyphs come straight from egui's
//! coverage-AA font atlas, so the output is anti-aliased vector text — never the
//! 5x7 software bitmap font that the old engine_runner editor face used.

use egui::{Context, RawInput, TextureId};
use std::collections::HashMap;

/// A CPU texture atlas (premultiplied sRGBA, row-major).
struct Texture {
    w: usize,
    h: usize,
    px: Vec<[u8; 4]>,
}

/// Render an egui UI closure headlessly to an RGBA8 buffer.
///
/// `size_px` is `[w, h]` in physical pixels at `pixels_per_point = 1.0`.
/// `bg` is the clear color. Returns row-major RGBA (4 bytes/pixel).
pub fn render_ui(
    ctx: &Context,
    size_px: [usize; 2],
    bg: [u8; 4],
    mut run_ui: impl FnMut(&Context),
) -> Vec<u8> {
    let [w, h] = size_px;
    let raw = RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(w as f32, h as f32),
        )),
        ..Default::default()
    };
    // Run twice: first frame builds the font atlas + lays out; the second frame
    // paints with everything resolved (animations settled enough for a still).
    // Texture deltas are ACCUMULATED across both frames — the font atlas is
    // typically uploaded on frame 1 and absent from frame 2's delta set.
    let mut textures: HashMap<TextureId, Texture> = HashMap::new();
    let first = ctx.run(raw.clone(), &mut run_ui);
    apply_texture_deltas(&mut textures, &first.textures_delta);
    let output = ctx.run(raw, &mut run_ui);
    apply_texture_deltas(&mut textures, &output.textures_delta);

    let primitives = ctx.tessellate(output.shapes, 1.0);

    let mut fb = vec![bg; w * h];
    for prim in &primitives {
        if let egui::epaint::Primitive::Mesh(mesh) = &prim.primitive {
            let clip = prim.clip_rect;
            let tex = textures.get(&mesh.texture_id);
            draw_mesh(&mut fb, w, h, mesh, tex, clip);
        }
        // Callback primitives (none in this headless path) are skipped.
    }

    let mut out = Vec::with_capacity(w * h * 4);
    for p in fb {
        out.extend_from_slice(&p);
    }
    out
}

/// Like [`render_ui`] but the caller supplies the [`RawInput`] for the FINAL
/// painted frame (e.g. to position the pointer for a hover test). A warm-up frame
/// with the same input runs first to settle layout / build the font atlas, then a
/// configurable number of additional frames advance egui animations (each frame
/// carries `predicted_dt` so `animation_time`-driven transitions progress). The
/// last frame is the one tessellated and rasterised.
pub fn render_ui_with_input(
    ctx: &Context,
    size_px: [usize; 2],
    bg: [u8; 4],
    raw: RawInput,
    extra_anim_frames: u32,
    mut run_ui: impl FnMut(&Context),
) -> Vec<u8> {
    let [w, h] = size_px;
    let mut textures: HashMap<TextureId, Texture> = HashMap::new();

    let mk_raw = || RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(w as f32, h as f32),
        )),
        predicted_dt: 1.0 / 60.0,
        ..raw.clone()
    };

    // Warm-up frame (font atlas + layout).
    let first = ctx.run(mk_raw(), &mut run_ui);
    apply_texture_deltas(&mut textures, &first.textures_delta);
    // Advance animations.
    for _ in 0..extra_anim_frames {
        let out = ctx.run(mk_raw(), &mut run_ui);
        apply_texture_deltas(&mut textures, &out.textures_delta);
    }
    // Final painted frame.
    let output = ctx.run(mk_raw(), &mut run_ui);
    apply_texture_deltas(&mut textures, &output.textures_delta);

    let primitives = ctx.tessellate(output.shapes, 1.0);
    let mut fb = vec![bg; w * h];
    for prim in &primitives {
        if let egui::epaint::Primitive::Mesh(mesh) = &prim.primitive {
            draw_mesh(&mut fb, w, h, mesh, textures.get(&mesh.texture_id), prim.clip_rect);
        }
    }
    let mut out = Vec::with_capacity(w * h * 4);
    for p in fb {
        out.extend_from_slice(&p);
    }
    out
}

fn apply_texture_deltas(
    textures: &mut HashMap<TextureId, Texture>,
    delta: &egui::epaint::textures::TexturesDelta,
) {
    for (id, image_delta) in &delta.set {
        let [iw, ih] = image_delta.image.size();
        let new_px: Vec<[u8; 4]> = match &image_delta.image {
            egui::epaint::ImageData::Color(c) => {
                c.pixels.iter().map(|p| [p.r(), p.g(), p.b(), p.a()]).collect()
            }
            egui::epaint::ImageData::Font(f) => {
                // Coverage -> premultiplied white sRGBA via egui's own gamma path.
                f.srgba_pixels(None)
                    .map(|p| [p.r(), p.g(), p.b(), p.a()])
                    .collect()
            }
        };
        match image_delta.pos {
            None => {
                textures.insert(*id, Texture { w: iw, h: ih, px: new_px });
            }
            Some([px, py]) => {
                if let Some(t) = textures.get_mut(id) {
                    for y in 0..ih {
                        for x in 0..iw {
                            let dx = px + x;
                            let dy = py + y;
                            if dx < t.w && dy < t.h {
                                t.px[dy * t.w + dx] = new_px[y * iw + x];
                            }
                        }
                    }
                }
            }
        }
    }
    for id in &delta.free {
        textures.remove(id);
    }
}

#[allow(clippy::too_many_arguments)]
fn draw_mesh(
    fb: &mut [[u8; 4]],
    w: usize,
    h: usize,
    mesh: &egui::epaint::Mesh,
    tex: Option<&Texture>,
    clip: egui::Rect,
) {
    let v = &mesh.vertices;
    for tri in mesh.indices.chunks_exact(3) {
        let a = &v[tri[0] as usize];
        let b = &v[tri[1] as usize];
        let c = &v[tri[2] as usize];
        raster_tri(fb, w, h, a, b, c, tex, clip);
    }
}

#[allow(clippy::too_many_arguments)]
fn raster_tri(
    fb: &mut [[u8; 4]],
    w: usize,
    h: usize,
    a: &egui::epaint::Vertex,
    b: &egui::epaint::Vertex,
    c: &egui::epaint::Vertex,
    tex: Option<&Texture>,
    clip: egui::Rect,
) {
    let (ax, ay) = (a.pos.x, a.pos.y);
    let (bx, by) = (b.pos.x, b.pos.y);
    let (cx, cy) = (c.pos.x, c.pos.y);

    let min_x = ax.min(bx).min(cx).max(clip.min.x).max(0.0).floor() as i32;
    let max_x = ax.max(bx).max(cx).min(clip.max.x).min(w as f32).ceil() as i32;
    let min_y = ay.min(by).min(cy).max(clip.min.y).max(0.0).floor() as i32;
    let max_y = ay.max(by).max(cy).min(clip.max.y).min(h as f32).ceil() as i32;
    if max_x <= min_x || max_y <= min_y {
        return;
    }

    let area = edge(ax, ay, bx, by, cx, cy);
    if area.abs() < 1e-6 {
        return;
    }
    let inv_area = 1.0 / area;

    for py in min_y..max_y {
        for px in min_x..max_x {
            let fx = px as f32 + 0.5;
            let fy = py as f32 + 0.5;
            // Barycentric weights (winding-agnostic — egui mixes both).
            let w0 = edge(bx, by, cx, cy, fx, fy) * inv_area;
            let w1 = edge(cx, cy, ax, ay, fx, fy) * inv_area;
            let w2 = edge(ax, ay, bx, by, fx, fy) * inv_area;
            if (w0 < 0.0 || w1 < 0.0 || w2 < 0.0) && (w0 > 0.0 || w1 > 0.0 || w2 > 0.0) {
                continue; // outside triangle
            }

            // Interpolate vertex color.
            let vc = [
                lerp3(a.color.r(), b.color.r(), c.color.r(), w0, w1, w2),
                lerp3(a.color.g(), b.color.g(), c.color.g(), w0, w1, w2),
                lerp3(a.color.b(), b.color.b(), c.color.b(), w0, w1, w2),
                lerp3(a.color.a(), b.color.a(), c.color.a(), w0, w1, w2),
            ];

            // Sample texture (font atlas / color); the default white pixel of
            // egui's atlas makes solid fills work too.
            let src = if let Some(t) = tex {
                let u = a.uv.x * w0 + b.uv.x * w1 + c.uv.x * w2;
                let vv = a.uv.y * w0 + b.uv.y * w1 + c.uv.y * w2;
                let tx = ((u * t.w as f32) as i32).clamp(0, t.w as i32 - 1) as usize;
                let ty = ((vv * t.h as f32) as i32).clamp(0, t.h as i32 - 1) as usize;
                let tp = t.px[ty * t.w + tx];
                // egui vertex color modulates the (premultiplied) texel.
                [
                    mul8(vc[0], tp[0]),
                    mul8(vc[1], tp[1]),
                    mul8(vc[2], tp[2]),
                    mul8(vc[3], tp[3]),
                ]
            } else {
                vc
            };

            let idx = py as usize * w + px as usize;
            fb[idx] = over(src, fb[idx]);
        }
    }
}

#[inline]
fn edge(ax: f32, ay: f32, bx: f32, by: f32, px: f32, py: f32) -> f32 {
    (bx - ax) * (py - ay) - (by - ay) * (px - ax)
}

#[inline]
fn lerp3(a: u8, b: u8, c: u8, w0: f32, w1: f32, w2: f32) -> u8 {
    ((a as f32 * w0 + b as f32 * w1 + c as f32 * w2).round()).clamp(0.0, 255.0) as u8
}

#[inline]
fn mul8(a: u8, b: u8) -> u8 {
    ((a as u32 * b as u32 + 127) / 255) as u8
}

/// Source-over compositing of a premultiplied-alpha `src` onto opaque `dst`.
#[inline]
fn over(src: [u8; 4], dst: [u8; 4]) -> [u8; 4] {
    let sa = src[3] as u32;
    let inv = 255 - sa;
    [
        (src[0] as u32 + dst[0] as u32 * inv / 255) as u8,
        (src[1] as u32 + dst[1] as u32 * inv / 255) as u8,
        (src[2] as u32 + dst[2] as u32 * inv / 255) as u8,
        255,
    ]
}

/// Write an RGBA8 buffer as a PNG using only `flate2`'s zlib (no image crate).
/// ~20-line zlib-only writer per the design.
pub fn write_png(path: &str, rgba: &[u8], w: u32, h: u32) -> std::io::Result<()> {
    use std::io::Write;
    fn crc32(bytes: &[u8]) -> u32 {
        let mut crc = 0xFFFF_FFFFu32;
        for &b in bytes {
            crc ^= b as u32;
            for _ in 0..8 {
                let mask = (crc & 1).wrapping_neg();
                crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
            }
        }
        !crc
    }
    fn chunk(out: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
        out.extend_from_slice(&(data.len() as u32).to_be_bytes());
        out.extend_from_slice(kind);
        out.extend_from_slice(data);
        let mut crc_in = Vec::with_capacity(4 + data.len());
        crc_in.extend_from_slice(kind);
        crc_in.extend_from_slice(data);
        out.extend_from_slice(&crc32(&crc_in).to_be_bytes());
    }

    // Filter type 0 per scanline.
    let mut raw = Vec::with_capacity((w * h * 4 + h) as usize);
    let stride = (w * 4) as usize;
    for y in 0..h as usize {
        raw.push(0u8);
        raw.extend_from_slice(&rgba[y * stride..(y + 1) * stride]);
    }
    let compressed = deflate_zlib(&raw);

    let mut png = Vec::new();
    png.extend_from_slice(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]);
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&w.to_be_bytes());
    ihdr.extend_from_slice(&h.to_be_bytes());
    ihdr.extend_from_slice(&[8, 6, 0, 0, 0]); // 8-bit, RGBA
    chunk(&mut png, b"IHDR", &ihdr);
    chunk(&mut png, b"IDAT", &compressed);
    chunk(&mut png, b"IEND", &[]);

    let mut f = std::fs::File::create(path)?;
    f.write_all(&png)?;
    Ok(())
}

/// zlib-wrapped stored (uncompressed) DEFLATE stream. Avoids pulling a deflate
/// dependency: PNG decoders accept stored blocks. Adler-32 trailer included.
fn deflate_zlib(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    out.push(0x78); // CMF
    out.push(0x01); // FLG (no dict, fastest)
    // Stored blocks, max 65535 bytes each.
    let mut i = 0;
    while i < data.len() {
        let block = (data.len() - i).min(0xFFFF);
        let is_last = i + block >= data.len();
        out.push(if is_last { 1 } else { 0 });
        out.extend_from_slice(&(block as u16).to_le_bytes());
        out.extend_from_slice(&(!(block as u16)).to_le_bytes());
        out.extend_from_slice(&data[i..i + block]);
        i += block;
    }
    // Adler-32 of the uncompressed data.
    let (mut a, mut b) = (1u32, 0u32);
    for &byte in data {
        a = (a + byte as u32) % 65521;
        b = (b + a) % 65521;
    }
    out.extend_from_slice(&((b << 16) | a).to_be_bytes());
    out
}

/// Convenience for tests: render `run_ui` and return the RGBA buffer plus dims.
pub fn render_default(
    size_px: [usize; 2],
    tokens: &vox_ui::Tokens,
    run_ui: impl FnMut(&Context),
) -> Vec<u8> {
    let ctx = Context::default();
    vox_ui::design::icons::install(&ctx);
    vox_ui::egui_theme::apply(&ctx, tokens);
    let bg = tokens.color("surface.bg.0");
    render_ui(&ctx, size_px, bg, run_ui)
}

/// Count pixels whose RGB differs from `bg` by more than `thresh` in any
/// channel (the snapshot's "non-background pixels" metric).
pub fn non_background_fraction(rgba: &[u8], bg: [u8; 4], thresh: u8) -> f32 {
    let n = rgba.len() / 4;
    if n == 0 {
        return 0.0;
    }
    let mut count = 0usize;
    for px in rgba.chunks_exact(4) {
        let d = (0..3)
            .map(|i| (px[i] as i32 - bg[i] as i32).unsigned_abs())
            .max()
            .unwrap_or(0);
        if d > thresh as u32 {
            count += 1;
        }
    }
    count as f32 / n as f32
}

/// The old 5x7 software glyph bitmaps emit pixels that are either fully the
/// text color or fully background — never intermediate coverage. Anti-aliased
/// vector text produces a continuum of grayscale luminances along glyph edges.
/// Returns the number of DISTINCT luminance levels found inside `rect`.
pub fn distinct_luminance_levels(
    rgba: &[u8],
    w: usize,
    rect: (usize, usize, usize, usize),
) -> usize {
    let (x0, y0, x1, y1) = rect;
    let mut seen = [false; 256];
    for y in y0..y1 {
        for x in x0..x1 {
            let idx = (y * w + x) * 4;
            if idx + 2 >= rgba.len() {
                continue;
            }
            let lum = (rgba[idx] as u32 * 30 + rgba[idx + 1] as u32 * 59 + rgba[idx + 2] as u32 * 11)
                / 100;
            seen[lum as usize] = true;
        }
    }
    seen.iter().filter(|s| **s).count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use vox_ui::Tokens;

    #[test]
    fn solid_fill_renders_token_color() {
        // A CentralPanel filled with accent.base must produce that color in the
        // CPU buffer — proves the mesh rasterizer + texture atlas path works.
        let tokens = Tokens::default();
        let accent = tokens.color("accent.base");
        let rgba = render_default([64, 64], &tokens, |ctx| {
            egui::CentralPanel::default()
                .frame(egui::Frame::NONE.fill(egui::Color32::from_rgb(
                    accent[0], accent[1], accent[2],
                )))
                .show(ctx, |ui| {
                    ui.allocate_space(ui.available_size());
                });
        });
        let center = ((32 * 64) + 32) * 4;
        let got = [rgba[center], rgba[center + 1], rgba[center + 2]];
        assert!(
            (got[0] as i32 - accent[0] as i32).abs() <= 4
                && (got[1] as i32 - accent[1] as i32).abs() <= 4
                && (got[2] as i32 - accent[2] as i32).abs() <= 4,
            "center pixel {got:?} != accent {accent:?}"
        );
    }

    #[test]
    fn text_is_antialiased_not_bitmap() {
        // The design's test 3: render the type ramp (title 20 / body 13 /
        // caption 11) and assert >16 distinct grayscale luminance levels — a
        // continuum only AA vector glyphs produce (the 5x7 path is binary).
        let tokens = Tokens::default();
        let bg = tokens.color("surface.bg.0");
        let w = 400usize;
        let h = 200usize;
        let rgba = render_default([w, h], &tokens, |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                ui.label(egui::RichText::new("Ochroma").size(20.0));
                ui.label(egui::RichText::new("Properties").size(13.0));
                ui.label(egui::RichText::new("caption text").size(11.0));
            });
        });
        let levels = distinct_luminance_levels(&rgba, w, (0, 0, w, h));
        assert!(
            levels > 16,
            "only {levels} distinct luminance levels — text looks like a bitmap font, not AA"
        );
        let _ = bg;
    }

    #[test]
    fn png_roundtrips_size() {
        let tokens = Tokens::default();
        let rgba = render_default([32, 16], &tokens, |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| ui.label("hi"));
        });
        let path = std::env::temp_dir().join("ochroma_cpu_render_test.png");
        write_png(path.to_str().unwrap(), &rgba, 32, 16).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(&bytes[0..8], &[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]);
        assert!(bytes.len() > 50, "png suspiciously small: {} bytes", bytes.len());
    }
}
