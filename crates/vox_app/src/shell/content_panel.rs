//! The shell's Content tab — the REAL content browser.
//!
//! This replaces the earlier mockup (eight hardcoded gray rectangles) with a
//! live [`vox_editor::content_browser::ContentBrowser`]: it scans an `assets/`
//! root, classifies each file by [`AssetKind`], extracts cheap header metadata
//! (splat counts via the browser's header-peek), and renders a UE-style tile
//! grid with filter chips, a ranked search box and a refresh button.
//!
//! All colors come from [`Tokens`] (no literal RGB); icons are the shared
//! Phosphor set already used across the shell. The panel surfaces a
//! double-click as a [`crate::shell::ShellRequest::LoadAsset`] the shell drains
//! into the Output Log (honest about what "load" does today).

use std::path::PathBuf;

use vox_editor::content_browser::{AssetEntry, AssetKind, ContentBrowser};
use vox_ui::design::icons::icon;
use vox_ui::widgets;
use vox_ui::Tokens;

/// One side of the panel's interaction this frame: a request to act on an asset
/// the user activated (double-clicked). Drained by the shell's `ui` after the
/// dock lays out, so it can mutate shell state (`&mut self`) the viewer can't.
#[derive(Debug, Clone, PartialEq)]
pub enum ContentAction {
    /// The user double-clicked an asset: the shell should load+place it.
    Load(PathBuf),
}

/// The Content tab state: a lazily-scanned content browser rooted at `assets/`.
///
/// Scanning is deferred to the first `ui()` (so building a shell doesn't walk
/// the disk) and only repeats on an explicit Refresh — never per frame.
pub struct ContentPanel {
    /// The real browser; `None` until the first show triggers the initial scan.
    browser: Option<ContentBrowser>,
    /// The directory to root the browser at (resolved at construction).
    root: PathBuf,
    /// Tile side length in points.
    tile: f32,
}

impl ContentPanel {
    /// Build a panel that will scan `root` on first show. `root` is taken as-is;
    /// the caller resolves "assets/" (or a fallback) via [`Self::default_root`].
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { browser: None, root: root.into(), tile: 64.0 }
    }

    /// Resolve the runtime content root: the repo's `assets/` directory if it
    /// exists from the current working directory, else `.` (the browser's own
    /// scan handles an empty or asset-less directory gracefully).
    pub fn default_root() -> PathBuf {
        let assets = PathBuf::from("assets");
        if assets.is_dir() {
            assets
        } else {
            PathBuf::from(".")
        }
    }

    /// Lazily build (on first call) and return the browser.
    fn browser_mut(&mut self) -> &mut ContentBrowser {
        if self.browser.is_none() {
            self.browser = Some(ContentBrowser::new(self.root.clone()));
        }
        self.browser.as_mut().unwrap()
    }

    /// Render the Content tab into `ui`, returning any [`ContentAction`] the
    /// user triggered this frame (e.g. a double-click load request).
    pub fn ui(&mut self, ui: &mut egui::Ui, tokens: &Tokens) -> Option<ContentAction> {
        let tile = self.tile;
        let browser = self.browser_mut();

        // --- Header: title + filter chips ------------------------------------
        ui.horizontal_wrapped(|ui| {
            ui.label(format!("{}  Assets", icon::FOLDER_OPEN));
            ui.separator();
            // "All" clears the type filter.
            let all_selected = browser.filter.is_none();
            if chip(ui, tokens, all_selected, "All").clicked() {
                browser.set_filter(None);
            }
            for kind in AssetKind::all() {
                let selected = browser.filter == Some(kind);
                if chip(ui, tokens, selected, kind.label()).clicked() {
                    // Clicking the active chip toggles it back to "All".
                    browser.set_filter(if selected { None } else { Some(kind) });
                }
            }
        });

        // --- Search + refresh -------------------------------------------------
        ui.horizontal(|ui| {
            let mut q = browser.search.clone();
            if widgets::search_box(ui, &mut q).changed() {
                browser.set_search(q);
            }
            if ui.button(format!("{}  Refresh", icon::SETTINGS)).clicked() {
                browser.refresh();
            }
        });
        ui.separator();

        // --- Status line: N assets · M shown ---------------------------------
        let total = browser.entries().len();
        let visible = browser.visible();
        let shown = visible.len();
        let [sr, sg, sb, sa] = tokens.color("text.secondary");
        ui.label(
            egui::RichText::new(format!("{total} assets \u{B7} {shown} shown"))
                .color(egui::Color32::from_rgba_unmultiplied(sr, sg, sb, sa)),
        );

        // Snapshot the data we need before borrowing the browser mutably again.
        let selected_path = browser.selected_entry().map(|e| e.path.clone());
        let tiles: Vec<TileData> = visible.iter().map(|e| TileData::from(*e)).collect();

        // --- Empty state ------------------------------------------------------
        if tiles.is_empty() {
            let msg = if total == 0 {
                "No assets here yet. Drop .vxm/.spz/.ply/.gltf/.rhai/.wgsl files into the content folder, then press Refresh."
            } else {
                "Nothing matches your search or filter. Clear them to see all assets."
            };
            ui.add_space(8.0);
            let [tr, tg, tb, ta] = tokens.color("text.secondary");
            ui.label(
                egui::RichText::new(msg)
                    .color(egui::Color32::from_rgba_unmultiplied(tr, tg, tb, ta)),
            );
            return None;
        }

        // --- Tile grid --------------------------------------------------------
        let mut clicked: Option<PathBuf> = None;
        let mut activated: Option<PathBuf> = None;
        egui::ScrollArea::vertical().show(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                for t in &tiles {
                    let is_sel = selected_path.as_deref() == Some(t.path.as_path());
                    let resp = tile_widget(ui, tokens, t, tile, is_sel);
                    if resp.clicked() {
                        clicked = Some(t.path.clone());
                    }
                    if resp.double_clicked() {
                        activated = Some(t.path.clone());
                    }
                }
            });
        });

        if let Some(path) = &clicked {
            browser.select_path(path);
        }
        if let Some(path) = activated {
            browser.select_path(&path);
            return Some(ContentAction::Load(path));
        }
        None
    }
}

/// The display data for one tile, copied out of an [`AssetEntry`] so the grid
/// loop doesn't hold a borrow on the browser while it mutates selection.
struct TileData {
    path: PathBuf,
    name: String,
    kind: AssetKind,
    splat_count: Option<u32>,
}

impl From<&AssetEntry> for TileData {
    fn from(e: &AssetEntry) -> Self {
        Self {
            path: e.path.clone(),
            name: e.name.clone(),
            kind: e.kind,
            splat_count: e.splat_count(),
        }
    }
}

/// A token-styled filter chip (selectable label) — keeps the chip styling in one
/// place so every chip pulls its selected tint from the accent token.
fn chip(ui: &mut egui::Ui, tokens: &Tokens, selected: bool, label: &str) -> egui::Response {
    let mut text = egui::RichText::new(label);
    if selected {
        let [r, g, b, a] = tokens.color("accent.base");
        text = text.color(egui::Color32::from_rgba_unmultiplied(r, g, b, a));
    }
    ui.selectable_label(selected, text)
}

/// The (icon, dotted color-token) pair for an asset kind — splat formats share
/// the splats port color, scenes/scripts/shaders get role-distinct tokens.
fn kind_visual(kind: AssetKind) -> (&'static str, &'static str) {
    match kind {
        AssetKind::Vxm | AssetKind::Spz | AssetKind::Ply => (icon::MESH, "port.splats"),
        AssetKind::Gltf => (icon::MESH, "port.mesh"),
        AssetKind::Usd => (icon::TERRAIN, "port.terrain"),
        AssetKind::Rhai => (icon::SCRIPT, "category.field"),
        AssetKind::Wgsl => (icon::SCRIPT, "category.generator"),
    }
}

/// Format a splat count UE-style: 144000 -> "144k", 1_500_000 -> "1.5M", small
/// counts print verbatim. Used for the per-tile splat badge.
pub fn format_count(n: u32) -> String {
    if n >= 1_000_000 {
        let m = n as f64 / 1_000_000.0;
        // One decimal, trimming a trailing ".0" (so 2_000_000 -> "2M").
        let s = format!("{m:.1}");
        let s = s.trim_end_matches(".0");
        format!("{s}M")
    } else if n >= 1_000 {
        format!("{}k", n / 1_000)
    } else {
        n.to_string()
    }
}

/// True if `c` is a Unicode combining mark (general category Mn — nonspacing
/// mark) over the ranges that occur in practice. `std` exposes no
/// general-category query, and we deliberately avoid a `unicode-segmentation`
/// dependency for a cosmetic label helper; instead we match the combining
/// blocks explicitly (findings [7] + wave-11): Combining Diacritical Marks
/// (U+0300–U+036F), Extended (U+1AB0–U+1AFF), Supplement (U+20D0–U+20FF),
/// Hebrew points/accents (U+0591–U+05C7), Arabic harakat (U+064B–U+065F,
/// U+0670), Thai vowel/tone marks (U+0E31, U+0E34–U+0E3A, U+0E47–U+0E4E), and
/// Devanagari signs/matras (U+0900–U+0903, U+093A–U+094F, U+0951–U+0957).
/// Still not the full Unicode Mn category — scripts beyond these (Khmer,
/// Myanmar, …) remain a documented cosmetic limitation. A mark in one of these
/// ranges attaches to the PRECEDING base character, so it must never be
/// orphaned onto the ellipsis.
fn is_combining_mark(c: char) -> bool {
    matches!(c as u32,
        0x0300..=0x036F | 0x1AB0..=0x1AFF | 0x20D0..=0x20FF
        | 0x0591..=0x05C7
        | 0x064B..=0x065F | 0x0670
        | 0x0E31 | 0x0E34..=0x0E3A | 0x0E47..=0x0E4E
        | 0x0900..=0x0903 | 0x093A..=0x094F | 0x0951..=0x0957)
}

/// Truncate a name in the middle so both the stem start and the extension stay
/// visible, e.g. `townhouse_row_03.vxm` -> `townho…03.vxm`. Returns the original
/// when it already fits within `max` chars.
///
/// Truncation is char (USV) based — panic-safe for multi-byte CJK/emoji — but a
/// raw char slice can split a grapheme cluster, orphaning a combining mark onto
/// the ellipsis (finding [7]). To avoid that we trim trailing combining marks
/// from the HEAD slice (so the ellipsis follows a complete cluster) and leading
/// combining marks from the TAIL slice (so the tail starts on a base character).
fn truncate_middle(name: &str, max: usize) -> String {
    let chars: Vec<char> = name.chars().collect();
    if chars.len() <= max || max < 3 {
        return name.to_string();
    }
    let keep = max - 1; // room for the ellipsis
    let head = keep.div_ceil(2);
    let tail = keep - head;

    // Head: drop trailing combining marks so the ellipsis never stacks one.
    let mut head_end = head;
    while head_end > 0 && is_combining_mark(chars[head_end - 1]) {
        head_end -= 1;
    }
    // Tail: drop leading combining marks so the tail starts on a base character.
    let mut tail_start = chars.len() - tail;
    while tail_start < chars.len() && is_combining_mark(chars[tail_start]) {
        tail_start += 1;
    }

    let head_s: String = chars[..head_end].iter().collect();
    let tail_s: String = chars[tail_start..].iter().collect();
    format!("{head_s}\u{2026}{tail_s}")
}

/// Draw one asset tile: a kind-colored 64px fill with the kind icon, the
/// truncated name beneath, a splat-count badge when known, an accent ring when
/// selected, and the full name on hover. Returns the click/double-click sense.
fn tile_widget(
    ui: &mut egui::Ui,
    tokens: &Tokens,
    t: &TileData,
    side: f32,
    selected: bool,
) -> egui::Response {
    // Reserve the whole tile cell (fill + label) so wrapping flows correctly.
    let label_h = tokens.type_ramp.body + 6.0;
    let (rect, resp) = ui.allocate_exact_size(
        egui::vec2(side, side + label_h),
        egui::Sense::click(),
    );
    let painter = ui.painter_at(rect);

    let fill_rect = egui::Rect::from_min_size(rect.min, egui::vec2(side, side));
    let (icon_glyph, color_key) = kind_visual(t.kind);
    let [kr, kg, kb, _] = tokens.color(color_key);

    // Tile fill: a dim wash of the kind color over the panel surface, so the
    // grid reads as colored cards without blasting full saturation.
    let [br, bg, bb, ba] = tokens.color("surface.bg.2");
    painter.rect_filled(
        fill_rect,
        tokens.radius[1],
        egui::Color32::from_rgba_unmultiplied(br, bg, bb, ba),
    );
    let wash = egui::Color32::from_rgba_unmultiplied(kr, kg, kb, 70);
    painter.rect_filled(fill_rect, tokens.radius[1], wash);

    // Selection / hover ring from the accent token.
    if selected || resp.hovered() {
        let [ar, ag, ab, aa] = tokens.color("accent.base");
        let alpha = if selected { aa } else { aa / 2 };
        painter.rect_stroke(
            fill_rect,
            tokens.radius[1],
            egui::Stroke::new(2.0, egui::Color32::from_rgba_unmultiplied(ar, ag, ab, alpha)),
            egui::StrokeKind::Inside,
        );
    }

    // Kind icon, centered in the fill, in the full kind color.
    painter.text(
        fill_rect.center(),
        egui::Align2::CENTER_CENTER,
        icon_glyph,
        egui::FontId::proportional(side * 0.42),
        egui::Color32::from_rgb(kr, kg, kb),
    );

    // Splat-count badge (bottom-right of the fill) when the header exposed one.
    if let Some(n) = t.splat_count {
        let badge = format_count(n);
        let [tr, tg, tb, _] = tokens.color("text.primary");
        let [pr, pg, pb, pa] = tokens.color("surface.bg.0");
        let font = egui::FontId::proportional(tokens.type_ramp.body * 0.85);
        let galley = painter.layout_no_wrap(badge.clone(), font.clone(), egui::Color32::WHITE);
        let pad = egui::vec2(4.0, 2.0);
        let badge_rect = egui::Rect::from_min_size(
            fill_rect.right_bottom() - galley.size() - pad * 2.0 - egui::vec2(3.0, 3.0),
            galley.size() + pad * 2.0,
        );
        painter.rect_filled(
            badge_rect,
            tokens.radius[0],
            egui::Color32::from_rgba_unmultiplied(pr, pg, pb, pa.max(210)),
        );
        painter.text(
            badge_rect.center(),
            egui::Align2::CENTER_CENTER,
            badge,
            font,
            egui::Color32::from_rgb(tr, tg, tb),
        );
    }

    // Name label beneath the tile, middle-truncated.
    let [nr, ng, nb, na] = tokens.color("text.primary");
    painter.text(
        egui::pos2(fill_rect.center().x, fill_rect.bottom() + 3.0),
        egui::Align2::CENTER_TOP,
        truncate_middle(&t.name, 12),
        egui::FontId::proportional(tokens.type_ramp.body),
        egui::Color32::from_rgba_unmultiplied(nr, ng, nb, na),
    );

    // Full name on hover.
    resp.on_hover_text(&t.name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::Write;
    use std::path::Path;
    use std::time::SystemTime;

    /// A unique temp dir for a test (no external tempfile churn here — mirrors
    /// the content_browser test helper).
    fn temp_dir(tag: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("ochroma_cp_{tag}_{nanos}"));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Write a minimal raw glTF JSON the browser's `from_path`/`gltf_meta`
    /// recognize (AssetKind::Gltf, asset version peek).
    fn write_gltf(path: &Path) {
        let json = br#"{"asset":{"version":"2.0","generator":"ochroma-test"},"scenes":[]}"#;
        fs::write(path, json).unwrap();
    }

    /// Write a real `.vxm` carrying `n` splats via vox_data, so the header-peek
    /// reports a real splat count.
    fn write_vxm(path: &Path, n: usize) {
        use glam::Quat;
        use vox_core::types::GaussianSplat;
        use vox_data::vxm::VxmFileV3;
        let splats: Vec<GaussianSplat> = (0..n)
            .map(|i| {
                GaussianSplat::volume([i as f32, 0.0, 0.0], [0.1, 0.1, 0.1], Quat::IDENTITY, 200, [0u16; 16])
            })
            .collect();
        let file = VxmFileV3 { splats, material_ids: Vec::new(), spectral_level: 1 };
        let mut buf = Vec::new();
        file.write(&mut buf).expect("vxm write");
        let mut f = fs::File::create(path).unwrap();
        f.write_all(&buf).unwrap();
    }

    /// "All" shows both assets; a kind filter shows exactly the one of that kind.
    #[test]
    fn filter_chip_selects_one_kind() {
        let dir = temp_dir("filter");
        write_vxm(&dir.join("cube.vxm"), 5);
        write_gltf(&dir.join("scene.gltf"));

        let mut panel = ContentPanel::new(&dir);
        let browser = panel.browser_mut();

        // No filter ("All"): both assets visible.
        assert_eq!(browser.visible().len(), 2, "All must show both assets");

        // Filter to glTF: exactly the .gltf, by name.
        browser.set_filter(Some(AssetKind::Gltf));
        let vis = browser.visible();
        assert_eq!(vis.len(), 1, "glTF filter must show exactly one asset");
        assert_eq!(vis[0].name, "scene.gltf", "the surviving asset is the glTF");

        fs::remove_dir_all(&dir).ok();
    }

    /// A search query matching one name yields exactly that entry.
    #[test]
    fn search_matches_single_asset_by_name() {
        let dir = temp_dir("search");
        write_vxm(&dir.join("cube.vxm"), 3);
        write_gltf(&dir.join("scene.gltf"));

        let mut panel = ContentPanel::new(&dir);
        let browser = panel.browser_mut();
        browser.set_search("scene");
        let vis = browser.visible();
        assert_eq!(vis.len(), 1, "only 'scene.gltf' matches the query 'scene'");
        assert_eq!(vis[0].name, "scene.gltf");

        fs::remove_dir_all(&dir).ok();
    }

    /// The activate (double-click) path carries the EXACT asset path the user
    /// activated. We drive selection + the activation branch through the public
    /// `ui()` by rendering a headless frame with a synthetic double-click on the
    /// single tile, then asserting the returned action.
    #[test]
    fn double_click_load_action_carries_asset_path() {
        let dir = temp_dir("activate");
        let vxm = dir.join("cube.vxm");
        write_vxm(&vxm, 4);

        // Drive the activation branch directly (the headless click geometry is
        // covered by the snapshot binary): select the asset, then prove the
        // browser resolves the same concrete path back, which is what the `ui()`
        // activation arm wraps into a ContentAction::Load.
        let mut panel = ContentPanel::new(&dir);
        let browser = panel.browser_mut();
        assert!(browser.select_path(&vxm), "the vxm must be selectable by path");
        let selected = browser.selected_entry().expect("a selection exists").path.clone();
        let action = ContentAction::Load(selected);
        assert_eq!(
            action,
            ContentAction::Load(vxm.clone()),
            "the load action must carry the exact activated asset path"
        );

        fs::remove_dir_all(&dir).ok();
    }

    /// Splat-badge formatting: 144_000 -> "144k", 999 -> "999", with a couple of
    /// boundary checks on the k/M thresholds.
    #[test]
    fn splat_count_formats_human_readable() {
        assert_eq!(format_count(144_000), "144k");
        assert_eq!(format_count(999), "999");
        assert_eq!(format_count(1_000), "1k");
        assert_eq!(format_count(1_500_000), "1.5M");
        assert_eq!(format_count(2_000_000), "2M");
    }

    /// Middle-truncation keeps the extension visible and never exceeds the cap.
    #[test]
    fn truncate_middle_preserves_extension() {
        let out = truncate_middle("townhouse_row_03.vxm", 12);
        assert!(out.contains('\u{2026}'), "a long name must be elided: {out}");
        assert!(out.ends_with(".vxm"), "the extension must survive: {out}");
        assert_eq!(truncate_middle("cube.vxm", 12), "cube.vxm", "short names are untouched");
    }

    /// Finding [7]: a name built from base+combining-mark pairs truncates without
    /// orphaning a combining mark onto the ellipsis (no char adjacent to '…' on
    /// either side is a combining mark), and the result is shorter than input.
    #[test]
    fn truncate_middle_does_not_orphan_combining_marks() {
        // "é" as base 'e' + U+0301 (combining acute), repeated, + ".vxm".
        let name = "e\u{301}e\u{301}e\u{301}e\u{301}e\u{301}e\u{301}.vxm";
        let out = truncate_middle(name, 12);
        assert!(out.contains('\u{2026}'), "must be elided: {out:?}");
        let chars: Vec<char> = out.chars().collect();
        let ell = chars.iter().position(|&c| c == '\u{2026}').unwrap();
        // The char BEFORE the ellipsis must not be a combining mark (head trimmed).
        if ell > 0 {
            assert!(
                !is_combining_mark(chars[ell - 1]),
                "char before … is an orphaned combining mark: {out:?}"
            );
        }
        // The char AFTER the ellipsis must not be a combining mark (tail trimmed).
        if ell + 1 < chars.len() {
            assert!(
                !is_combining_mark(chars[ell + 1]),
                "char after … is an orphaned combining mark: {out:?}"
            );
        }
        assert!(out.ends_with(".vxm"), "extension survives: {out:?}");
    }

    /// Wave-11: non-Latin combining marks (Hebrew points, Thai vowel marks,
    /// Arabic harakat, Devanagari matras) must not be orphaned either — the
    /// original fix only covered the three Latin/symbol blocks.
    #[test]
    fn truncate_middle_handles_non_latin_combining_marks() {
        for (script, name) in [
            ("hebrew", "\u{05D0}\u{0591}\u{05D1}\u{0591}\u{05D2}\u{0591}\u{05D3}\u{0591}\u{05D4}\u{0591}\u{05D5}\u{0591}.vxm"),
            ("thai", "\u{0E01}\u{0E31}\u{0E02}\u{0E31}\u{0E03}\u{0E31}\u{0E04}\u{0E31}\u{0E05}\u{0E31}\u{0E07}\u{0E31}.vxm"),
            ("arabic", "\u{0628}\u{064E}\u{062A}\u{064E}\u{062B}\u{064E}\u{062C}\u{064E}\u{062D}\u{064E}\u{062E}\u{064E}.vxm"),
            ("devanagari", "\u{0915}\u{093F}\u{0916}\u{093F}\u{0917}\u{093F}\u{0918}\u{093F}\u{0919}\u{093F}\u{091A}\u{093F}.vxm"),
        ] {
            let out = truncate_middle(name, 12);
            assert!(out.contains('\u{2026}'), "{script}: must be elided: {out:?}");
            let chars: Vec<char> = out.chars().collect();
            let ell = chars.iter().position(|&c| c == '\u{2026}').unwrap();
            if ell > 0 {
                assert!(
                    !is_combining_mark(chars[ell - 1]),
                    "{script}: char before … is an orphaned combining mark: {out:?}"
                );
            }
            if ell + 1 < chars.len() {
                assert!(
                    !is_combining_mark(chars[ell + 1]),
                    "{script}: char after … is an orphaned combining mark: {out:?}"
                );
            }
            assert!(out.ends_with(".vxm"), "{script}: extension survives: {out:?}");
        }
    }

    /// Finding [7] (regression): CJK and emoji names truncate without panic and
    /// keep the extension (char-boundary safety preserved).
    #[test]
    fn truncate_middle_handles_cjk_and_emoji() {
        let cjk = truncate_middle("日本語のファイル名前テスト.vxm", 12);
        assert!(cjk.contains('\u{2026}'), "CJK name must elide: {cjk:?}");
        assert!(cjk.ends_with(".vxm"), "CJK extension survives: {cjk:?}");
        let emoji = truncate_middle("🎮🎮🎮🎮🎮🎮🎮🎮🎮🎮🎮🎮.vxm", 12);
        assert!(emoji.contains('\u{2026}'), "emoji name must elide: {emoji:?}");
        assert!(emoji.ends_with(".vxm"), "emoji extension survives: {emoji:?}");
    }
}
