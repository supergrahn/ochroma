//! UiTree — the single retained, styleable, hot-reloadable game-UI tree.
//!
//! This is Ochroma's convergence target for game UI: one retained hierarchy
//! ([`UiNode`]) styled through named classes ([`StyleSheet`], the USS analogue),
//! authored as a declarative document ([`UiDoc`], the UXML analogue), laid out
//! by a flexbox-lite solver, hit-tested for input, and software-rasterised so it
//! is headless-provable. It works WITHOUT the `game-ui` feature (pure CPU); when
//! `game-ui` is on, [`taffy_parity`] cross-checks the plain-math layout against
//! taffy.
//!
//! Design (Unity UI Toolkit shape):
//! - **Structure** is [`UiNode`] (`id` / `kind` / `style` / `children`).
//! - **Styling** is separated from structure: a node carries inline [`Style`]
//!   overrides plus a list of `classes`; a [`StyleSheet`] maps class names to
//!   [`Style`] blocks. Resolution is sheet-classes (in order) then inline.
//! - **Documents** ([`UiDoc`]) are serde TOML/JSON; [`UiTree::reload`] diffs by
//!   `id` and preserves runtime state (slider values, focus) for unchanged ids.
//! - **Rendering** mirrors the established software-rasterise idiom
//!   (`VelloCtxCpu::rasterize_into`, `vox_core::game_ui::burn_text`).
//! - **Events** mirror the codebase's drainable widget-event idiom.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use vox_core::game_ui::{CHAR_H, CHAR_STRIDE};

use crate::text;

/// Upper bound on a resolved `font_scale`. `font_scale` is author-controlled
/// (an unbounded `u32` straight out of `UiDoc` JSON); without a ceiling, a
/// hostile/typo'd value multiplied by a long label's character count overflows
/// `u32` text-width math (debug panic / release garbage). 64x the 5x7 base font
/// is already absurdly large for any real UI, so we clamp there at resolution.
pub const MAX_FONT_SCALE: u32 = 64;

// ---------------------------------------------------------------------------
// Style — the resolvable visual properties (the USS-property analogue).
// ---------------------------------------------------------------------------

/// One edge inset (left, top, right, bottom) in layout pixels.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub struct Edges {
    pub left: f32,
    pub top: f32,
    pub right: f32,
    pub bottom: f32,
}

impl Edges {
    pub const fn all(v: f32) -> Self {
        Self { left: v, top: v, right: v, bottom: v }
    }
}

/// Anchor of a node inside its parent's content box (for absolutely-placed
/// nodes and for the root inside the viewport).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum Anchor {
    #[default]
    TopLeft,
    TopCenter,
    TopRight,
    CenterLeft,
    Center,
    CenterRight,
    BottomLeft,
    BottomCenter,
    BottomRight,
}

/// How a container arranges its flex children.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum FlexDir {
    #[default]
    Column,
    Row,
}

/// A patch of style properties. Every field is optional so that class blocks
/// and inline overrides compose by "last writer wins" (the USS cascade).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub struct Style {
    /// Fixed width in px; `None` means "driven by flex grow / parent".
    pub width: Option<f32>,
    /// Fixed height in px; `None` means "driven by flex grow / parent".
    pub height: Option<f32>,
    pub margin: Option<Edges>,
    pub padding: Option<Edges>,
    /// Flex grow weight inside a flex container (0 = don't grow).
    pub grow: Option<f32>,
    pub flex_dir: Option<FlexDir>,
    pub anchor: Option<Anchor>,
    /// Background / fill colour, straight-alpha RGBA in `0..=1`.
    pub color: Option<[f32; 4]>,
    /// Progress/slider FILL colour, straight-alpha RGBA in `0..=1`. When set,
    /// `ProgressBar`/`Slider` paint their value portion with this exact colour;
    /// when `None` the fill is derived by brightening `color` (the old default).
    /// Lets a dark track carry a distinct bright fill (e.g. dark + amber).
    pub fill_color: Option<[f32; 4]>,
    /// Text colour, RGB `0..=255` (matches `burn_text`).
    pub text_color: Option<[u8; 3]>,
    /// Text scale multiplier over the 5x7 base font (1 = native).
    pub font_scale: Option<u32>,
}

/// A fully-resolved style — no `None`s, ready for layout and rendering.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResolvedStyle {
    pub width: Option<f32>,
    pub height: Option<f32>,
    pub margin: Edges,
    pub padding: Edges,
    pub grow: f32,
    pub flex_dir: FlexDir,
    pub anchor: Anchor,
    pub color: [f32; 4],
    /// `None` means "derive the fill from `color`"; `Some` is an explicit fill.
    pub fill_color: Option<[f32; 4]>,
    pub text_color: [u8; 3],
    pub font_scale: u32,
}

impl Default for ResolvedStyle {
    fn default() -> Self {
        Self {
            width: None,
            height: None,
            margin: Edges::default(),
            padding: Edges::default(),
            grow: 0.0,
            flex_dir: FlexDir::Column,
            anchor: Anchor::TopLeft,
            color: [0.0, 0.0, 0.0, 0.0],
            fill_color: None,
            text_color: [220, 222, 230],
            font_scale: 1,
        }
    }
}

impl ResolvedStyle {
    fn apply(&mut self, s: &Style) {
        if let Some(v) = s.width {
            self.width = Some(v);
        }
        if let Some(v) = s.height {
            self.height = Some(v);
        }
        if let Some(v) = s.margin {
            self.margin = v;
        }
        if let Some(v) = s.padding {
            self.padding = v;
        }
        if let Some(v) = s.grow {
            self.grow = v;
        }
        if let Some(v) = s.flex_dir {
            self.flex_dir = v;
        }
        if let Some(v) = s.anchor {
            self.anchor = v;
        }
        if let Some(v) = s.color {
            self.color = v;
        }
        if let Some(v) = s.fill_color {
            self.fill_color = Some(v);
        }
        if let Some(v) = s.text_color {
            self.text_color = v;
        }
        if let Some(v) = s.font_scale {
            // Clamp author-controlled scale to a sane ceiling (>=1) so downstream
            // text-width math cannot overflow on hostile inputs.
            self.font_scale = v.clamp(1, MAX_FONT_SCALE);
        }
    }
}

// ---------------------------------------------------------------------------
// StyleSheet — named style classes (the USS analogue).
// ---------------------------------------------------------------------------

/// A named collection of [`Style`] blocks. Nodes reference these by class name;
/// resolution applies the listed classes in order, then the node's inline style.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StyleSheet {
    pub classes: HashMap<String, Style>,
}

impl StyleSheet {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_class(mut self, name: &str, style: Style) -> Self {
        self.classes.insert(name.to_string(), style);
        self
    }

    pub fn set(&mut self, name: &str, style: Style) {
        self.classes.insert(name.to_string(), style);
    }

    /// Resolve a node's effective style: classes (in order), then inline.
    pub fn resolve(&self, classes: &[String], inline: &Style) -> ResolvedStyle {
        let mut rs = ResolvedStyle::default();
        for c in classes {
            if let Some(block) = self.classes.get(c) {
                rs.apply(block);
            }
        }
        rs.apply(inline);
        rs
    }
}

// ---------------------------------------------------------------------------
// Node kinds.
// ---------------------------------------------------------------------------

/// The widget variety of a node. The leaf-vs-container distinction is implied:
/// `Panel` is the generic container; the rest are leaves with widget state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum UiKind {
    /// Generic styled container; lays out its children via `flex_dir`.
    Panel,
    Label { text: String },
    Button { label: String, on_click: String },
    Slider { value: f32, min: f32, max: f32, on_change: String },
    ProgressBar { value: f32 },
    /// Image-ish: a tinted rect standing in for a sampled texture (`path` is
    /// the asset reference; CPU render fills with the resolved colour x tint).
    Image { path: String, tint: [f32; 4] },
}

/// A retained node: identity, kind, styling, and children.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UiNode {
    pub id: String,
    pub kind: UiKind,
    #[serde(default)]
    pub classes: Vec<String>,
    #[serde(default)]
    pub style: Style,
    /// Whether the node renders / is hit-testable. Authors omit this for shown
    /// nodes (defaults to `true`) and write `"visible": false` to hide.
    #[serde(default = "default_true")]
    pub visible: bool,
    #[serde(default)]
    pub children: Vec<UiNode>,
}

fn default_true() -> bool {
    true
}

impl UiNode {
    pub fn new(id: &str, kind: UiKind) -> Self {
        Self {
            id: id.to_string(),
            kind,
            classes: Vec::new(),
            style: Style::default(),
            visible: true,
            children: Vec::new(),
        }
    }

    pub fn with_class(mut self, class: &str) -> Self {
        self.classes.push(class.to_string());
        self
    }

    pub fn with_style(mut self, style: Style) -> Self {
        self.style = style;
        self
    }

    pub fn with_children(mut self, children: Vec<UiNode>) -> Self {
        self.children = children;
        self
    }
}

// ---------------------------------------------------------------------------
// Document (the UXML analogue).
// ---------------------------------------------------------------------------

/// A declarative UI document: the style sheet plus the root node hierarchy.
/// Serde-loadable from TOML or JSON; this is what hot-reload watches.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UiDoc {
    #[serde(default)]
    pub stylesheet: StyleSheet,
    pub root: UiNode,
}

impl UiDoc {
    pub fn from_json(s: &str) -> Result<Self, String> {
        let doc: Self = serde_json::from_str(s).map_err(|e| e.to_string())?;
        // Duplicate node ids silently collapse in the id-keyed layout map —
        // a node renders at the wrong rect and hit-testing misroutes (review
        // finding). An id collision is an authoring error: reject it loudly.
        let mut seen = std::collections::HashSet::new();
        let mut stack = vec![&doc.root];
        while let Some(n) = stack.pop() {
            if !seen.insert(n.id.as_str()) {
                return Err(format!("duplicate node id '{}' in UI doc", n.id));
            }
            stack.extend(n.children.iter());
        }
        Ok(doc)
    }

    pub fn to_json(&self) -> Result<String, String> {
        serde_json::to_string_pretty(self).map_err(|e| e.to_string())
    }
}

// ---------------------------------------------------------------------------
// Layout — flexbox-lite producing absolute rects.
// ---------------------------------------------------------------------------

/// An absolute, resolved rectangle in viewport pixels: `[x, y, w, h]`.
pub type Rect = [f32; 4];

/// Resolved layout: maps node id -> absolute rect. Ids are assumed unique.
#[derive(Debug, Clone, Default)]
pub struct Layout {
    pub rects: HashMap<String, Rect>,
}

impl Layout {
    pub fn rect(&self, id: &str) -> Option<Rect> {
        self.rects.get(id).copied()
    }
}

fn anchor_origin(anchor: Anchor, outer: Rect, w: f32, h: f32) -> (f32, f32) {
    let [ox, oy, ow, oh] = outer;
    let x = match anchor {
        Anchor::TopLeft | Anchor::CenterLeft | Anchor::BottomLeft => ox,
        Anchor::TopCenter | Anchor::Center | Anchor::BottomCenter => ox + (ow - w) / 2.0,
        Anchor::TopRight | Anchor::CenterRight | Anchor::BottomRight => ox + ow - w,
    };
    let y = match anchor {
        Anchor::TopLeft | Anchor::TopCenter | Anchor::TopRight => oy,
        Anchor::CenterLeft | Anchor::Center | Anchor::CenterRight => oy + (oh - h) / 2.0,
        Anchor::BottomLeft | Anchor::BottomCenter | Anchor::BottomRight => oy + oh - h,
    };
    (x, y)
}

/// Compute absolute rects for the whole tree, in plain math, flexbox-lite.
///
/// Algorithm per container (the node's content box = its rect minus padding):
/// - Children are stacked along `flex_dir` (Column = +y, Row = +x).
/// - A child's cross-axis size fills the content box (minus its margins) unless
///   the child fixes that dimension via `width`/`height`.
/// - Main-axis space left after fixed-size children is split among `grow`
///   children in proportion to their grow weights. Margins are subtracted from
///   each child's box on both axes.
pub fn compute_layout(tree: &UiTree, viewport: [f32; 2]) -> Layout {
    let mut layout = Layout::default();
    let viewport_rect = [0.0, 0.0, viewport[0], viewport[1]];
    let root_style = tree.sheet.resolve(&tree.root.classes, &tree.root.style);
    // Root size: explicit, else fills the viewport.
    let rw = root_style.width.unwrap_or(viewport[0]);
    let rh = root_style.height.unwrap_or(viewport[1]);
    let (rx, ry) = anchor_origin(root_style.anchor, viewport_rect, rw, rh);
    place(tree, &tree.root, [rx, ry, rw, rh], &mut layout);
    layout
}

fn place(tree: &UiTree, node: &UiNode, rect: Rect, layout: &mut Layout) {
    layout.rects.insert(node.id.clone(), rect);
    if node.children.is_empty() {
        return;
    }
    let style = tree.sheet.resolve(&node.classes, &node.style);
    let pad = style.padding;
    let content = [
        rect[0] + pad.left,
        rect[1] + pad.top,
        (rect[2] - pad.left - pad.right).max(0.0),
        (rect[3] - pad.top - pad.bottom).max(0.0),
    ];

    // Resolve each child's style once.
    let child_styles: Vec<ResolvedStyle> = node
        .children
        .iter()
        .map(|c| tree.sheet.resolve(&c.classes, &c.style))
        .collect();

    let row = matches!(style.flex_dir, FlexDir::Row);
    // Main-axis content extent (minus each child's main-axis margins).
    let main_total = if row { content[2] } else { content[3] };
    let mut fixed_used = 0.0f32;
    let mut grow_sum = 0.0f32;
    for cs in &child_styles {
        let m = cs.margin;
        let margin_main = if row { m.left + m.right } else { m.top + m.bottom };
        let fixed = if row { cs.width } else { cs.height };
        match fixed {
            Some(v) => fixed_used += v + margin_main,
            None => {
                fixed_used += margin_main;
                grow_sum += cs.grow.max(0.0);
            }
        }
    }
    let free = (main_total - fixed_used).max(0.0);

    let mut cursor = if row { content[0] } else { content[1] };
    for (child, cs) in node.children.iter().zip(child_styles.iter()) {
        let m = cs.margin;
        // Cross-axis: fill content (minus cross margins) unless fixed.
        let (cross_off, cross_avail) = if row {
            (m.top, (content[3] - m.top - m.bottom).max(0.0))
        } else {
            (m.left, (content[2] - m.left - m.right).max(0.0))
        };
        let cross_fixed = if row { cs.height } else { cs.width };
        let cross_size = cross_fixed.unwrap_or(cross_avail);

        // Main-axis size.
        let fixed = if row { cs.width } else { cs.height };
        let main_size = match fixed {
            Some(v) => v,
            None => {
                if grow_sum > 0.0 {
                    free * (cs.grow.max(0.0) / grow_sum)
                } else {
                    0.0
                }
            }
        };

        let margin_lead = if row { m.left } else { m.top };
        let main_pos = cursor + margin_lead;

        let child_rect = if row {
            [main_pos, content[1] + cross_off, main_size, cross_size]
        } else {
            [content[0] + cross_off, main_pos, cross_size, main_size]
        };

        place(tree, child, child_rect, layout);

        let margin_main = if row { m.left + m.right } else { m.top + m.bottom };
        cursor += main_size + margin_main;
    }
}

// ---------------------------------------------------------------------------
// Events (drainable widget-event idiom).
// ---------------------------------------------------------------------------

/// An interaction outcome, drained by the host each frame.
#[derive(Debug, Clone, PartialEq)]
pub enum UiEvent {
    Clicked { id: String, action: String },
    ValueChanged { id: String, value: f32 },
}

// ---------------------------------------------------------------------------
// The retained tree.
// ---------------------------------------------------------------------------

/// The single retained game-UI tree: root hierarchy + active style sheet +
/// runtime interaction state (focus) + a drainable event queue.
pub struct UiTree {
    pub root: UiNode,
    pub sheet: StyleSheet,
    /// Id of the currently-focused node (drawn brighter), if any.
    pub focused: Option<String>,
    events: Vec<UiEvent>,
}

impl UiTree {
    pub fn new(root: UiNode) -> Self {
        Self { root, sheet: StyleSheet::new(), focused: None, events: Vec::new() }
    }

    pub fn from_doc(doc: UiDoc) -> Self {
        Self { root: doc.root, sheet: doc.stylesheet, focused: None, events: Vec::new() }
    }

    pub fn with_sheet(mut self, sheet: StyleSheet) -> Self {
        self.sheet = sheet;
        self
    }

    pub fn node_count(&self) -> usize {
        fn count(n: &UiNode) -> usize {
            1 + n.children.iter().map(count).sum::<usize>()
        }
        count(&self.root)
    }

    pub fn find(&self, id: &str) -> Option<&UiNode> {
        fn rec<'a>(n: &'a UiNode, id: &str) -> Option<&'a UiNode> {
            if n.id == id {
                return Some(n);
            }
            n.children.iter().find_map(|c| rec(c, id))
        }
        rec(&self.root, id)
    }

    pub fn find_mut(&mut self, id: &str) -> Option<&mut UiNode> {
        fn rec<'a>(n: &'a mut UiNode, id: &str) -> Option<&'a mut UiNode> {
            if n.id == id {
                return Some(n);
            }
            n.children.iter_mut().find_map(|c| rec(c, id))
        }
        rec(&mut self.root, id)
    }

    /// Read a slider's current value, if `id` names a slider.
    pub fn slider_value(&self, id: &str) -> Option<f32> {
        match &self.find(id)?.kind {
            UiKind::Slider { value, .. } => Some(*value),
            _ => None,
        }
    }

    pub fn set_slider_value(&mut self, id: &str, v: f32) -> bool {
        if let Some(node) = self.find_mut(id)
            && let UiKind::Slider { value, min, max, .. } = &mut node.kind
        {
            *value = v.clamp(*min, *max);
            return true;
        }
        false
    }

    pub fn set_label(&mut self, id: &str, text: &str) -> bool {
        if let Some(node) = self.find_mut(id)
            && let UiKind::Label { text: t } = &mut node.kind
        {
            *t = text.to_string();
            return true;
        }
        false
    }

    /// Drain accumulated events (clears the queue).
    pub fn drain_events(&mut self) -> Vec<UiEvent> {
        std::mem::take(&mut self.events)
    }

    // --- Hot-reload --------------------------------------------------------

    /// Replace the tree from a new document, **preserving runtime state**
    /// (slider values, progress, focus) for any node id whose kind is
    /// structurally the same. Unchanged ids keep their live value; new ids take
    /// the doc value; removed ids are dropped (focus cleared if it pointed at
    /// one).
    pub fn reload(&mut self, doc: &UiDoc) {
        // Snapshot live state by id before replacing.
        let mut live_sliders: HashMap<String, f32> = HashMap::new();
        let mut live_progress: HashMap<String, f32> = HashMap::new();
        collect_state(&self.root, &mut live_sliders, &mut live_progress);

        let mut new_root = doc.root.clone();
        merge_state(&mut new_root, &live_sliders, &live_progress);

        self.root = new_root;
        self.sheet = doc.stylesheet.clone();

        // Preserve focus only if the id still exists.
        if let Some(f) = &self.focused
            && self.find(f).is_none()
        {
            self.focused = None;
        }
    }
}

fn collect_state(
    node: &UiNode,
    sliders: &mut HashMap<String, f32>,
    progress: &mut HashMap<String, f32>,
) {
    match &node.kind {
        UiKind::Slider { value, .. } => {
            sliders.insert(node.id.clone(), *value);
        }
        UiKind::ProgressBar { value } => {
            progress.insert(node.id.clone(), *value);
        }
        _ => {}
    }
    for c in &node.children {
        collect_state(c, sliders, progress);
    }
}

fn merge_state(
    node: &mut UiNode,
    sliders: &HashMap<String, f32>,
    progress: &HashMap<String, f32>,
) {
    match &mut node.kind {
        UiKind::Slider { value, min, max, .. } => {
            if let Some(v) = sliders.get(&node.id) {
                *value = v.clamp(*min, *max);
            }
        }
        UiKind::ProgressBar { value } => {
            if let Some(v) = progress.get(&node.id) {
                *value = *v;
            }
        }
        _ => {}
    }
    for c in &mut node.children {
        merge_state(c, sliders, progress);
    }
}

// ---------------------------------------------------------------------------
// Hit-testing.
// ---------------------------------------------------------------------------

/// Return the id of the deepest visible node whose rect contains `point`
/// (top-most child wins). `None` if the point is outside the whole tree.
pub fn hit_test(tree: &UiTree, layout: &Layout, point: [f32; 2]) -> Option<String> {
    fn inside(rect: Rect, p: [f32; 2]) -> bool {
        p[0] >= rect[0] && p[0] < rect[0] + rect[2] && p[1] >= rect[1] && p[1] < rect[1] + rect[3]
    }
    fn rec(node: &UiNode, layout: &Layout, p: [f32; 2]) -> Option<String> {
        if !node.visible {
            return None;
        }
        let rect = layout.rect(&node.id)?;
        if !inside(rect, p) {
            return None;
        }
        // A child (later = drawn on top) takes precedence over the parent.
        for c in node.children.iter().rev() {
            if let Some(hit) = rec(c, layout, p) {
                return Some(hit);
            }
        }
        Some(node.id.clone())
    }
    rec(&tree.root, layout, point)
}

/// Resolve a click at `point`: emits a `Clicked` for a hit button (and focuses
/// it). Returns the hit node id, if any. Events are read via [`UiTree::drain_events`].
pub fn click(tree: &mut UiTree, layout: &Layout, point: [f32; 2]) -> Option<String> {
    let hit = hit_test(tree, layout, point)?;
    if let Some(node) = tree.find(&hit)
        && let UiKind::Button { on_click, .. } = &node.kind
    {
        let action = on_click.clone();
        tree.events.push(UiEvent::Clicked { id: hit.clone(), action });
        tree.focused = Some(hit.clone());
    }
    Some(hit)
}

// ---------------------------------------------------------------------------
// Rendering (CPU software-rasterise, mirroring VelloCtxCpu + burn_text).
// ---------------------------------------------------------------------------

fn blend_rect(pixels: &mut [[u8; 4]], w: u32, h: u32, rect: Rect, color: [f32; 4]) {
    let a = color[3].clamp(0.0, 1.0);
    if a <= 0.0 {
        return;
    }
    let x0 = (rect[0].floor() as i64).clamp(0, w as i64);
    let y0 = (rect[1].floor() as i64).clamp(0, h as i64);
    let x1 = ((rect[0] + rect[2]).ceil() as i64).clamp(0, w as i64);
    let y1 = ((rect[1] + rect[3]).ceil() as i64).clamp(0, h as i64);
    if x0 >= x1 || y0 >= y1 {
        return;
    }
    let (sr, sg, sb) = (color[0].clamp(0.0, 1.0), color[1].clamp(0.0, 1.0), color[2].clamp(0.0, 1.0));
    let inv = 1.0 - a;
    for y in y0..y1 {
        for x in x0..x1 {
            let idx = (y * w as i64 + x) as usize;
            if idx >= pixels.len() {
                continue;
            }
            let d = pixels[idx];
            let (dr, dg, db, da) =
                (d[0] as f32 / 255.0, d[1] as f32 / 255.0, d[2] as f32 / 255.0, d[3] as f32 / 255.0);
            pixels[idx] = [
                ((sr * a + dr * inv) * 255.0 + 0.5) as u8,
                ((sg * a + dg * inv) * 255.0 + 0.5) as u8,
                ((sb * a + db * inv) * 255.0 + 0.5) as u8,
                ((a + da * inv) * 255.0 + 0.5) as u8,
            ];
        }
    }
}

/// Brighten an RGB triple toward white by `t` in `0..=1` (used for focus).
fn brighten(c: [u8; 3], t: f32) -> [u8; 3] {
    let t = t.clamp(0.0, 1.0);
    [
        (c[0] as f32 + (255.0 - c[0] as f32) * t) as u8,
        (c[1] as f32 + (255.0 - c[1] as f32) * t) as u8,
        (c[2] as f32 + (255.0 - c[2] as f32) * t) as u8,
    ]
}

fn brighten4(c: [f32; 4], t: f32) -> [f32; 4] {
    let t = t.clamp(0.0, 1.0);
    [c[0] + (1.0 - c[0]) * t, c[1] + (1.0 - c[1]) * t, c[2] + (1.0 - c[2]) * t, c[3]]
}

/// Centre a string within `rect` and render it at the resolved scale via the
/// [`crate::text`] module (real parley/swash text under `game-ui`, 5x7 bitmap
/// fallback otherwise). `scale` is the 5x7 multiplier; the text module maps it
/// to a pixel font height (`CHAR_H * scale`) so both paths size the same box.
fn draw_text_centered(
    pixels: &mut [[u8; 4]],
    w: u32,
    h: u32,
    rect: Rect,
    label: &str,
    color: [u8; 3],
    scale: u32,
) {
    let scale = scale.max(1);
    // Author-controlled text is capped before any shaping/rasterization: a
    // megabyte label would stall the UI for minutes (parley shapes the whole
    // string) while nothing beyond a few thousand chars can ever be visible.
    const MAX_DRAW_CHARS: usize = 4096;
    let label = if label.chars().count() > MAX_DRAW_CHARS {
        &label[..label.char_indices().nth(MAX_DRAW_CHARS).map(|(i, _)| i).unwrap_or(label.len())]
    } else {
        label
    };
    // Approximate text extent for centring. The bitmap font's metrics
    // (stride/CHAR_H) are exact for the fallback path and a close-enough box for
    // the proportional parley path (centring need not be sub-pixel).
    // Saturating u64 math: label length and font_scale are author-controlled
    // (a hostile doc with a megabyte label must not overflow — review finding;
    // the MAX_FONT_SCALE clamp alone still wrapped at ~11M chars).
    let tw = if label.is_empty() {
        0
    } else {
        (label.chars().count() as u64)
            .saturating_mul(CHAR_STRIDE as u64)
            .saturating_mul(scale as u64)
            .saturating_sub(scale as u64)
            .min(u32::MAX as u64) as u32
    };
    let th = CHAR_H * scale;
    let tx = (rect[0] + (rect[2] - tw as f32) / 2.0).max(0.0);
    let ty = (rect[1] + (rect[3] - th as f32) / 2.0).max(0.0);
    text::draw_text(pixels, w, h, [tx, ty], label, color, th as f32);
}

/// Software-rasterise the whole tree into an RGBA8 buffer (row-major,
/// `pixels.len() >= w*h`). The focused node renders measurably brighter.
pub fn rasterize_into(tree: &UiTree, layout: &Layout, pixels: &mut [[u8; 4]], w: u32, h: u32) {
    fn rec(
        tree: &UiTree,
        node: &UiNode,
        layout: &Layout,
        pixels: &mut [[u8; 4]],
        w: u32,
        h: u32,
    ) {
        if !node.visible {
            return;
        }
        let Some(rect) = layout.rect(&node.id) else {
            return;
        };
        let style = tree.sheet.resolve(&node.classes, &node.style);
        let focused = tree.focused.as_deref() == Some(node.id.as_str());

        match &node.kind {
            UiKind::Panel => {
                blend_rect(pixels, w, h, rect, style.color);
            }
            UiKind::Image { tint, .. } => {
                let c = [
                    style.color[0] * tint[0],
                    style.color[1] * tint[1],
                    style.color[2] * tint[2],
                    if style.color[3] <= 0.0 { tint[3] } else { style.color[3] * tint[3] },
                ];
                blend_rect(pixels, w, h, rect, c);
            }
            UiKind::Label { text } => {
                if style.color[3] > 0.0 {
                    blend_rect(pixels, w, h, rect, style.color);
                }
                let tc = if focused { brighten(style.text_color, 0.5) } else { style.text_color };
                draw_text_centered(pixels, w, h, rect, text, tc, style.font_scale);
            }
            UiKind::Button { label, .. } => {
                let bg = if focused { brighten4(style.color, 0.4) } else { style.color };
                blend_rect(pixels, w, h, rect, bg);
                let tc = if focused { brighten(style.text_color, 0.4) } else { style.text_color };
                draw_text_centered(pixels, w, h, rect, label, tc, style.font_scale);
            }
            UiKind::Slider { value, min, max, .. } => {
                // Track, then a fill proportional to value, then a knob.
                blend_rect(pixels, w, h, rect, style.color);
                let t = ((value - min) / (max - min)).clamp(0.0, 1.0);
                let fill = [rect[0], rect[1], rect[2] * t, rect[3]];
                // Explicit fill colour if set, else brighten the track colour.
                let fill_color = style.fill_color.unwrap_or_else(|| brighten4(style.color, 0.5));
                blend_rect(pixels, w, h, fill, fill_color);
            }
            UiKind::ProgressBar { value } => {
                blend_rect(pixels, w, h, rect, style.color);
                let t = value.clamp(0.0, 1.0);
                let fill = [rect[0], rect[1], rect[2] * t, rect[3]];
                let fill_color = style.fill_color.unwrap_or_else(|| brighten4(style.color, 0.6));
                blend_rect(pixels, w, h, fill, fill_color);
            }
        }

        for c in &node.children {
            rec(tree, c, layout, pixels, w, h);
        }
    }
    rec(tree, &tree.root, layout, pixels, w, h);
}

// ---------------------------------------------------------------------------
// taffy parity (feature-gated proof that plain-math agrees with taffy).
// ---------------------------------------------------------------------------

/// Lay out a single column of `grow`-equal children with taffy and return their
/// rects, for cross-checking the plain-math solver. Only available with the
/// `game-ui` feature (taffy is an opt-in dep).
#[cfg(feature = "game-ui")]
pub fn taffy_column_rects(viewport_w: f32, viewport_h: f32, n: usize) -> Vec<Rect> {
    use taffy::prelude::*;
    let mut t: TaffyTree<()> = TaffyTree::new();
    let children: Vec<NodeId> = (0..n)
        .map(|_| {
            t.new_leaf(Style {
                flex_grow: 1.0,
                size: Size { width: Dimension::Auto, height: Dimension::Auto },
                ..Default::default()
            })
            .expect("leaf")
        })
        .collect();
    let root = t
        .new_with_children(
            Style {
                display: Display::Flex,
                flex_direction: FlexDirection::Column,
                size: Size { width: Dimension::Length(viewport_w), height: Dimension::Length(viewport_h) },
                ..Default::default()
            },
            &children,
        )
        .expect("root");
    t.compute_layout(
        root,
        Size { width: AvailableSpace::Definite(viewport_w), height: AvailableSpace::Definite(viewport_h) },
    )
    .expect("compute");
    children
        .iter()
        .map(|c| {
            let l = t.layout(*c).expect("layout");
            // taffy reports child location relative to parent (here, origin).
            [l.location.x, l.location.y, l.size.width, l.size.height]
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn luma_at(pixels: &[[u8; 4]], w: u32, x: u32, y: u32) -> f32 {
        let p = pixels[(y * w + x) as usize];
        0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32
    }

    fn col_child(id: &str) -> UiNode {
        let mut n = UiNode::new(id, UiKind::Panel);
        n.style.grow = Some(1.0);
        n
    }

    /// Column of 3 grow-equal children in 600px splits into 200px rects.
    #[test]
    fn column_three_grow_equal_splits_exactly() {
        let root = UiNode::new("root", UiKind::Panel)
            .with_children(vec![col_child("a"), col_child("b"), col_child("c")]);
        let tree = UiTree::new(root);
        let layout = compute_layout(&tree, [400.0, 600.0]);
        let a = layout.rect("a").unwrap();
        let b = layout.rect("b").unwrap();
        let c = layout.rect("c").unwrap();
        println!("a={a:?} b={b:?} c={c:?}");
        assert_eq!(a, [0.0, 0.0, 400.0, 200.0]);
        assert_eq!(b, [0.0, 200.0, 400.0, 200.0]);
        assert_eq!(c, [0.0, 400.0, 400.0, 200.0]);
    }

    /// Margins and padding shift children by exact pixels.
    #[test]
    fn padding_and_margin_shift_by_exact_pixels() {
        let mut root = UiNode::new("root", UiKind::Panel);
        root.style.padding = Some(Edges::all(10.0));
        let mut child = col_child("a");
        child.style.margin = Some(Edges { left: 5.0, top: 7.0, right: 5.0, bottom: 0.0 });
        root = root.with_children(vec![child]);
        let tree = UiTree::new(root);
        let layout = compute_layout(&tree, [400.0, 600.0]);
        let a = layout.rect("a").unwrap();
        println!("a={a:?}");
        // x = pad.left(10) + margin.left(5) = 15
        // y = pad.top(10) + margin.top(7) = 17
        // w = 400 - pad.l(10) - pad.r(10) - margin.l(5) - margin.r(5) = 370
        // h = (600 - pad.t(10) - pad.b(10)) - margin.t(7) - margin.b(0) = 573
        assert_eq!(a, [15.0, 17.0, 370.0, 573.0]);
    }

    /// Same tree, two themes -> a button's pixels differ in the themed channel.
    #[test]
    fn two_themes_differ_in_button_color_channel() {
        let root = UiNode::new("root", UiKind::Panel).with_children(vec![{
            let mut b = UiNode::new("btn", UiKind::Button { label: "GO".into(), on_click: "go".into() })
                .with_class("primary");
            b.style.grow = Some(1.0);
            b
        }]);

        let blue = StyleSheet::new().with_class(
            "primary",
            Style { color: Some([0.0, 0.0, 1.0, 1.0]), ..Default::default() },
        );
        let red = StyleSheet::new().with_class(
            "primary",
            Style { color: Some([1.0, 0.0, 0.0, 1.0]), ..Default::default() },
        );

        let (w, h) = (64u32, 64u32);
        let render = |sheet: StyleSheet| {
            let tree = UiTree::new(root.clone()).with_sheet(sheet);
            let layout = compute_layout(&tree, [w as f32, h as f32]);
            let mut px = vec![[0u8; 4]; (w * h) as usize];
            rasterize_into(&tree, &layout, &mut px, w, h);
            px
        };
        let pb = render(blue);
        let pr = render(red);
        // Sample a button-interior pixel (avoid the centered glyphs).
        let i = (4 * w + 4) as usize;
        println!("blue px={:?} red px={:?}", pb[i], pr[i]);
        assert_eq!(pb[i], [0, 0, 255, 255]);
        assert_eq!(pr[i], [255, 0, 0, 255]);
        assert_ne!(pb[i][0], pr[i][0]);
        assert_ne!(pb[i][2], pr[i][2]);
    }

    /// Hot-reload preserves slider value while picking up a changed label.
    #[test]
    /// A hostile doc (megabyte label x clamped-max font_scale) must render
    /// without arithmetic overflow — saturating width math (review finding;
    /// the font_scale clamp alone still wrapped at ~11M chars).
    #[test]
    fn hostile_label_length_does_not_overflow() {
        let huge = "x".repeat(12_000_000);
        let doc = UiDoc {
            stylesheet: StyleSheet::default(),
            root: UiNode {
                id: "root".into(),
                kind: UiKind::Label { text: huge },
                classes: vec![],
                style: Style { font_scale: Some(999_999), ..Default::default() },
                visible: true,
                children: vec![],
            },
        };
        let tree = UiTree::from_doc(doc);
        let layout = compute_layout(&tree, [320.0, 200.0]);
        let mut px = vec![[0u8; 4]; 320 * 200];
        rasterize_into(&tree, &layout, &mut px, 320, 200); // must not panic
    }

    /// Duplicate ids are an authoring error: from_json rejects them loudly
    /// instead of letting the layout map silently collapse them.
    #[test]
    fn duplicate_ids_are_rejected_at_parse() {
        let json = r#"{"root":{"id":"r","kind":{"Panel":null},"children":[
            {"id":"x","kind":{"Label":{"text":"a"}}},
            {"id":"x","kind":{"Label":{"text":"b"}}}]}}"#;
        match UiDoc::from_json(json) {
            Err(e) => assert!(e.contains("duplicate node id 'x'"), "names the id: {e}"),
            Ok(_) => panic!("duplicate ids must be rejected"),
        }
    }

    #[test]
    fn reload_preserves_slider_value_and_picks_up_label() {
        let doc_v1 = r#"{
            "stylesheet": { "classes": {} },
            "root": {
                "id": "root", "kind": "Panel",
                "children": [
                    { "id": "vol", "kind": { "Slider": { "value": 0.3, "min": 0.0, "max": 1.0, "on_change": "set_vol" } } },
                    { "id": "lbl", "kind": { "Label": { "text": "OLD" } } }
                ]
            }
        }"#;
        let doc1 = UiDoc::from_json(doc_v1).unwrap();
        let mut tree = UiTree::from_doc(doc1);
        // Runtime set the slider to 0.7.
        assert!(tree.set_slider_value("vol", 0.7));
        assert_eq!(tree.slider_value("vol"), Some(0.7));

        // New doc: only the unrelated label changed; slider doc value differs (0.3).
        let doc_v2 = r#"{
            "stylesheet": { "classes": {} },
            "root": {
                "id": "root", "kind": "Panel",
                "children": [
                    { "id": "vol", "kind": { "Slider": { "value": 0.3, "min": 0.0, "max": 1.0, "on_change": "set_vol" } } },
                    { "id": "lbl", "kind": { "Label": { "text": "NEW" } } }
                ]
            }
        }"#;
        let doc2 = UiDoc::from_json(doc_v2).unwrap();
        tree.reload(&doc2);

        // Slider value preserved across reload.
        assert_eq!(tree.slider_value("vol"), Some(0.7), "slider value lost on reload");

        // The new label text actually renders (text-extraction via burn_text).
        let label = tree.find("lbl").unwrap();
        match &label.kind {
            UiKind::Label { text } => assert_eq!(text, "NEW"),
            _ => panic!("lbl is not a label"),
        }
        // Pixel proof the new text rasterises: render and count lit text pixels.
        let mut sheet = StyleSheet::new();
        sheet.set("lbltxt", Style { text_color: Some([255, 255, 255]), font_scale: Some(2), ..Default::default() });
        if let Some(n) = tree.find_mut("lbl") {
            n.classes.push("lbltxt".into());
            n.style.height = Some(40.0);
            n.style.grow = Some(1.0);
        }
        let (w, h) = (200u32, 80u32);
        let layout = compute_layout(&tree, [w as f32, h as f32]);
        let mut px = vec![[0u8; 4]; (w * h) as usize];
        rasterize_into(&tree, &layout, &mut px, w, h);
        // Count lit text pixels. The text path may be the blocky 5x7 bitmap
        // (fully-opaque white) OR real anti-aliased parley glyphs (partial
        // coverage at edges), so we count any pixel with meaningful ink rather
        // than requiring full saturation — both paths must light real glyphs.
        let lit = px
            .iter()
            .filter(|p| p[0] > 60 && p[1] > 60 && p[2] > 60)
            .count();
        println!("NEW label lit text pixels = {lit}");
        assert!(lit > 30, "expected the NEW label glyphs to rasterise, got {lit} lit px");
    }

    /// Hit-test: inside a button returns its id; 2px outside returns parent/none.
    #[test]
    fn hit_test_inside_and_outside_button() {
        let mut btn = UiNode::new("btn", UiKind::Button { label: "X".into(), on_click: "x".into() });
        btn.style.width = Some(100.0);
        btn.style.height = Some(40.0);
        btn.style.anchor = Some(Anchor::TopLeft);
        // Root is a panel filling the viewport; the button sits at top-left,
        // a single non-grow fixed child.
        let root = UiNode::new("root", UiKind::Panel).with_children(vec![btn]);
        let tree = UiTree::new(root);
        let layout = compute_layout(&tree, [400.0, 300.0]);
        let btn_rect = layout.rect("btn").unwrap();
        println!("btn_rect={btn_rect:?}");

        // Inside the button.
        let inside = [btn_rect[0] + 10.0, btn_rect[1] + 10.0];
        assert_eq!(hit_test(&tree, &layout, inside).as_deref(), Some("btn"));

        // 2px past the button's right edge -> falls on the root panel.
        let just_outside = [btn_rect[0] + btn_rect[2] + 2.0, btn_rect[1] + 10.0];
        assert_eq!(hit_test(&tree, &layout, just_outside).as_deref(), Some("root"));

        // Far outside the whole tree -> none.
        assert_eq!(hit_test(&tree, &layout, [10_000.0, 10_000.0]), None);
    }

    /// Full menu doc rasterises >N non-bg px; focused button is brighter.
    #[test]
    fn menu_renders_and_focused_button_is_brighter() {
        let doc = r#"{
            "stylesheet": { "classes": {
                "bg":   { "color": [0.07, 0.07, 0.10, 1.0] },
                "btn":  { "color": [0.15, 0.30, 0.60, 1.0], "grow": 1.0, "margin": { "left": 20.0, "top": 8.0, "right": 20.0, "bottom": 8.0 } }
            } },
            "root": {
                "id": "root", "kind": "Panel", "classes": ["bg"],
                "children": [
                    { "id": "b_start",   "kind": { "Button": { "label": "START",   "on_click": "start" } },   "classes": ["btn"] },
                    { "id": "b_options", "kind": { "Button": { "label": "OPTIONS", "on_click": "options" } }, "classes": ["btn"] },
                    { "id": "b_quit",    "kind": { "Button": { "label": "QUIT",    "on_click": "quit" } },    "classes": ["btn"] }
                ]
            }
        }"#;
        let mut tree = UiTree::from_doc(UiDoc::from_json(doc).unwrap());
        tree.focused = Some("b_start".into());

        let (w, h) = (320u32, 240u32);
        let layout = compute_layout(&tree, [w as f32, h as f32]);
        let mut px = vec![[0u8; 4]; (w * h) as usize];
        // Fill with the bg colour first so "non-background" is well-defined.
        rasterize_into(&tree, &layout, &mut px, w, h);

        // Root bg class [0.07,0.07,0.10] rasterises to [18,18,26].
        let bg = [18u8, 18, 26, 255];
        let non_bg = px.iter().filter(|p| {
            (p[0] as i32 - bg[0] as i32).abs() > 8
                || (p[1] as i32 - bg[1] as i32).abs() > 8
                || (p[2] as i32 - bg[2] as i32).abs() > 8
        }).count();
        println!("menu non-background px = {non_bg}");
        assert!(non_bg > 5000, "menu should paint many non-bg px, got {non_bg}");

        // Focused (b_start) vs unfocused (b_quit): compare interior luminance.
        let r_start = layout.rect("b_start").unwrap();
        let r_quit = layout.rect("b_quit").unwrap();
        let sx = (r_start[0] + 4.0) as u32;
        let sy = (r_start[1] + r_start[3] / 2.0) as u32;
        let qx = (r_quit[0] + 4.0) as u32;
        let qy = (r_quit[1] + r_quit[3] / 2.0) as u32;
        let l_focus = luma_at(&px, w, sx, sy);
        let l_normal = luma_at(&px, w, qx, qy);
        println!("focused luma={l_focus} unfocused luma={l_normal}");
        assert!(l_focus > l_normal + 10.0, "focused button must be brighter: {l_focus} vs {l_normal}");
    }

    /// Document JSON round-trips structure + style classes.
    #[test]
    fn doc_json_round_trip() {
        let doc = r#"{
            "stylesheet": { "classes": { "p": { "color": [1.0, 0.0, 0.0, 1.0] } } },
            "root": { "id": "root", "kind": "Panel", "classes": ["p"],
                "children": [ { "id": "t", "kind": { "Label": { "text": "Hi" } } } ] }
        }"#;
        let parsed = UiDoc::from_json(doc).unwrap();
        let s = parsed.to_json().unwrap();
        let again = UiDoc::from_json(&s).unwrap();
        let tree = UiTree::from_doc(again);
        assert_eq!(tree.node_count(), 2);
        assert!(tree.find("t").is_some());
        assert_eq!(tree.sheet.classes.get("p").unwrap().color, Some([1.0, 0.0, 0.0, 1.0]));
    }

    /// Feature-gated: plain-math and taffy agree within 1px on the 3-child column.
    #[cfg(feature = "game-ui")]
    #[test]
    fn taffy_parity_three_child_column() {
        let root = UiNode::new("root", UiKind::Panel)
            .with_children(vec![col_child("a"), col_child("b"), col_child("c")]);
        let tree = UiTree::new(root);
        let layout = compute_layout(&tree, [400.0, 600.0]);
        let mine = [layout.rect("a").unwrap(), layout.rect("b").unwrap(), layout.rect("c").unwrap()];
        let taffy = taffy_column_rects(400.0, 600.0, 3);
        println!("plain={mine:?}\ntaffy={taffy:?}");
        for (m, t) in mine.iter().zip(taffy.iter()) {
            for k in 0..4 {
                assert!((m[k] - t[k]).abs() <= 1.0, "axis {k}: plain {} vs taffy {}", m[k], t[k]);
            }
        }
    }
}
