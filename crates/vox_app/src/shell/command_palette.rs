//! The command palette — the AI-native ONE-command-surface (design UX
//! Principle 2, SOTA item 17).
//!
//! Every editor action (menu item, toolbar button, and later every plugin and
//! AI tool-call) is a registered [`Command`] in a [`CommandRegistry`]. The
//! palette (Ctrl+K), the keyboard, the menus, and the toolbar ALL invoke the
//! same registry — nothing is operable by hand that an agent can't drive, and
//! vice versa. This registry surface is therefore the future AI tool-call
//! surface: `{ id, title, category, shortcut }` + a `run` callback.
//!
//! The palette is a centered modal overlay with a fuzzy-ranked result list,
//! arrow-key selection, and Enter-to-execute.

use std::rc::Rc;
use vox_ui::Tokens;

/// A single registered command — the unit of the one-command-surface. The
/// `run` callback is the executable handle (the AI tool-call shape later).
pub struct Command {
    pub id: String,
    pub title: String,
    pub category: String,
    /// Human-readable shortcut hint (e.g. `"Ctrl+K"`), shown right-aligned.
    pub shortcut: String,
    /// The action. Shared+interior-mutable so a command can flip a flag a test
    /// (or the host) observes after Enter.
    pub run: Rc<dyn Fn()>,
}

impl Command {
    pub fn new(
        id: impl Into<String>,
        title: impl Into<String>,
        category: impl Into<String>,
        shortcut: impl Into<String>,
        run: impl Fn() + 'static,
    ) -> Self {
        Command {
            id: id.into(),
            title: title.into(),
            category: category.into(),
            shortcut: shortcut.into(),
            run: Rc::new(run),
        }
    }
}

/// The registry of all editor commands — the single dispatch surface.
#[derive(Default)]
pub struct CommandRegistry {
    pub commands: Vec<Command>,
}

impl CommandRegistry {
    pub fn new() -> Self {
        CommandRegistry { commands: Vec::new() }
    }

    /// Register a command. **Same-id registration REPLACES the existing command
    /// in place** (it does not append a shadowed duplicate), mirroring the
    /// subgraph registry's `register_subgraph`. Re-registering under an id that
    /// already exists swaps the title/category/shortcut/callback to the new
    /// command, so [`run`](Self::run) dispatches the new version and
    /// [`search`](Self::search) returns exactly one hit for that id. This keeps
    /// menus, palette Enter, and `run(id)` from ever diverging under a collision.
    pub fn add(&mut self, cmd: Command) -> &mut Self {
        if let Some(existing) = self.commands.iter_mut().find(|c| c.id == cmd.id) {
            *existing = cmd;
        } else {
            self.commands.push(cmd);
        }
        self
    }

    /// Invoke a command by id (the path menus/toolbar use — they route THROUGH
    /// the registry, never calling logic directly). Returns whether it existed.
    pub fn run(&self, id: &str) -> bool {
        if let Some(c) = self.commands.iter().find(|c| c.id == id) {
            (c.run)();
            true
        } else {
            false
        }
    }

    /// Fuzzy-rank commands against `query`, best-first. The ranker mirrors the
    /// editor registry ladder: exact > prefix > word-prefix > substring >
    /// subsequence; ties broken by shorter title then alphabetically. Empty
    /// query returns every command in registration order.
    pub fn search(&self, query: &str) -> Vec<&Command> {
        if query.trim().is_empty() {
            return self.commands.iter().collect();
        }
        let q = query.to_lowercase();
        let mut scored: Vec<(i32, usize, &Command)> = self
            .commands
            .iter()
            .filter_map(|c| fuzzy_score(&c.title, &q).map(|s| (s, c.title.len(), c)))
            .collect();
        // Higher score first; then shorter title; then alphabetical.
        scored.sort_by(|a, b| {
            b.0.cmp(&a.0)
                .then(a.1.cmp(&b.1))
                .then(a.2.title.cmp(&b.2.title))
        });
        scored.into_iter().map(|(_, _, c)| c).collect()
    }
}

/// Score a `title` against a lowercased `query`. Returns `None` if no match.
/// Larger is better. The ladder makes `addw` rank `Add to world` (a word-start
/// subsequence) above an unrelated substring elsewhere.
fn fuzzy_score(title: &str, q: &str) -> Option<i32> {
    let t = title.to_lowercase();
    if t == q {
        return Some(1000);
    }
    if t.starts_with(q) {
        return Some(800);
    }
    // Word-prefix: query matches the initials/word-starts of the title
    // (e.g. "addw" => "Add to world").
    if let Some(s) = word_prefix_score(&t, q) {
        return Some(s);
    }
    if t.contains(q) {
        return Some(400);
    }
    subsequence_score(&t, q).map(|s| 200 + s)
}

/// If `q` matches the concatenation of leading characters of consecutive words
/// (allowing the first word to absorb a multi-char run), score it highly. E.g.
/// title "Add to world" -> word starts "a","t","w" plus the first word's body:
/// "addw" matches "add"(word0 body) + "w"(word2 start).
fn word_prefix_score(title: &str, q: &str) -> Option<i32> {
    let words: Vec<&str> = title.split_whitespace().collect();
    if words.is_empty() {
        return None;
    }
    // Greedy: consume q across words, each word may match a prefix of itself.
    let mut qi = 0;
    let qb = q.as_bytes();
    let mut matched_words: i32 = 0;
    for w in &words {
        if qi >= qb.len() {
            break;
        }
        let wb = w.as_bytes();
        // How many leading chars of this word match q from qi?
        let mut k = 0;
        while qi + k < qb.len() && k < wb.len() && qb[qi + k] == wb[k] {
            k += 1;
        }
        if k > 0 {
            qi += k;
            matched_words += 1;
        }
    }
    if qi == qb.len() && matched_words >= 1 {
        // Reward covering more words (tighter acronym match).
        Some(600 + matched_words * 10)
    } else {
        None
    }
}

/// In-order subsequence match; score rewards contiguous runs.
fn subsequence_score(title: &str, q: &str) -> Option<i32> {
    let (tb, qb) = (title.as_bytes(), q.as_bytes());
    let mut ti = 0;
    let mut qi = 0;
    let mut run = 0;
    let mut score = 0i32;
    while ti < tb.len() && qi < qb.len() {
        if tb[ti] == qb[qi] {
            run += 1;
            score += run; // contiguity bonus
            qi += 1;
        } else {
            run = 0;
        }
        ti += 1;
    }
    if qi == qb.len() {
        Some(score)
    } else {
        None
    }
}

/// Which half of the dual-mode "Ask Ochroma" surface is active.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PaletteMode {
    /// Fuzzy command search over the registry (the classic palette).
    #[default]
    Command,
    /// Natural-language intent: the typed sentence is parsed into actions and
    /// executed through the same registry + graph. "Ask Ochroma generates."
    Intent,
}

/// What the palette produced this frame, handed back to the shell to act on.
/// Command execution happens inside the palette (it has the registry); an intent
/// submission is returned as text because executing it needs `&mut EditorShell`.
#[derive(Debug, Clone, Default)]
pub enum PaletteOutcome {
    /// Nothing actionable this frame.
    #[default]
    None,
    /// A command was executed by Enter; carries its id (for tests/telemetry).
    CommandRun(String),
    /// An intent sentence was submitted (Enter in intent mode). The shell parses
    /// + executes it and appends the receipt to the assistant log.
    IntentSubmitted(String),
}

/// Live UI state of the open palette (query text + highlighted row + mode).
#[derive(Default)]
pub struct PaletteState {
    pub open: bool,
    pub query: String,
    pub selected: usize,
    /// Command vs intent mode (Tab toggles; a query that isn't a command prefix
    /// match also hints intent mode in the UI).
    pub mode: PaletteMode,
}

impl PaletteState {
    pub fn toggle(&mut self) {
        self.open = !self.open;
        if self.open {
            self.query.clear();
            self.selected = 0;
            self.mode = PaletteMode::Command;
        }
    }

    /// Render the centered dual-mode "Ask Ochroma" modal over `ctx`. Tab toggles
    /// command/intent mode. In command mode, Up/Down + Enter run the highlighted
    /// command (in place). In intent mode, Enter SUBMITS the typed sentence as a
    /// [`PaletteOutcome::IntentSubmitted`] for the shell to parse+execute. The
    /// `assistant_log` (newest last) renders as a receipt strip at the bottom in
    /// both modes — every executed intent leaves a human-readable line there.
    pub fn ui(
        &mut self,
        ctx: &egui::Context,
        t: &Tokens,
        registry: &CommandRegistry,
        assistant_log: &[String],
    ) -> PaletteOutcome {
        if !self.open {
            return PaletteOutcome::None;
        }

        // Keyboard: Esc closes; Tab toggles mode; Up/Down move; Enter executes.
        let (up, down, enter, esc, tab) = ctx.input(|i| {
            (
                i.key_pressed(egui::Key::ArrowUp),
                i.key_pressed(egui::Key::ArrowDown),
                i.key_pressed(egui::Key::Enter),
                i.key_pressed(egui::Key::Escape),
                i.key_pressed(egui::Key::Tab),
            )
        });
        if tab {
            self.mode = match self.mode {
                PaletteMode::Command => PaletteMode::Intent,
                PaletteMode::Intent => PaletteMode::Command,
            };
            self.selected = 0;
        }

        let results = registry.search(&self.query);
        let n = results.len();
        if self.mode == PaletteMode::Command {
            if down && n > 0 {
                self.selected = (self.selected + 1).min(n - 1);
            }
            if up {
                self.selected = self.selected.saturating_sub(1);
            }
            if self.selected >= n {
                self.selected = n.saturating_sub(1);
            }
        }

        if esc {
            self.open = false;
            return PaletteOutcome::None;
        }

        let mut outcome = PaletteOutcome::None;
        if enter {
            match self.mode {
                PaletteMode::Command => {
                    if let Some(c) = results.get(self.selected) {
                        (c.run)();
                        outcome = PaletteOutcome::CommandRun(c.id.clone());
                    }
                    self.open = false;
                }
                PaletteMode::Intent => {
                    let text = self.query.trim().to_string();
                    if !text.is_empty() {
                        outcome = PaletteOutcome::IntentSubmitted(text);
                    }
                    // Stay open in intent mode (a conversation) but clear the line.
                    self.query.clear();
                    self.selected = 0;
                }
            }
        }

        // Dim backdrop.
        let screen = ctx.screen_rect();
        egui::Area::new(egui::Id::new("palette_backdrop"))
            .fixed_pos(screen.min)
            .order(egui::Order::Foreground)
            .show(ctx, |ui| {
                ui.painter()
                    .rect_filled(screen, 0.0, egui::Color32::from_black_alpha(140));
            });

        // Centered modal.
        let modal_w = 560.0;
        let modal = egui::Rect::from_center_size(
            egui::pos2(screen.center().x, screen.top() + screen.height() * 0.30),
            egui::vec2(modal_w, 420.0),
        );
        let intent_mode = self.mode == PaletteMode::Intent;
        egui::Area::new(egui::Id::new("palette_modal"))
            .order(egui::Order::Foreground)
            .fixed_pos(modal.min)
            .show(ctx, |ui| {
                let [r, g, b, _] = t.color("surface.bg.2");
                egui::Frame::NONE
                    .fill(egui::Color32::from_rgb(r, g, b))
                    .corner_radius(t.radius[1])
                    .stroke(egui::Stroke::new(1.0, c32(t, "accent.base")))
                    .inner_margin(egui::Margin::same(12))
                    .show(ui, |ui| {
                        ui.set_width(modal_w - 24.0);
                        // Mode header + a hint that Tab toggles.
                        ui.horizontal(|ui| {
                            let title = if intent_mode {
                                "Ask Ochroma — Intent"
                            } else {
                                "Ask Ochroma — Command"
                            };
                            ui.label(
                                egui::RichText::new(format!(
                                    "{}  {title}",
                                    vox_ui::design::icons::icon::SEARCH
                                ))
                                .color(c32(t, if intent_mode { "accent.base" } else { "text.secondary" }))
                                .strong(),
                            );
                            ui.with_layout(
                                egui::Layout::right_to_left(egui::Align::Center),
                                |ui| {
                                    ui.label(
                                        egui::RichText::new("Tab: switch mode")
                                            .small()
                                            .color(c32(t, "text.secondary")),
                                    );
                                },
                            );
                        });
                        let hint = if intent_mode {
                            "Describe what you want… e.g. set terrain resolution to 128"
                        } else {
                            "Type a command…"
                        };
                        let edit = egui::TextEdit::singleline(&mut self.query)
                            .hint_text(hint)
                            .desired_width(modal_w - 24.0)
                            .font(egui::FontId::proportional(t.type_ramp.heading));
                        let resp = ui.add(edit);
                        resp.request_focus();
                        ui.separator();

                        if intent_mode {
                            ui.label(
                                egui::RichText::new(
                                    "Press Enter — I'll generate the edit, not just navigate.",
                                )
                                .color(c32(t, "text.secondary")),
                            );
                        } else {
                            // Command-mode result rows.
                            for (i, c) in results.iter().enumerate() {
                                let sel = i == self.selected;
                                let row = egui::Rect::from_min_size(
                                    ui.cursor().min,
                                    egui::vec2(modal_w - 24.0, 26.0),
                                );
                                if sel {
                                    ui.painter().rect_filled(row, t.radius[0], c32(t, "accent.dim"));
                                }
                                ui.horizontal(|ui| {
                                    ui.add_space(6.0);
                                    ui.label(
                                        egui::RichText::new(&c.title)
                                            .color(c32(t, "text.primary"))
                                            .strong(),
                                    );
                                    ui.with_layout(
                                        egui::Layout::right_to_left(egui::Align::Center),
                                        |ui| {
                                            if !c.shortcut.is_empty() {
                                                ui.label(
                                                    egui::RichText::new(&c.shortcut)
                                                        .monospace()
                                                        .color(c32(t, "text.secondary")),
                                                );
                                            }
                                            ui.label(
                                                egui::RichText::new(&c.category)
                                                    .small()
                                                    .color(c32(t, "text.secondary")),
                                            );
                                        },
                                    );
                                });
                                ui.add_space(2.0);
                            }
                            if results.is_empty() {
                                ui.label(
                                    egui::RichText::new("No matching command")
                                        .color(c32(t, "text.disabled")),
                                );
                            }
                        }

                        // === Assistant history strip (receipts) — both modes ===
                        if !assistant_log.is_empty() {
                            ui.add_space(6.0);
                            ui.separator();
                            ui.label(
                                egui::RichText::new(format!(
                                    "{}  Assistant",
                                    vox_ui::design::icons::icon::SEARCH
                                ))
                                .small()
                                .color(c32(t, "text.secondary")),
                            );
                            // Newest receipts last; show up to the last 4 on a raised
                            // chip so the receipt text region is clearly lit.
                            let start = assistant_log.len().saturating_sub(4);
                            for line in &assistant_log[start..] {
                                let chip = egui::Rect::from_min_size(
                                    ui.cursor().min,
                                    egui::vec2(modal_w - 24.0, 22.0),
                                );
                                let [cr, cg, cb, _] = t.color("surface.bg.3");
                                ui.painter().rect_filled(
                                    chip,
                                    t.radius[0],
                                    egui::Color32::from_rgb(cr, cg, cb),
                                );
                                ui.horizontal(|ui| {
                                    ui.add_space(6.0);
                                    ui.label(
                                        egui::RichText::new(line)
                                            .color(c32(t, "status.success"))
                                            .monospace(),
                                    );
                                });
                                ui.add_space(2.0);
                            }
                        }
                    });
            });

        outcome
    }
}

fn c32(t: &Tokens, key: &str) -> egui::Color32 {
    let [r, g, b, a] = t.color(key);
    egui::Color32::from_rgba_unmultiplied(r, g, b, a)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    fn registry_with_flag() -> (CommandRegistry, Rc<RefCell<bool>>) {
        let flag = Rc::new(RefCell::new(false));
        let mut reg = CommandRegistry::new();
        let f = flag.clone();
        reg.add(Command::new(
            "world.add",
            "Add to world",
            "Create",
            "Ctrl+A",
            move || *f.borrow_mut() = true,
        ));
        reg.add(Command::new("forge.terrain", "Forge: Terrain", "Forge", "", || {}));
        reg.add(Command::new("view.wireframe", "Toggle wireframe", "View", "", || {}));
        reg.add(Command::new("file.save", "Save world", "File", "Ctrl+S", || {}));
        (reg, flag)
    }

    #[test]
    fn fuzzy_addw_ranks_add_to_world_first() {
        let (reg, _flag) = registry_with_flag();
        let hits = reg.search("addw");
        assert!(!hits.is_empty(), "no hits for 'addw'");
        assert_eq!(
            hits[0].id, "world.add",
            "'addw' should rank 'Add to world' first, got {:?}",
            hits.iter().map(|c| &c.id).collect::<Vec<_>>()
        );
    }

    #[test]
    fn enter_fires_the_command_callback() {
        // The registry's run() executes the command's callback (the path Enter
        // takes): asserting the bound flag flips proves the AI-tool-call surface
        // actually executes, not just matches.
        let (reg, flag) = registry_with_flag();
        assert!(!*flag.borrow(), "flag must start false");
        let ran = reg.run("world.add");
        assert!(ran, "world.add must exist");
        assert!(*flag.borrow(), "running world.add must flip the flag");
    }

    #[test]
    fn menus_route_through_registry() {
        // A non-existent id returns false (menus can't invoke phantom actions);
        // a real id returns true. Proves the single dispatch surface.
        let (reg, _flag) = registry_with_flag();
        assert!(!reg.run("does.not.exist"));
        assert!(reg.run("file.save"));
    }

    #[test]
    fn empty_query_lists_all_in_order() {
        let (reg, _flag) = registry_with_flag();
        let hits = reg.search("");
        assert_eq!(hits.len(), 4);
        assert_eq!(hits[0].id, "world.add");
    }

    #[test]
    fn duplicate_id_registration_replaces_in_place() {
        // Registering a v2 command under an id already taken by v1 must REPLACE
        // v1 (not append a shadowed duplicate): run(id) fires v2's callback, and
        // search shows exactly one hit for that id.
        let which = Rc::new(RefCell::new(0u32));
        let mut reg = CommandRegistry::new();
        let w1 = which.clone();
        reg.add(Command::new(
            "tool.run",
            "Run Tool v1",
            "Tools",
            "",
            move || *w1.borrow_mut() = 1,
        ));
        let w2 = which.clone();
        reg.add(Command::new(
            "tool.run",
            "Run Tool v2",
            "Tools",
            "",
            move || *w2.borrow_mut() = 2,
        ));

        // Exactly one command carries this id (no shadowed duplicate).
        assert_eq!(
            reg.commands.iter().filter(|c| c.id == "tool.run").count(),
            1,
            "duplicate id must collapse to a single entry"
        );
        // The surviving entry is v2 (title swapped in place).
        let entry = reg.commands.iter().find(|c| c.id == "tool.run").unwrap();
        assert_eq!(entry.title, "Run Tool v2", "v2 must replace v1's metadata");

        // run(id) fires v2's callback, not v1's.
        assert!(reg.run("tool.run"));
        assert_eq!(*which.borrow(), 2, "run(id) must dispatch v2, got {}", *which.borrow());

        // search("Run Tool") returns exactly one hit (no duplicate row).
        let hits = reg.search("Run Tool");
        assert_eq!(
            hits.iter().filter(|c| c.id == "tool.run").count(),
            1,
            "search must show one hit for the replaced id"
        );
        assert_eq!(hits[0].title, "Run Tool v2");
    }
}
