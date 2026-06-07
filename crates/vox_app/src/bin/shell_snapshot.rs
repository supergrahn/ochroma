//! Headless snapshot of the Ochroma editor SHELL.
//!
//! Renders the full docked shell (menu bar + Phosphor icon toolbar + tabbed
//! dock panels + status bar) at 1920x1080 to a PNG, with NO GPU and NO window —
//! the egui frame is tessellated and rasterized on the CPU (see
//! `shell::cpu_render`). The output is dark, tokenized, icon-led, and contains
//! ZERO bitmap glyphs (all text is egui's AA vector atlas).
//!
//! Usage:
//!   cargo run -p vox_app --bin shell_snapshot -- --shot /tmp/shell.png

use vox_app::shell::{cpu_render, EditorShell};
use vox_ui::Tokens;

fn main() {
    let mut shot_path = "/tmp/shell.png".to_string();
    let mut theme = "dark".to_string();
    let mut tab = String::new();
    let mut palette = false;
    let mut grow_tree = false;
    let mut forge_terrain = false;
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--shot" => {
                i += 1;
                if i < args.len() {
                    shot_path = args[i].clone();
                }
            }
            "--theme" => {
                i += 1;
                if i < args.len() {
                    theme = args[i].clone();
                }
            }
            "--tab" => {
                i += 1;
                if i < args.len() {
                    tab = args[i].clone();
                }
            }
            "--palette" => palette = true,
            "--grow-tree" => grow_tree = true,
            "--forge-terrain" => forge_terrain = true,
            _ => {}
        }
        i += 1;
    }

    let tokens = match theme.as_str() {
        "light" => Tokens::load("assets/ui/ochroma_light.theme.json")
            .unwrap_or_else(|_| Tokens::default()),
        _ => Tokens::load("assets/ui/ochroma.theme.json").unwrap_or_default(),
    };

    let w = 1920usize;
    let h = 1080usize;
    let bg = tokens.color("surface.bg.0");

    let mut shell = EditorShell::new(tokens.clone());
    // Install BOTH real plugins so their tabs + command categories coexist
    // (Crucible + Forge) — the two-plugin proof.
    shell.install_plugin(Box::new(vox_app::shell::plugins::CruciblePlugin::new()));
    // Forge wired to the shell's terrain-sink so "Raise terrain" plants real splats.
    shell.install_forge();
    // FloraPrime wired to the shell's grow-sink so "Grow tree" plants real splats.
    shell.install_floraprime();
    match tab.as_str() {
        "node_graph" => shell.focus_node_graph(),
        "content" => shell.focus_content(),
        "crucible" => shell.focus_plugin_tab(vox_app::shell::plugins::CRUCIBLE_TAB),
        "forge" => shell.focus_plugin_tab(vox_app::shell::plugins::FORGE_TAB),
        "floraprime" => shell.focus_plugin_tab(vox_app::shell::plugins::FLORAPRIME_TAB),
        // Default: the central tab is the REAL rendered viewport.
        _ => shell.focus_viewport(),
    }
    if forge_terrain {
        // Press "Raise terrain" headlessly: cook a heightfield patch and plant it
        // into the live world so the viewport shows its real landform splats.
        shell.raise_terrain_headless(0);
        eprintln!(
            "[shell_snapshot] raised terrain: {} things in the world, {} overlay splats",
            shell.entities.len(),
            shell.overlay.len()
        );
    }
    if grow_tree {
        // Press "Grow tree" headlessly: plant a Silver Birch (broadleaf, species 0)
        // into the live world so the viewport shows its real splats in the shot.
        shell.grow_tree_headless("Silver Birch", "broadleaf", 0);
        eprintln!(
            "[shell_snapshot] grew a tree: {} things in the world, {} overlay splats",
            shell.entities.len(),
            shell.overlay.len()
        );
    }
    if palette {
        // Drive a scripted intent so the assistant receipt strip is LIT in the
        // snapshot pixels (proves the generative loop end-to-end, not just the
        // empty palette). The receipt renders as e.g. "Set terrain.resolution 64 -> 128".
        let receipt = shell.run_intent("set terrain resolution to 128");
        eprintln!("[shell_snapshot] scripted intent receipt: {receipt}");
        shell.palette.mode = vox_app::shell::command_palette::PaletteMode::Intent;
        shell.open_palette();
    }
    let ctx = egui::Context::default();
    vox_ui::design::icons::install(&ctx);
    vox_ui::egui_theme::apply(&ctx, &tokens);

    let rgba = cpu_render::render_ui(&ctx, [w, h], bg, |ctx| {
        // Re-open the palette each frame (the snapshot harness runs >1 frame and
        // the modal would otherwise consume/close on a synthetic Enter).
        if palette {
            shell.palette.open = true;
        }
        shell.ui(ctx);
    });

    cpu_render::write_png(&shot_path, &rgba, w as u32, h as u32)
        .unwrap_or_else(|e| panic!("failed to write {shot_path}: {e}"));

    let bytes = std::fs::metadata(&shot_path).map(|m| m.len()).unwrap_or(0);
    let nonbg = cpu_render::non_background_fraction(&rgba, bg, 6) * 100.0;
    let tab_note = if tab.is_empty() { "default".to_string() } else { tab.clone() };
    println!(
        "[shell_snapshot] wrote {shot_path} ({} bytes), {nonbg:.1}% non-background pixels, theme={theme}, tab={tab_note}, palette={palette}, 1920x1080",
        bytes
    );
    println!(
        "[shell_snapshot] dock layout: left=World | center-top=Viewport center-bottom=Node Graph | right=Properties | bottom=Content+Output Log; menu bar + Phosphor icon toolbar + status bar 'All systems healthy'"
    );
}
