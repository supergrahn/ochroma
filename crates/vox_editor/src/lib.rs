// Editor crate — node graph, gizmos, terrain editor, viewport UI, content browser.
pub mod content_browser;
pub mod gizmo_interaction;
pub mod node_graph;
pub mod node_thumbnail;
pub mod nodes;
pub mod registry;
pub mod subgraph;
pub mod templates;
pub mod editor_panel;
// Node-graph editor windows, moved out of vox_render (the renderer must not pull
// the UI/node-DAG stack). Gated on `crucible` because OchrGraph + the material
// node library live behind vox_nodes' crucible backend.
#[cfg(feature = "crucible")]
pub mod anim_editor_ui;
#[cfg(feature = "crucible")]
pub mod material_editor_ui;
#[cfg(feature = "crucible")]
pub mod vfx_editor_ui;
mod registry_tests;
