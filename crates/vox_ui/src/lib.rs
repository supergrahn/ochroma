pub mod canvas;
pub mod game_hud;
pub mod game_menu;
pub mod game_widgets;
pub mod layout;
pub mod node_graph_widget;
pub mod spectral_hud;
pub mod text;
pub mod theme;
pub mod ui_tree;
pub mod vello_ctx;

pub use game_hud::GameHud;
pub use game_menu::GameMenu;
pub use game_widgets::{GameWidgets, ResourceRow, WidgetCmd};
pub use layout::{LayoutTree, LayoutNodeId};
pub use spectral_hud::{SpectralHUD, SpectralRadianceCache};
// The single retained, styleable, hot-reloadable game-UI tree (the rank-#11
// convergence target). Replaces the former crate-root `UiRoot`/`UiNode`/`Theme`
// stub enum, which were an unused parallel sketch of the same idea.
pub use ui_tree::{
    click, compute_layout, hit_test, rasterize_into, Anchor, Edges, FlexDir, Layout, ResolvedStyle,
    Style, StyleSheet, UiDoc, UiEvent, UiKind, UiNode, UiTree,
};
