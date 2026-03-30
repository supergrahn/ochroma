//! Taffy-backed flexbox layout for vox_ui game widgets.

#[cfg(feature = "game-ui")]
use taffy::prelude::*;

#[cfg(feature = "game-ui")]
pub type LayoutNodeId = NodeId;

#[cfg(not(feature = "game-ui"))]
pub type LayoutNodeId = u64;

#[cfg(feature = "game-ui")]
pub struct LayoutTree {
    tree: TaffyTree,
    root: Option<NodeId>,
}

#[cfg(feature = "game-ui")]
impl LayoutTree {
    pub fn new() -> Self {
        Self { tree: TaffyTree::new(), root: None }
    }

    /// Create a root row-flex container. Sets it as the tree root.
    pub fn add_row_container(&mut self, width: f32, height: f32) -> LayoutNodeId {
        let node = self.tree.new_with_children(
            Style {
                display: Display::Flex,
                flex_direction: FlexDirection::Row,
                size: Size {
                    width:  Dimension::Length(width),
                    height: Dimension::Length(height),
                },
                ..Default::default()
            },
            &[],
        ).expect("taffy row container");
        self.root = Some(node);
        node
    }

    /// Add a flex child that grows proportionally.
    pub fn add_flex_child(&mut self, parent: LayoutNodeId, flex_grow: f32) -> LayoutNodeId {
        let child = self.tree.new_leaf(Style {
            flex_grow,
            size: Size { width: Dimension::Auto, height: Dimension::Auto },
            ..Default::default()
        }).expect("taffy flex child");
        self.tree.add_child(parent, child).expect("add child");
        child
    }

    /// Resolve layout for the entire tree.
    pub fn resolve(&mut self, viewport_width: f32, viewport_height: f32) {
        if let Some(root) = self.root {
            self.tree.compute_layout(
                root,
                Size {
                    width:  AvailableSpace::Definite(viewport_width),
                    height: AvailableSpace::Definite(viewport_height),
                },
            ).expect("taffy compute_layout");
        }
    }

    /// Resolved rect for a node: [x, y, width, height].
    pub fn rect(&self, node: LayoutNodeId) -> Option<[f32; 4]> {
        let layout = self.tree.layout(node).ok()?;
        Some([
            layout.location.x,
            layout.location.y,
            layout.size.width,
            layout.size.height,
        ])
    }
}

#[cfg(feature = "game-ui")]
impl Default for LayoutTree {
    fn default() -> Self { Self::new() }
}

#[cfg(not(feature = "game-ui"))]
pub struct LayoutTree;

#[cfg(not(feature = "game-ui"))]
impl LayoutTree {
    pub fn new() -> Self { Self }
    pub fn add_row_container(&mut self, _w: f32, _h: f32) -> LayoutNodeId { 0 }
    pub fn add_flex_child(&mut self, _parent: LayoutNodeId, _flex_grow: f32) -> LayoutNodeId { 0 }
    pub fn resolve(&mut self, _w: f32, _h: f32) {}
    pub fn rect(&self, _node: LayoutNodeId) -> Option<[f32; 4]> { Some([0.0; 4]) }
}

#[cfg(not(feature = "game-ui"))]
impl Default for LayoutTree {
    fn default() -> Self { Self::new() }
}

#[cfg(all(test, feature = "game-ui"))]
mod tests {
    use super::*;

    #[test]
    fn layout_tree_root_fills_container() {
        let mut tree = LayoutTree::new();
        let root = tree.add_row_container(800.0, 600.0);
        tree.resolve(800.0, 600.0);
        let rect = tree.rect(root).expect("root rect");
        assert!((rect[2] - 800.0).abs() < 1.0, "root width={}", rect[2]);
        assert!((rect[3] - 600.0).abs() < 1.0, "root height={}", rect[3]);
    }

    #[test]
    fn two_children_share_width() {
        let mut tree = LayoutTree::new();
        let root = tree.add_row_container(400.0, 100.0);
        let a    = tree.add_flex_child(root, 1.0);
        let b    = tree.add_flex_child(root, 1.0);
        tree.resolve(400.0, 100.0);
        let ra   = tree.rect(a).unwrap();
        let rb   = tree.rect(b).unwrap();
        println!("child_a_width={} child_b_width={}", ra[2], rb[2]);
        assert!((ra[2] - 200.0).abs() < 2.0, "child a width={}", ra[2]);
        assert!((rb[2] - 200.0).abs() < 2.0, "child b width={}", rb[2]);
    }

    #[test]
    fn resolve_twice_is_stable() {
        let mut tree = LayoutTree::new();
        let root = tree.add_row_container(640.0, 480.0);
        tree.resolve(640.0, 480.0);
        let r1 = tree.rect(root).unwrap();
        tree.resolve(640.0, 480.0);
        let r2 = tree.rect(root).unwrap();
        assert!((r1[2] - r2[2]).abs() < 1e-4);
    }
}
