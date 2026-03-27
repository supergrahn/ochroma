use vox_nn::street_layout::*;

#[test]
fn generate_road_graph_has_main_street() {
    let graph = generate_road_graph(100.0, 8.0, 25.0, 42);
    assert!(graph.nodes.len() >= 2, "Need at least start and end");
    assert!(!graph.edges.is_empty());
    assert!(graph.edges.iter().any(|e| e.road_type == RoadType::MainStreet));
}

#[test]
fn generate_road_graph_has_side_streets() {
    let graph = generate_road_graph(100.0, 8.0, 25.0, 42);
    let side_count = graph.edges.iter().filter(|e| e.road_type == RoadType::SideStreet).count();
    assert!(side_count >= 2, "Expected side streets, got {}", side_count);
}

#[test]
fn generate_road_graph_is_deterministic() {
    let a = generate_road_graph(100.0, 8.0, 25.0, 42);
    let b = generate_road_graph(100.0, 8.0, 25.0, 42);
    assert_eq!(a.nodes.len(), b.nodes.len());
    assert_eq!(a.edges.len(), b.edges.len());
}

#[test]
fn building_plots_generated_along_roads() {
    let graph = generate_road_graph(100.0, 8.0, 25.0, 42);
    let plots = generate_building_plots(&graph, 6.0, 12.0);
    assert!(plots.len() > 10, "Expected many plots, got {}", plots.len());
}

#[test]
fn building_plots_face_road() {
    let graph = generate_road_graph(100.0, 8.0, 25.0, 42);
    let plots = generate_building_plots(&graph, 6.0, 12.0);
    // All plots should have a valid facing angle
    for plot in &plots {
        assert!(plot.facing_angle.is_finite());
    }
}
