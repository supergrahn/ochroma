use vox_data::osm_import::*;

const SAMPLE_JSON: &str = r#"{
    "nodes": [
        { "id": 1, "lat": 51.5074, "lon": -0.1278, "tags": {} },
        { "id": 2, "lat": 51.5080, "lon": -0.1270, "tags": {} }
    ],
    "ways": [
        {
            "id": 100,
            "node_refs": [1, 2],
            "tags": { "highway": "residential", "name": "Baker Street" }
        }
    ]
}"#;

const BUILDING_JSON: &str = r#"{
    "nodes": [
        { "id": 1, "lat": 51.5074, "lon": -0.1278, "tags": {} },
        { "id": 2, "lat": 51.5074, "lon": -0.1268, "tags": {} },
        { "id": 3, "lat": 51.5084, "lon": -0.1268, "tags": {} },
        { "id": 4, "lat": 51.5084, "lon": -0.1278, "tags": {} }
    ],
    "ways": [
        {
            "id": 200,
            "node_refs": [1, 2, 3, 4],
            "tags": { "building": "yes", "addr:street": "221B Baker Street" }
        }
    ]
}"#;

#[test]
fn parse_json_nodes_and_ways() {
    let data = parse_osm_json(SAMPLE_JSON).unwrap();
    assert_eq!(data.nodes.len(), 2);
    assert_eq!(data.ways.len(), 1);
    assert_eq!(data.nodes[0].id, 1);
    assert!((data.nodes[0].lat - 51.5074).abs() < 1e-4);
    assert_eq!(data.ways[0].node_refs, vec![1, 2]);
}

#[test]
fn extract_road_from_data() {
    let data = parse_osm_json(SAMPLE_JSON).unwrap();
    let roads = extract_roads(&data);
    assert_eq!(roads.len(), 1);
    assert_eq!(roads[0].name, "Baker Street");
    assert_eq!(roads[0].road_type, "residential");
    assert_eq!(roads[0].points.len(), 2);
    // First point should be at origin (0,0) since it IS the origin node.
    assert!(roads[0].points[0].0.abs() < 0.01);
    assert!(roads[0].points[0].1.abs() < 0.01);
}

#[test]
fn extract_building_from_data() {
    let data = parse_osm_json(BUILDING_JSON).unwrap();
    let buildings = extract_buildings(&data);
    assert_eq!(buildings.len(), 1);
    assert_eq!(buildings[0].address, "221B Baker Street");
    assert_eq!(buildings[0].building_type, "yes");
    // Footprint area should be positive and reasonable (roughly 70m x 110m).
    assert!(buildings[0].footprint_area > 0.0);
}

#[test]
fn coordinate_conversion() {
    // Same point as origin should yield (0,0).
    let (x, y) = lat_lon_to_local(51.5074, -0.1278, 51.5074, -0.1278);
    assert!(x.abs() < 0.001);
    assert!(y.abs() < 0.001);

    // Moving ~111m north (roughly 0.001 degree latitude).
    let (x2, y2) = lat_lon_to_local(51.5084, -0.1278, 51.5074, -0.1278);
    assert!(x2.abs() < 0.01); // No east-west movement.
    assert!(y2 > 100.0 && y2 < 120.0); // ~111m north.
}

#[test]
fn empty_data_handling() {
    let json = r#"{ "nodes": [], "ways": [] }"#;
    let data = parse_osm_json(json).unwrap();
    assert!(data.nodes.is_empty());
    assert!(data.ways.is_empty());
    assert!(extract_roads(&data).is_empty());
    assert!(extract_buildings(&data).is_empty());
}

#[test]
fn invalid_json_returns_error() {
    assert!(parse_osm_json("not json").is_err());
    assert!(parse_osm_json(r#"{ "nodes": "bad" }"#).is_err());
}
