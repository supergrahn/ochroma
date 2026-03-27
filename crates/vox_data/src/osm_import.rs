use std::collections::HashMap;

/// A single OSM node with geographic coordinates and optional tags.
#[derive(Debug, Clone)]
pub struct OsmNode {
    pub id: i64,
    pub lat: f64,
    pub lon: f64,
    pub tags: HashMap<String, String>,
}

/// An OSM way representing roads, building outlines, etc.
#[derive(Debug, Clone)]
pub struct OsmWay {
    pub id: i64,
    pub node_refs: Vec<i64>,
    pub tags: HashMap<String, String>,
}

/// Container for parsed OSM data.
#[derive(Debug, Clone)]
pub struct OsmData {
    pub nodes: Vec<OsmNode>,
    pub ways: Vec<OsmWay>,
}

impl OsmData {
    /// Look up a node by id.
    pub fn node_by_id(&self, id: i64) -> Option<&OsmNode> {
        self.nodes.iter().find(|n| n.id == id)
    }
}

/// A road extracted from OSM data.
#[derive(Debug, Clone)]
pub struct OsmRoad {
    pub name: String,
    pub road_type: String,
    pub points: Vec<(f32, f32)>,
}

/// A building extracted from OSM data.
#[derive(Debug, Clone)]
pub struct OsmBuilding {
    pub address: String,
    pub building_type: String,
    pub centroid: (f32, f32),
    pub footprint_area: f32,
}

/// Convert latitude/longitude to local metres using a simple Mercator projection
/// relative to an origin point.
pub fn lat_lon_to_local(lat: f64, lon: f64, origin_lat: f64, origin_lon: f64) -> (f32, f32) {
    const EARTH_RADIUS: f64 = 6_378_137.0; // WGS84 semi-major axis in metres

    let dlat = (lat - origin_lat).to_radians();
    let dlon = (lon - origin_lon).to_radians();
    let cos_lat = origin_lat.to_radians().cos();

    let x = (dlon * EARTH_RADIUS * cos_lat) as f32;
    let y = (dlat * EARTH_RADIUS) as f32;

    (x, y)
}

/// Parse a simplified OSM JSON format into `OsmData`.
///
/// Expected format:
/// ```json
/// {
///   "nodes": [{ "id": 1, "lat": 51.5, "lon": -0.1, "tags": { "name": "x" } }],
///   "ways": [{ "id": 100, "node_refs": [1, 2], "tags": { "highway": "residential" } }]
/// }
/// ```
pub fn parse_osm_json(json: &str) -> Result<OsmData, String> {
    let value: serde_json::Value =
        serde_json::from_str(json).map_err(|e| format!("JSON parse error: {e}"))?;

    let nodes_arr = value
        .get("nodes")
        .and_then(|v| v.as_array())
        .ok_or_else(|| "Missing or invalid 'nodes' array".to_string())?;

    let ways_arr = value
        .get("ways")
        .and_then(|v| v.as_array())
        .ok_or_else(|| "Missing or invalid 'ways' array".to_string())?;

    let mut nodes = Vec::with_capacity(nodes_arr.len());
    for node_val in nodes_arr {
        let id = node_val
            .get("id")
            .and_then(|v| v.as_i64())
            .ok_or_else(|| "Node missing 'id'".to_string())?;
        let lat = node_val
            .get("lat")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| format!("Node {id} missing 'lat'"))?;
        let lon = node_val
            .get("lon")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| format!("Node {id} missing 'lon'"))?;

        let mut tags = HashMap::new();
        if let Some(tags_obj) = node_val.get("tags").and_then(|v| v.as_object()) {
            for (k, v) in tags_obj {
                if let Some(s) = v.as_str() {
                    tags.insert(k.clone(), s.to_string());
                }
            }
        }

        nodes.push(OsmNode { id, lat, lon, tags });
    }

    let mut ways = Vec::with_capacity(ways_arr.len());
    for way_val in ways_arr {
        let id = way_val
            .get("id")
            .and_then(|v| v.as_i64())
            .ok_or_else(|| "Way missing 'id'".to_string())?;

        let node_refs = way_val
            .get("node_refs")
            .and_then(|v| v.as_array())
            .ok_or_else(|| format!("Way {id} missing 'node_refs'"))?
            .iter()
            .filter_map(|v| v.as_i64())
            .collect();

        let mut tags = HashMap::new();
        if let Some(tags_obj) = way_val.get("tags").and_then(|v| v.as_object()) {
            for (k, v) in tags_obj {
                if let Some(s) = v.as_str() {
                    tags.insert(k.clone(), s.to_string());
                }
            }
        }

        ways.push(OsmWay {
            id,
            node_refs,
            tags,
        });
    }

    Ok(OsmData { nodes, ways })
}

/// Extract roads from parsed OSM data (ways tagged with `highway=*`).
pub fn extract_roads(data: &OsmData) -> Vec<OsmRoad> {
    let mut roads = Vec::new();

    // Use the first node as the projection origin, or (0,0) if empty.
    let (origin_lat, origin_lon) = data
        .nodes
        .first()
        .map(|n| (n.lat, n.lon))
        .unwrap_or((0.0, 0.0));

    for way in &data.ways {
        if let Some(road_type) = way.tags.get("highway") {
            let name = way
                .tags
                .get("name")
                .cloned()
                .unwrap_or_default();

            let points: Vec<(f32, f32)> = way
                .node_refs
                .iter()
                .filter_map(|nid| data.node_by_id(*nid))
                .map(|n| lat_lon_to_local(n.lat, n.lon, origin_lat, origin_lon))
                .collect();

            roads.push(OsmRoad {
                name,
                road_type: road_type.clone(),
                points,
            });
        }
    }

    roads
}

/// Extract buildings from parsed OSM data (ways tagged with `building=*`).
/// Computes centroid and approximate footprint area using the shoelace formula.
pub fn extract_buildings(data: &OsmData) -> Vec<OsmBuilding> {
    let mut buildings = Vec::new();

    let (origin_lat, origin_lon) = data
        .nodes
        .first()
        .map(|n| (n.lat, n.lon))
        .unwrap_or((0.0, 0.0));

    for way in &data.ways {
        if let Some(building_type) = way.tags.get("building") {
            let address = way
                .tags
                .get("addr:street")
                .or_else(|| way.tags.get("name"))
                .cloned()
                .unwrap_or_default();

            let points: Vec<(f32, f32)> = way
                .node_refs
                .iter()
                .filter_map(|nid| data.node_by_id(*nid))
                .map(|n| lat_lon_to_local(n.lat, n.lon, origin_lat, origin_lon))
                .collect();

            if points.is_empty() {
                continue;
            }

            // Compute centroid.
            let n = points.len() as f32;
            let cx = points.iter().map(|(x, _)| x).sum::<f32>() / n;
            let cy = points.iter().map(|(_, y)| y).sum::<f32>() / n;

            // Compute area via shoelace formula.
            let mut area = 0.0f32;
            for i in 0..points.len() {
                let j = (i + 1) % points.len();
                area += points[i].0 * points[j].1;
                area -= points[j].0 * points[i].1;
            }
            let footprint_area = area.abs() / 2.0;

            buildings.push(OsmBuilding {
                address,
                building_type: building_type.clone(),
                centroid: (cx, cy),
                footprint_area,
            });
        }
    }

    buildings
}
