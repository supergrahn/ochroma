use vox_data::mega_geometry::ReadyMegaGeometry;

fn fixture() -> ReadyMegaGeometry {
    let mut program = vec![0_u32; 32];
    program[0] = 1;
    program[1] = 128;
    program[2] = 1;
    program[9] = 0;
    program[14] = 0;
    program[15] = 1;
    program[16] = 0;
    program[17] = 1;
    program[31] = 1;
    let mut deformation = vec![0_u32; 16];
    deformation[0] = 1;
    deformation[1] = 64;
    let mut template = vec![0_u32; 28];
    template[0] = 1;
    template[1] = 112;
    template[2] = 1;
    ReadyMegaGeometry::from_certified_programs(
        1,
        2,
        vec![0, 1],
        [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ],
        program,
        deformation,
        template,
        vec![0],
        vec![-1.0, -1.0, -1.0, 1.0, 1.0, 1.0],
        vec![
            1,
            64,
            0,
            (8 << 16) | 1,
            0,
            0,
            0,
            1,
            0,
            1,
            (-1.0f32).to_bits(),
            (-1.0f32).to_bits(),
            (-1.0f32).to_bits(),
            1.0f32.to_bits(),
            1.0f32.to_bits(),
            1.0f32.to_bits(),
        ],
        vec![],
        vec![1, 32, 0, 0, 0, 14, 1.0e-5f32.to_bits(), 1],
        vec![
            -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 2.0,
        ],
    )
    .unwrap()
}

#[test]
fn ready_mega_geometry_round_trips_deterministically() {
    let source = fixture();
    let first = serde_json::to_vec(&source).unwrap();
    let decoded: ReadyMegaGeometry = serde_json::from_slice(&first).unwrap();
    decoded.validate().unwrap();
    assert_eq!(serde_json::to_vec(&decoded).unwrap(), first);
    assert_eq!(decoded.source_triangle_count(), 2);
}

#[test]
fn schema_6_payload_without_visibility_fabric_remains_native_scalar_compatible() {
    let mut value = serde_json::to_value(fixture()).unwrap();
    value["schema"] = serde_json::json!(6_u32);
    let object = value.as_object_mut().unwrap();
    object.remove("visibility_cell_words");
    object.remove("visibility_cell_children");
    object.remove("visibility_coefficient_words");
    object.remove("visibility_coefficient_parameters");

    let payload: ReadyMegaGeometry = serde_json::from_value(value).unwrap();
    payload.validate().unwrap();
    assert!(payload.visibility_cell_words().is_empty());
    assert!(payload.visibility_coefficient_words().is_empty());
}

#[test]
fn schema_6_payload_cannot_smuggle_partial_visibility_fabric_state() {
    let mut value = serde_json::to_value(fixture()).unwrap();
    value["schema"] = serde_json::json!(6_u32);
    value["visibility_coefficient_words"] = serde_json::json!([]);

    let payload: ReadyMegaGeometry = serde_json::from_value(value).unwrap();
    assert!(payload.validate().is_err());
}

#[test]
fn unreleased_schema_7_visibility_coefficients_require_runtime_rederivation() {
    let mut value = serde_json::to_value(fixture()).unwrap();
    value["schema"] = serde_json::json!(7_u32);

    let payload: ReadyMegaGeometry = serde_json::from_value(value).unwrap();
    let error = payload.validate().unwrap_err().to_string();
    assert!(
        error.contains("schema 7"),
        "stale schema must produce an actionable recook boundary: {error}"
    );
}

#[test]
fn ochroma_derives_runtime_schedule_and_content_address() {
    let first = fixture();
    let second = fixture();
    assert_eq!(first.content_hash(), second.content_hash());
    assert_eq!(first.surface_correspondence_words(), &[0, 0, 0, 0, 0, 1]);
    assert_eq!(first.page_hierarchy_words()[0], u32::MAX);
}

#[test]
fn selectors_cannot_escape_a_program_template_dictionary() {
    let mut value = serde_json::to_value(fixture()).unwrap();
    value["selectors"] = serde_json::json!([1]);
    let payload: ReadyMegaGeometry = serde_json::from_value(value).unwrap();
    assert!(payload.validate().is_err());
}

#[test]
fn page_hierarchy_must_remain_a_complete_tree() {
    let mut value = serde_json::to_value(fixture()).unwrap();
    value["page_hierarchy_words"][0] = serde_json::json!(0_u32);
    let payload: ReadyMegaGeometry = serde_json::from_value(value).unwrap();
    assert!(payload.validate().is_err());
}
