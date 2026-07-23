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
    ReadyMegaGeometry::new(
        [7; 32],
        1,
        2,
        vec![0, 1],
        [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ],
        program,
        deformation,
        template,
        vec![0],
        vec![-1.0, -1.0, -1.0, 1.0, 1.0, 1.0],
        vec![0, 0, 0, 0, 0, 1],
        vec![u32::MAX, u32::MAX, u32::MAX, 0, 1, (-1.0f32).to_bits(), (-1.0f32).to_bits(), (-1.0f32).to_bits(), 1.0f32.to_bits(), 1.0f32.to_bits(), 1.0f32.to_bits()],
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
