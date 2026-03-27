use vox_nn::nl_commands::{parse_command, ParsedCommand};

#[test]
fn build_near_school() {
    let cmd = parse_command("build a park near the school").unwrap();
    match cmd {
        ParsedCommand::BuildStructure { what, where_near } => {
            assert_eq!(what, "park");
            assert_eq!(where_near.unwrap(), "school");
        }
        _ => panic!("Expected BuildStructure, got {:?}", cmd),
    }
}

#[test]
fn place_structure_no_location() {
    let cmd = parse_command("build hospital").unwrap();
    match cmd {
        ParsedCommand::BuildStructure { what, where_near } => {
            assert_eq!(what, "hospital");
            assert!(where_near.is_none());
        }
        _ => panic!("Expected BuildStructure"),
    }
}

#[test]
fn zone_residential() {
    let cmd = parse_command("zone residential").unwrap();
    match cmd {
        ParsedCommand::ModifyZone {
            zone_type,
            location,
        } => {
            assert_eq!(zone_type, "residential");
            assert!(location.is_none());
        }
        _ => panic!("Expected ModifyZone, got {:?}", cmd),
    }
}

#[test]
fn zone_area_as_type() {
    let cmd = parse_command("zone downtown as commercial").unwrap();
    match cmd {
        ParsedCommand::ModifyZone {
            zone_type,
            location,
        } => {
            assert_eq!(zone_type, "commercial");
            assert_eq!(location.unwrap(), "downtown");
        }
        _ => panic!("Expected ModifyZone"),
    }
}

#[test]
fn show_traffic_overlay() {
    let cmd = parse_command("show traffic overlay").unwrap();
    match cmd {
        ParsedCommand::ShowOverlay { overlay_type } => {
            assert!(overlay_type.contains("traffic"));
        }
        _ => panic!("Expected ShowOverlay, got {:?}", cmd),
    }
}

#[test]
fn show_traffic_shorthand() {
    let cmd = parse_command("show traffic").unwrap();
    match cmd {
        ParsedCommand::ShowOverlay { overlay_type } => {
            assert_eq!(overlay_type, "traffic");
        }
        _ => panic!("Expected ShowOverlay, got {:?}", cmd),
    }
}

#[test]
fn go_to_downtown() {
    let cmd = parse_command("go to downtown").unwrap();
    match cmd {
        ParsedCommand::CameraMove {
            target_description,
        } => {
            assert_eq!(target_description, "downtown");
        }
        _ => panic!("Expected CameraMove, got {:?}", cmd),
    }
}

#[test]
fn show_me_location() {
    let cmd = parse_command("show me the harbor").unwrap();
    match cmd {
        ParsedCommand::CameraMove {
            target_description,
        } => {
            assert_eq!(target_description, "harbor");
        }
        _ => panic!("Expected CameraMove, got {:?}", cmd),
    }
}

#[test]
fn why_citizens_unhappy() {
    let cmd = parse_command("why are citizens unhappy").unwrap();
    match cmd {
        ParsedCommand::QueryInfo { question } => {
            assert!(question.contains("unhappy"));
        }
        _ => panic!("Expected QueryInfo, got {:?}", cmd),
    }
}

#[test]
fn what_query() {
    let cmd = parse_command("what is the population").unwrap();
    match cmd {
        ParsedCommand::QueryInfo { question } => {
            assert!(question.contains("population"));
        }
        _ => panic!("Expected QueryInfo"),
    }
}

#[test]
fn unrecognised_returns_none() {
    assert!(parse_command("asdf gibberish xyz").is_none());
}

#[test]
fn empty_returns_none() {
    assert!(parse_command("").is_none());
    assert!(parse_command("   ").is_none());
}
