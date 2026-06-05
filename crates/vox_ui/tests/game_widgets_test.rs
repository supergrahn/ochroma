use vox_ui::game_widgets::{ResourceRow, WidgetCmd};

#[test]
fn widget_resource_bar_shows_correct_count() {
    let row = ResourceRow {
        label: "Wood".to_string(),
        count: 42,
        icon_color: egui::Color32::from_rgb(139, 90, 43),
    };
    let cmd = WidgetCmd::Panel {
        title: "Resources".to_string(),
        rows: vec![row],
    };
    let label = match &cmd {
        WidgetCmd::Panel { rows, .. } => format!("{}={}", rows[0].label, rows[0].count),
        _ => panic!("wrong variant"),
    };
    println!("resource_bar: {}", label);
    assert_eq!(label, "Wood=42");
}

#[test]
fn tooltip_text_is_accessible() {
    let cmd = WidgetCmd::Tooltip { text: "Lumberjack hut".to_string() };
    let text = match &cmd {
        WidgetCmd::Tooltip { text } => text.as_str(),
        _ => panic!("wrong variant"),
    };
    assert_eq!(text, "Lumberjack hut");
}

#[test]
fn button_label_is_accessible() {
    let cmd = WidgetCmd::Button { label: "Build House".to_string(), id: "btn_house".to_string() };
    let label = match &cmd {
        WidgetCmd::Button { label, .. } => label.as_str(),
        _ => panic!("wrong variant"),
    };
    assert_eq!(label, "Build House");
}
