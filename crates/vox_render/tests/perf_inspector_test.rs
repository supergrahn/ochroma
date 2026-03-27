use vox_render::perf_inspector::*;

#[test]
fn record_frames() {
    let mut inspector = PerfInspector::new();
    assert_eq!(inspector.frame_count(), 0);

    inspector.record_frame(PerfSnapshot {
        frame: FrameBreakdown {
            total_ms: 16.0,
            sort_ms: 2.0,
            cull_ms: 1.0,
            render_ms: 10.0,
            ui_ms: 2.0,
            sim_ms: 1.0,
        },
        vram: VramBreakdown::default(),
        entities: EntityBreakdown::default(),
    });

    assert_eq!(inspector.frame_count(), 1);
    assert!(inspector.latest().is_some());
}

#[test]
fn average_calculation() {
    let mut inspector = PerfInspector::new();

    // Record 3 frames with different timings.
    for total in [12.0f32, 15.0, 18.0] {
        inspector.record_frame(PerfSnapshot {
            frame: FrameBreakdown {
                total_ms: total,
                sort_ms: total * 0.1,
                cull_ms: total * 0.05,
                render_ms: total * 0.6,
                ui_ms: total * 0.15,
                sim_ms: total * 0.1,
            },
            vram: VramBreakdown::default(),
            entities: EntityBreakdown::default(),
        });
    }

    let avg = inspector.average_over(3);
    assert!((avg.total_ms - 15.0).abs() < 0.01);
    assert!((avg.sort_ms - 1.5).abs() < 0.01);

    // Average over more frames than available should still work.
    let avg_all = inspector.average_over(100);
    assert!((avg_all.total_ms - 15.0).abs() < 0.01);
}

#[test]
fn json_export() {
    let mut inspector = PerfInspector::new();
    inspector.record_frame(PerfSnapshot {
        frame: FrameBreakdown {
            total_ms: 16.0,
            sort_ms: 2.0,
            cull_ms: 1.0,
            render_ms: 10.0,
            ui_ms: 2.0,
            sim_ms: 1.0,
        },
        vram: VramBreakdown {
            total_mb: 512.0,
            splats_mb: 256.0,
            textures_mb: 128.0,
            buffers_mb: 128.0,
        },
        entities: EntityBreakdown {
            total: 1000,
            buildings: 200,
            citizens: 500,
            vehicles: 100,
            trees: 150,
            props: 50,
        },
    });

    let json = inspector.export_json();
    assert!(json.contains("total_ms"));
    assert!(json.contains("splats_mb"));
    assert!(json.contains("buildings"));

    // Valid JSON.
    let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert!(parsed.is_array());
}

#[test]
fn breakdown_sum_matches_total() {
    let breakdown = FrameBreakdown {
        total_ms: 16.0,
        sort_ms: 2.0,
        cull_ms: 1.0,
        render_ms: 10.0,
        ui_ms: 2.0,
        sim_ms: 1.0,
    };

    assert!((breakdown.component_sum() - breakdown.total_ms).abs() < 0.01);
}

#[test]
fn toggle_visibility() {
    let mut inspector = PerfInspector::new();
    assert!(!inspector.visible);

    inspector.toggle();
    assert!(inspector.visible);

    inspector.toggle();
    assert!(!inspector.visible);
}

#[test]
fn empty_average_returns_default() {
    let inspector = PerfInspector::new();
    let avg = inspector.average_over(10);
    assert!((avg.total_ms - 0.0).abs() < 0.001);
}
