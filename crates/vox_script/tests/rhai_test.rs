use vox_script::rhai_runtime::RhaiRuntime;

#[test]
fn load_and_run_script() {
    let mut rt = RhaiRuntime::new();
    let idx = rt.load_script("test", r#"
        log("Hello from Rhai!");
        let x = 40 + 2;
        x
    "#).unwrap();
    assert_eq!(rt.script_count(), 1);
    rt.run(idx).unwrap();
}

#[test]
fn call_engine_functions() {
    let mut rt = RhaiRuntime::new();
    rt.load_script("test", r#"
        fn on_start() {
            log("Script started");
            play_sound("click.wav", 0.8);
        }

        fn on_update(dt) {
            let speed = 5.0;
            set_position(speed * dt, 0.0, 0.0);
        }
    "#).unwrap();
    // Scripts compile without error
    assert_eq!(rt.script_count(), 1);
}

#[test]
fn eval_expression() {
    let rt = RhaiRuntime::new();
    let result = rt.eval("40 + 2").unwrap();
    assert_eq!(result, "42");
}

#[test]
fn distance_function() {
    let rt = RhaiRuntime::new();
    let result = rt.eval("distance(0.0, 0.0, 0.0, 3.0, 4.0, 0.0)").unwrap();
    assert_eq!(result, "5.0");
}

#[test]
fn hot_reload_from_file() {
    let dir = std::env::temp_dir().join("ochroma_rhai_test");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let path = dir.join("test_script.rhai");
    std::fs::write(&path, r#"log("version 1");"#).unwrap();

    let mut rt = RhaiRuntime::new();
    let idx = rt.load_script_file("test", &path).unwrap();
    rt.run(idx).unwrap();

    // Modify the file
    std::fs::write(&path, r#"log("version 2");"#).unwrap();

    // Hot reload
    rt.reload(idx).unwrap();
    rt.run(idx).unwrap(); // Should print "version 2"

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn invalid_script_returns_error() {
    let mut rt = RhaiRuntime::new();
    let result = rt.load_script("bad", "this is not valid rhai {{{{");
    assert!(result.is_err());
}

#[test]
fn script_names_listed() {
    let mut rt = RhaiRuntime::new();
    rt.load_script("alpha", "let x = 1;").unwrap();
    rt.load_script("beta", "let y = 2;").unwrap();
    let names = rt.script_names();
    assert!(names.contains(&"alpha"));
    assert!(names.contains(&"beta"));
}
