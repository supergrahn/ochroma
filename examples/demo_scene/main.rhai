// Demo scene script — loaded by engine at startup when --demo flag is passed.
// Demonstrates: scripted entity movement, sound playback, log output, hot-reload.
//
// Edit this file while `cargo run --bin ochroma -- --demo` is running.
// Changes apply within 1 second (hot-reload via mtime polling).

let frame = 0;

fn on_start() {
    log("Demo scene started! Edit this file to see hot-reload in action.");
}

fn on_update(dt) {
    frame += 1;
    if frame % 120 == 0 {
        log("Demo: frame " + frame);
    }
}
