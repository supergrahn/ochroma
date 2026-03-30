//! hello_splat — minimal entry point for the WebGPU browser demo.

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(start)]
pub fn wasm_main() {
    console_error_panic_hook::set_once();
    web_sys::console::log_1(&"Ochroma hello_splat loaded".into());
}

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    println!("hello_splat: native mode.");
}
