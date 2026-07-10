fn main() {
    // The image_dds `encode` feature pulls intel_tex_2 (ISPC BCn encoder),
    // whose prebuilt objects need the C++ runtime (`__gxx_personality_v0`).
    // Same fix urban_horizon's build.rs applies for its own image_dds encode
    // dependency. MSVC targets link the C++ runtime automatically.
    let target = std::env::var("TARGET").unwrap_or_default();
    if target.contains("linux") || target.contains("windows-gnu") {
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }
}
