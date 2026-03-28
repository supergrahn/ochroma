use std::path::Path;
use vox_tools::build::{BuildConfig, BuildManifest, BuildTarget};
use vox_tools::turnaround::run_turnaround;

// --- BuildManifest tests ---

#[test]
fn windows_binary_has_exe_extension() {
    let manifest = BuildManifest::new("MyGame", BuildTarget::Windows, BuildConfig::Release);
    assert_eq!(manifest.output_binary_name(), "MyGame.exe");
}

#[test]
fn linux_binary_has_no_extension() {
    let manifest = BuildManifest::new("MyGame", BuildTarget::Linux, BuildConfig::Release);
    assert_eq!(manifest.output_binary_name(), "MyGame");
}

#[test]
fn windows_target_triple() {
    let manifest = BuildManifest::new("Game", BuildTarget::Windows, BuildConfig::Release);
    assert_eq!(manifest.cargo_target_triple(), "x86_64-pc-windows-msvc");
}

#[test]
fn linux_target_triple() {
    let manifest = BuildManifest::new("Game", BuildTarget::Linux, BuildConfig::Release);
    assert_eq!(manifest.cargo_target_triple(), "x86_64-unknown-linux-gnu");
}

#[test]
fn macos_target_triple() {
    let manifest = BuildManifest::new("Game", BuildTarget::MacOS, BuildConfig::Release);
    assert_eq!(manifest.cargo_target_triple(), "aarch64-apple-darwin");
}

#[test]
fn steamos_target_triple_matches_linux() {
    let manifest = BuildManifest::new("Game", BuildTarget::SteamOS, BuildConfig::Release);
    assert_eq!(manifest.cargo_target_triple(), "x86_64-unknown-linux-gnu");
}

#[test]
fn debug_profile_is_dev() {
    let manifest = BuildManifest::new("Game", BuildTarget::Linux, BuildConfig::Debug);
    assert_eq!(manifest.cargo_profile(), "dev");
}

#[test]
fn release_profile_is_release() {
    let manifest = BuildManifest::new("Game", BuildTarget::Linux, BuildConfig::Release);
    assert_eq!(manifest.cargo_profile(), "release");
}

#[test]
fn shipping_profile_is_release() {
    let manifest = BuildManifest::new("Game", BuildTarget::Linux, BuildConfig::Shipping);
    assert_eq!(manifest.cargo_profile(), "release");
}

#[test]
fn build_command_contains_target_and_profile() {
    let manifest = BuildManifest::new("Game", BuildTarget::Windows, BuildConfig::Debug);
    let cmd = manifest.build_command();
    assert!(cmd.contains("x86_64-pc-windows-msvc"));
    assert!(cmd.contains("dev"));
    assert!(cmd.contains("cargo build"));
}

#[test]
fn manifest_defaults_are_sane() {
    let manifest = BuildManifest::new("TestGame", BuildTarget::Linux, BuildConfig::Release);
    assert_eq!(manifest.game_name, "TestGame");
    assert_eq!(manifest.version, "0.1.0");
    assert!(manifest.icon.is_none());
    assert!(manifest.features.is_empty());
}

// --- Turnaround tests ---

#[test]
fn turnaround_nonexistent_views_returns_error() {
    let views = Path::new("/tmp/ochroma_test_nonexistent_views_dir_zzz");
    let output = Path::new("/tmp/ochroma_test_output.vxm");
    let result = run_turnaround(views, output, None);
    assert!(result.is_err(), "Should fail when views directory does not exist");
    let err_msg = format!("{}", result.unwrap_err());
    assert!(err_msg.contains("does not exist"), "Error should mention missing views path");
}

// --- BuildConfig equality ---

#[test]
fn build_configs_are_distinct() {
    assert_ne!(BuildConfig::Debug, BuildConfig::Release);
    assert_ne!(BuildConfig::Release, BuildConfig::Shipping);
    assert_ne!(BuildConfig::Debug, BuildConfig::Shipping);
}
