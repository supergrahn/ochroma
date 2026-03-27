use vox_nn::llm_client::{LlmClient, LlmProvider};
use vox_nn::text_to_city::*;

#[test]
fn generate_victorian_district() {
    let client = LlmClient::new(LlmProvider::Mock);
    let district = generate_district_from_prompt(&client, "A Victorian London street", 42).unwrap();
    assert!(!district.buildings.is_empty(), "Should generate buildings");
    assert!(district.total_splats > 0, "Should have splats");
    println!(
        "Generated {} buildings, {} total splats in {}ms",
        district.buildings.len(),
        district.total_splats,
        district.generation_time_ms
    );
}

#[test]
fn generate_modern_district() {
    let client = LlmClient::new(LlmProvider::Mock);
    let district =
        generate_district_from_prompt(&client, "A modern city boulevard", 99).unwrap();
    assert!(!district.buildings.is_empty());
}

#[test]
fn generation_is_fast() {
    let client = LlmClient::new(LlmProvider::Mock);
    let district = generate_district_from_prompt(&client, "A street", 1).unwrap();
    assert!(
        district.generation_time_ms < 5000,
        "Should generate in under 5 seconds"
    );
}

#[test]
fn different_prompts_different_results() {
    let client = LlmClient::new(LlmProvider::Mock);
    let a = generate_district_from_prompt(&client, "A Victorian street", 1).unwrap();
    let b = generate_district_from_prompt(&client, "A modern boulevard", 1).unwrap();
    // Different layouts should produce different JSON
    assert_ne!(a.layout_json, b.layout_json);
}
