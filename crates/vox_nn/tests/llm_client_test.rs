use vox_nn::llm_client::*;

#[test]
fn mock_client_returns_victorian_layout() {
    let client = LlmClient::new(LlmProvider::Mock);
    let result = prompt_to_layout(&client, "A Victorian London street").unwrap();
    assert!(result.contains("victorian_terraced"));
    assert!(result.contains("lamp_post"));
}

#[test]
fn mock_client_returns_modern_layout() {
    let client = LlmClient::new(LlmProvider::Mock);
    let result = prompt_to_layout(&client, "A modern city boulevard").unwrap();
    assert!(result.contains("modern"));
}

#[test]
fn mock_client_generates_splat_rule() {
    let client = LlmClient::new(LlmProvider::Mock);
    let rule = prompt_to_rule(&client, "A tall brick apartment building").unwrap();
    assert!(rule.contains("[rule]"));
    assert!(rule.contains("brick"));
}

#[test]
fn llm_prompt_builder() {
    let prompt = LlmPrompt::new("system", "user").with_format("JSON");
    assert_eq!(prompt.format_hint, Some("JSON".to_string()));
    assert_eq!(prompt.temperature, 0.3);
}

#[test]
fn mock_response_has_model_name() {
    let client = LlmClient::new(LlmProvider::Mock);
    let prompt = LlmPrompt::new("system", "tell me something");
    let response = client.complete(&prompt).unwrap();
    assert_eq!(response.model, "deterministic-stub");
    assert!(response.tokens_used > 0);
}

#[test]
fn same_prompt_yields_identical_layout() {
    let client = LlmClient::new(LlmProvider::Mock);
    let a = prompt_to_layout(&client, "A Victorian London street near the docks").unwrap();
    let b = prompt_to_layout(&client, "A Victorian London street near the docks").unwrap();
    assert_eq!(a, b, "deterministic stub must reproduce the exact same layout for the same prompt");
    assert!(!a.is_empty());
}

#[test]
fn different_prompts_yield_different_layouts() {
    let client = LlmClient::new(LlmProvider::Mock);
    // Both prompts select the Victorian template (contain "Victorian"), but the
    // prompt-hash seed must still make the concrete layouts differ.
    let a = prompt_to_layout(&client, "A Victorian London street near the docks").unwrap();
    let b = prompt_to_layout(&client, "A Victorian terrace on a quiet hill").unwrap();
    assert_ne!(
        a, b,
        "different prompts (same style template) must still produce different layouts"
    );
}

#[test]
fn ollama_fallback_is_deterministic_and_labeled() {
    // The Ollama provider falls back to the deterministic stub when the server
    // is unreachable. It must still be reproducible and labeled as a stub.
    let client = LlmClient::new(LlmProvider::Ollama {
        url: "http://localhost:11434".to_string(),
        model: "llama3.2".to_string(),
    });
    let prompt = LlmPrompt::new("sys", "A modern boulevard with glass towers");
    let r1 = client.complete(&prompt).unwrap();
    let r2 = client.complete(&prompt).unwrap();
    assert_eq!(r1.text, r2.text, "ollama fallback must be deterministic");
    assert_eq!(r1.model, "deterministic-stub", "fallback must be labeled deterministic-stub");
}

#[test]
fn deterministic_seed_appears_in_layout() {
    // The chosen seed is prompt-derived and embedded in the output so the
    // layout is auditable/reproducible. Two distinct prompts get distinct seeds.
    let client = LlmClient::new(LlmProvider::Mock);
    let a = prompt_to_layout(&client, "Victorian alpha").unwrap();
    let b = prompt_to_layout(&client, "Victorian beta").unwrap();
    assert!(a.contains("layout_seed"), "layout must record its deterministic seed: {a}");
    assert!(b.contains("layout_seed"));
    assert_ne!(a, b);
}
