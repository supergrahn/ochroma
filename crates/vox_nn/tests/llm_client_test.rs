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
    assert_eq!(response.model, "mock");
    assert!(response.tokens_used > 0);
}
