use crate::llm_client::{LlmClient, prompt_to_layout};
use vox_data::proc_gs_advanced::*;

/// Result of text-to-city generation.
pub struct GeneratedDistrict {
    pub prompt: String,
    pub layout_json: String,
    pub buildings: Vec<(glam::Vec3, Vec<vox_core::types::GaussianSplat>)>,
    pub props: Vec<(glam::Vec3, Vec<vox_core::types::GaussianSplat>)>,
    pub trees: Vec<(glam::Vec3, Vec<vox_core::types::GaussianSplat>)>,
    pub total_splats: usize,
    pub generation_time_ms: u128,
}

/// Generate a district from a text prompt.
pub fn generate_district_from_prompt(
    client: &LlmClient,
    prompt: &str,
    seed: u64,
) -> Result<GeneratedDistrict, String> {
    let start = std::time::Instant::now();

    // Step 1: Get layout from LLM
    let layout_json = prompt_to_layout(client, prompt)?;

    // Step 2: Parse layout (simplified -- extract building positions from JSON)
    let buildings_data = extract_building_positions(&layout_json);
    let props_data = extract_prop_positions(&layout_json);
    let veg_data = extract_vegetation_positions(&layout_json);

    // Step 3: Generate each building using Proc-GS
    let mut buildings = Vec::new();
    let mut total_splats = 0;

    for (i, (pos, _rule_name, building_seed)) in buildings_data.iter().enumerate() {
        let splats = vox_data::proc_gs::emit_splats_simple(
            seed.wrapping_add(*building_seed),
            6.0 + (i as f32 % 3.0) * 2.0, // vary width
            12.0,
        );
        total_splats += splats.len();
        buildings.push((glam::Vec3::new(pos[0], pos[1], pos[2]), splats));
    }

    // Step 4: Generate props
    let mut props = Vec::new();
    for (pos, asset_type) in &props_data {
        let splats = if asset_type.contains("lamp") {
            generate_lamp_post(seed + 1000, 4.5)
        } else if asset_type.contains("bench") {
            generate_bench(seed + 2000)
        } else {
            generate_lamp_post(seed + 3000, 3.0)
        };
        total_splats += splats.len();
        props.push((glam::Vec3::new(pos[0], pos[1], pos[2]), splats));
    }

    // Step 5: Generate vegetation
    let mut trees = Vec::new();
    for (pos, _rule) in &veg_data {
        let splats = generate_tree(seed + 5000, 8.0, 3.0);
        total_splats += splats.len();
        trees.push((glam::Vec3::new(pos[0], pos[1], pos[2]), splats));
    }

    let elapsed = start.elapsed().as_millis();

    Ok(GeneratedDistrict {
        prompt: prompt.to_string(),
        layout_json,
        buildings,
        props,
        trees,
        total_splats,
        generation_time_ms: elapsed,
    })
}

/// Simple JSON parser to extract building positions (no serde_json dependency needed).
fn extract_building_positions(json: &str) -> Vec<([f32; 3], String, u64)> {
    let mut results = Vec::new();
    let mut seed_counter = 0u64;
    let mut in_slots = false;
    for line in json.lines() {
        let line = line.trim();
        if line.contains("\"slots\"") {
            in_slots = true;
            continue;
        }
        if line.contains("\"props\"") || line.contains("\"vegetation\"") {
            in_slots = false;
            continue;
        }

        if in_slots && line.contains("\"position\"") && line.contains('[') {
            if let Some(start) = line.find('[') {
                if let Some(end) = line.find(']') {
                    let nums: Vec<f32> = line[start + 1..end]
                        .split(',')
                        .filter_map(|s| s.trim().parse().ok())
                        .collect();
                    if nums.len() >= 3 {
                        seed_counter += 1;
                        results.push((
                            [nums[0], nums[1], nums[2]],
                            "building".to_string(),
                            seed_counter,
                        ));
                    }
                }
            }
        }
    }

    // If we didn't find anything in a "slots" section, fall back to scanning all positions
    if results.is_empty() {
        seed_counter = 0;
        for line in json.lines() {
            let line = line.trim();
            if line.contains("\"position\"") && line.contains('[') {
                if let Some(start) = line.find('[') {
                    if let Some(end) = line.find(']') {
                        let nums: Vec<f32> = line[start + 1..end]
                            .split(',')
                            .filter_map(|s| s.trim().parse().ok())
                            .collect();
                        if nums.len() >= 3 {
                            seed_counter += 1;
                            results.push((
                                [nums[0], nums[1], nums[2]],
                                "building".to_string(),
                                seed_counter,
                            ));
                        }
                    }
                }
            }
        }
    }

    results
}

fn extract_prop_positions(json: &str) -> Vec<([f32; 3], String)> {
    let mut results = Vec::new();
    let mut in_props = false;
    let mut current_pos: Option<[f32; 3]> = None;

    for line in json.lines() {
        let line = line.trim();
        if line.contains("\"props\"") {
            in_props = true;
            continue;
        }
        if in_props && line.contains("\"vegetation\"") {
            in_props = false;
            continue;
        }

        if in_props {
            if line.contains("\"position\"") && line.contains('[') {
                if let Some(start) = line.find('[') {
                    if let Some(end) = line.find(']') {
                        let nums: Vec<f32> = line[start + 1..end]
                            .split(',')
                            .filter_map(|s| s.trim().parse().ok())
                            .collect();
                        if nums.len() >= 3 {
                            current_pos = Some([nums[0], nums[1], nums[2]]);
                        }
                    }
                }
            }
            if line.contains("\"asset\"") {
                let asset = line
                    .split('"')
                    .nth(3)
                    .unwrap_or("lamp")
                    .to_string();
                let pos = current_pos.unwrap_or([0.0, 0.0, 0.0]);
                results.push((pos, asset));
                current_pos = None;
            }
        }
    }

    results
}

fn extract_vegetation_positions(json: &str) -> Vec<([f32; 3], String)> {
    let mut results = Vec::new();
    let mut in_veg = false;
    let mut current_pos: Option<[f32; 3]> = None;

    for line in json.lines() {
        let line = line.trim();
        if line.contains("\"vegetation\"") {
            in_veg = true;
            continue;
        }

        if in_veg {
            if line.contains("\"position\"") && line.contains('[') {
                if let Some(start) = line.find('[') {
                    if let Some(end) = line.find(']') {
                        let nums: Vec<f32> = line[start + 1..end]
                            .split(',')
                            .filter_map(|s| s.trim().parse().ok())
                            .collect();
                        if nums.len() >= 3 {
                            current_pos = Some([nums[0], nums[1], nums[2]]);
                        }
                    }
                }
            }
            if line.contains("\"rule\"") {
                let rule = line
                    .split('"')
                    .nth(3)
                    .unwrap_or("tree")
                    .to_string();
                let pos = current_pos.unwrap_or([0.0, 0.0, 0.0]);
                results.push((pos, rule));
                current_pos = None;
            }
        }
    }

    results
}
