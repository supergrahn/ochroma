use crate::desc::AgentStateDesc;
use crate::node_graph::{AgentNodeGraph, AgentNodeKind, AgentNodeRegistry};
use crate::compute::layout_source;

#[derive(Debug, thiserror::Error)]
pub enum CodegenError {
    #[error("cycle detected in graph")]
    Cycle,
    #[error("unregistered custom node kind: {0}")]
    UnregisteredCustomNode(String),
    #[error("validation error: {0}")]
    Validation(String),
}

pub struct WgslSource {
    pub source:      String,
    pub entry_point: String,  // always "agent_update"
}

pub struct AgentShaderGen;

impl AgentShaderGen {
    pub fn generate(
        graph: &AgentNodeGraph,
        registry: &AgentNodeRegistry,
        desc: &AgentStateDesc,
    ) -> Result<WgslSource, CodegenError> {
        let order = graph.topological_order().map_err(|_| CodegenError::Cycle)?;

        // Assign a variable name to each output pin
        let mut pin_var: std::collections::HashMap<u32, String> = std::collections::HashMap::new();
        let mut var_counter = 0usize;
        for &nid in &order {
            let node = graph.nodes().iter().find(|n| n.id() == nid).unwrap();
            for &pin in node.output_pins() {
                pin_var.insert(pin, format!("_v{}", var_counter));
                var_counter += 1;
            }
        }

        // Build pin → source var map from connections
        let mut pin_src: std::collections::HashMap<u32, String> = std::collections::HashMap::new();
        for c in graph.connections() {
            if let Some(src_var) = pin_var.get(&c.src_pin) {
                pin_src.insert(c.dst_pin, src_var.clone());
            }
        }

        let mut body = String::new();
        for &nid in &order {
            let node = graph.nodes().iter().find(|n| n.id() == nid).unwrap();
            let inputs: Vec<String> = node.input_pins().iter()
                .map(|p| pin_src.get(p).cloned().unwrap_or_else(|| "0.0".to_string()))
                .collect();
            let output = node.output_pins().first()
                .and_then(|p| pin_var.get(p))
                .cloned()
                .unwrap_or_default();

            let fragment = emit_node(node.kind(), &inputs, &output, registry, desc)?;
            body.push_str(&fragment);
            body.push('\n');
        }

        let bindings = layout_source(desc);
        let source = format!(
            "{bindings}\n\
             @compute @workgroup_size(64)\n\
             fn agent_update(@builtin(global_invocation_id) gid: vec3<u32>) {{\n\
             let i = gid.x;\n\
             if i >= uniforms.agent_count {{ return; }}\n\
             if (agent_flags[i] & 1u) == 0u {{ return; }}\n\
             {body}\
             }}\n"
        );

        Ok(WgslSource { source, entry_point: "agent_update".to_string() })
    }
}

fn emit_node(
    kind: &AgentNodeKind,
    inputs: &[String],
    output: &str,
    registry: &AgentNodeRegistry,
    _desc: &AgentStateDesc,
) -> Result<String, CodegenError> {
    let s = match kind {
        AgentNodeKind::OnUpdate => String::new(),
        AgentNodeKind::OnSpectralThreshold { .. } => String::new(),
        AgentNodeKind::GetPosition => format!(
            "var {output} = vec3<f32>(positions_in[i*3u], positions_in[i*3u+1u], positions_in[i*3u+2u]);"),
        AgentNodeKind::GetVelocity => format!(
            "var {output} = vec3<f32>(velocities_in[i*3u], velocities_in[i*3u+1u], velocities_in[i*3u+2u]);"),
        AgentNodeKind::AgentId => format!("var {output} = i;"),
        AgentNodeKind::GetTime => format!("var {output} = uniforms.time;"),
        AgentNodeKind::ReadCustom { slot } => format!(
            "var {output} = custom[i * uniforms.custom_floats + {slot}u];"),
        AgentNodeKind::SampleSpectral { band } => format!(
            "var {output} = spectral_samples[i * 16u + {band}u];"),
        AgentNodeKind::QueryNeighbours { .. } => String::new(),
        AgentNodeKind::NeighbourCount => String::new(),
        AgentNodeKind::NeighbourPosition { .. } => String::new(),
        AgentNodeKind::Add => format!("var {output} = {} + {};", inputs[0], inputs[1]),
        AgentNodeKind::Sub => format!("var {output} = {} - {};", inputs[0], inputs[1]),
        AgentNodeKind::Mul => format!("var {output} = {} * {};", inputs[0], inputs[1]),
        AgentNodeKind::Div => format!("var {output} = {} / {};", inputs[0], inputs[1]),
        AgentNodeKind::Normalize => format!("var {output} = normalize({});", inputs[0]),
        AgentNodeKind::Length => format!("var {output} = length({});", inputs[0]),
        AgentNodeKind::Distance => format!("var {output} = distance({}, {});", inputs[0], inputs[1]),
        AgentNodeKind::Lerp => format!("var {output} = mix({}, {}, {});", inputs[0], inputs[1], inputs[2]),
        AgentNodeKind::Clamp => format!("var {output} = clamp({}, {}, {});", inputs[0], inputs[1], inputs[2]),
        AgentNodeKind::Select => String::new(),
        AgentNodeKind::Noise => format!("var {output} = fract(sin({} * 127.1) * 43758.5453);", inputs[0]),
        AgentNodeKind::Compare { op } => {
            let cmp = match op {
                crate::node_graph::CompareOp::Lt => "<",
                crate::node_graph::CompareOp::Le => "<=",
                crate::node_graph::CompareOp::Gt => ">",
                crate::node_graph::CompareOp::Ge => ">=",
                crate::node_graph::CompareOp::Eq => "==",
                crate::node_graph::CompareOp::Ne => "!=",
            };
            format!("var {output} = ({} {cmp} {});", inputs[0], inputs[1])
        }
        AgentNodeKind::And => format!("var {output} = {} && {};", inputs[0], inputs[1]),
        AgentNodeKind::Or  => format!("var {output} = {} || {};", inputs[0], inputs[1]),
        AgentNodeKind::Not => format!("var {output} = !{};", inputs[0]),
        AgentNodeKind::Branch => String::new(),
        AgentNodeKind::SetVelocity => format!(
            "positions_out[i*3u]    = positions_in[i*3u]    + {0}.x * uniforms.dt;\n\
             positions_out[i*3u+1u] = positions_in[i*3u+1u] + {0}.y * uniforms.dt;\n\
             positions_out[i*3u+2u] = positions_in[i*3u+2u] + {0}.z * uniforms.dt;\n\
             velocities_out[i*3u]   = {0}.x;\n\
             velocities_out[i*3u+1u] = {0}.y;\n\
             velocities_out[i*3u+2u] = {0}.z;", inputs.get(0).map(|s| s.as_str()).unwrap_or("0.0")),
        AgentNodeKind::AddVelocity => format!(
            "velocities_out[i*3u]   = velocities_in[i*3u]   + {0}.x;\n\
             velocities_out[i*3u+1u] = velocities_in[i*3u+1u] + {0}.y;\n\
             velocities_out[i*3u+2u] = velocities_in[i*3u+2u] + {0}.z;",
            inputs.get(0).map(|s| s.as_str()).unwrap_or("0.0")),
        AgentNodeKind::WriteCustom { slot } => format!(
            "custom[i * uniforms.custom_floats + {slot}u] = {};",
            inputs.get(0).map(|s| s.as_str()).unwrap_or("0.0")),
        AgentNodeKind::RequestCpuAttention =>
            "agent_flags[i] = agent_flags[i] | 2u;".to_string(),
        AgentNodeKind::SpectralDot => format!(
            "var {output} = dot(vec4<f32>({0}.x, {0}.y, {0}.z, {0}.w), \
                                vec4<f32>({1}.x, {1}.y, {1}.z, {1}.w));",
            inputs.get(0).map(|s| s.as_str()).unwrap_or("0.0"),
            inputs.get(1).map(|s| s.as_str()).unwrap_or("0.0")),
        AgentNodeKind::SampleSpectralCurve => String::new(),
        AgentNodeKind::SpectralBand { band } => format!(
            "var {output} = {}.band{band};",
            inputs.get(0).map(|s| s.as_str()).unwrap_or("0.0")),
        AgentNodeKind::Custom { kind_name } => {
            let frag = registry.get(kind_name)
                .ok_or_else(|| CodegenError::UnregisteredCustomNode(kind_name.clone()))?;
            let mut code = frag.0.clone();
            for (idx, input) in inputs.iter().enumerate() {
                code = code.replace(&format!("{{input_{idx}}}"), input);
            }
            code.replace("{output}", output)
        }
    };
    Ok(s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::test_device;
    use crate::node_graph::{AgentNodeGraph, AgentNodeKind, AgentNodeRegistry};
    use crate::desc::AgentStateDesc;
    use crate::compute::{AgentComputePipeline, ShaderSource};

    fn minimal_desc() -> AgentStateDesc {
        AgentStateDesc { agent_count: 4, custom_floats: 0, spectral: false, spatial_hash: None }
    }

    #[test]
    fn single_node_graph_generates_wgsl() {
        let registry = AgentNodeRegistry::new();
        let desc = minimal_desc();
        let mut g = AgentNodeGraph::new("test");
        g.add_node(AgentNodeKind::OnUpdate, [0.0, 0.0]);
        let result = AgentShaderGen::generate(&g, &registry, &desc);
        assert!(result.is_ok(), "single-node graph must generate: {:?}", result.err());
        let src = result.unwrap();
        assert!(src.source.contains("agent_update"), "must have entry point");
        assert!(src.source.contains("@compute"),    "must have @compute attribute");
    }

    #[test]
    fn node_graph_compiles_to_valid_wgsl() {
        let Some((device, _queue)) = test_device() else { return; };
        let registry = AgentNodeRegistry::new();
        let desc = minimal_desc();
        let mut g = AgentNodeGraph::new("normalize_pos");
        let a = g.add_node(AgentNodeKind::GetPosition, [0.0, 0.0]);
        let b = g.add_node(AgentNodeKind::Normalize,   [100.0, 0.0]);
        let c = g.add_node(AgentNodeKind::SetVelocity, [200.0, 0.0]);

        let a_out = g.nodes().iter().find(|n| n.id() == a).unwrap().output_pins()[0];
        let b_in  = g.nodes().iter().find(|n| n.id() == b).unwrap().input_pins()[0];
        let b_out = g.nodes().iter().find(|n| n.id() == b).unwrap().output_pins()[0];
        let c_in  = g.nodes().iter().find(|n| n.id() == c).unwrap().input_pins()[0];
        g.connect(a, a_out, b, b_in).unwrap();
        g.connect(b, b_out, c, c_in).unwrap();

        let wgsl = AgentShaderGen::generate(&g, &registry, &desc).expect("codegen");
        let result = AgentComputePipeline::new(&device,
            ShaderSource::Wgsl(wgsl.source), &desc);
        assert!(result.is_ok(), "generated WGSL must compile in wgpu: {:?}", result.err());
    }

    #[test]
    fn custom_node_fragment_is_substituted() {
        let mut registry = AgentNodeRegistry::new();
        registry.register("Double", crate::node_graph::SlangFragment(
            "var {output} = {input_0} * 2.0;".to_string()
        ));
        let desc = minimal_desc();
        let mut g = AgentNodeGraph::new("custom");
        g.add_node(AgentNodeKind::Custom { kind_name: "Double".to_string() }, [0.0, 0.0]);
        let src = AgentShaderGen::generate(&g, &registry, &desc).expect("codegen");
        assert!(src.source.contains("* 2.0"), "custom fragment must be inlined");
    }

    #[test]
    fn unregistered_custom_node_returns_error() {
        let registry = AgentNodeRegistry::new();
        let desc = minimal_desc();
        let mut g = AgentNodeGraph::new("bad");
        g.add_node(AgentNodeKind::Custom { kind_name: "NotRegistered".to_string() }, [0.0, 0.0]);
        let result = AgentShaderGen::generate(&g, &registry, &desc);
        assert!(matches!(result, Err(CodegenError::UnregisteredCustomNode(_))));
    }
}
