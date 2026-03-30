#[cfg(feature = "crucible")]
use crucible_core::node::{CrucibleNode, NodeDescriptor, PortSpec};
#[cfg(feature = "crucible")]
use crucible_core::port::{PortData, PortDataType, PortMap, ParamValue};
#[cfg(feature = "crucible")]
use crucible_core::error::CookError;

// ---------------------------------------------------------------------------
// FloatConstNode
// ---------------------------------------------------------------------------

#[cfg(feature = "crucible")]
pub struct FloatConstNode {
    pub value: f64,
}

#[cfg(feature = "crucible")]
impl FloatConstNode {
    pub fn new(value: f64) -> Self { Self { value } }
}

#[cfg(feature = "crucible")]
impl CrucibleNode for FloatConstNode {
    fn descriptor(&self) -> NodeDescriptor {
        NodeDescriptor {
            type_name: "FloatConst",
            inputs: vec![],
            outputs: vec![PortSpec { name: "out", data_type: PortDataType::Scalar, optional: false }],
        }
    }
    fn set_param(&mut self, key: &str, value: ParamValue) -> Result<(), CookError> {
        match key {
            "value" => {
                self.value = value.as_float_coerce().ok_or_else(|| CookError::InvalidParam {
                    key: key.into(),
                    reason: format!("'value' must be a number, got {:?}", value),
                })?;
                Ok(())
            }
            _ => Err(CookError::UnknownParam { key: key.into(), node: "FloatConst".into() }),
        }
    }
    fn cook(&self, _inputs: PortMap) -> Result<PortMap, CookError> {
        let mut out = PortMap::default();
        out.insert("out".into(), PortData::Scalar(self.value));
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// MultiplyNode
// ---------------------------------------------------------------------------

#[cfg(feature = "crucible")]
pub struct MultiplyNode;

#[cfg(feature = "crucible")]
impl CrucibleNode for MultiplyNode {
    fn descriptor(&self) -> NodeDescriptor {
        NodeDescriptor {
            type_name: "Multiply",
            inputs: vec![
                PortSpec { name: "a", data_type: PortDataType::Scalar, optional: false },
                PortSpec { name: "b", data_type: PortDataType::Scalar, optional: false },
            ],
            outputs: vec![PortSpec { name: "out", data_type: PortDataType::Scalar, optional: false }],
        }
    }
    fn set_param(&mut self, key: &str, _: ParamValue) -> Result<(), CookError> {
        Err(CookError::UnknownParam { key: key.into(), node: "Multiply".into() })
    }
    fn cook(&self, inputs: PortMap) -> Result<PortMap, CookError> {
        let a = inputs.get("a").and_then(|p| p.as_scalar())
            .ok_or_else(|| CookError::MissingInput("a".into()))?;
        let b = inputs.get("b").and_then(|p| p.as_scalar())
            .ok_or_else(|| CookError::MissingInput("b".into()))?;
        let mut out = PortMap::default();
        out.insert("out".into(), PortData::Scalar(a * b));
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// AddNode
// ---------------------------------------------------------------------------

#[cfg(feature = "crucible")]
pub struct AddNode;

#[cfg(feature = "crucible")]
impl CrucibleNode for AddNode {
    fn descriptor(&self) -> NodeDescriptor {
        NodeDescriptor {
            type_name: "Add",
            inputs: vec![
                PortSpec { name: "a", data_type: PortDataType::Scalar, optional: false },
                PortSpec { name: "b", data_type: PortDataType::Scalar, optional: false },
            ],
            outputs: vec![PortSpec { name: "out", data_type: PortDataType::Scalar, optional: false }],
        }
    }
    fn set_param(&mut self, key: &str, _: ParamValue) -> Result<(), CookError> {
        Err(CookError::UnknownParam { key: key.into(), node: "Add".into() })
    }
    fn cook(&self, inputs: PortMap) -> Result<PortMap, CookError> {
        let a = inputs.get("a").and_then(|p| p.as_scalar())
            .ok_or_else(|| CookError::MissingInput("a".into()))?;
        let b = inputs.get("b").and_then(|p| p.as_scalar())
            .ok_or_else(|| CookError::MissingInput("b".into()))?;
        let mut out = PortMap::default();
        out.insert("out".into(), PortData::Scalar(a + b));
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// LerpNode
// ---------------------------------------------------------------------------

#[cfg(feature = "crucible")]
pub struct LerpNode;

#[cfg(feature = "crucible")]
impl CrucibleNode for LerpNode {
    fn descriptor(&self) -> NodeDescriptor {
        NodeDescriptor {
            type_name: "Lerp",
            inputs: vec![
                PortSpec { name: "a", data_type: PortDataType::Scalar, optional: false },
                PortSpec { name: "b", data_type: PortDataType::Scalar, optional: false },
                PortSpec { name: "t", data_type: PortDataType::Scalar, optional: false },
            ],
            outputs: vec![PortSpec { name: "out", data_type: PortDataType::Scalar, optional: false }],
        }
    }
    fn set_param(&mut self, key: &str, _: ParamValue) -> Result<(), CookError> {
        Err(CookError::UnknownParam { key: key.into(), node: "Lerp".into() })
    }
    fn cook(&self, inputs: PortMap) -> Result<PortMap, CookError> {
        let a = inputs.get("a").and_then(|p| p.as_scalar())
            .ok_or_else(|| CookError::MissingInput("a".into()))?;
        let b = inputs.get("b").and_then(|p| p.as_scalar())
            .ok_or_else(|| CookError::MissingInput("b".into()))?;
        let t = inputs.get("t").and_then(|p| p.as_scalar())
            .ok_or_else(|| CookError::MissingInput("t".into()))?;
        let mut out = PortMap::default();
        out.insert("out".into(), PortData::Scalar(a + (b - a) * t));
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// OneMinusNode
// ---------------------------------------------------------------------------

#[cfg(feature = "crucible")]
pub struct OneMinusNode;

#[cfg(feature = "crucible")]
impl CrucibleNode for OneMinusNode {
    fn descriptor(&self) -> NodeDescriptor {
        NodeDescriptor {
            type_name: "OneMinus",
            inputs: vec![PortSpec { name: "input", data_type: PortDataType::Scalar, optional: false }],
            outputs: vec![PortSpec { name: "out", data_type: PortDataType::Scalar, optional: false }],
        }
    }
    fn set_param(&mut self, key: &str, _: ParamValue) -> Result<(), CookError> {
        Err(CookError::UnknownParam { key: key.into(), node: "OneMinus".into() })
    }
    fn cook(&self, inputs: PortMap) -> Result<PortMap, CookError> {
        let v = inputs.get("input").and_then(|p| p.as_scalar())
            .ok_or_else(|| CookError::MissingInput("input".into()))?;
        let mut out = PortMap::default();
        out.insert("out".into(), PortData::Scalar(1.0 - v));
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// MaterialOutputNode
// ---------------------------------------------------------------------------

#[cfg(feature = "crucible")]
pub struct MaterialOutputNode;

#[cfg(feature = "crucible")]
impl CrucibleNode for MaterialOutputNode {
    fn descriptor(&self) -> NodeDescriptor {
        NodeDescriptor {
            type_name: "MaterialOutput",
            inputs: vec![
                PortSpec { name: "base_r",    data_type: PortDataType::Scalar, optional: true },
                PortSpec { name: "base_g",    data_type: PortDataType::Scalar, optional: true },
                PortSpec { name: "base_b",    data_type: PortDataType::Scalar, optional: true },
                PortSpec { name: "roughness", data_type: PortDataType::Scalar, optional: true },
                PortSpec { name: "metallic",  data_type: PortDataType::Scalar, optional: true },
                PortSpec { name: "emission",  data_type: PortDataType::Scalar, optional: true },
            ],
            outputs: vec![PortSpec { name: "material", data_type: PortDataType::Null, optional: false }],
        }
    }
    fn set_param(&mut self, key: &str, _: ParamValue) -> Result<(), CookError> {
        Err(CookError::UnknownParam { key: key.into(), node: "MaterialOutput".into() })
    }
    fn cook(&self, _inputs: PortMap) -> Result<PortMap, CookError> {
        let mut out = PortMap::default();
        out.insert("material".into(), PortData::Null);
        Ok(out)
    }
}

#[cfg(all(test, feature = "crucible"))]
mod tests {
    use super::*;

    #[test]
    fn float_const_outputs_value() {
        let n = FloatConstNode::new(2.5);
        let out = n.cook(PortMap::default()).unwrap();
        assert!((out["out"].as_scalar().unwrap() - 2.5).abs() < 1e-9);
    }

    #[test]
    fn multiply_node_multiplies() {
        let n = MultiplyNode;
        let mut inputs = PortMap::default();
        inputs.insert("a".into(), PortData::Scalar(3.0));
        inputs.insert("b".into(), PortData::Scalar(4.0));
        let out = n.cook(inputs).unwrap();
        assert!((out["out"].as_scalar().unwrap() - 12.0).abs() < 1e-9);
    }

    #[test]
    fn add_node_adds() {
        let n = AddNode;
        let mut inputs = PortMap::default();
        inputs.insert("a".into(), PortData::Scalar(1.0));
        inputs.insert("b".into(), PortData::Scalar(2.0));
        let out = n.cook(inputs).unwrap();
        assert!((out["out"].as_scalar().unwrap() - 3.0).abs() < 1e-9);
    }

    #[test]
    fn lerp_node_lerps() {
        let n = LerpNode;
        let mut inputs = PortMap::default();
        inputs.insert("a".into(), PortData::Scalar(0.0));
        inputs.insert("b".into(), PortData::Scalar(10.0));
        inputs.insert("t".into(), PortData::Scalar(0.5));
        let out = n.cook(inputs).unwrap();
        assert!((out["out"].as_scalar().unwrap() - 5.0).abs() < 1e-9);
    }

    #[test]
    fn one_minus_node() {
        let n = OneMinusNode;
        let mut inputs = PortMap::default();
        inputs.insert("input".into(), PortData::Scalar(0.3));
        let out = n.cook(inputs).unwrap();
        assert!((out["out"].as_scalar().unwrap() - 0.7).abs() < 1e-9);
    }

    #[test]
    fn material_output_cooks_to_null() {
        let n = MaterialOutputNode;
        let out = n.cook(PortMap::default()).unwrap();
        assert!(matches!(out["material"], PortData::Null));
    }

    #[test]
    fn float_const_set_param_value() {
        let mut n = FloatConstNode::new(1.0);
        n.set_param("value", ParamValue::Float(7.0)).unwrap();
        let out = n.cook(PortMap::default()).unwrap();
        assert!((out["out"].as_scalar().unwrap() - 7.0).abs() < 1e-9);
    }
}
