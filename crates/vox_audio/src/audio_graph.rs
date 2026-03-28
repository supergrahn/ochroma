use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioGraph {
    pub name: String,
    pub nodes: Vec<AudioGraphNode>,
    pub connections: Vec<AudioConnection>,
    pub output_node_id: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioGraphNode {
    pub id: u32,
    pub node_type: AudioNodeType,
    pub position: [f32; 2],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AudioNodeType {
    // Sources
    WavPlayer { path: String, looping: bool },
    Oscillator { waveform: Waveform, frequency: f32 },
    Noise { noise_type: NoiseType },
    // Effects
    Gain { volume: f32 },
    Pan { pan: f32 },
    LowPass { cutoff: f32 },
    HighPass { cutoff: f32 },
    Delay { time: f32, feedback: f32 },
    Reverb { room_size: f32, damping: f32 },
    // Mixing
    Mixer { channel_count: u32 },
    // Modulation
    LFO { frequency: f32, amplitude: f32, waveform: Waveform },
    Envelope { attack: f32, decay: f32, sustain: f32, release: f32 },
    // Parameters
    GameParameter { name: String },
    // Output
    Output,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Waveform {
    Sine,
    Square,
    Triangle,
    Sawtooth,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum NoiseType {
    White,
    Pink,
    Brown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioConnection {
    pub from_node: u32,
    pub from_output: String,
    pub to_node: u32,
    pub to_input: String,
}

impl AudioGraph {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            nodes: Vec::new(),
            connections: Vec::new(),
            output_node_id: 0,
        }
    }

    pub fn add_node(&mut self, node_type: AudioNodeType, pos: [f32; 2]) -> u32 {
        let id = self.nodes.len() as u32;
        self.nodes.push(AudioGraphNode {
            id,
            node_type,
            position: pos,
        });
        id
    }

    pub fn connect(&mut self, from: u32, from_pin: &str, to: u32, to_pin: &str) {
        self.connections.push(AudioConnection {
            from_node: from,
            from_output: from_pin.to_string(),
            to_node: to,
            to_input: to_pin.to_string(),
        });
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn save_json(&self) -> Result<String, String> {
        serde_json::to_string_pretty(self).map_err(|e| e.to_string())
    }

    pub fn load_json(json: &str) -> Result<Self, String> {
        serde_json::from_str(json).map_err(|e| e.to_string())
    }

    /// Generate a simple ambient loop graph: WavPlayer -> Gain -> Output.
    pub fn ambient_loop(wav_path: &str, volume: f32) -> Self {
        let mut graph = Self::new("ambient_loop");
        let wav = graph.add_node(
            AudioNodeType::WavPlayer {
                path: wav_path.to_string(),
                looping: true,
            },
            [0.0, 0.0],
        );
        let gain = graph.add_node(AudioNodeType::Gain { volume }, [200.0, 0.0]);
        let out = graph.add_node(AudioNodeType::Output, [400.0, 0.0]);
        graph.output_node_id = out;
        graph.connect(wav, "audio", gain, "input");
        graph.connect(gain, "audio", out, "input");
        graph
    }

    /// Generate a footstep system graph: two WavPlayers -> Mixer -> Gain -> Output.
    pub fn footstep_system(walk_path: &str, run_path: &str) -> Self {
        let mut graph = Self::new("footstep_system");
        let walk = graph.add_node(
            AudioNodeType::WavPlayer {
                path: walk_path.to_string(),
                looping: false,
            },
            [0.0, 0.0],
        );
        let run = graph.add_node(
            AudioNodeType::WavPlayer {
                path: run_path.to_string(),
                looping: false,
            },
            [0.0, 100.0],
        );
        let mixer = graph.add_node(
            AudioNodeType::Mixer { channel_count: 2 },
            [200.0, 50.0],
        );
        let gain = graph.add_node(AudioNodeType::Gain { volume: 1.0 }, [400.0, 50.0]);
        let out = graph.add_node(AudioNodeType::Output, [600.0, 50.0]);
        graph.output_node_id = out;
        graph.connect(walk, "audio", mixer, "input_0");
        graph.connect(run, "audio", mixer, "input_1");
        graph.connect(mixer, "audio", gain, "input");
        graph.connect(gain, "audio", out, "input");
        graph
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_graph() {
        let graph = AudioGraph::new("test");
        assert_eq!(graph.name, "test");
        assert_eq!(graph.node_count(), 0);
        assert!(graph.connections.is_empty());
    }

    #[test]
    fn add_nodes_and_connections() {
        let mut graph = AudioGraph::new("test");
        let osc = graph.add_node(
            AudioNodeType::Oscillator {
                waveform: Waveform::Sine,
                frequency: 440.0,
            },
            [0.0, 0.0],
        );
        let gain = graph.add_node(AudioNodeType::Gain { volume: 0.5 }, [100.0, 0.0]);
        let out = graph.add_node(AudioNodeType::Output, [200.0, 0.0]);
        graph.connect(osc, "audio", gain, "input");
        graph.connect(gain, "audio", out, "input");
        assert_eq!(graph.node_count(), 3);
        assert_eq!(graph.connections.len(), 2);
        assert_eq!(graph.connections[0].from_node, osc);
        assert_eq!(graph.connections[1].to_node, out);
    }

    #[test]
    fn json_round_trip() {
        let mut graph = AudioGraph::new("round_trip");
        graph.add_node(
            AudioNodeType::Noise {
                noise_type: NoiseType::Pink,
            },
            [0.0, 0.0],
        );
        graph.add_node(AudioNodeType::Output, [100.0, 0.0]);
        graph.connect(0, "audio", 1, "input");
        graph.output_node_id = 1;

        let json = graph.save_json().unwrap();
        let loaded = AudioGraph::load_json(&json).unwrap();
        assert_eq!(loaded.name, "round_trip");
        assert_eq!(loaded.node_count(), 2);
        assert_eq!(loaded.connections.len(), 1);
        assert_eq!(loaded.output_node_id, 1);
    }

    #[test]
    fn ambient_loop_preset() {
        let graph = AudioGraph::ambient_loop("ambient.wav", 0.8);
        assert_eq!(graph.name, "ambient_loop");
        assert_eq!(graph.node_count(), 3); // wav, gain, output
        assert_eq!(graph.connections.len(), 2);
        // Check the wav node is looping
        match &graph.nodes[0].node_type {
            AudioNodeType::WavPlayer { path, looping } => {
                assert_eq!(path, "ambient.wav");
                assert!(*looping);
            }
            _ => panic!("Expected WavPlayer"),
        }
    }

    #[test]
    fn footstep_preset() {
        let graph = AudioGraph::footstep_system("walk.wav", "run.wav");
        assert_eq!(graph.name, "footstep_system");
        assert_eq!(graph.node_count(), 5); // walk, run, mixer, gain, output
        assert_eq!(graph.connections.len(), 4);
    }
}
