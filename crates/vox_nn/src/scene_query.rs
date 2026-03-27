use std::collections::HashMap;

/// Description of a single entity in the game world.
#[derive(Debug, Clone)]
pub struct EntityDescription {
    pub name: String,
    pub entity_type: String,
    pub position: [f32; 3],
    pub details: HashMap<String, String>,
}

/// Engine for natural language queries about the game world.
#[derive(Debug, Default)]
pub struct SceneQueryEngine {
    entities: HashMap<u64, EntityDescription>,
}

impl SceneQueryEngine {
    pub fn new() -> Self {
        Self {
            entities: HashMap::new(),
        }
    }

    /// Register an entity description.
    pub fn register(&mut self, id: u64, desc: EntityDescription) {
        self.entities.insert(id, desc);
    }

    /// Remove an entity.
    pub fn unregister(&mut self, id: u64) {
        self.entities.remove(&id);
    }

    /// Return the number of registered entities.
    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }

    /// Generate a human-readable description of a single entity.
    pub fn describe_entity(&self, id: u64) -> Option<String> {
        let desc = self.entities.get(&id)?;
        let mut text = format!(
            "{} (type: {}) at ({:.1}, {:.1}, {:.1})",
            desc.name, desc.entity_type, desc.position[0], desc.position[1], desc.position[2]
        );
        if !desc.details.is_empty() {
            let detail_strs: Vec<String> = desc
                .details
                .iter()
                .map(|(k, v)| format!("{}: {}", k, v))
                .collect();
            text.push_str(&format!(" [{}]", detail_strs.join(", ")));
        }
        Some(text)
    }

    /// Find entities matching a query string by type, name, or tag.
    pub fn find_entities_matching(&self, query: &str) -> Vec<(u64, &EntityDescription)> {
        if query.is_empty() {
            return Vec::new();
        }
        let q = query.to_lowercase();
        self.entities
            .iter()
            .filter(|(_, desc)| {
                desc.name.to_lowercase().contains(&q)
                    || desc.entity_type.to_lowercase().contains(&q)
                    || desc
                        .details
                        .values()
                        .any(|v| v.to_lowercase().contains(&q))
                    || desc
                        .details
                        .keys()
                        .any(|k| k.to_lowercase().contains(&q))
            })
            .map(|(&id, desc)| (id, desc))
            .collect()
    }

    /// AI-generated suggestion for a problem (mock: pattern-match common problems).
    pub fn suggest_action(&self, problem: &str) -> String {
        let p = problem.to_lowercase();

        if p.contains("traffic") || p.contains("congestion") {
            return "Consider adding public transit routes or widening roads in congested areas."
                .to_string();
        }
        if p.contains("power") || p.contains("electricity") || p.contains("blackout") {
            return "Build additional power plants or upgrade the grid to handle increased demand."
                .to_string();
        }
        if p.contains("crime") || p.contains("safety") {
            return "Place more police stations and improve street lighting in affected districts."
                .to_string();
        }
        if p.contains("pollution") || p.contains("environment") {
            return "Add green spaces and transition to renewable energy sources.".to_string();
        }
        if p.contains("unhappy") || p.contains("happiness") || p.contains("satisfaction") {
            return "Improve services (healthcare, education, parks) near residential areas."
                .to_string();
        }
        if p.contains("fire") {
            return "Ensure fire station coverage within response-time radius of all buildings."
                .to_string();
        }
        if p.contains("water") || p.contains("flood") {
            return "Upgrade water infrastructure and add drainage systems in flood-prone areas."
                .to_string();
        }

        format!(
            "Analyse the situation further. {} entities currently registered in the world.",
            self.entities.len()
        )
    }

    /// Generate a caption describing the current scene.
    pub fn generate_scene_caption(
        &self,
        entity_ids: &[u64],
        time_of_day: &str,
        weather: &str,
    ) -> String {
        let entities: Vec<&EntityDescription> = entity_ids
            .iter()
            .filter_map(|id| self.entities.get(id))
            .collect();

        let count = entities.len();
        if count == 0 {
            return format!("An empty scene at {} under {}.", time_of_day, weather);
        }

        // Collect unique entity types.
        let mut types: Vec<String> = entities.iter().map(|e| e.entity_type.clone()).collect();
        types.sort();
        types.dedup();

        let type_summary = if types.len() <= 3 {
            types.join(", ")
        } else {
            format!("{} and {} more types", types[..2].join(", "), types.len() - 2)
        };

        format!(
            "A scene with {} entities ({}) at {} under {}.",
            count, type_summary, time_of_day, weather
        )
    }
}
