use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Type of relationship between two citizens.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RelationshipType {
    Family,
    Friend,
    Coworker,
    Neighbour,
}

/// A directed relationship from one citizen to another.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Relationship {
    pub rel_type: RelationshipType,
    /// Strength of the relationship, clamped to [0.0, 1.0].
    pub strength: f32,
    /// Number of interactions recorded.
    pub interactions: u32,
    /// Age of relationship in game-years.
    pub age_years: f32,
}

impl Relationship {
    pub fn new(rel_type: RelationshipType, strength: f32) -> Self {
        Self {
            rel_type,
            strength: strength.clamp(0.0, 1.0),
            interactions: 0,
            age_years: 0.0,
        }
    }

    /// Record an interaction, boosting strength.
    pub fn interact(&mut self, boost: f32) {
        self.interactions += 1;
        self.strength = (self.strength + boost).clamp(0.0, 1.0);
    }

    /// Decay strength over time. Returns true if relationship should be pruned.
    pub fn decay(&mut self, dt_years: f32) -> bool {
        self.age_years += dt_years;
        // Family bonds decay very slowly
        let rate = match self.rel_type {
            RelationshipType::Family => 0.01,
            RelationshipType::Friend => 0.05,
            RelationshipType::Coworker => 0.08,
            RelationshipType::Neighbour => 0.06,
        };
        self.strength = (self.strength - rate * dt_years).max(0.0);
        self.strength < 0.01
    }
}

/// Social network as an adjacency list: citizen_id -> [(target_id, Relationship)].
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SocialNetwork {
    adjacency: HashMap<u32, Vec<(u32, Relationship)>>,
}

impl SocialNetwork {
    pub fn new() -> Self {
        Self {
            adjacency: HashMap::new(),
        }
    }

    /// Add a bidirectional relationship between two citizens.
    pub fn add_relationship(&mut self, a: u32, b: u32, rel_type: RelationshipType, strength: f32) {
        self.adjacency
            .entry(a)
            .or_default()
            .push((b, Relationship::new(rel_type, strength)));
        self.adjacency
            .entry(b)
            .or_default()
            .push((a, Relationship::new(rel_type, strength)));
    }

    /// Get all relationships for a citizen.
    pub fn relationships_of(&self, citizen_id: u32) -> &[(u32, Relationship)] {
        self.adjacency
            .get(&citizen_id)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Get friends (Friend-type relationships) of a citizen.
    pub fn friends_of(&self, citizen_id: u32) -> Vec<u32> {
        self.relationships_of(citizen_id)
            .iter()
            .filter(|(_, r)| r.rel_type == RelationshipType::Friend)
            .map(|(id, _)| *id)
            .collect()
    }

    /// Get neighbours of a citizen.
    pub fn neighbours_of(&self, citizen_id: u32) -> Vec<u32> {
        self.relationships_of(citizen_id)
            .iter()
            .filter(|(_, r)| r.rel_type == RelationshipType::Neighbour)
            .map(|(id, _)| *id)
            .collect()
    }

    /// Get all citizen ids in the network.
    pub fn all_citizens(&self) -> Vec<u32> {
        self.adjacency.keys().copied().collect()
    }

    /// Total number of edges (bidirectional counted once).
    pub fn edge_count(&self) -> usize {
        let total: usize = self.adjacency.values().map(|v| v.len()).sum();
        total / 2
    }

    /// Propagate influence: happy citizens boost their connected neighbours' satisfaction.
    /// Returns a map of citizen_id -> satisfaction delta.
    pub fn influence_propagation(&self, satisfaction: &HashMap<u32, f32>) -> HashMap<u32, f32> {
        let mut deltas: HashMap<u32, f32> = HashMap::new();

        for (&citizen_id, edges) in &self.adjacency {
            let my_sat = satisfaction.get(&citizen_id).copied().unwrap_or(0.5);
            for (target_id, relationship) in edges {
                // Influence is proportional to relationship strength and satisfaction differential
                let target_sat = satisfaction.get(target_id).copied().unwrap_or(0.5);
                // Only positive influence from happier neighbours
                if my_sat > target_sat {
                    let influence = (my_sat - target_sat) * relationship.strength * 0.1;
                    *deltas.entry(*target_id).or_default() += influence;
                }
            }
        }

        deltas
    }

    /// Find communities using a simple connected-component approach on strong relationships.
    /// Returns a list of communities, where each community is a set of citizen ids.
    pub fn find_communities(&self, min_strength: f32) -> Vec<Vec<u32>> {
        let citizens: Vec<u32> = self.all_citizens();
        let mut visited: HashMap<u32, bool> = citizens.iter().map(|&id| (id, false)).collect();
        let mut communities = Vec::new();

        for &start in &citizens {
            if visited[&start] {
                continue;
            }
            // BFS from this citizen using only strong edges
            let mut community = Vec::new();
            let mut queue = vec![start];
            visited.insert(start, true);

            while let Some(current) = queue.pop() {
                community.push(current);
                if let Some(edges) = self.adjacency.get(&current) {
                    for (neighbour, rel) in edges {
                        if rel.strength >= min_strength && !visited[neighbour] {
                            visited.insert(*neighbour, true);
                            queue.push(*neighbour);
                        }
                    }
                }
            }

            community.sort();
            communities.push(community);
        }

        // Sort communities by size (largest first)
        communities.sort_by(|a, b| b.len().cmp(&a.len()));
        communities
    }

    /// Advance all relationships by dt game-years, pruning dead ones.
    pub fn tick(&mut self, dt_years: f32) {
        for edges in self.adjacency.values_mut() {
            edges.retain_mut(|(_, rel)| !rel.decay(dt_years));
        }
    }

    /// Remove a citizen from the network entirely.
    pub fn remove_citizen(&mut self, citizen_id: u32) {
        self.adjacency.remove(&citizen_id);
        for edges in self.adjacency.values_mut() {
            edges.retain(|(id, _)| *id != citizen_id);
        }
    }
}
