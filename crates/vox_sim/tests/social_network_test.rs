use std::collections::HashMap;
use vox_sim::social_network::{RelationshipType, SocialNetwork};

#[test]
fn test_add_relationship_bidirectional() {
    let mut net = SocialNetwork::new();
    net.add_relationship(1, 2, RelationshipType::Friend, 0.8);

    assert_eq!(net.friends_of(1), vec![2]);
    assert_eq!(net.friends_of(2), vec![1]);
    assert_eq!(net.edge_count(), 1);
}

#[test]
fn test_multiple_relationships() {
    let mut net = SocialNetwork::new();
    net.add_relationship(1, 2, RelationshipType::Friend, 0.8);
    net.add_relationship(1, 3, RelationshipType::Coworker, 0.5);
    net.add_relationship(1, 4, RelationshipType::Neighbour, 0.6);
    net.add_relationship(2, 3, RelationshipType::Family, 0.9);

    assert_eq!(net.relationships_of(1).len(), 3);
    assert_eq!(net.edge_count(), 4);
}

#[test]
fn test_friends_of_filters_by_type() {
    let mut net = SocialNetwork::new();
    net.add_relationship(1, 2, RelationshipType::Friend, 0.8);
    net.add_relationship(1, 3, RelationshipType::Coworker, 0.5);
    net.add_relationship(1, 4, RelationshipType::Friend, 0.7);

    let friends = net.friends_of(1);
    assert_eq!(friends.len(), 2);
    assert!(friends.contains(&2));
    assert!(friends.contains(&4));
}

#[test]
fn test_neighbours_of() {
    let mut net = SocialNetwork::new();
    net.add_relationship(1, 2, RelationshipType::Neighbour, 0.6);
    net.add_relationship(1, 3, RelationshipType::Neighbour, 0.4);
    net.add_relationship(1, 4, RelationshipType::Friend, 0.7);

    let neighbours = net.neighbours_of(1);
    assert_eq!(neighbours.len(), 2);
    assert!(neighbours.contains(&2));
    assert!(neighbours.contains(&3));
}

#[test]
fn test_influence_propagation_happy_boosts_unhappy() {
    let mut net = SocialNetwork::new();
    net.add_relationship(1, 2, RelationshipType::Friend, 1.0);

    let mut satisfaction = HashMap::new();
    satisfaction.insert(1, 0.9); // happy
    satisfaction.insert(2, 0.3); // unhappy

    let deltas = net.influence_propagation(&satisfaction);

    // Citizen 2 should receive a positive delta from citizen 1
    let delta_2 = deltas.get(&2).copied().unwrap_or(0.0);
    assert!(
        delta_2 > 0.0,
        "Unhappy citizen should receive positive influence, got {}",
        delta_2
    );

    // Citizen 1 should not receive influence (already happier)
    let delta_1 = deltas.get(&1).copied().unwrap_or(0.0);
    assert_eq!(
        delta_1, 0.0,
        "Happy citizen should not receive influence from unhappy neighbour"
    );
}

#[test]
fn test_influence_strength_matters() {
    let mut net = SocialNetwork::new();
    net.add_relationship(1, 2, RelationshipType::Friend, 1.0); // strong bond
    net.add_relationship(1, 3, RelationshipType::Friend, 0.1); // weak bond

    let mut satisfaction = HashMap::new();
    satisfaction.insert(1, 0.9);
    satisfaction.insert(2, 0.3);
    satisfaction.insert(3, 0.3);

    let deltas = net.influence_propagation(&satisfaction);

    let delta_2 = deltas.get(&2).copied().unwrap_or(0.0);
    let delta_3 = deltas.get(&3).copied().unwrap_or(0.0);

    assert!(
        delta_2 > delta_3,
        "Stronger relationship should provide more influence: {} vs {}",
        delta_2,
        delta_3
    );
}

#[test]
fn test_find_communities_single_cluster() {
    let mut net = SocialNetwork::new();
    net.add_relationship(1, 2, RelationshipType::Friend, 0.8);
    net.add_relationship(2, 3, RelationshipType::Friend, 0.8);
    net.add_relationship(3, 4, RelationshipType::Friend, 0.8);

    let communities = net.find_communities(0.5);
    assert_eq!(communities.len(), 1);
    assert_eq!(communities[0].len(), 4);
}

#[test]
fn test_find_communities_two_clusters() {
    let mut net = SocialNetwork::new();
    // Cluster A
    net.add_relationship(1, 2, RelationshipType::Friend, 0.8);
    net.add_relationship(2, 3, RelationshipType::Friend, 0.8);
    // Cluster B (disconnected at min_strength)
    net.add_relationship(10, 11, RelationshipType::Friend, 0.9);
    // Weak bridge (below threshold)
    net.add_relationship(3, 10, RelationshipType::Coworker, 0.2);

    let communities = net.find_communities(0.5);
    assert_eq!(
        communities.len(),
        2,
        "Should find 2 communities, got: {:?}",
        communities
    );
}

#[test]
fn test_remove_citizen() {
    let mut net = SocialNetwork::new();
    net.add_relationship(1, 2, RelationshipType::Friend, 0.8);
    net.add_relationship(2, 3, RelationshipType::Friend, 0.8);

    net.remove_citizen(2);

    assert!(net.friends_of(1).is_empty());
    assert!(net.friends_of(3).is_empty());
    assert_eq!(net.relationships_of(2).len(), 0);
}

#[test]
fn test_relationship_decay() {
    let mut net = SocialNetwork::new();
    net.add_relationship(1, 2, RelationshipType::Coworker, 0.3);

    // Advance time significantly
    net.tick(10.0);

    // Weak coworker relationship should have decayed to zero and been pruned
    assert!(
        net.relationships_of(1).is_empty(),
        "Weak relationship should have been pruned after decay"
    );
}

#[test]
fn test_family_bonds_decay_slowly() {
    let mut net = SocialNetwork::new();
    net.add_relationship(1, 2, RelationshipType::Family, 0.9);

    // Advance 5 years — family bond should survive
    net.tick(5.0);

    assert!(
        !net.relationships_of(1).is_empty(),
        "Family bond should survive 5 years of decay"
    );
}

#[test]
fn social_adjacency_iteration_is_id_sorted() {
    // Insert citizen ids in deliberately scrambled (non-ascending) order so that
    // a HashMap-backed adjacency would return them in arbitrary process-stable order
    // and could silently pass a run-twice check.  BTreeMap MUST yield them sorted.
    let mut net = SocialNetwork::new();
    net.add_relationship(500, 100, RelationshipType::Friend, 0.8);
    net.add_relationship(200, 100, RelationshipType::Coworker, 0.5);
    net.add_relationship(300, 500, RelationshipType::Neighbour, 0.6);
    net.add_relationship(50, 200, RelationshipType::Family, 0.9);

    let citizens = net.all_citizens();
    let mut sorted = citizens.clone();
    sorted.sort_unstable();

    assert_eq!(
        citizens, sorted,
        "all_citizens() must return ids in ascending order (BTreeMap guarantee); \
         non-sorted output means the adjacency map reverted to HashMap"
    );
}

#[test]
fn influence_propagation_returns_id_sorted_deltas() {
    // A happy hub (id 100) influences several unhappy neighbours whose ids are
    // ADDED in scrambled order. The returned delta map MUST iterate id-sorted so a
    // downstream caller that folds the deltas does so deterministically — a
    // HashMap-backed return would iterate in arbitrary process-stable order and
    // silently break replay once this feeds CitySim::tick.
    let mut net = SocialNetwork::new();
    for &t in &[50u32, 200, 30, 150] {
        net.add_relationship(100, t, RelationshipType::Friend, 1.0);
    }
    let mut satisfaction = HashMap::new();
    satisfaction.insert(100, 0.9); // happy hub
    for &t in &[50u32, 200, 30, 150] {
        satisfaction.insert(t, 0.2); // unhappy neighbours
    }

    let deltas = net.influence_propagation(&satisfaction);
    let keys: Vec<u32> = deltas.keys().copied().collect();

    // Exact id-sorted key order (BTreeMap guarantee). A HashMap return would
    // almost surely scramble these four ids and fail this assertion.
    assert_eq!(
        keys,
        vec![30, 50, 150, 200],
        "influence_propagation deltas must iterate id-sorted; got {keys:?} — a \
         HashMap return lets a downstream fold of these deltas go nondeterministic"
    );
    // Real computed outcome, not a stub: every neighbour got a positive influence.
    for &t in &[30u32, 50, 150, 200] {
        assert!(
            deltas[&t] > 0.0,
            "neighbour {t} should receive positive influence, got {}",
            deltas[&t]
        );
    }
}
