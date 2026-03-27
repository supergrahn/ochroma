use rand::prelude::*;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CityHistory {
    pub city_name: String,
    pub founding_year: i32,
    pub founding_reason: String,
    pub eras: Vec<HistoricalEra>,
    pub landmarks: Vec<Landmark>,
    pub district_names: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoricalEra {
    pub name: String,
    pub start_year: i32,
    pub end_year: i32,
    pub architectural_style: String,
    pub population_peak: u32,
    pub major_event: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Landmark {
    pub name: String,
    pub year_built: i32,
    pub landmark_type: String,
    pub significance: String,
}

/// Generate a complete procedural city history from a seed.
pub fn generate_history(seed: u64) -> CityHistory {
    let mut rng = StdRng::seed_from_u64(seed);

    let founding_reasons = [
        "river crossing",
        "mining settlement",
        "trading post",
        "military fort",
        "fishing village",
        "railway junction",
    ];
    let styles = [
        "Classical",
        "Medieval",
        "Victorian",
        "Art Deco",
        "Modernist",
        "Post-Modern",
        "Neo-Traditional",
    ];
    let events = [
        "Great Fire destroyed the old quarter",
        "Railway connection brought rapid growth",
        "Industrial revolution transformed the economy",
        "University founded, attracting scholars",
        "Flood devastated the riverside district",
        "New bridge connected north and south banks",
        "Trade agreement doubled merchant traffic",
        "Reform movement improved worker conditions",
    ];
    let district_prefixes = [
        "North", "South", "East", "West", "Old", "New", "Upper", "Lower", "Central", "River",
        "Hill", "Market", "Castle", "Bridge", "Mill", "Church",
    ];
    let district_suffixes = [
        "gate", "wick", "field", "bury", "ton", "ford", "ham", "wood", "dale", "mere", "brook",
        "stead",
    ];
    let city_prefixes = [
        "Ash", "Oak", "Iron", "Stone", "Green", "Silver", "Gold", "Raven", "Wolf", "Bear",
        "River", "Lake", "Marsh", "Hill", "Vale",
    ];
    let city_suffixes = [
        "ford", "bridge", "bury", "ton", "field", "haven", "port", "wick", "dale", "stead",
    ];
    let landmark_types = [
        "Cathedral",
        "Town Hall",
        "Market Hall",
        "Bridge",
        "Monument",
        "University",
        "Castle",
        "Library",
        "Theatre",
        "Station",
    ];

    let city_name = format!(
        "{}{}",
        city_prefixes[rng.random_range(0..city_prefixes.len())],
        city_suffixes[rng.random_range(0..city_suffixes.len())]
    );
    let founding_year = rng.random_range(800..1800i32);
    let founding_reason =
        founding_reasons[rng.random_range(0..founding_reasons.len())].to_string();

    // Generate 3-5 historical eras
    let num_eras = rng.random_range(3..6usize);
    let mut eras = Vec::new();
    let mut year = founding_year;
    for i in 0..num_eras {
        let duration = rng.random_range(50..200i32);
        let style = styles[i.min(styles.len() - 1)].to_string();
        let pop = rng.random_range(500..50000u32) * (i as u32 + 1);
        eras.push(HistoricalEra {
            name: format!("{} Period", style),
            start_year: year,
            end_year: year + duration,
            architectural_style: style,
            population_peak: pop,
            major_event: events[rng.random_range(0..events.len())].to_string(),
        });
        year += duration;
    }

    // Generate landmarks
    let num_landmarks = rng.random_range(3..8usize);
    let landmarks: Vec<Landmark> = (0..num_landmarks)
        .map(|_| {
            let lt = landmark_types[rng.random_range(0..landmark_types.len())];
            Landmark {
                name: format!("{} {}", city_name, lt),
                year_built: rng.random_range(founding_year..2020i32),
                landmark_type: lt.to_string(),
                significance: format!(
                    "Built during the {} era",
                    eras[rng.random_range(0..eras.len())].name
                ),
            }
        })
        .collect();

    // Generate district names
    let num_districts = rng.random_range(5..12usize);
    let district_names: Vec<String> = (0..num_districts)
        .map(|_| {
            format!(
                "{}{}",
                district_prefixes[rng.random_range(0..district_prefixes.len())],
                district_suffixes[rng.random_range(0..district_suffixes.len())]
            )
        })
        .collect();

    CityHistory {
        city_name,
        founding_year,
        founding_reason,
        eras,
        landmarks,
        district_names,
    }
}

/// Generate a citizen name from a cultural context.
pub fn generate_citizen_name(seed: u64) -> (String, String) {
    let mut rng = StdRng::seed_from_u64(seed);
    let first_names = [
        "Alice", "Bob", "Clara", "David", "Emma", "Frank", "Grace", "Henry", "Iris", "James",
        "Kate", "Leo", "Mary", "Noah", "Olive", "Paul", "Quinn", "Rose", "Sam", "Tom",
    ];
    let last_names = [
        "Smith", "Brown", "Wilson", "Taylor", "Clark", "Hall", "Young", "King", "Wright",
        "Green", "Baker", "Hill", "Wood", "Turner", "Evans", "Cooper", "Ward", "Morris", "Reed",
        "Gray",
    ];

    let first = first_names[rng.random_range(0..first_names.len())].to_string();
    let last = last_names[rng.random_range(0..last_names.len())].to_string();
    (first, last)
}
