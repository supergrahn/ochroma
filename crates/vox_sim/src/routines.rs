use crate::citizen::{Citizen, DailyState, LifecycleStage};

/// Update citizen daily states based on time of day.
/// Returns list of (citizen_id, new_destination) for agents that need to move.
pub fn update_routines(citizens: &mut [Citizen], hour: f32) -> Vec<(u32, [f32; 2])> {
    let mut movements = Vec::new();

    for citizen in citizens.iter_mut() {
        if citizen.lifecycle != LifecycleStage::Worker {
            continue;
        }

        let new_state = match hour as u32 {
            7 | 8 if citizen.daily_state == DailyState::AtHome => {
                if let Some(wp) = citizen.workplace {
                    movements.push((citizen.id, [wp as f32 * 10.0, 0.0]));
                }
                DailyState::Commuting
            }
            9..=16 if citizen.daily_state == DailyState::Commuting => DailyState::AtWork,
            17 if citizen.daily_state == DailyState::AtWork => {
                if let Some(res) = citizen.residence {
                    movements.push((citizen.id, [res as f32 * 10.0, 0.0]));
                }
                DailyState::Returning
            }
            18..=19 if citizen.daily_state == DailyState::Returning => {
                if citizen.id % 3 == 0 {
                    DailyState::Shopping
                } else {
                    DailyState::AtHome
                }
            }
            20.. if citizen.daily_state == DailyState::Shopping => DailyState::AtHome,
            _ => citizen.daily_state,
        };

        citizen.daily_state = new_state;
    }

    movements
}
