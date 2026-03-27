use std::collections::VecDeque;
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct Notification {
    pub message: String,
    pub category: NotificationCategory,
    pub created: Instant,
    pub duration_secs: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NotificationCategory {
    Info,
    Milestone,
    Warning,
    Disaster,
}

pub struct NotificationManager {
    pub notifications: VecDeque<Notification>,
    pub max_visible: usize,
}

impl NotificationManager {
    pub fn new(max_visible: usize) -> Self {
        Self {
            notifications: VecDeque::new(),
            max_visible,
        }
    }

    pub fn push(&mut self, message: String, category: NotificationCategory) {
        self.notifications.push_back(Notification {
            message,
            category,
            created: Instant::now(),
            duration_secs: 5.0,
        });
    }

    pub fn tick(&mut self) {
        self.notifications
            .retain(|n| n.created.elapsed().as_secs_f32() < n.duration_secs);
    }

    pub fn visible(&self) -> impl Iterator<Item = &Notification> {
        self.notifications.iter().rev().take(self.max_visible)
    }

    pub fn show(&self, ctx: &egui::Context) {
        let mut offset = 0.0;
        for notif in self.visible() {
            let color = match notif.category {
                NotificationCategory::Info => egui::Color32::from_rgb(100, 180, 255),
                NotificationCategory::Milestone => egui::Color32::from_rgb(255, 215, 0),
                NotificationCategory::Warning => egui::Color32::from_rgb(255, 165, 0),
                NotificationCategory::Disaster => egui::Color32::from_rgb(255, 80, 80),
            };
            egui::Area::new(egui::Id::new(format!("notif_{}", offset as u32)))
                .fixed_pos(egui::pos2(10.0, 50.0 + offset))
                .show(ctx, |ui| {
                    ui.colored_label(color, &notif.message);
                });
            offset += 25.0;
        }
    }
}
