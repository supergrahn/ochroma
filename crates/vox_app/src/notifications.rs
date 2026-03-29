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

// ── NotificationQueue (f32-based TTL, testable) ───────────────────────────

#[derive(Debug, Clone)]
pub struct QueuedNotification {
    pub message: String,
    pub ttl: f32,
    pub initial_ttl: f32,
}

pub struct NotificationQueue {
    pub items: std::collections::VecDeque<QueuedNotification>,
    pub max_visible: usize,
}

impl NotificationQueue {
    pub fn new(max_visible: usize) -> Self {
        Self { items: std::collections::VecDeque::new(), max_visible }
    }

    pub fn push(&mut self, message: String, ttl: f32) {
        self.items.push_back(QueuedNotification {
            message, ttl, initial_ttl: ttl,
        });
    }

    pub fn tick(&mut self, dt: f32) {
        for item in self.items.iter_mut() { item.ttl -= dt; }
        self.items.retain(|n| n.ttl > 0.0);
    }

    pub fn show(&self, ctx: &egui::Context) {
        let screen = ctx.screen_rect();
        let right_x = screen.max.x - 10.0;
        let mut y = 50.0;
        for (i, notif) in self.items.iter().rev().take(self.max_visible).enumerate() {
            let alpha = (notif.ttl / notif.initial_ttl).clamp(0.0, 1.0);
            let color = egui::Color32::from_rgba_unmultiplied(255, 255, 255, (alpha * 255.0) as u8);
            egui::Area::new(egui::Id::new(format!("toast_{}", i)))
                .fixed_pos(egui::pos2(right_x - 250.0, y))
                .show(ctx, |ui| { ui.colored_label(color, &notif.message); });
            y += 25.0;
        }
    }
}

impl Default for NotificationQueue {
    fn default() -> Self { Self::new(5) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn notification_queue_push_adds_items() {
        let mut q = NotificationQueue::new(5);
        q.push("Hello".into(), 3.0);
        assert_eq!(q.items.len(), 1);
        q.push("World".into(), 2.0);
        assert_eq!(q.items.len(), 2);
    }

    #[test]
    fn notification_queue_tick_removes_expired() {
        let mut q = NotificationQueue::new(5);
        q.push("Short".into(), 1.0);
        q.push("Long".into(), 5.0);
        q.tick(1.5);
        assert_eq!(q.items.len(), 1);
        assert_eq!(q.items[0].message, "Long");
    }

    #[test]
    fn notification_queue_tick_decrements_ttl() {
        let mut q = NotificationQueue::new(5);
        q.push("Test".into(), 5.0);
        q.tick(2.0);
        assert!((q.items[0].ttl - 3.0).abs() < 1e-5);
    }

    #[test]
    fn notification_queue_empty_after_all_expire() {
        let mut q = NotificationQueue::new(5);
        q.push("A".into(), 1.0);
        q.push("B".into(), 2.0);
        q.tick(3.0);
        assert_eq!(q.items.len(), 0);
    }
}
