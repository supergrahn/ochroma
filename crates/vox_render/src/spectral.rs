use glam::Mat4;

#[derive(Debug, Clone)]
pub struct RenderCamera {
    pub view: Mat4,
    pub proj: Mat4,
}

impl RenderCamera {
    pub fn view_proj(&self) -> Mat4 {
        self.proj * self.view
    }
}
