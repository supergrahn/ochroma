//! VelloCtx — wgpu-backed Vello scene renderer for game UI.
//!
//! VelloCtx owns a vello::Renderer and accumulates draw commands each frame.
//! VelloCtxCpu is a headless equivalent for unit tests (no GPU required).

// --- Draw command enum (shared by GPU and CPU paths) ----------------------

#[derive(Debug, Clone)]
pub enum DrawCmd {
    FillRect { rect: [f32; 4], color: [f32; 4] },
}

// --- CPU test stub --------------------------------------------------------

/// Headless VelloCtx for unit tests — accumulates DrawCmd without a GPU.
pub struct VelloCtxCpu {
    commands: Vec<DrawCmd>,
    width:    u32,
    height:   u32,
}

impl VelloCtxCpu {
    pub fn new(width: u32, height: u32) -> Self {
        Self { commands: Vec::new(), width, height }
    }

    pub fn begin_frame(&mut self) {
        self.commands.clear();
    }

    pub fn fill_rect(&mut self, rect: [f32; 4], color: [f32; 4]) {
        self.commands.push(DrawCmd::FillRect { rect, color });
    }

    pub fn commands(&self) -> &[DrawCmd] {
        &self.commands
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.width  = width;
        self.height = height;
    }

    pub fn width(&self)  -> u32 { self.width  }
    pub fn height(&self) -> u32 { self.height }
}

// --- GPU VelloCtx (feature-gated) ----------------------------------------

#[cfg(feature = "game-ui")]
pub struct VelloCtx {
    renderer: vello::Renderer,
    scene:    vello::Scene,
    width:    u32,
    height:   u32,
}

#[cfg(feature = "game-ui")]
impl VelloCtx {
    pub fn new(
        device: &vello::wgpu::Device,
        _queue: &vello::wgpu::Queue,
        width:  u32,
        height: u32,
        surface_format: vello::wgpu::TextureFormat,
    ) -> Result<Self, vello::Error> {
        let renderer = vello::Renderer::new(
            device,
            vello::RendererOptions {
                surface_format: Some(surface_format),
                use_cpu:        false,
                antialiasing_support: vello::AaSupport::area_only(),
                num_init_threads: std::num::NonZeroUsize::new(1),
            },
        )?;
        Ok(Self { renderer, scene: vello::Scene::new(), width, height })
    }

    pub fn begin_frame(&mut self) {
        self.scene = vello::Scene::new();
    }

    pub fn fill_rect(&mut self, rect: [f32; 4], color: [f32; 4]) {
        use vello::kurbo::{Affine, Rect};
        use vello::peniko::{Brush, Color, Fill};
        // peniko::Color is AlphaColor<Srgb>; construct via AlphaColor::new([r, g, b, a])
        let vello_color = Color::new([color[0], color[1], color[2], color[3]]);
        let vello_rect = Rect::new(
            rect[0] as f64, rect[1] as f64,
            (rect[0] + rect[2]) as f64, (rect[1] + rect[3]) as f64,
        );
        self.scene.fill(
            Fill::NonZero,
            Affine::IDENTITY,
            &Brush::Solid(vello_color),
            None,
            &vello_rect,
        );
    }

    pub fn end_frame(
        &mut self,
        device: &vello::wgpu::Device,
        queue:  &vello::wgpu::Queue,
        surface_view: &vello::wgpu::TextureView,
    ) -> Result<(), vello::Error> {
        self.renderer.render_to_texture(
            device,
            queue,
            &self.scene,
            surface_view,
            &vello::RenderParams {
                base_color:          vello::peniko::color::palette::css::BLACK,
                width:               self.width,
                height:              self.height,
                antialiasing_method: vello::AaConfig::Area,
            },
        )
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.width  = width;
        self.height = height;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vello_ctx_cpu_starts_empty() {
        let ctx = VelloCtxCpu::new(800, 600);
        assert_eq!(ctx.commands().len(), 0);
    }

    #[test]
    fn fill_rect_appends_command() {
        let mut ctx = VelloCtxCpu::new(800, 600);
        ctx.fill_rect([10.0, 20.0, 100.0, 30.0], [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(ctx.commands().len(), 1);
        match &ctx.commands()[0] {
            DrawCmd::FillRect { rect, color } => {
                println!("rect[0]={} color[0]={}", rect[0], color[0]);
                assert!((rect[0] - 10.0).abs() < 1e-6);
                assert!((color[0] - 1.0).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn fill_rect_multiple_commands() {
        let mut ctx = VelloCtxCpu::new(800, 600);
        for _ in 0..16 { ctx.fill_rect([0.0; 4], [0.0; 4]); }
        assert_eq!(ctx.commands().len(), 16);
    }

    #[test]
    fn begin_frame_clears_commands() {
        let mut ctx = VelloCtxCpu::new(800, 600);
        ctx.fill_rect([0.0; 4], [0.0; 4]);
        assert_eq!(ctx.commands().len(), 1);
        ctx.begin_frame();
        assert_eq!(ctx.commands().len(), 0);
    }

    #[test]
    fn resize_updates_dimensions() {
        let mut ctx = VelloCtxCpu::new(800, 600);
        ctx.resize(1920, 1080);
        println!("width={} height={}", ctx.width(), ctx.height());
        assert_eq!(ctx.width(), 1920);
        assert_eq!(ctx.height(), 1080);
    }
}
