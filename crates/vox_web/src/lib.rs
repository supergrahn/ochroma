//! vox_web — WebGPU/WASM bring-up for Ochroma.
//!
//! This crate is the browser entry point. It is intentionally kept free of the
//! native-only engine crates so it compiles cleanly to `wasm32-unknown-unknown`.
//!
//! What it does today (this milestone):
//!   * requests a WebGPU adapter + device in the browser,
//!   * acquires the `#ochroma-canvas` HTML canvas as a wgpu surface,
//!   * configures that surface, and
//!   * runs a `requestAnimationFrame` render loop that CLEARS the canvas to a
//!     recognizable non-black color every frame.
//!
//! A cleared frame proves the full adapter → device → surface → queue pipeline
//! works in-browser. Porting the `vox_render` spectral splat pipeline to wasm is
//! a separate, larger effort (see [`NEXT_STEP`]) and is intentionally NOT done
//! here.

/// The HTML id of the canvas this crate renders into. Must match `web/index.html`.
pub const CANVAS_ID: &str = "ochroma-canvas";

/// Honest note about what the next milestone is, surfaced in the browser console
/// so anyone poking at the build knows a clear-color frame is the deliverable and
/// real splat rendering is still to come.
pub const NEXT_STEP: &str =
    "vox_web clears the canvas via WebGPU. Next: port vox_render's spectral splat \
     pipeline (shaders + buffers) to wasm32 — a separate multi-day effort.";

/// The render-loop frame description — pure data, no GPU types, so it can be
/// unit-tested on the host with `cargo test`. The wasm render loop turns this
/// into an actual `wgpu` clear pass.
pub mod frame {
    /// A linear-space RGBA clear color, components in `[0.0, 1.0]`.
    ///
    /// These map 1:1 onto `wgpu::Color { r, g, b, a }`.
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct ClearColor {
        pub r: f64,
        pub g: f64,
        pub b: f64,
        pub a: f64,
    }

    impl ClearColor {
        /// True if this color is visibly distinct from pure black, i.e. at least
        /// one RGB channel is meaningfully lit. Used to assert "recognizable,
        /// non-black" in tests and as a runtime invariant before submitting a
        /// frame.
        pub fn is_recognizable_non_black(&self) -> bool {
            const EPS: f64 = 1.0 / 255.0;
            self.r > EPS || self.g > EPS || self.b > EPS
        }
    }

    /// Ochroma's brand "spectral teal" — the recognizable clear color the canvas
    /// is filled with. Deliberately bright and non-grey so that "did the GPU
    /// actually paint?" is obvious at a glance in the browser.
    ///
    /// sRGB ~ (0.0, 0.62, 0.58); stored here in the linear values wgpu expects
    /// for a `*-unorm-srgb` surface (sRGB → linear: `((c + 0.055)/1.055)^2.4`).
    pub const SPECTRAL_TEAL: ClearColor = ClearColor {
        r: 0.0,
        g: 0.342_647,
        b: 0.295_572,
        a: 1.0,
    };

    /// Compute the clear color for a given frame index.
    ///
    /// We keep a subtle, deterministic green-channel pulse keyed off the frame
    /// index so that a *live* render loop is visibly animating (not a single
    /// painted frame that could be a fluke), while staying firmly in the teal
    /// family and never collapsing to black.
    ///
    /// The pulse is a triangle wave over a 120-frame period with amplitude 0.06,
    /// applied to the green channel and clamped into `[0, 1]`.
    pub fn clear_color_for_frame(frame_index: u64) -> ClearColor {
        const PERIOD: u64 = 120;
        const AMPLITUDE: f64 = 0.06;

        // Triangle wave in [0, 1] from the frame phase.
        let phase = (frame_index % PERIOD) as f64 / PERIOD as f64; // [0, 1)
        let tri = 1.0 - (2.0 * phase - 1.0).abs(); // 0 → 1 → 0
        let pulse = (tri - 0.5) * 2.0 * AMPLITUDE; // [-AMP, +AMP]

        let base = SPECTRAL_TEAL;
        ClearColor {
            r: base.r,
            g: (base.g + pulse).clamp(0.0, 1.0),
            b: base.b,
            a: 1.0,
        }
    }
}

#[cfg(target_arch = "wasm32")]
mod web {
    use crate::{frame, CANVAS_ID, NEXT_STEP};
    use std::cell::RefCell;
    use std::rc::Rc;
    use wasm_bindgen::prelude::*;
    use wasm_bindgen::JsCast;

    impl From<frame::ClearColor> for wgpu::Color {
        fn from(c: frame::ClearColor) -> Self {
            wgpu::Color {
                r: c.r,
                g: c.g,
                b: c.b,
                a: c.a,
            }
        }
    }

    /// Everything needed to render one clear frame, owned for the lifetime of the
    /// page.
    struct Renderer {
        surface: wgpu::Surface<'static>,
        device: wgpu::Device,
        queue: wgpu::Queue,
        config: wgpu::SurfaceConfiguration,
        frame_index: u64,
    }

    impl Renderer {
        /// Render one frame: acquire the next swapchain texture and run a render
        /// pass whose `LoadOp::Clear` paints the whole canvas. Returns `Ok(())`
        /// once the frame has been submitted and presented.
        fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
            let color = frame::clear_color_for_frame(self.frame_index);
            // Runtime guard: never silently submit a black frame.
            debug_assert!(
                color.is_recognizable_non_black(),
                "clear color collapsed to black"
            );

            let surface_texture = self.surface.get_current_texture()?;
            let view = surface_texture
                .texture
                .create_view(&wgpu::TextureViewDescriptor::default());

            let mut encoder =
                self.device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("ochroma-clear-encoder"),
                    });

            {
                let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("ochroma-clear-pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(color.into()),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                // Dropping `_pass` records the clear; no draw calls yet (splats TBD).
            }

            self.queue.submit(std::iter::once(encoder.finish()));
            surface_texture.present();
            self.frame_index = self.frame_index.wrapping_add(1);
            Ok(())
        }
    }

    fn document() -> web_sys::Document {
        web_sys::window()
            .expect("no global `window` — vox_web must run in a browser")
            .document()
            .expect("window has no document")
    }

    fn canvas() -> web_sys::HtmlCanvasElement {
        document()
            .get_element_by_id(CANVAS_ID)
            .unwrap_or_else(|| panic!("no element with id `#{CANVAS_ID}` in the page"))
            .dyn_into::<web_sys::HtmlCanvasElement>()
            .expect("`#ochroma-canvas` is not a <canvas> element")
    }

    /// Set the `#status` element's text, if present. This is how we flip the page
    /// to "Ready" — but only after the device is genuinely acquired.
    fn set_status(text: &str) {
        if let Some(el) = document().get_element_by_id("status") {
            el.set_text_content(Some(text));
        }
    }

    /// Size the canvas's drawing buffer to its CSS pixel size so the surface
    /// configuration matches what the user actually sees. Returns `(width,
    /// height)` in device pixels, clamped to at least 1×1.
    fn size_canvas(canvas: &web_sys::HtmlCanvasElement) -> (u32, u32) {
        let dpr = web_sys::window()
            .map(|w| w.device_pixel_ratio())
            .unwrap_or(1.0)
            .max(1.0);
        let css_w = canvas.client_width().max(1) as f64;
        let css_h = canvas.client_height().max(1) as f64;
        let w = ((css_w * dpr) as u32).max(1);
        let h = ((css_h * dpr) as u32).max(1);
        canvas.set_width(w);
        canvas.set_height(h);
        (w, h)
    }

    /// Async bring-up: instance → surface → adapter → device → configured surface.
    async fn build_renderer() -> Renderer {
        let canvas = canvas();
        let (width, height) = size_canvas(&canvas);

        // WebGPU-only instance — no GL fallback, this milestone targets WebGPU.
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::BROWSER_WEBGPU,
            ..Default::default()
        });

        let surface = instance
            .create_surface(wgpu::SurfaceTarget::Canvas(canvas))
            .expect("failed to create surface from #ochroma-canvas");

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: Some(&surface),
            })
            .await
            .expect("no WebGPU adapter — does this browser have WebGPU enabled?");

        log::info!("vox_web: adapter acquired: {:?}", adapter.get_info());

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("ochroma-device"),
                    required_features: wgpu::Features::empty(),
                    // Downlevel WebGPU defaults — broadest browser compatibility.
                    required_limits: wgpu::Limits::downlevel_webgl2_defaults()
                        .using_resolution(adapter.limits()),
                    memory_hints: wgpu::MemoryHints::default(),
                },
                None,
            )
            .await
            .expect("failed to acquire WebGPU device");

        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width,
            height,
            present_mode: wgpu::PresentMode::Fifo,
            desired_maximum_frame_latency: 2,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&device, &config);

        Renderer {
            surface,
            device,
            queue,
            config,
            frame_index: 0,
        }
    }

    /// Shared, re-armable `requestAnimationFrame` callback slot — the canonical
    /// wasm rAF pattern where the closure re-schedules itself.
    type RafCallback = Rc<RefCell<Option<Closure<dyn FnMut()>>>>;

    /// Drive the render loop via `requestAnimationFrame`. We keep the renderer in
    /// an `Rc<RefCell<_>>` and re-arm the rAF callback from inside itself.
    fn start_render_loop(renderer: Renderer) {
        let renderer = Rc::new(RefCell::new(renderer));

        // Two handles to the same closure slot so the callback can re-schedule
        // itself.
        let cb: RafCallback = Rc::new(RefCell::new(None));
        let cb_clone = cb.clone();

        let renderer_for_cb = renderer.clone();
        *cb.borrow_mut() = Some(Closure::new(move || {
            {
                let mut r = renderer_for_cb.borrow_mut();
                match r.render() {
                    Ok(()) => {}
                    Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                        // Reconfigure with the last-known config and try again next frame.
                        let config = r.config.clone();
                        r.surface.configure(&r.device, &config);
                    }
                    Err(wgpu::SurfaceError::OutOfMemory) => {
                        log::error!("vox_web: surface out of memory — stopping render loop");
                        return; // do not re-arm
                    }
                    Err(e) => {
                        log::warn!("vox_web: dropped frame: {e:?}");
                    }
                }
            }
            request_animation_frame(cb_clone.borrow().as_ref().unwrap());
        }));

        request_animation_frame(cb.borrow().as_ref().unwrap());
    }

    fn request_animation_frame(cb: &Closure<dyn FnMut()>) {
        web_sys::window()
            .expect("no window")
            .request_animation_frame(cb.as_ref().unchecked_ref())
            .expect("requestAnimationFrame failed");
    }

    /// The wasm entry point, invoked by the wasm-bindgen `start` shim.
    pub fn run() {
        console_error_panic_hook::set_once();
        // Best-effort console logger; ignore if already initialized.
        let _ = console_log::init_with_level(log::Level::Info);

        log::info!("vox_web: starting WebGPU bring-up");
        log::info!("vox_web: {NEXT_STEP}");
        set_status("Initialising WebGPU…");

        wasm_bindgen_futures::spawn_local(async move {
            let renderer = build_renderer().await;
            // Device is genuinely acquired and the surface configured: only NOW
            // do we flip the page to "Ready".
            set_status("Ready");
            log::info!("vox_web: WebGPU device acquired — clearing canvas every frame");
            start_render_loop(renderer);
        });
    }
}

/// Entry point called automatically by the JS glue generated by wasm-bindgen.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn wasm_main() {
    web::run();
}

#[cfg(test)]
mod tests {
    use super::frame::{clear_color_for_frame, ClearColor, SPECTRAL_TEAL};

    #[test]
    fn brand_teal_is_recognizable_non_black() {
        assert!(
            SPECTRAL_TEAL.is_recognizable_non_black(),
            "brand clear color must be visibly non-black"
        );
        // Green is the dominant lit channel; red is fully off.
        assert_eq!(SPECTRAL_TEAL.r, 0.0);
        assert!(SPECTRAL_TEAL.g > SPECTRAL_TEAL.b);
    }

    #[test]
    fn pure_black_is_not_recognizable() {
        let black = ClearColor {
            r: 0.0,
            g: 0.0,
            b: 0.0,
            a: 1.0,
        };
        assert!(!black.is_recognizable_non_black());
    }

    #[test]
    fn frame_zero_matches_base_teal() {
        // At frame 0 the triangle-wave pulse is exactly -AMPLITUDE (phase 0),
        // so green = base.g - 0.06. Check the exact computed value.
        let c = clear_color_for_frame(0);
        assert_eq!(c.r, SPECTRAL_TEAL.r);
        assert_eq!(c.b, SPECTRAL_TEAL.b);
        let expected_g = SPECTRAL_TEAL.g - 0.06;
        assert!(
            (c.g - expected_g).abs() < 1e-9,
            "frame 0 green = {}, expected {expected_g}",
            c.g
        );
    }

    #[test]
    fn pulse_peaks_at_period_midpoint() {
        // Period is 120; the triangle wave peaks at frame 60 (phase 0.5),
        // giving green = base.g + AMPLITUDE.
        let mid = clear_color_for_frame(60);
        let expected_g = SPECTRAL_TEAL.g + 0.06;
        assert!(
            (mid.g - expected_g).abs() < 1e-9,
            "midpoint green = {}, expected {expected_g}",
            mid.g
        );
        // And it is the maximum across a full period.
        let max_g = (0..120)
            .map(|i| clear_color_for_frame(i).g)
            .fold(f64::MIN, f64::max);
        assert!(
            (mid.g - max_g).abs() < 1e-9,
            "frame 60 should hold the peak green of the period"
        );
    }

    #[test]
    fn every_frame_in_period_is_non_black() {
        // The animated clear must NEVER collapse to black on any frame.
        for i in 0..240u64 {
            let c = clear_color_for_frame(i);
            assert!(
                c.is_recognizable_non_black(),
                "frame {i} clear color {c:?} collapsed to black"
            );
            // Pulse stays within [-0.06, +0.06] of base green.
            assert!(
                (c.g - SPECTRAL_TEAL.g).abs() <= 0.06 + 1e-9,
                "frame {i} green {} exceeds expected pulse band",
                c.g
            );
        }
    }

    #[test]
    fn frame_index_wraps_by_period() {
        // Frame N and frame N+120 produce identical colors (deterministic loop).
        for i in [0u64, 7, 59, 60, 119] {
            assert_eq!(
                clear_color_for_frame(i),
                clear_color_for_frame(i + 120),
                "color must repeat every 120 frames"
            );
        }
    }
}
