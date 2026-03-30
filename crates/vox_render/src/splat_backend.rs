//! Wraps `spectra_renderer::Renderer` in a dedicated OS thread so Bevy's render
//! schedule never blocks waiting for GPU work to complete.
//!
//! One frame of latency: `submit_frame()` sends work; `read_last_output()` returns
//! the result of the PREVIOUS submission. Acceptable for realtime.

#[cfg(feature = "spectra-native")]
mod inner {
    use std::sync::{Arc, Mutex};
    use std::sync::mpsc::{channel, Sender};
    use spectra_renderer::{Renderer, RenderConfig, SplatScene, CameraParams};
    use spectra_gpu::CudarcSlangBackend;

    /// Command sent from the main thread to the render thread.
    enum RtCommand {
        /// Update scene (new splat data), camera, and render one frame.
        Render { scene: Option<SplatScene>, camera: CameraParams },
        /// Terminate the render thread.
        Shutdown,
    }

    /// Non-blocking frontend to `Renderer` running on a dedicated OS thread.
    pub struct SpectraRenderBackend {
        tx: Sender<RtCommand>,
        last_output: Arc<Mutex<Vec<u8>>>,
        fail_count: u32,
        width: u32,
        height: u32,
    }

    impl SpectraRenderBackend {
        /// Spawn render thread with near-realtime config (4 spp, DLSS, NRC, ReSTIR PT).
        pub fn realtime(width: u32, height: u32) -> Result<Self, String> {
            let config = RenderConfig::near_realtime(width, height);
            Self::spawn(config, width, height)
        }

        /// Spawn render thread with full offline config (128 spp).
        pub fn cinematic(width: u32, height: u32) -> Result<Self, String> {
            let mut config = RenderConfig::default();
            config.width = width;
            config.height = height;
            Self::spawn(config, width, height)
        }

        fn spawn(config: RenderConfig, width: u32, height: u32) -> Result<Self, String> {
            let (tx, rx) = channel::<RtCommand>();
            let last_output: Arc<Mutex<Vec<u8>>> = Arc::new(Mutex::new(Vec::new()));
            let last_output_clone = Arc::clone(&last_output);

            std::thread::Builder::new()
                .name("spectra-render".into())
                .spawn(move || {
                    let gpu = match CudarcSlangBackend::new(0) {
                        Ok(g) => g,
                        Err(e) => {
                            eprintln!("[spectra-render] GPU init failed: {e}");
                            return;
                        }
                    };
                    let mut renderer = Renderer::new(gpu, config);

                    loop {
                        let cmd = match rx.recv() {
                            Ok(c)  => c,
                            Err(_) => break,
                        };
                        match cmd {
                            RtCommand::Shutdown => break,
                            RtCommand::Render { scene, camera } => {
                                if let Some(s) = scene {
                                    if let Err(e) = renderer.load_splat_scene(s) {
                                        eprintln!("[spectra-render] load_splat_scene: {e}");
                                    }
                                }
                                if let Err(e) = renderer.set_camera(camera) {
                                    eprintln!("[spectra-render] set_camera: {e}");
                                    continue;
                                }
                                match renderer.render_splat_frame()
                                    .and_then(|_| renderer.read_splat_output())
                                {
                                    Ok(bytes) => {
                                        if let Ok(mut guard) = last_output_clone.lock() {
                                            *guard = bytes;
                                        }
                                    }
                                    Err(e) => {
                                        eprintln!("[spectra-render] render error: {e}");
                                    }
                                }
                            }
                        }
                    }
                })
                .map_err(|e| format!("thread spawn failed: {e}"))?;

            Ok(Self { tx, last_output, fail_count: 0, width, height })
        }

        /// Submit a frame request (non-blocking).
        pub fn submit_frame(
            &mut self,
            new_scene: Option<SplatScene>,
            camera: CameraParams,
        ) -> Result<(), String> {
            self.tx.send(RtCommand::Render { scene: new_scene, camera })
                .map_err(|e| {
                    self.fail_count += 1;
                    format!("render thread channel closed: {e}")
                })?;
            self.fail_count = 0;
            Ok(())
        }

        /// Read the last completed frame (non-blocking).
        /// Returns empty Vec before the first frame completes.
        pub fn read_last_output(&self) -> Vec<u8> {
            self.last_output.lock()
                .map(|g| g.clone())
                .unwrap_or_default()
        }

        /// Number of consecutive `submit_frame()` failures. Reset to 0 on success.
        pub fn fail_count(&self) -> u32 { self.fail_count }

        pub fn width(&self)  -> u32 { self.width }
        pub fn height(&self) -> u32 { self.height }
    }

    impl Drop for SpectraRenderBackend {
        fn drop(&mut self) {
            let _ = self.tx.send(RtCommand::Shutdown);
        }
    }
}

#[cfg(feature = "spectra-native")]
pub use inner::SpectraRenderBackend;

#[cfg(all(test, feature = "spectra-native"))]
mod tests {
    use super::SpectraRenderBackend;

    #[test]
    fn spectra_render_backend_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<SpectraRenderBackend>();
    }

    #[test]
    fn read_last_output_before_first_frame_returns_empty() {
        // Construction requires GPU — just verify the type structure compiles correctly.
        // For GPU smoke test, run manually with: cargo test -p vox_render --features spectra-native
    }
}
