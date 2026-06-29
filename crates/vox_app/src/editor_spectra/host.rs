//! Spectra-stack host for `ochroma_editor` (built under `--features spectra`).
//!
//! LAW: ochroma_editor renders + presents ENTIRELY through the Spectra path
//! tracer + Spectra present stack. NO wgpu surface, NO egui-wgpu, NO
//! SoftwareRasteriser anywhere in this path.
//!
//! Pipeline per frame:
//!   1. Run `EditorShell::ui(ctx)` -> tessellate (egui).
//!   2. Paint egui into an OFFSCREEN Vulkan color image on the path tracer's own
//!      ash device (`VulkanSlangBackend`, owned by the Spectra present) via
//!      `egui-ash-renderer` (render-pass path; device is VK 1.2, no
//!      dynamic-rendering).
//!   3. Copy that image to a host RGBA buffer.
//!   4. Hand the host RGBA to `spectra_present` (windowed: `present_frame`;
//!      headless `--frames/--shot`: `VulkanPresent::new_offscreen` + readback).
//!
//! Step 1 of the editor rebuild keeps the Viewport tab solid; the egui chrome
//! (menus, tabs, toolbar, panels) renders + presents on the Spectra device.

use std::sync::Arc;

use ash::vk;
use egui_ash_renderer::{Options, Renderer};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use spectra_gpu::vulkan_backend::VulkanSlangBackend;
use spectra_present::{
    select_present, Present, PresentBackend, PresentChoice, PresentFrame, VulkanPresent,
};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

use vox_app::shell::{cpu_render, EditorShell};
use vox_ui::Tokens;

use crate::{build_shell, load_tokens, Cli};

const WINDOW_WIDTH: u32 = 1600;
const WINDOW_HEIGHT: u32 = 900;
const WINDOW_TITLE: &str = "Ochroma Editor";
/// egui paints into this UNORM color image; egui-ash-renderer's shader emits
/// sRGB-encoded color when `srgb_framebuffer=false`, so an UNORM target carries
/// already-sRGB bytes — exactly what `PresentFrame::from_rgba` wants.
const UI_FORMAT: vk::Format = vk::Format::R8G8B8A8_UNORM;

/// The offscreen egui render target + its readback staging, owned on the
/// Spectra device. Rebuilt on resize.
struct UiTarget {
    width: u32,
    height: u32,
    image: vk::Image,
    image_mem: vk::DeviceMemory,
    view: vk::ImageView,
    framebuffer: vk::Framebuffer,
    /// Host-visible buffer the rendered image is copied into each frame.
    readback: vk::Buffer,
    readback_mem: vk::DeviceMemory,
    readback_size: u64,
}

impl UiTarget {
    fn new(backend: &VulkanSlangBackend, render_pass: vk::RenderPass, w: u32, h: u32) -> Self {
        let device = backend.vk_device();
        let image = unsafe {
            device.create_image(
                &vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(UI_FORMAT)
                    .extent(vk::Extent3D { width: w, height: h, depth: 1 })
                    .mip_levels(1)
                    .array_layers(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(
                        vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
                    )
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .initial_layout(vk::ImageLayout::UNDEFINED),
                None,
            )
        }
        .expect("create ui image");

        let req = unsafe { device.get_image_memory_requirements(image) };
        let mem_type = find_memory_type(
            backend.vk_mem_props(),
            req.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );
        let image_mem = unsafe {
            device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(req.size)
                    .memory_type_index(mem_type),
                None,
            )
        }
        .expect("alloc ui image mem");
        unsafe { device.bind_image_memory(image, image_mem, 0) }.expect("bind ui image mem");

        let view = unsafe {
            device.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(UI_FORMAT)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    }),
                None,
            )
        }
        .expect("create ui view");

        let attachments = [view];
        let framebuffer = unsafe {
            device.create_framebuffer(
                &vk::FramebufferCreateInfo::default()
                    .render_pass(render_pass)
                    .attachments(&attachments)
                    .width(w)
                    .height(h)
                    .layers(1),
                None,
            )
        }
        .expect("create ui framebuffer");

        let readback_size = (w as u64) * (h as u64) * 4;
        let readback = unsafe {
            device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(readback_size)
                    .usage(vk::BufferUsageFlags::TRANSFER_DST)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )
        }
        .expect("create readback buffer");
        let rreq = unsafe { device.get_buffer_memory_requirements(readback) };
        let rmem_type = find_memory_type(
            backend.vk_mem_props(),
            rreq.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );
        let readback_mem = unsafe {
            device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(rreq.size)
                    .memory_type_index(rmem_type),
                None,
            )
        }
        .expect("alloc readback mem");
        unsafe { device.bind_buffer_memory(readback, readback_mem, 0) }
            .expect("bind readback mem");

        Self {
            width: w,
            height: h,
            image,
            image_mem,
            view,
            framebuffer,
            readback,
            readback_mem,
            readback_size,
        }
    }

    fn destroy(&mut self, device: &ash::Device) {
        unsafe {
            device.destroy_framebuffer(self.framebuffer, None);
            device.destroy_image_view(self.view, None);
            device.destroy_image(self.image, None);
            device.free_memory(self.image_mem, None);
            device.destroy_buffer(self.readback, None);
            device.free_memory(self.readback_mem, None);
        }
    }
}

fn find_memory_type(
    props: &vk::PhysicalDeviceMemoryProperties,
    type_bits: u32,
    flags: vk::MemoryPropertyFlags,
) -> u32 {
    for i in 0..props.memory_type_count {
        if (type_bits & (1 << i)) != 0
            && props.memory_types[i as usize].property_flags.contains(flags)
        {
            return i;
        }
    }
    panic!("no suitable Vulkan memory type for {flags:?}");
}

/// Build the one-attachment render pass egui paints into. Clears to the theme
/// background, leaves the image in TRANSFER_SRC_OPTIMAL for the readback copy.
fn create_ui_render_pass(device: &ash::Device) -> vk::RenderPass {
    let color = vk::AttachmentDescription::default()
        .format(UI_FORMAT)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
    let color_refs = [vk::AttachmentReference::default()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)];
    let subpass = vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&color_refs);
    let deps = [
        vk::SubpassDependency::default()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE),
        vk::SubpassDependency::default()
            .src_subpass(0)
            .dst_subpass(vk::SUBPASS_EXTERNAL)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_stage_mask(vk::PipelineStageFlags::TRANSFER)
            .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ),
    ];
    let attachments = [color];
    let subpasses = [subpass];
    unsafe {
        device.create_render_pass(
            &vk::RenderPassCreateInfo::default()
                .attachments(&attachments)
                .subpasses(&subpasses)
                .dependencies(&deps),
            None,
        )
    }
    .expect("create ui render pass")
}

/// Everything bound to the Spectra device for egui rendering.
///
/// FIELD ORDER IS LOAD-BEARING for teardown: Rust drops fields in declaration
/// order, and `present` (the `VulkanPresent`) OWNS the ash device — its Drop
/// destroys the device. `egui_renderer` (egui-ash-renderer) has its own Drop
/// that touches the device, so it MUST drop BEFORE `present`. Hence `present` is
/// declared LAST. Our own resources are freed in the explicit `Drop` below,
/// which runs before any field drops, while `present` (and the device) is alive.
struct SpectraGui {
    egui_renderer: Renderer,
    render_pass: vk::RenderPass,
    target: UiTarget,
    /// Owned clone of the ash device (egui-ash-renderer holds its own clone too).
    device: ash::Device,
    queue: vk::Queue,
    command_pool: vk::CommandPool,
    /// One-shot command buffer reused each frame for the egui pass + copy.
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
    display: (u32, u32),
    /// MUST be last: owns + destroys the ash device on drop.
    present: PresentBackend,
}

impl SpectraGui {
    /// Windowed: build the Spectra present over `window`, then the egui renderer
    /// on the present's own device.
    fn new_windowed(window: &Arc<Window>, bg: [u8; 4]) -> Self {
        let backend = VulkanSlangBackend::new(0).expect("VulkanSlangBackend::new(0)");
        let display = (WINDOW_WIDTH, WINDOW_HEIGHT);
        let present = select_present(PresentChoice::Vulkan, window, backend, display, display)
            .expect("select_present (vulkan)");
        Self::from_present(present, display, bg)
    }

    /// Headless (`--frames/--shot`): `VulkanPresent::new_offscreen` (no window),
    /// then the egui renderer on its device. The presented image is read back.
    fn new_offscreen(w: u32, h: u32, bg: [u8; 4]) -> Self {
        let backend = VulkanSlangBackend::new(0).expect("VulkanSlangBackend::new(0)");
        let present = PresentBackend::Vulkan(
            VulkanPresent::new_offscreen(backend, (w, h), (w, h))
                .expect("VulkanPresent::new_offscreen"),
        );
        Self::from_present(present, (w, h), bg)
    }

    fn from_present(present: PresentBackend, display: (u32, u32), _bg: [u8; 4]) -> Self {
        let backend = present
            .vulkan_backend()
            .expect("vulkan present backend for egui");
        let device = backend.vk_device().clone();
        let queue = backend.vk_queue();
        let command_pool = backend.vk_command_pool();

        let render_pass = create_ui_render_pass(&device);
        let egui_renderer = Renderer::with_default_allocator(
            backend.vk_instance(),
            backend.vk_physical_device(),
            device.clone(),
            render_pass,
            Options { in_flight_frames: 1, srgb_framebuffer: false, ..Default::default() },
        )
        .expect("egui-ash-renderer init");

        let target = UiTarget::new(backend, render_pass, display.0, display.1);

        let cmd = unsafe {
            device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
        }
        .expect("alloc cmd")[0];
        let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }
            .expect("create fence");

        Self {
            present,
            egui_renderer,
            render_pass,
            target,
            device,
            queue,
            command_pool,
            cmd,
            fence,
            display,
        }
    }

    fn resize(&mut self, w: u32, h: u32) {
        if w == 0 || h == 0 || (w == self.target.width && h == self.target.height) {
            return;
        }
        unsafe { self.device.device_wait_idle().ok() };
        self.target.destroy(&self.device);
        let backend = self.present.vulkan_backend().expect("vulkan backend");
        self.target = UiTarget::new(backend, self.render_pass, w, h);
        self.present.resize((w, h)).expect("present resize");
        self.display = (w, h);
    }

    /// Paint the tessellated egui frame into the offscreen image, copy it to the
    /// host readback buffer, and return the host RGBA bytes (display.w*h*4).
    fn paint_to_host(
        &mut self,
        full_output: &egui::FullOutput,
        tris: &[egui::ClippedPrimitive],
        bg: [u8; 4],
        ppp: f32,
    ) -> Vec<u8> {
        let (w, h) = (self.target.width, self.target.height);

        // 1) Upload egui font/color atlas deltas onto the Spectra device.
        self.egui_renderer
            .set_textures(self.queue, self.command_pool, &full_output.textures_delta.set)
            .expect("egui set_textures");

        // 2) Record: render pass (clear=bg) -> egui draw -> copy image to host.
        unsafe {
            self.device
                .reset_command_buffer(self.cmd, vk::CommandBufferResetFlags::empty())
                .ok();
            self.device
                .begin_command_buffer(
                    self.cmd,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .expect("begin cmd");

            let clear = [vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [
                        srgb_to_linear(bg[0]),
                        srgb_to_linear(bg[1]),
                        srgb_to_linear(bg[2]),
                        1.0,
                    ],
                },
            }];
            self.device.cmd_begin_render_pass(
                self.cmd,
                &vk::RenderPassBeginInfo::default()
                    .render_pass(self.render_pass)
                    .framebuffer(self.target.framebuffer)
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: vk::Extent2D { width: w, height: h },
                    })
                    .clear_values(&clear),
                vk::SubpassContents::INLINE,
            );
            self.egui_renderer
                .cmd_draw(self.cmd, vk::Extent2D { width: w, height: h }, ppp, tris)
                .expect("egui cmd_draw");
            self.device.cmd_end_render_pass(self.cmd);

            // Image is now TRANSFER_SRC_OPTIMAL (render pass final layout).
            let region = vk::BufferImageCopy::default()
                .buffer_offset(0)
                .buffer_row_length(0)
                .buffer_image_height(0)
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                .image_extent(vk::Extent3D { width: w, height: h, depth: 1 });
            self.device.cmd_copy_image_to_buffer(
                self.cmd,
                self.target.image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                self.target.readback,
                &[region],
            );

            self.device.end_command_buffer(self.cmd).expect("end cmd");

            self.device.reset_fences(&[self.fence]).ok();
            let cmds = [self.cmd];
            let submit = vk::SubmitInfo::default().command_buffers(&cmds);
            self.device
                .queue_submit(self.queue, &[submit], self.fence)
                .expect("queue submit");
            self.device
                .wait_for_fences(&[self.fence], true, u64::MAX)
                .expect("wait fence");
        }

        // 3) Free atlas deltas now that the submit completed.
        self.egui_renderer
            .free_textures(&full_output.textures_delta.free)
            .expect("egui free_textures");

        // 4) Read the host buffer.
        let mut rgba = vec![0u8; self.target.readback_size as usize];
        unsafe {
            let ptr = self
                .device
                .map_memory(
                    self.target.readback_mem,
                    0,
                    self.target.readback_size,
                    vk::MemoryMapFlags::empty(),
                )
                .expect("map readback") as *const u8;
            std::ptr::copy_nonoverlapping(ptr, rgba.as_mut_ptr(), rgba.len());
            self.device.unmap_memory(self.target.readback_mem);
        }
        rgba
    }
}

impl Drop for SpectraGui {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().ok();
            self.target.destroy(&self.device);
            self.device.destroy_render_pass(self.render_pass, None);
            self.device.destroy_fence(self.fence, None);
            // command_pool is owned by the backend; do not destroy it here.
        }
    }
}

/// Convert an sRGB-encoded 8-bit channel to a linear 0..1 value for the Vulkan
/// clear color (the clear is interpreted in the attachment's color space).
fn srgb_to_linear(c: u8) -> f32 {
    let c = c as f32 / 255.0;
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

pub struct EditorHost {
    window: Option<Arc<Window>>,
    gui: Option<SpectraGui>,
    egui_ctx: egui::Context,
    egui_state: Option<egui_winit::State>,
    shell: EditorShell,
    tokens: Tokens,
    bg: [u8; 4],
    cli: Cli,
    frames_rendered: u32,
    reset_history: bool,
}

impl EditorHost {
    pub fn new(cli: Cli) -> Self {
        let tokens = load_tokens(cli.light);
        let bg = tokens.color("surface.bg.0");
        let shell = build_shell(tokens.clone(), &cli);
        Self {
            window: None,
            gui: None,
            egui_ctx: egui::Context::default(),
            egui_state: None,
            shell,
            tokens,
            bg,
            cli,
            frames_rendered: 0,
            reset_history: true,
        }
    }

    /// Headless proof: render `frames` frames offscreen on the Spectra stack and
    /// (optionally) write the last presented image to a PNG. No window, no event
    /// loop — uses `VulkanPresent::new_offscreen` + readback.
    pub fn run_headless(mut self) {
        let frames = self.cli.frames.unwrap_or(1).max(1);
        let (w, h) = (WINDOW_WIDTH, WINDOW_HEIGHT);

        // egui input + theme (same dark, icon-led shell as the windowed host).
        vox_ui::design::icons::install(&self.egui_ctx);
        vox_ui::egui_theme::apply(&self.egui_ctx, &self.tokens);
        self.egui_ctx.set_pixels_per_point(1.0);

        let mut gui = SpectraGui::new_offscreen(w, h, self.bg);
        println!(
            "[ochroma_editor] Spectra present (offscreen): backend={:?} device={}",
            gui.present.kind(),
            gui.present.device_name()
        );

        let mut last_rgba: Vec<u8> = Vec::new();
        for f in 0..frames {
            let raw_input = egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::pos2(0.0, 0.0),
                    egui::vec2(w as f32, h as f32),
                )),
                ..Default::default()
            };
            let full_output = self.egui_ctx.run(raw_input, |ctx| self.shell.ui(ctx));
            let tris = self
                .egui_ctx
                .tessellate(full_output.shapes.clone(), 1.0);
            last_rgba = gui.paint_to_host(&full_output, &tris, self.bg, 1.0);
            // Present the host RGBA through the Spectra present stack each frame.
            let reset = f == 0;
            gui.present
                .present_frame(&PresentFrame::from_rgba(&last_rgba, (w, h), (w, h), reset))
                .expect("present_frame (offscreen)");
        }

        // Prefer the image the present stack actually presented (BGRA->RGBA
        // normalized for us); fall back to our painted RGBA if readback is empty.
        let presented = match &gui.present {
            PresentBackend::Vulkan(v) => {
                v.read_presented_rgba(v.last_image_index()).ok()
            }
            #[allow(unreachable_patterns)]
            _ => None,
        };
        let rgba = match presented {
            Some(p) if p.len() == (w * h * 4) as usize => p,
            _ => last_rgba,
        };

        if let Some(path) = self.cli.shot.clone() {
            if let Some(parent) = std::path::Path::new(&path).parent()
                && !parent.as_os_str().is_empty()
            {
                let _ = std::fs::create_dir_all(parent);
            }
            if let Err(e) = cpu_render::write_png(&path, &rgba, w, h) {
                eprintln!("[ochroma_editor] failed to write {path}: {e}");
                std::process::exit(1);
            }
            let nonbg = cpu_render::non_background_fraction(&rgba, self.bg, 6) * 100.0;
            let bytes = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
            println!(
                "[ochroma_editor] wrote {path} ({bytes} bytes), {nonbg:.1}% non-background pixels, {w}x{h}, present=spectra-vulkan-offscreen"
            );
        } else {
            let nonbg = cpu_render::non_background_fraction(&rgba, self.bg, 6) * 100.0;
            println!(
                "[ochroma_editor] rendered {frames} frames offscreen, {nonbg:.1}% non-background pixels, {w}x{h}"
            );
        }
        println!("[ochroma_editor] rendered {frames} frames (--frames {frames}), exiting 0");
    }

    fn redraw(&mut self, event_loop: &ActiveEventLoop) {
        let window = match &self.window {
            Some(w) => w.clone(),
            None => return,
        };
        if self.gui.is_none() {
            return;
        }

        let ppp = window.scale_factor() as f32;
        let (w, h) = self.gui.as_ref().unwrap().display;

        let raw_input = self
            .egui_state
            .as_mut()
            .expect("egui_state")
            .take_egui_input(&window);
        let full_output = self.egui_ctx.run(raw_input, |ctx| self.shell.ui(ctx));
        self.egui_state
            .as_mut()
            .unwrap()
            .handle_platform_output(&window, full_output.platform_output.clone());
        let tris = self.egui_ctx.tessellate(full_output.shapes.clone(), ppp);

        let bg = self.bg;
        let gui = self.gui.as_mut().unwrap();
        let rgba = gui.paint_to_host(&full_output, &tris, bg, ppp);
        let reset = std::mem::take(&mut self.reset_history);
        if let Err(e) =
            gui.present
                .present_frame(&PresentFrame::from_rgba(&rgba, (w, h), (w, h), reset))
        {
            eprintln!("[ochroma_editor] present error: {e}; rebuilding swapchain");
            gui.present.resize((w, h)).ok();
            self.reset_history = true;
        }

        self.frames_rendered += 1;
        if let Some(n) = self.cli.frames
            && self.frames_rendered >= n
        {
            println!(
                "[ochroma_editor] rendered {} frames (--frames {}), exiting 0",
                self.frames_rendered, n
            );
            event_loop.exit();
            return;
        }
        window.request_redraw();
    }
}

impl ApplicationHandler for EditorHost {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title(WINDOW_TITLE)
            .with_inner_size(winit::dpi::PhysicalSize::new(WINDOW_WIDTH, WINDOW_HEIGHT));
        let window = Arc::new(
            event_loop
                .create_window(attrs)
                .expect("Failed to create window"),
        );

        vox_ui::design::icons::install(&self.egui_ctx);
        vox_ui::egui_theme::apply(&self.egui_ctx, &self.tokens);

        let egui_state = egui_winit::State::new(
            self.egui_ctx.clone(),
            egui::ViewportId::ROOT,
            &*window,
            Some(window.scale_factor() as f32),
            None,
            None,
        );

        let gui = SpectraGui::new_windowed(&window, self.bg);
        println!(
            "[ochroma_editor] Spectra present (windowed): backend={:?} device={}",
            gui.present.kind(),
            gui.present.device_name()
        );
        println!("[ochroma_editor] editor shell live on the Spectra stack — dock: World | Viewport | Node Graph | Properties | Content | Output Log + Crucible/Forge/FloraPrime plugin tabs. Ctrl+K opens the command palette.");

        self.egui_state = Some(egui_state);
        self.gui = Some(gui);
        self.window = Some(window.clone());
        window.request_redraw();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        if let Some(egui_state) = &mut self.egui_state
            && let Some(window) = &self.window
        {
            let _ = egui_state.on_window_event(window, &event);
        }
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if size.width > 0
                    && size.height > 0
                    && let Some(gui) = &mut self.gui
                {
                    gui.resize(size.width, size.height);
                    self.reset_history = true;
                }
            }
            WindowEvent::RedrawRequested => self.redraw(event_loop),
            _ => {}
        }
    }
}

/// Entry for the spectra-stack editor. Headless when `--frames` is set, else a
/// live window driven by the Spectra present stack.
pub fn run(cli: Cli) {
    if cli.frames.is_some() {
        // Headless proof path: no event loop needed.
        EditorHost::new(cli).run_headless();
        return;
    }
    let event_loop = EventLoop::new().expect("Failed to create event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut host = EditorHost::new(cli);
    event_loop.run_app(&mut host).expect("Event loop failed");
}

// raw-window-handle bound used by select_present is satisfied by winit's Window.
#[allow(dead_code)]
fn _assert_window_handles<W: HasWindowHandle + HasDisplayHandle>() {}
