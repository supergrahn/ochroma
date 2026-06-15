//! The **Sequencer** panel — the in-editor cinematic timeline (Unreal-Sequencer
//! class), driving the pure [`CameraSequence`] from `vox_render::cine`.
//!
//! This is the editor face of the cinematic camera tool: a frame ruler, one row
//! per animated track with diamond keys, a transport (play / scrub / loop), the
//! photographic DoF / focus / shutter inspectors, a **Key from viewport** button
//! that snapshots the live orbit camera into an eye keyframe, and a **Render
//! Sequence** button that runs the same deterministic offline export the
//! `cine_export` bin uses.
//!
//! The pixel↔frame geometry and hit-testing live in [`SequencerView`] so they
//! are unit-testable headless (no egui context, no GPU). [`ui`] is the egui
//! painter that draws the timeline and wires interaction to a `CinePlayer`
//! playhead and the export entry point.
//!
//! The panel follows the **panel-architecture decision**: layout, the per-track
//! data map, and hit-testing are co-located here rather than living as arms of a
//! central enum. The host registers it in `host.rs`.

use vox_render::{CameraSequence, CineFraming, KeyRef};

/// Which animated track a row represents. The position lanes are split into
/// X/Y/Z (one diamond row each) so a key can be selected per component; the
/// scalar tracks are single-value rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Track {
    EyeX,
    EyeY,
    EyeZ,
    Target,
    Focus,
}

impl Track {
    /// All tracks in top-to-bottom row order.
    pub const ALL: [Track; 5] = [
        Track::EyeX,
        Track::EyeY,
        Track::EyeZ,
        Track::Target,
        Track::Focus,
    ];

    /// Row index (0 = top), used to compute the row's vertical center.
    pub fn row_index(self) -> usize {
        Track::ALL.iter().position(|&t| t == self).unwrap_or(0)
    }

    /// Friendly label drawn in the track header.
    pub fn label(self) -> &'static str {
        match self {
            Track::EyeX => "Eye.X",
            Track::EyeY => "Eye.Y",
            Track::EyeZ => "Eye.Z",
            Track::Target => "Target",
            Track::Focus => "Focus",
        }
    }
}

/// Pixel↔frame geometry for the timeline. Pure; no egui context required, so the
/// hit-testing is fully unit-testable headless.
#[derive(Debug, Clone, Copy)]
pub struct SequencerView {
    /// Horizontal pixels per timeline frame.
    pub px_per_frame: f32,
    /// X pixel of frame 0 (the ruler origin / left edge of the track area).
    pub ruler_x0: f32,
    /// Y pixel of the top of the first track row.
    pub rows_y0: f32,
    /// Vertical pixels per track row.
    pub row_h: f32,
    /// Pixel radius within which a click counts as hitting a key diamond.
    pub key_pick_radius: f32,
}

impl SequencerView {
    /// New view with the given scale and ruler origin, sensible row defaults.
    pub fn new(px_per_frame: f32, ruler_x0: f32) -> Self {
        Self {
            px_per_frame,
            ruler_x0,
            rows_y0: 0.0,
            row_h: 18.0,
            key_pick_radius: 6.0,
        }
    }

    /// X pixel of a (possibly fractional) frame.
    pub fn frame_x(&self, frame: f32) -> f32 {
        self.ruler_x0 + frame * self.px_per_frame
    }

    /// Inverse of [`frame_x`](Self::frame_x): pixel → fractional frame.
    pub fn x_frame(&self, x: f32) -> f32 {
        (x - self.ruler_x0) / self.px_per_frame
    }

    /// Vertical center of a track's row.
    pub fn row_y(&self, track: Track) -> f32 {
        self.rows_y0 + (track.row_index() as f32 + 0.5) * self.row_h
    }

    /// The keys of a track as `(frame, value)`, read off the sequence.
    pub fn track_keys(&self, seq: &CameraSequence, track: Track) -> Vec<KeyRef> {
        match track {
            Track::EyeX => seq.eye_x_keys(),
            Track::EyeY => seq.eye_y_keys(),
            Track::EyeZ => seq.eye_z_keys(),
            Track::Target => seq.target_keys_ref(),
            Track::Focus => seq.focus_keys(),
        }
    }

    /// Hit-test a click against a track's key diamonds. Returns the index (into
    /// `track_keys`) of the nearest key whose diamond contains the click within
    /// `key_pick_radius`, or `None` if the click missed every key.
    ///
    /// The vertical test uses the row center; the horizontal test uses the key's
    /// frame pixel. This is the inverse of the diamond draw in [`ui`].
    pub fn hit_key(
        &self,
        seq: &CameraSequence,
        track: Track,
        pos: egui::Pos2,
    ) -> Option<usize> {
        let row_y = self.row_y(track);
        if (pos.y - row_y).abs() > self.row_h * 0.5 + self.key_pick_radius {
            return None;
        }
        let keys = self.track_keys(seq, track);
        let mut best: Option<(usize, f32)> = None;
        for (i, k) in keys.iter().enumerate() {
            let kx = self.frame_x(k.frame() as f32);
            let dx = (pos.x - kx).abs();
            if dx <= self.key_pick_radius {
                match best {
                    Some((_, bd)) if bd <= dx => {}
                    _ => best = Some((i, dx)),
                }
            }
        }
        best.map(|(i, _)| i)
    }
}

/// State the Sequencer panel owns across frames: the authored sequence, the
/// transport playhead (in frames), play/loop flags, the current scale/scroll,
/// and the selected key. The host stores one of these per Sequencer tab.
pub struct SequencerState {
    /// The authored cinematic. The single source of truth.
    pub seq: CameraSequence,
    /// Transport playhead, in frames (fractional while scrubbing).
    pub playhead: f64,
    /// Whether the transport is playing.
    pub playing: bool,
    /// Whether playback loops at the end.
    pub looping: bool,
    /// Pixels per frame (timeline zoom).
    pub px_per_frame: f32,
    /// X pixel of frame 0.
    pub ruler_x0: f32,
    /// Currently selected `(track, key index)`, if any.
    pub selected: Option<(Track, usize)>,
    /// Last requested export, surfaced to the host to actually run.
    pub render_request: Option<RenderRequest>,
}

/// A request to render the sequence offline, raised by the **Render Sequence**
/// button and drained by the host (which owns the Spectra backend).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderRequest {
    pub out_dir: String,
    pub fps: u32,
    pub spp: u32,
    pub width: u32,
    pub height: u32,
}

impl SequencerState {
    /// New panel state wrapping an authored sequence.
    pub fn new(seq: CameraSequence) -> Self {
        Self {
            seq,
            playhead: 0.0,
            playing: false,
            looping: true,
            px_per_frame: 4.0,
            ruler_x0: 120.0,
            selected: None,
            render_request: None,
        }
    }

    /// The geometry view derived from the current zoom/scroll.
    pub fn view(&self) -> SequencerView {
        SequencerView::new(self.px_per_frame, self.ruler_x0)
    }

    /// Advance the playhead by `dt_frames` when playing, wrapping if looping.
    /// Pure; deterministic given the same `dt_frames` (no wall-clock).
    pub fn advance(&mut self, dt_frames: f64) {
        if !self.playing {
            return;
        }
        let dur = self.seq.duration() as f64;
        self.playhead += dt_frames;
        if self.playhead > dur {
            if self.looping {
                self.playhead = if dur > 0.0 { self.playhead % dur } else { 0.0 };
            } else {
                self.playhead = dur;
                self.playing = false;
            }
        }
    }

    /// Snapshot a live viewport framing into an eye key at the current playhead.
    /// Backs the **Key from viewport** button.
    pub fn key_from_viewport(&mut self, framing: &CineFraming, fov_y: f32) {
        let frame = self.playhead.round() as i32;
        self.seq.set_key_from_framing(framing, frame, fov_y);
    }
}

/// Draw the Sequencer timeline and wire interaction.
///
/// Returns the [`SequencerView`] used this frame (so the caller can correlate
/// painter geometry). Scrubbing/keys mutate `state.playhead` and `state.seq`;
/// the **Render Sequence** button sets `state.render_request` for the host to
/// drain. `live_framing` is the current viewport orbit (for **Key from
/// viewport**); pass `None` to disable that button.
pub fn ui(
    ui: &mut egui::Ui,
    state: &mut SequencerState,
    live_framing: Option<(CineFraming, f32)>,
) -> SequencerView {
    let avail = ui.available_rect_before_wrap();
    let ruler_h = 18.0;
    let mut view = state.view();
    view.rows_y0 = avail.top() + ruler_h + 2.0;

    // Transport row.
    ui.horizontal(|ui| {
        let label = if state.playing { "Pause" } else { "Play" };
        if ui.button(label).clicked() {
            state.playing = !state.playing;
        }
        if ui.button("Stop").clicked() {
            state.playing = false;
            state.playhead = 0.0;
        }
        ui.checkbox(&mut state.looping, "Loop");
        ui.label(format!(
            "frame {:.1} / {}",
            state.playhead,
            state.seq.duration()
        ));
        if let Some((framing, fov_y)) = live_framing {
            if ui.button("Key from viewport").clicked() {
                state.key_from_viewport(&framing, fov_y);
            }
        }
        if ui.button("Render Sequence").clicked() {
            state.render_request = Some(RenderRequest {
                out_dir: "out".to_string(),
                fps: state.seq.fps().fps().round() as u32,
                spp: state.seq.config().default_spp_export,
                width: 1920,
                height: 1080,
            });
        }
    });

    let (rect, resp) =
        ui.allocate_exact_size(egui::vec2(avail.width(), 8.0 * view.row_h), egui::Sense::click_and_drag());
    let painter = ui.painter_at(rect);
    let visuals = ui.visuals().clone();

    // Frame ruler ticks every 30 frames.
    let dur = state.seq.duration().max(1);
    let mut f = 0;
    while f <= dur {
        let x = view.frame_x(f as f32);
        painter.line_segment(
            [egui::pos2(x, rect.top()), egui::pos2(x, rect.top() + ruler_h)],
            egui::Stroke::new(1.0, visuals.weak_text_color()),
        );
        painter.text(
            egui::pos2(x + 2.0, rect.top()),
            egui::Align2::LEFT_TOP,
            f.to_string(),
            egui::FontId::proportional(9.0),
            visuals.weak_text_color(),
        );
        f += 30;
    }

    // Per-track rows + diamond keys.
    for track in Track::ALL {
        let row_y = view.row_y(track);
        painter.text(
            egui::pos2(rect.left() + 4.0, row_y),
            egui::Align2::LEFT_CENTER,
            track.label(),
            egui::FontId::proportional(10.0),
            visuals.text_color(),
        );
        for k in view.track_keys(&state.seq, track) {
            let kx = view.frame_x(k.frame() as f32);
            let c = egui::pos2(kx, row_y);
            let r = 4.0;
            let pts = vec![
                egui::pos2(c.x, c.y - r),
                egui::pos2(c.x + r, c.y),
                egui::pos2(c.x, c.y + r),
                egui::pos2(c.x - r, c.y),
            ];
            painter.add(egui::Shape::convex_polygon(
                pts,
                visuals.selection.bg_fill,
                egui::Stroke::new(1.0, visuals.text_color()),
            ));
        }
    }

    // Playhead.
    let ph_x = view.frame_x(state.playhead as f32);
    painter.line_segment(
        [
            egui::pos2(ph_x, rect.top()),
            egui::pos2(ph_x, rect.bottom()),
        ],
        egui::Stroke::new(1.5, visuals.warn_fg_color),
    );

    // Interaction: drag in the ruler band scrubs; click on a key selects.
    if let Some(p) = resp.interact_pointer_pos() {
        if resp.dragged() && p.y <= rect.top() + ruler_h {
            let frame = view.x_frame(p.x).clamp(0.0, dur as f32);
            state.playhead = frame as f64;
        } else if resp.clicked() {
            let mut hit = None;
            for track in Track::ALL {
                if let Some(i) = view.hit_key(&state.seq, track, p) {
                    hit = Some((track, i));
                    break;
                }
            }
            state.selected = hit;
        }
    }

    view
}

#[cfg(test)]
mod tests {
    use super::*;
    use vox_render::eye_from_framing;

    const DEMO_JSON: &str = r#"{ "fps":{"num":60,"den":1}, "duration":120,
        "eye_keys":[[0,[0,0,0]],[30,[1,2,0]],[90,[5,2,0]],[120,[6,0,0]]],
        "target_keys":[[0,[0,0,0]],[120,[0,0,0]]],
        "focus_distance":[[0,10.0],[30,50.0]],
        "aperture_fstop":[[0,2.8]], "focal_length_mm":[[0,50.0]],
        "fov_y_deg":[[0,45.0]] }"#;

    #[test]
    fn sequencer_hit_test() {
        let mut seq = CameraSequence::load_json(DEMO_JSON).unwrap();
        let view = SequencerView::new(/*px_per_frame*/ 4.0, /*ruler_x0*/ 10.0);
        // A click at the x-pixel for frame 30 hits that key.
        let click_x = view.ruler_x0 + 30.0 * view.px_per_frame;
        let hit = view.hit_key(
            &seq,
            Track::EyeX,
            egui::pos2(click_x, view.row_y(Track::EyeX)),
        );
        let hit_frame = hit.map(|k| seq.eye_x_keys()[k].frame());
        println!("hit key frame = {:?}", hit_frame);
        assert_eq!(hit_frame, Some(30));

        // Key-from-viewport writes the framing-derived eye.
        let framing = CineFraming {
            focus: glam::vec3(0.0, 0.0, 0.0),
            radius: 10.0,
            zoom: 1.0,
            yaw: 0.0,
            pitch: 0.5,
        };
        seq.set_key_from_framing(&framing, /*frame*/ 45, /*fov_y*/ 0.785);
        let p = {
            seq.rebuild_arc_lut();
            seq.eval(45.0)
        };
        let expect = eye_from_framing(&framing);
        println!("eye {:?} vs expect {:?}", p.eye(), expect);
        assert!(
            (p.eye() - expect).length() < 1e-3,
            "eye {:?} vs {:?}",
            p.eye(),
            expect
        );
    }

    #[test]
    fn advance_loops_at_duration() {
        let seq = CameraSequence::load_json(DEMO_JSON).unwrap();
        let mut st = SequencerState::new(seq);
        st.playing = true;
        st.looping = true;
        st.playhead = 119.0;
        st.advance(5.0); // 124 -> wraps modulo 120 -> 4.0
        assert!((st.playhead - 4.0).abs() < 1e-9, "playhead {}", st.playhead);
    }
}
