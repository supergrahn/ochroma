# Plan: SOTA road tools

Status: Research DONE · Foundational tool SHIPPED (`civitas e8695d8`) · Waves 2–8 below.
Repo: the city builder at `~/Ochroma/projects/urban_horizon`. Hooks: `src/road/mod.rs` (`RoadNetwork`, dedup, `to_road_graph`), `src/command/mod.rs` (dispatch + `ViewState`), `src/ui/build_tools.rs` (palette), `src/render_gpu/hud.rs` (draw + `*_at`), `src/bin/play.rs` (input).

## Why (CS2 research, ranked by player value)
1. Continuous/chained drawing (no phantom dead-ends, no re-snap micro-gaps). 2. Six draw modes incl. **grid** (a block in 3 clicks) and complex-curve splines. 3. Automatic intersection generation + premade interchanges. 4. Roundabout & highway-ramp auto-geometry the traffic AI uses. 5. Parallel/offset mode. 6. Seven independently-toggleable snap categories (incl. guideline ghost-lines, zone-cell). 7. Stepped elevation + live slope %. 8. In-place replace/upgrade. 9. Live length+angle readout.

## Shipped (Wave 1)
Roads HUD category (Avenue/Street/Alley) · click-anchor → rubber-band w/ live length+angle chip → click-commit → chained continuous draw · node-snap to endpoints · `S` toggles 15° angle-snap · right-click cancel. `RoadNetwork::angle_snap`/`snap_road_end`. 22 road tests (5 new, real geometric assertions).

## Waves 2–8 (ordered by remaining CS2 delta)

### Wave 2 — Curve mode (biggest remaining delta)
`RoadDraft::Curve{start,control,end}` (3-click: 2nd click = Bézier handle); `RoadNetwork::add_curve_segment(start,control,end,class,segments)` subdivides into straight sub-segments; HUD draws the arc.
**Done When:** `cargo test -p urban_horizon --lib road::curve` prints `3-click bezier -> 8 sub-segments, shared interior nodes, max chord error < 0.25 m`.

### Wave 3 — Auto intersection splitting
Segment-vs-segment crossing test in `add_segment` (or a `split_crossings()` pass): a new segment crossing an existing one splits BOTH at the crossing and makes a 4-way node (today's dedup is endpoint-only).
**Done When:** `cargo test -p urban_horizon --lib road::crossing_split` prints `2 crossing segments -> 4 segments, 1 shared degree-4 node at the intersection`.

### Wave 4 — Replace/upgrade tool
`RoadNetwork::replace_segment(seg_idx,new_class)` + a `BuildTool::RoadReplace` mode; HUD highlights hovered segment, click swaps class, junctions preserved.
**Done When:** `cargo test -p urban_horizon --lib road::replace` prints `Street->Avenue keeps both end nodes + neighbour links, class changed`.

### Wave 5 — Parallel/offset mode
`ViewState.road_parallel_offset: Option<f32>`; `CommitRoad` lays a mirrored second segment at the perpendicular offset; HUD renders both rubber-bands.
**Done When:** `cargo test -p urban_horizon --lib road::parallel` prints `offset 8 m -> 2 parallel segments, perpendicular spacing == 8.0 ± 0.01`.

### Wave 6 — Precision bearing overlay
Expose the already-computed bearing as a HUD chip at the cursor (`047°`), matching CS2's angle readout.
**Done When:** running `play`, dragging a road shows a live bearing chip; `road::bearing` test asserts `dir (1,0) -> 090°, (0,1) -> 000°`.

### Wave 7 — Elevation / grade
`RoadNode.y: f32`; scroll-wheel adjusts endpoint elevation during drag w/ live `%grade`; segments with `y!=0` render as bridge/cut, tunnel auto-trigger past a threshold.
**Done When:** `cargo test -p urban_horizon --lib road::grade` prints `+10 m over 100 m run -> 10% grade, segment flagged Bridge`.

### Wave 8 — Delete tool
`BuildTool::RoadDelete` highlights nearest segment, click removes via `game.remove_road(seg_idx)` logging `RemoveRoad{seg_idx}` for undo.
**Done When:** `cargo test -p urban_horizon --lib road::delete` prints `remove mid-segment -> network loses 1 segment, undo restores it, replay-exact`.

## Notes
All ops on `RoadNetwork` are deterministic + replay-safe (action-log). Each wave: implement on `RoadNetwork` + wire the tool in `command`/`play.rs`/`hud.rs` + real-assertion tests, no `assert!(is_some())`.
