# Tools UX — road + terrain/terraforming, from the player's seat

Game: **Haven Horizon**. We have the road mechanisms (waves 1-8) and the 3D-SDF terraform mechanisms (carve/union/smooth). This doc is about the *experience* — how a player actually wields them — and the gap between "the mechanism exists" and "it feels good." Reference: CS2 road tools are loved for *feel*, not feature-count.

## Player's mental model
- **Roads:** "I draw lines and the city wires itself up." The player should never hand-build a junction, never re-pick the tool mid-draw, and always *see* what will happen before committing.
- **Terraforming:** "I sculpt the land like clay." Raise/lower is table-stakes; the differentiator is **digging in 3D** — tunnels, caves, cliff faces, overhangs — which no mainstream builder does. The UX must make a 3D op feel as natural as a 2D brush.

---
## ROAD TOOLS — UX

**1. One tool, modal — not eight tools.** The player picks "Road," then the *class* (Avenue/Street/Alley) and the *mode* (straight/curve) are lightweight toggles, not separate palette entries. Replace/Delete are verbs on the road tool, not siblings. Today we have 6 palette tiles (3 classes × straight + curve/replace/delete) — that's tool-soup. Collapse to: class selector + mode chips (Straight · Curve · Grid) + a verb row (Upgrade · Delete) that operate on hover.

**2. Continuous drawing is the default, and it must be obvious.** After committing a segment the tool stays live and the next click extends the chain (we have this). Make the chain *visible*: the committed spine stays highlighted, the pending segment rubber-bands from the last node. Right-click ends the chain (not the tool). Esc cancels the pending segment only.

**3. Confidence before commit — the live readout is the hero.** While dragging: length (m), bearing (NNN°), and grade (%) in a chip *at the cursor*, plus a 90°/45° snap "lock" indicator when angle-snap engages. We have length+bearing+grade; the missing piece is the **snap-lock visual** (the chip turns gold / a tick line appears) so the player feels the snap catch.

**4. Snapping the player can SEE.** CS2's most-praised invisible feature: ghost **guidelines** projected from existing road ends/angles, and node highlights when you're about to connect. We snap to nodes but don't *show* the target. Add: a glowing disc on the node you'll snap to, and faint extension/perpendicular guidelines from nearby roads. This is the single biggest "feels pro" upgrade.

**5. Intersections just happen — confirm them.** Crossing an existing road auto-splits into a junction (wave 3). The player needs a beat of feedback: the new node pulses, the junction footprint flashes. No dialog, no mode.

**6. Modifiers are discoverable, not memorized.** Angle-snap (S), parallel (L), elevation (scroll) are powerful but invisible. Show a small modifier legend on the road tool ("S snap · L parallel · scroll raise") and reflect active modifiers as lit chips. Parallel mode draws *both* rubber-bands live; elevation shows pylon/cut ghosts as you scroll.

**7. The "oops" path is frictionless.** Undo (already action-logged), Delete-on-hover, Upgrade-in-place (drag over a run to repaint its class). Hover highlight in the delete/replace color before the click — we have this; keep it.

**Road UX gaps to close (priority):** (a) snap-target + guideline visualization; (b) snap-lock feedback on the readout; (c) collapse the palette to class+mode+verb; (d) junction-made confirmation pulse; (e) modifier legend.

---
## TERRAIN / TERRAFORMING — UX

**1. It's a brush, always.** One terrain tool; everything is brush size + strength + falloff (soft edge) + a live footprint decal projected on the ground (a ring that hugs the surface, including up cliff faces). The player paints; the land responds in real time (sub-frame — the design guarantees it). Size = bracket keys / scroll; strength = shift-scroll.

**2. The operation is a verb the player picks, and the 2D/3D split must be intuitive:**
- **Raise / Lower** — the classic vertical push; the brush adds/removes along the surface normal. (Maps to union/subtract near the surface.)
- **Flatten** — click to sample a target height, then paint it flat. (Plateau/build-pad.)
- **Smooth** — average the neighborhood (our SMOOTH_UNION). For cleaning up.
- **Carve / Excavate (the 3D verb)** — digs *into* the volume regardless of surface: hold and drag *into* a hillside to bore a tunnel/cave; the brush subtracts a 3D capsule along the drag. This is the cave-maker. Metaphor: "drill."
- **Build / Extrude (the 3D verb)** — adds solid where you paint, including *over* empty space → overhangs and arches. Metaphor: "stack clay."
- **Cliff / Steepen** — a specialized push that drives the surface toward vertical (steepens the local gradient) for clean cliff faces.

**3. The camera teaches the 3D ops.** A flat top-down camera makes caves/overhangs invisible and unusable. When the terrain tool is active (especially Carve/Build/Cliff), bias toward an orbitable 3/4 view and let the player tumble freely; the brush projects onto the first surface the cursor ray hits (so you can paint the *underside* of an overhang or the *back wall* of a cave). This is the make-or-break for 3D terraforming feeling possible at all.

**4. Show depth, not just position.** A height/depth readout at the cursor (+/− m from where you started the stroke), a contour or slope overlay toggle, and for Carve a faint cross-section ghost of what the drill will remove. The player must *see* into the volume they're editing.

**5. Constraints are visible and kind.** Edits blocked under buildings/roads → the blocked footprint tints red and the brush "skips" it rather than silently failing. Water line, protected zones same. (Care-first ethos: the tool explains itself.)

**6. Cost + reversibility.** Terraform has a cost (money/time, sim-driven) shown live as you stroke; full undo (the edit log replays). A stroke is one undo unit, not one-voxel-per-undo.

**Terrain UX gaps to close (priority):** (a) the brush + live footprint decal + size/strength controls; (b) the verb set with Carve/Build/Cliff as first-class (the 3D differentiators); (c) ray-hit-first-surface painting + orbit bias so undersides/backwalls are reachable; (d) depth/cross-section feedback; (e) blocked-region red tint; (f) live cost + stroke-granular undo.

---
## Shared principles
- **Never a modal dialog mid-tool.** Everything is direct manipulation + lightweight on-canvas chips/legends.
- **Always show the result before commit** (rubber-band / brush footprint / cross-section ghost).
- **The novel 3D ops (caves, cliffs, overhangs) are the marketing money-shots** — their UX deserves the most polish, because no competitor has them and they're the hardest to make legible.
- Both tools already have replay-safe mechanisms; the work here is **feedback, visualization, and mode ergonomics** — the layer between "it computes the right thing" and "it feels great."

## Next implementation steps (post-design)
1. Road: snap-target/guideline visualization + snap-lock readout (the top "feels pro" win).
2. Terrain: the brush model + live footprint decal + the verb set (Raise/Lower/Flatten/Smooth/Carve/Build/Cliff), wired to `apply_terraform_stamp`.
3. Both: in-game tool input + GPU upload (the deferred last-mile from the road/terrain workflows), honoring the GPU-first law (brush eval, footprint, extraction on GPU).
