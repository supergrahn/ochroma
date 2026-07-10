---
name: design-real-building
description: Design a photoreal-quality building and author it as a Forge directive/blueprint — imagine the photograph first, then rebuild every detail as geometry+zones. Use for EVERY building authored for Urban Horizon, from a single-family house to a supertall; every typology gets the same realism, detail and quality floor. Pairs with forge-assets (the how); this skill is the WHAT and the bar.
---

# Design a real building (the photograph-first method)

Every building starts as an imagined photograph and ends as a Forge asset that would
survive being placed next to that photograph. One quality floor for all typologies:
**a garden shed and a supertall get the same rules, only different budgets.**
**BINDING SPEC: `urban_horizon/docs/reference/realistic-buildings-forge-spec.md`** — the
adversarially-verified 6-layer realism study with the 10 ordered priorities and exact
Forge file:line changes. Read it BEFORE authoring or touching emitters; its priority
list IS the emitter backlog order, and its REFUTED list is binding (no GreekRevival,
subtract unimplemented, soot/efflorescence need new fields). A building that ignores a
spec layer the emitters already support (reveals, sills, cornices, string courses,
tripartite fields) fails the task.
Companion skill: `forge-assets` (ops, CLI, signatures). Reference bar:
`urban_horizon/docs/reference/buildings/skyscraper-detail-bar-analysis.md`.

## Step 0 — Instruction intake (any "create/make/add building(s)" request)

When the user asks for a building or a building set — however briefly ("add a fire
station", "we need malls") — the request is ALWAYS expanded to the full detail
inventory before any authoring: identify the governing typology schedule (Step 3),
write the complete photograph prompt (Step 1), and enumerate every Step-2 floor item
plus the schedule's program-specific details for THIS building. The user naming a
building type is a request for the WHOLE building — entrance, roofscape, materials,
relief, variants, night state — never just a massed shape with the right label. If
the request is a SET, each member gets its own prompt and the set gets a shared
palette/era logic so it reads as one commission, not clones.

## Step 1 — Write the photograph prompt (before ANY geometry)

Write, in prose, the image-generation prompt for the building as if commissioning a photo
of the real thing. It MUST answer all nine. If you cannot answer one, you are not ready
to author:

1. **Program & era** — what is it, who uses it, when was it built? ("1970s 3-storey brick
   walk-up, 6 flats, landlord-maintained" — era decides proportions, materials, wear.)
2. **Site & orientation** — corner or mid-block? Which face meets the street? Where does
   weather come from? (drives entrance face, weathering gradient, balcony side)
3. **Structure** — what holds it up, and where does that show? (bearing walls → punched
   openings + lintels; frame → curtain wall + expressed slabs; the structure must be
   legible in the geometry)
4. **The module** — floor height + opening rhythm. House ~2.7-3.0m; office 3.6-4.0m.
   EVERYTHING (windows, bands, mullions, courses) derives from this one module.
5. **Massing in three acts** — base (how it meets the ground), middle (the repeating
   shaft/body), top (how it ends: eaves, parapet, cornice, crown, roofscape). Name all
   three; "extruded footprint with a flat lid" is an automatic fail.
6. **Materials, exactly four or more**, each with a different light answer: e.g. glass
   (smooth/transmissive), frame metal (satin), field material (brick/stucco/siding —
   rough, textured), base material (stone/concrete — rougher, darker). Name real
   materials with real colors ("iron-spot brown brick, cream painted timber trim").
7. **The entrance event** — every building meets the street with an event: porch, stoop,
   canopy, lobby, colonnade. Describe it as its own composition.
8. **Light states** — how it looks at noon, golden hour, and 22:00 (which windows glow,
   what spills onto the street).
9. **Age & care** — New/Aged/Weathered/Derelict + WHERE wear concentrates (sills, base
   splash zone, parapet streaks, downpipe runs).

## Step 2 — The universal detail floor (geometry, never texture)

Non-negotiable minimums at EVERY scale. Anything on this list must be MESH with real
depth; painting it on is task failure:

| Detail | Minimum | Applies to |
|---|---|---|
| Opening reveals (window/door set-back into wall) | ≥ 0.15m | everything |
| Sills + lintels/headers on punched openings | modeled, ≥ 0.05m projection | everything |
| Mullion/transom or glazing-bar relief | ≥ 0.06m | everything glazed |
| Base/plinth line (material change + projection at ground) | modeled course | everything |
| Roof edge (eave overhang / parapet with coping / cornice) | ≥ 0.2m projection or height | everything |
| Entrance assembly (door + surround + canopy/porch/stoop) | modeled, own zone | everything |
| Roofscape furniture (chimney/vents on houses; plant, screens, access box on flats+) | ≥ 1 element per 200m² roof | everything |
| Material zones with DISTINCT albedo and roughness | ≥ 4 | everything |
| Spandrel/slab-edge expression on framed buildings | own zone (MAT_SPANDREL) | mid-rise up |
| Lit-window zoning for dusk (clustered, seed-stable, 10-20%) | vision vs lit split | everything with interiors |
| Weathering masks aligned to Step-1 wear map | per-vertex, from cook | everything |
| AABB honesty | mesh XZ = declared footprint ±10% | everything (the D1 law) |

**GROUND-DATUM CONTRACT (how buildings meet streets — never per-building sockets):**
the LOT owns the datum, the building owns the plinth. Rules:
1. A lot fronting a curbed road has datum = the walk top of its primary frontage
   (road_lift + authored curb.height_m); the lot surface/pad is graded TO that datum,
   so the ground floor arrives at back-of-sidewalk level automatically.
2. A lot with NO curb/sidewalk (rural lane, mountable-curb industrial yard) has
   datum = terrain (or road_lift for paved yards) — the same building works there
   unchanged because of rule 3.
3. Every building carries its modeled plinth/foundation (already a Step-2 floor item)
   sized ≥0.3m with the entrance detailed for BOTH arrivals: flush threshold at datum
   (urban) and 1-3 steps down to grade (unpaved). The plinth is the tolerance band —
   it absorbs datum differences and slope, so ONE asset serves every street type.
4. Buildings never compensate with per-asset height offsets/sockets; if a facade
   meets the ground wrong, fix the lot datum or the plinth, never nudge the instance.

**Variation law:** every archetype ships ≥3 variants (mirror, material palette swap,
roofscape/porch permutation) so a street never repeats a silhouette within 5 lots.

**Surface relief law (every material zone, every building):** walls must read as
physical surfaces under raking light, not printed color. Every non-glass zone ships
the FULL map set at 2k — albedo + normal + roughness + **displacement** — with an
AUTHORED `displacement_scale` per material (never a shared default): brick/stone
mortar joints 8-15mm, stucco/render 2-4mm, clapboard/siding lap 10-20mm, concrete
board-form 3-6mm, roof tiles/slates 10-25mm. `normal_scale` 1.0 unless the source map
is authored hot. The renderer's POM/cone-step relief does the work — the authoring
duty is real displacement maps + true scales. Witness: a street-level raking-light
crop (sun ≤25° to the facade) must show mortar/lap shadows; a flat-reading wall at
that angle fails the task. Macro relief ≥0.15m (reveals, bands, courses that big)
stays GEOMETRY per the table above — displacement covers only sub-0.15m texture
relief; never demote a geometry-scale detail to a map.

**MATERIAL REALISM LAW (every zone, no exceptions — shared with design-real-terrain):**
every material must be physically true, not plausible-ish:
1. **Measured sources only** for physical surfaces: real scanned PBR sets (PolyHaven/
   ambientCG class) at 2k+, full map set (albedo/normal/roughness/displacement + AO
   where the set ships it). Hand-invented flat RGB is permitted ONLY for painted/
   coated zones (trim, doors, renders) — and then within measured paint ranges.
2. **Albedo stays in the measured range** (linear): charcoal/asphalt ≥0.04, fresh
   snow ≤0.9; common walls — brick 0.25-0.45, concrete 0.35-0.55, limestone
   0.55-0.75, wood 0.3-0.5. Nothing pure black, nothing pure white, no neon.
3. **Metallic is binary physics**: 1.0 for real bare metal (steel, copper, aluminium,
   zinc), 0.0 for EVERYTHING else — painted metal is a dielectric (metallic 0). The
   0.1-0.4 "sort of metallic" middle is a lie the law forbids.
4. **Roughness carries the material's story**: glass 0.03-0.08, polished stone
   0.15-0.3, painted wood 0.4-0.6, brick/concrete 0.7-0.9, weathered wood 0.8-0.95;
   wet/worn states MOVE roughness (rain-wet -0.3, hand-polished edges -0.2 via masks),
   never a uniform slider.
5. **Glass is real glass**: transmission 0.9, IOR 1.5, thin_walled for panes; tinted/
   low-e via slight color, never opacity hacks. Emissive only for actual light
   sources (signage, lit panes) at plausible luminance — nothing else glows.
6. **Wear is caused, not decorated**: weathering masks follow the Step-1 age map
   (splash zones, sills, drip paths, hand-touch polish); a pristine 1930s building
   or a rusted 5-year-old one needs a stated reason.
7. **Every value is auditable**: the directive records source URI + authored scales;
   the cook witness asserts maps exist and values sit in the class ranges above.

## Step 3 — Per-typology detail schedules (the same floor, scaled budgets)

- **Single-family house:** foundation plinth; siding/brick coursing direction; porch with
  posts + roof (not a flat slab); gable/hip with real rake + eave depth, barge boards;
  chimney; ≥2 window sizes (main vs bathroom/stair); shutters or trim per era; back door
  ≠ front door. Spec fields: `Residential{siding, porch, chimney, dormers, bay_windows}`.
- **Rowhouse/walk-up:** party-wall rhythm visible (downpipes or pilasters at joins);
  stoop + entrance surround per unit; cornice line; ground floor taller than uppers;
  bay windows or balconies on the street face only.
- **Commercial storefront:** ground floor 4.5-5.5m with full-height glazing + real
  mullions; signage band as its own zone (emissive-ready); awnings/canopy geometry;
  upper floors in a CONTRASTING system (punched windows over glass base); service door
  on the side/rear.
- **Mid-rise office/apartment slab:** expressed floor slabs or spandrel bands; regular
  bay rhythm from the module; podium if street-facing; mechanical penthouse, not a bare
  roof; balcony rules (apartments: yes, offices: no).
- **Tower/supertall:** tripartite massing mandatory (podium/shaft/crown); shaft rhythm
  from curtain-wall band ratio; crown is a designed volume (sail/setbacks/lattice/
  penthouse + mast), never a lid; base opens up (double-height lobby, colonnade);
  aviation light socket at the crown.
- **Mall / shopping center / big-box:** the box is honest but DRESSED: entrance
  pavilion as its own volume (double-height glazed, canopy, signage tower);
  facade rhythm from structural bays with material banding (never one flat
  field); loading docks + service yard on the rear face (doors, bumpers, ramp);
  rooftop is a FIFTH FACADE — RTU units, ducts, screens (it's what the player
  sees most); parking-field furniture (cart corrals, lamps) via plot props.
- **Hospital / large civic:** campus massing (podium + ward slabs/wings, not one
  box); double-height glazed entrance hall + canopy (drop-off event); regular
  ward window rhythm; rooftop plant + (hospitals) helipad socket; wayfinding
  scale signage zone; ambulance bay with its own canopy (hospitals).
- **Fire / police station:** the PROGRAM is the facade: apparatus bay doors
  (fire) / secure entry + sally port (police) dominate the street face at true
  door scale (4.5-5m bays); tower element per era (hose tower / comms mast);
  civic material palette (brick/stone base, restrained upper); flagpole socket;
  the entrance must read as public-facing even on a working building.
- **Government / town hall / library:** civic gravity through symmetry OR a
  deliberate modern asymmetry — never accidental; entrance elevated (steps +
  landing) with portico/colonnade or cantilevered canopy; taller ground floor;
  material budget one notch above neighbors (stone/fine brick + metal); plaza
  interface (steps, planters) at the base; clock/emblem/inscription zone.
- **Church / religious:** the silhouette IS the program — tower/spire/dome as a
  real volume with its own crown detail (cross/finial socket, louvred belfry
  openings); nave massing with true roof pitch and gable-end window (rose/
  lancet as geometry-framed glass zones); entrance porch/narthex volume;
  material: stone or brick with dressed trim courses. (Forge `ReligiousSpec`
  exists — drive it.)
- **Crematorium / cemetery chapel:** quiet dignity — low horizontal massing,
  one vertical accent (the flue/tower, honestly expressed, not hidden), covered
  approach walk (pergola/canopy), garden-facing glazed wall; muted material
  palette; the flue terminates in a designed cap, never a bare pipe.
- **UTILITY & INDUSTRIAL LAW (garbage, power, water/sewage, depots): the
  EQUIPMENT is the architecture — model it.** These read fake when they're
  boxes with industrial textures; they read real when the process is legible
  as geometry: waste = tipping hall (tall portal doors) + bunker + stack +
  weighbridge; power = turbine/reactor hall + stacks/cooling towers (real
  hyperboloid or fin-fan banks) + transformer yard (transformers, bus
  structures, insulators as place_part props) + perimeter fence; water/sewage
  = circular clarifier basins (open tanks with rail bridges), digester eggs/
  tanks, blower building, pipe racks between volumes. Every tank/stack/rack is
  a volume or place_part — never a decal. Safety dressing (railings, ladders,
  bollards, hazard-yellow zone accents) is the detail budget here, replacing
  the window rhythm other programs get. Night state: sparse sodium/LED work
  lights, not office windows.

## FULL-POWER LAW (mandatory — violating it fails the task)

Every catalog building is authored with Forge's COMPLETE toolbox, never a reduced gear:
- `hero_level: hero` for every asset that players see up close (all zonables/services);
  `supporting` is permitted ONLY for distant-filler explicitly marked as such in the
  prompt. `background` never ships in the catalog.
- ALL assembly channels on: STRUCTURE + ORNAMENT + DETAIL (the DETAIL atom channel is
  where sills, surrounds, brackets, downpipes live — omitting it is building blind).
- The blueprint node-DAG is the DEFAULT authoring surface for anything beyond a simple
  box (multi-volume massing, place_volume, sweep_profile, crown_kit, place_part for
  entrance columns/canopies/balconies as REAL parts). The flat directive alone is
  acceptable only when the photograph prompt genuinely describes a single-volume
  punched-window box — and even then DETAIL channel + place_part entrance are required.
- If an emitter can't reach the floor (the GAP list), the asset still uses every tool
  that exists, and the gap is recorded — never silently accepted.

## Step 4 — Decompose to Forge (the recreate)

Translate each Step-1/2/3 answer to ops (see `forge-assets` for signatures; gaps table in
the detail-bar analysis):
- Massing acts → `podium_tower`/`setback`/multi-volume union (`place_volume` on
  `roof_anchor`); curved plans → `footprint_curve` beziers / `sweep_profile`.
- Facade → `facade_assign` per volume: system (punched vs curtain), `window_density`
  0.75-0.95 for glass families, spandrel = MAT_SPANDREL, glass transmissive.
- Entrance/parts → `place_part` (door, canopy, porch posts) with sockets.
- Top → `crown_kit` + swept volumes; houses → real pitched roof solids.
- Materials → `MaterialSpec` per zone with real polyhaven:// URIs at 2k, distinct
  roughness values (glass 0.05 / metal 0.3 / field 0.7-0.9 / base 0.8+).
- `asset_output` with `assert_watertight` — always.
- FORBIDDEN: bare `BuildingParams`; texture-faking any Step-2 item; boolean subtract
  (compose voids as unions); density values that ignore the module.

## Step 5 — Self-check before cooking (all must be YES)

1. Does the mesh AABB match the declared footprint ±10%? (run the `slab_evidence` pattern)
2. Are there ≥4 zones with distinct albedo AND roughness in the output contract?
3. Is every Step-2 row satisfied IN GEOMETRY (spot-check reveals ≥0.15m from vertex data)?
4. Could you name the era and structure just from the silhouette + openings? If not,
   return to Step 1.
5. Would the building survive at 40m next to the reference photograph — same detail
   density at that distance? (gate-camera crop is the witness)
6. Do the 3 variants differ in silhouette or palette, not just hue?

## Directing Forge — verified wire facts (live-run proof 2026-07-10)

A skill-authored directive was run through `forge run building` and MEASURED (hero brick
walk-up, seed 7101, 8ms): 16k verts, 363 material zones, 9 distinct (color/rough/disp)
materials, real transmissive glass (alpha 0.15/rough 0.05), per-zone
albedo+normal+rough+displacement maps with per-material scales, per-vertex weathering
masks (edge_wear/moss/paint_peel/rust/water_stain), modeled window reveals (~0.10m
front-facade plane offset). The machinery is REAL. Wire gotchas (serde casing is MIXED —
budget one failed parse, read the error, it names the expected variants):
- `program`/`setting`/`condition`: PascalCase ("Residential", "Urban", "Aged")
- `hero_level`: lowercase ("background"|"supporting"|"hero")
- `channels`: SCREAMING_SNAKE (["STRUCTURE","ORNAMENT","DETAIL"])
- `spec`: tagged union — `{"type": "Residential", ...fields}`
- required: program, setting, style, condition, seed, footprint{shape,width,depth},
  floors, floor_height, channels, hero_level. Output: `payload.asset` (zones/sockets/
  weathering/bounds) + `payload.spatial` (positions/indices/uvs/tangents).

Known Forge deltas vs this skill's floor (author around them; queued as node requests):
- Window reveal measured ~0.10m vs the 0.15m floor — no exposed reveal-depth param yet.
- `displacement_scale` is CONDITION-derived (Aged→5-6mm), not material-class-derived —
  brick cannot yet author 8-15mm mortar depth; needs a MaterialSpec-level override.
- Bay windows/porch project OUTSIDE the declared footprint (measured +2m X / +4m Z on a
  16x14 body): the AABB±10% gate applies to the BODY; document projections per asset.

Cook via `game_asset_cook`, witness through the present path at the gate camera
(day + 22:00 once live time-of-day works), and record the verdict in the uplift plan.
