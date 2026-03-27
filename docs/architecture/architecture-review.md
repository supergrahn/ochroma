# Architecture Review — Critical Problems Found

After careful review, the architecture has 14 serious problems that will block us from reaching Unreal parity. These must be fixed BEFORE we dispatch agents.

## Problem 1: Two Competing Entity Systems

We have TWO entity systems that don't know about each other:

- `vox_core::ecs::SplatInstanceComponent` — Bevy ECS, used by old main.rs
- `vox_core::engine_runtime::Entity` — custom struct, used by engine_runner.rs

**The engine_runtime::Entity is NOT an ECS component.** It's a plain struct in a Vec. This means we lose all of Bevy's benefits: queries, systems, parallel iteration, archetypes.

**Fix:** Delete our custom Entity. Use Bevy ECS as THE entity system. The engine_runtime owns a `bevy_ecs::World` and schedules systems against it. Every system is a Bevy system function, not a method on a struct.

## Problem 2: Architecture Document ≠ Actual Code

The architecture describes a 9-phase loop, but the actual `engine_runtime.rs` tick() method does:
1. Increment frame counter
2. Run scripts (half-working)
3. AABB collision check
4. Advance time

That's 4 things, not 9. There's no rendering, no audio, no animation, no culling in the engine runtime. All of that is in engine_runner.rs (the binary), duplicated outside the engine.

**Fix:** The engine_runtime MUST own and execute all 9 phases. The binary (engine_runner.rs) should only handle window events and call `engine.tick()`. Everything else happens inside the engine.

## Problem 3: No Fixed Timestep

Physics and game logic should run at a fixed rate (e.g., 60Hz) independent of frame rate. If rendering runs at 30fps, physics should still step twice per frame. If rendering runs at 144fps, physics shouldn't step every frame.

The architecture doesn't address this at all. The current tick() takes `dt` and runs everything once.

**Fix:** Implement a fixed-timestep accumulator:
```
accumulator += frame_dt
while accumulator >= FIXED_DT:
    physics.step(FIXED_DT)
    scripts.update(FIXED_DT)
    accumulator -= FIXED_DT
render(interpolation_factor = accumulator / FIXED_DT)
```

## Problem 4: Renderer Has Never Been Verified on GPU

The GpuRasteriser and WGSL shader exist but we've NEVER seen their visual output. Every visual test uses the software rasteriser. The GPU path could produce garbage and we wouldn't know.

**Fix:** Before building shadows or any rendering feature:
1. Render a known scene with the GPU rasteriser
2. Save the output as an image
3. Compare with software rasteriser output
4. If they don't match, fix the shader FIRST

## Problem 5: Shadow Maps on CPU is a Non-Starter

The spec says to implement shadow maps on the software rasteriser. That means rendering the scene TWICE on CPU. With 239k splats taking 90+ seconds to render once, shadow maps would take 180+ seconds per frame. This is useless.

**Fix:** Shadow maps only make sense on GPU. Either:
- Implement in the WGSL shader (add a shadow pass)
- Or defer to Spectra integration (Spectra has real shadows)
- Don't implement CPU shadow maps at all

## Problem 6: Gaussian Splat Animation is Hard

The spec says "apply bone transforms to splat positions." But a Gaussian splat isn't just a point — it's an ellipsoid defined by position + covariance matrix. Transforming a splat requires transforming the covariance too, or the splat shape becomes wrong (a sphere stays spherical even when the bone rotates, which looks incorrect).

**Fix:** For v1, support RIGID animation only:
- Groups of splats move together as a rigid body (no deformation)
- A "bone" owns a group of splats and transforms all their positions
- No skinning, no blending between bones
- This is like animating Lego pieces — each piece moves rigidly

Deformable splat animation (proper skinning) is a research problem. Defer it.

## Problem 7: Editor Gizmos as Splats Won't Work

The spec says gizmos are splat clusters mixed into the scene. Problems:
- Gizmos will be occluded by scene geometry (hidden behind walls)
- Gizmos will be depth-sorted with scene splats (visual artifacts)
- Gizmo colours will go through spectral conversion (wrong colours)

**Fix:** Gizmos must be rendered in a SEPARATE overlay pass:
- After main render: draw gizmo lines/shapes on top
- Either as simple 2D lines projected to screen (burned into framebuffer like HUD text)
- Or as a second render pass with depth test disabled

For software rasteriser: draw gizmo arrows directly into the pixel buffer after the scene is rendered, using the same burn_text approach but for lines.

## Problem 8: No Asset Caching

If 50 trees reference the same tree.ply, the architecture implies loading it 50 times. We need reference-counted asset handles.

**Fix:** AssetManager uses a HashMap<PathBuf, Arc<AssetData>>. Load once, share the Arc. Handle tracks the path. When all handles drop, asset is unloaded.

## Problem 9: Spectra Integration is Undefined

The architecture lists "SpectraBackend" as a RenderBackend implementation, but:
- Spectra is Python + CUDA, not a Rust library
- The Rust crate (spectra-gaussian-render) can't be imported due to workspace conflicts
- We ported the EWA algorithm ourselves but it's not "Spectra"

**Fix:** Define the REAL integration path:
- Option A: Spectra runs as a subprocess. Engine sends scene data via shared memory or socket. Spectra renders and returns a framebuffer. This is how UE5 plugins work.
- Option B: We maintain our own Rust port of Spectra's algorithm (spectra_render.rs). It's "Spectra-quality" but not literally Spectra.
- Option C: The user integrates Spectra at the Python level for offline/high-quality, and we use our wgpu rasteriser for real-time.

Recommend: Option C for now. Document it clearly.

## Problem 10: ScriptContext is Too Limited

The current ScriptContext can set_position and play_sound, but it can't:
- Read other entity positions (needed for AI: "find nearest enemy")
- Access physics (needed: "am I on the ground?")
- Access time/frame info
- Access custom entity data (health, score, inventory)

**Fix:** Add to ScriptContext:
```rust
ctx.get_entity_position(other_id) → Option<Vec3>
ctx.get_entities_with_tag(tag) → Vec<u32>
ctx.is_grounded() → bool
ctx.get_time() → f32
ctx.get_dt() → f32
ctx.get_custom_data(key) → Option<String>
ctx.set_custom_data(key, value)
```

## Problem 11: GLTF-to-Splat Conversion Will Look Bad

Randomly placing splats on triangle surfaces produces a noisy, ugly result. It will NOT look like the original mesh. Game developers will compare this to Unreal's mesh rendering and be disappointed.

**Fix:** Be honest about what this is:
- GLTF import is for GEOMETRY REFERENCE, not final rendering
- The imported splat cloud shows the shape but not the material quality
- For production assets, train proper 3DGS from multi-view images of the mesh
- Alternatively: render the GLTF mesh from multiple views, then train 3DGS on those renders

For v1: import GLTF as a rough splat cloud for placement reference. Document that production assets should be proper 3DGS .ply files.

## Problem 12: Walking Sim and Demo Duplicate the Engine

engine_runner.rs, demo.rs, and walking_sim.rs each implement their own render loop, input handling, camera, etc. Changes to the engine don't propagate to the games.

**Fix:** After fixing the engine_runtime to own the full loop (Problem 2), rewrite the walking sim and demo as THIN scripts:
```rust
fn main() {
    let mut engine = EngineRuntime::new(config);
    engine.scripts.register("OrbCollector", || Box::new(OrbScript));
    engine.load_scene("walking_sim_level");
    engine.run();  // engine handles EVERYTHING
}
```

The game is just scripts + scene data. The engine does the rest.

## Problem 13: No Component System for Custom Data

Unreal has UActorComponent. Unity has MonoBehaviour. Our entities have fixed fields (position, rotation, scripts). But games need custom data: health, inventory, AI state, quest progress.

**Fix:** Add a `custom_data: HashMap<String, serde_json::Value>` to Entity. Scripts read/write it:
```rust
ctx.set_custom_data("health", json!(100));
let health: i64 = ctx.get_custom_data("health").as_i64().unwrap();
```

This is simple but extensible. Later we can add typed components.

## Problem 14: No Error Recovery in the Render Loop

If any system panics, the entire engine crashes. Unreal continues running even if a Blueprint errors.

**Fix:** Wrap each phase in catch_unwind or Result handling:
```rust
if let Err(e) = self.tick_scripts(dt) {
    eprintln!("[engine] Script error: {}", e);
    // continue running, skip this frame's scripts
}
```

## Revised Architecture Decisions

Based on this review:

1. **Use Bevy ECS** as the entity system (delete custom Entity struct)
2. **Engine runtime owns the full 9-phase loop** (not just scripts+physics)
3. **Fixed timestep** for physics/scripts, variable for rendering
4. **Verify GPU rendering** before building anything on top of it
5. **Shadows via GPU only** (or Spectra), not CPU
6. **Rigid-body animation only** for v1 (no splat deformation)
7. **Gizmos as 2D overlay**, not splats in the scene
8. **Asset caching** with reference counting
9. **Spectra is separate** — we use our own Rust rasteriser for real-time, Spectra for offline
10. **Richer ScriptContext** with entity queries and custom data
11. **GLTF import is rough reference** — real assets are trained .ply files
12. **Games are thin scripts** on top of the engine runtime
13. **Custom data via HashMap** on entities
14. **Error recovery** around each engine phase
