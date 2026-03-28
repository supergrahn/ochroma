# Assets Directory

Place your .ply Gaussian Splat files here.

## Test Scenes

The `test_scenes/` directory contains placeholder files for automated testing.
The PLY integration test (`crates/vox_data/tests/ply_integration_test.rs`)
generates a synthetic 1000-splat PLY in memory to verify the load-render
pipeline without requiring large external files.

## Getting Real Assets

1. **Train your own**: Use [nerfstudio](https://nerf.studio), [gsplat](https://docs.gsplat.studio), or the original [3DGS trainer](https://github.com/graphdeco-inria/gaussian-splatting)
2. **Download**: Sites like Polycam, Luma AI, and Scaniverse export to PLY format
3. **Procedural**: The engine generates terrain, buildings, and trees at runtime — no external assets needed for the default scene

## Loading in the Engine

```bash
# Load a PLY file in the full engine
cargo run --bin ochroma -- assets/my_scene.ply

# Load a PLY file in the interactive demo
cargo run --bin demo -- assets/my_scene.ply
```

## PLY Format Support

The engine loads standard binary little-endian PLY files with 3DGS properties:

- Positions: `x`, `y`, `z`
- Scales: `scale_0`, `scale_1`, `scale_2` (log space from training)
- Rotations: `rot_0`, `rot_1`, `rot_2`, `rot_3` (normalised quaternion)
- Opacity: `opacity` (logit space from training — sigmoid is applied on load)
- Colour: `f_dc_0`, `f_dc_1`, `f_dc_2` (spherical harmonic DC term)
- Optional higher-order SH: `f_rest_0` ... `f_rest_44`

Files exported directly from nerfstudio, gsplat, or the original 3DGS trainer
will load correctly. The loader converts training-space parameters to the
engine's spectral format on load.

## Asset Pipeline Tool

The `vox_tools` CLI provides asset processing:

```bash
# Import a GLTF/GLB file and convert to .vxm
cargo run --bin vox_tools -- import --input model.glb --output model.vxm

# Run a turnaround capture pipeline from photos
cargo run --bin vox_tools -- turnaround --views ./photos/ --output asset.vxm
```
