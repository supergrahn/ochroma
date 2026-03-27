# Assets Directory

Place your .ply Gaussian Splat files here.

## Getting Assets

1. **Train your own**: Use nerfstudio, gsplat, or the original 3DGS trainer
2. **Download**: Sites like Polycam, Luma AI export to PLY format
3. **Generate**: Use `cargo run --bin render_showcase` to generate procedural assets

## Loading in the Engine

```bash
cargo run --bin ochroma -- assets/my_scene.ply
```
