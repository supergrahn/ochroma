# Ochroma Engine v0.1.0

Spectral Gaussian Splatting Game Engine.

## Quick Start

    ./bin/ochroma                          # Launch engine with default scene
    ./bin/ochroma path/to/scene.ply        # Load a Gaussian splat scene
    ./bin/walking_sim                      # Play the example game

## Controls

    WASD          Move
    Space/Shift   Up/Down
    Right-click   Look around
    Left-click    Place object
    Tab           Toggle editor
    T             Time of day
    +/-           Exposure
    M             Tone mapping
    Q             DLSS quality
    F5            Quick save
    F9            Quick load
    F12           Screenshot
    Escape        Quit

## Creating Assets

Use any 3DGS training tool to create .ply files:
- nerfstudio, gsplat, original 3DGS trainer
- Polycam, Luma AI (export as PLY)
- Or use: ./bin/ochroma --import model.glb (converts mesh to splats)

## System Requirements

- Linux x64 (Windows support coming)
- Vulkan-capable GPU (NVIDIA recommended)
- 4GB RAM minimum
