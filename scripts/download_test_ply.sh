#!/bin/bash
# Download a real 3DGS .ply file for testing.
# These are from the original 3DGS paper (Kerbl et al. 2023).
#
# Usage: ./scripts/download_test_ply.sh
#
# After running, test with:
#   cargo run --bin ochroma -- assets/test_scenes/bicycle.ply

mkdir -p assets/test_scenes

echo "Downloading bicycle scene (3DGS paper)..."
echo "Note: You need to download from https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/"
echo "Place the .ply file at: assets/test_scenes/bicycle.ply"
echo ""
echo "Alternatively, use any .ply file from:"
echo "  - https://poly.cam (export as Gaussian Splat PLY)"
echo "  - https://lumalabs.ai (export as PLY)"
echo "  - nerfstudio training output"
echo "  - gsplat training output"
echo ""
echo "Then run: cargo run --bin ochroma -- assets/test_scenes/your_file.ply"
