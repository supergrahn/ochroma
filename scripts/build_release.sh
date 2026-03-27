#!/bin/bash
set -e
echo "Building Ochroma Engine (release)..."
cargo build --release --bin ochroma --bin walking_sim
echo "Build complete:"
ls -lh target/release/ochroma target/release/walking_sim
echo "Binary sizes:"
du -h target/release/ochroma target/release/walking_sim
