#!/usr/bin/env bash
set -e

echo "Building Ochroma Engine release..."
cargo build --release --bin ochroma --bin walking_sim

# Verify binaries were produced
ls -lh target/release/ochroma target/release/walking_sim

echo ""
echo "Binary sizes:"
du -h target/release/ochroma target/release/walking_sim

echo ""
echo "Build complete."
