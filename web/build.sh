#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$SCRIPT_DIR/.."
CRATE_DIR="$REPO_ROOT/crates/vox_web"
OUT_DIR="$SCRIPT_DIR/dist"

echo "Building hello_splat for WebGPU..."
wasm-pack build "$CRATE_DIR" \
  --target web \
  --out-dir "$OUT_DIR" \
  --release

echo "Renaming output to hello_splat..."
# wasm-pack names output after the crate (vox_web); rename to hello_splat
if [ -f "$OUT_DIR/vox_web_bg.wasm" ]; then
  mv "$OUT_DIR/vox_web_bg.wasm" "$OUT_DIR/hello_splat_bg.wasm"
  mv "$OUT_DIR/vox_web.js"      "$OUT_DIR/hello_splat.js"      2>/dev/null || true
  mv "$OUT_DIR/vox_web_bg.wasm.d.ts" "$OUT_DIR/hello_splat_bg.wasm.d.ts" 2>/dev/null || true
  mv "$OUT_DIR/vox_web.d.ts"    "$OUT_DIR/hello_splat.d.ts"    2>/dev/null || true
  # Fix the JS import reference
  sed -i 's/vox_web_bg\.wasm/hello_splat_bg.wasm/g' "$OUT_DIR/hello_splat.js" 2>/dev/null || true
fi

echo "Copying index.html..."
cp "$SCRIPT_DIR/index.html" "$OUT_DIR/index.html"

echo "Done. Serve with: python3 -m http.server 8080 --directory $OUT_DIR"
