#!/usr/bin/env bash
# build-spectra-native.sh — run a cargo command with the Slang runtime on the
# library path so the spectra-native feature builds.
#
# The spectra-renderer build scripts dlopen libslang.so at *build* time. Cargo's
# [env] table in .cargo/config.toml is applied to the compiled crates but does
# NOT reach the build-script loader's library search path, so LD_LIBRARY_PATH
# must be set in the *shell* environment that invokes cargo. This wrapper does
# exactly that.
#
# Usage:
#   scripts/build-spectra-native.sh check -p vox_render --features spectra-native
#   scripts/build-spectra-native.sh build --features spectra-native
#   scripts/build-spectra-native.sh test  -p vox_render --features spectra-native
#
# If no cargo subcommand is given, defaults to:
#   check -p vox_render --features spectra-native
#
# Alternative (system-wide, requires sudo once):
#   echo "$SLANG_LIB_DIR" | sudo tee /etc/ld.so.conf.d/slang.conf && sudo ldconfig
# after which LD_LIBRARY_PATH is no longer needed.

set -euo pipefail

# Resolve the Slang lib dir. Honour an existing SLANG_DIR if the caller set one.
SLANG_LIB_DIR="${SLANG_LIB_DIR:-${SLANG_DIR:+$SLANG_DIR/lib}}"
SLANG_LIB_DIR="${SLANG_LIB_DIR:-$HOME/.local/slang/lib}"

if [[ ! -e "$SLANG_LIB_DIR/libslang.so" ]]; then
    echo "error: libslang.so not found in '$SLANG_LIB_DIR'." >&2
    echo "       Set SLANG_LIB_DIR or SLANG_DIR to your Slang SDK install." >&2
    exit 1
fi

export LD_LIBRARY_PATH="$SLANG_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

if [[ $# -eq 0 ]]; then
    set -- check -p vox_render --features spectra-native
fi

echo "[build-spectra-native] LD_LIBRARY_PATH=$LD_LIBRARY_PATH" >&2
echo "[build-spectra-native] cargo $*" >&2
exec cargo "$@"
