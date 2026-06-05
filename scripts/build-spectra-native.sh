#!/usr/bin/env bash
# build-spectra-native.sh — run a cargo command with the Slang SDK wired up so
# the `spectra-native` feature of vox_render builds, links, AND runs.
#
# Two distinct Slang needs, satisfied here:
#
#  1. RUNTIME loader path (LD_LIBRARY_PATH): the spectra-renderer / spectra-gpu
#     build scripts dlopen libslang.so at *build* time, and the Vulkan backend
#     shells out to `slangc` + loads libslang.so at *run* time. Cargo's [env]
#     table in .cargo/config.toml does NOT reach the build-script loader, so the
#     lib dir must be on LD_LIBRARY_PATH in the *shell* environment.
#
#  2. LINK library (SLANG_DIR): the vendored `slang-sys` build.rs emits
#     `cargo:rustc-link-lib=dylib=slang-compiler` and searches `$SLANG_DIR/lib`.
#     A `check` never links, so it tolerates an SDK without libslang-compiler.so
#     (e.g. the 2024.14.5 SDK in ~/.local/slang whose vtable ABI the patched
#     shader-slang-sys targets). But `build` / `test` / `run` / `bench` link a
#     real binary and therefore REQUIRE an SDK that ships libslang-compiler.so.
#     We auto-pick such an SDK for linking commands and export SLANG_DIR to it,
#     overriding the .cargo/config.toml value for the duration of this command.
#
# Usage:
#   scripts/build-spectra-native.sh check -p vox_render --features spectra-native
#   scripts/build-spectra-native.sh test  -p vox_render --features spectra-native -- --nocapture
#
# If no cargo subcommand is given, defaults to:
#   check -p vox_render --features spectra-native

set -euo pipefail

# --- candidate SDK roots, in preference order -------------------------------
# Each entry is a Slang SDK root (with a lib/ subdir). The first that satisfies
# the requirement for the requested cargo command wins.
SDK_CANDIDATES=()
[[ -n "${SLANG_DIR:-}" ]]      && SDK_CANDIDATES+=("$SLANG_DIR")
SDK_CANDIDATES+=("$HOME/.local/slang" "$HOME/slang-sdk")

# Does this cargo subcommand link a final binary (and thus need slang-compiler)?
SUBCMD="${1:-check}"
case "$SUBCMD" in
    build|test|run|bench|install) LINKS=1 ;;
    *)                            LINKS=0 ;;
esac

pick_sdk() {
    # $1 = "link"  -> require libslang-compiler.so (and libslang.so)
    #      "check" -> require libslang.so only
    local need_compiler="$1"
    for root in "${SDK_CANDIDATES[@]}"; do
        local lib="$root/lib"
        [[ -e "$lib/libslang.so" ]] || continue
        if [[ "$need_compiler" == "link" ]]; then
            [[ -e "$lib/libslang-compiler.so" ]] || continue
        fi
        printf '%s' "$root"
        return 0
    done
    return 1
}

if [[ "$LINKS" == "1" ]]; then
    SDK_ROOT="$(pick_sdk link)" || {
        echo "error: no Slang SDK with libslang-compiler.so found for a linking" >&2
        echo "       command ($SUBCMD). Searched: ${SDK_CANDIDATES[*]}" >&2
        echo "       (a 'check' only needs libslang.so and would work.)" >&2
        exit 1
    }
else
    SDK_ROOT="$(pick_sdk check)" || {
        echo "error: no Slang SDK with libslang.so found." >&2
        echo "       Searched: ${SDK_CANDIDATES[*]}" >&2
        exit 1
    }
fi

SLANG_LIB_DIR="$SDK_ROOT/lib"

# Build-script link search + runtime loader path both point at the chosen SDK.
export SLANG_DIR="$SDK_ROOT"
export LD_LIBRARY_PATH="$SLANG_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

if [[ $# -eq 0 ]]; then
    set -- check -p vox_render --features spectra-native
fi

echo "[build-spectra-native] SLANG_DIR=$SLANG_DIR" >&2
echo "[build-spectra-native] LD_LIBRARY_PATH=$LD_LIBRARY_PATH" >&2
echo "[build-spectra-native] cargo $*" >&2
exec cargo "$@"
