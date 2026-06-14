<#
.SYNOPSIS
  One-command Windows build of the Ochroma GPU path-traced renderer (spectra-native, NVIDIA CUDA/DLSS).

.DESCRIPTION
  Auto-discovers every toolchain dependency the GPU build needs and sets the environment, then builds:
    - Slang  (shader compiler)         -> SLANG_DIR, slang\bin on PATH
    - CUDA Toolkit (nvcc/nvrtc)        -> CUDA_PATH, CUDA\vXX.Y\bin on PATH  (Slang PTX pass-through compiler)
    - VS LLVM (libclang.dll)           -> LIBCLANG_PATH                       (bindgen for shader-slang-sys)
    - SPECTRA_BACKEND = cuda           (NVIDIA backend, NOT vulkan)

  Nothing is version-pinned: CUDA picks the newest installed toolkit, libclang is found via vswhere.
  Override any auto-detected path by setting the matching env var before running.

.EXAMPLE
  powershell -NoProfile -ExecutionPolicy RemoteSigned -File scripts\build-windows-gpu.ps1
  powershell -NoProfile -File scripts\build-windows-gpu.ps1 -Release -Log C:\Users\tom_e\build_gpu.log
#>
[CmdletBinding()]
param(
    # Build profile. Default is dev (debug); pass -Release for an optimized build.
    [switch]$Release,
    # Optional path to tee full build output to (UTF-8).
    [string]$Log,
    # Slang install dir. Falls back to $env:SLANG_DIR, then C:\Users\<you>\slang.
    [string]$SlangDir,
    # Cargo package to build (default: vox_render, the GPU renderer lib).
    [string]$Package = "vox_render",
    # Optional binary target to build/run (implies the package that owns it).
    [string]$Bin,
    # After building, run the binary (requires -Bin).
    [switch]$Run
)

$ErrorActionPreference = "Stop"

function Write-Step($msg) { Write-Host "[build-gpu] $msg" -ForegroundColor Cyan }
function Die($msg) { Write-Host "[build-gpu] ERROR: $msg" -ForegroundColor Red; exit 1 }

# --- Slang ----------------------------------------------------------------
if (-not $SlangDir) { $SlangDir = $env:SLANG_DIR }
if (-not $SlangDir) { $SlangDir = Join-Path $env:USERPROFILE "slang" }
if (-not (Test-Path (Join-Path $SlangDir "bin"))) {
    Die "Slang not found at '$SlangDir' (no bin\). Set -SlangDir or `$env:SLANG_DIR."
}
$env:SLANG_DIR = $SlangDir
Write-Step "Slang:    $SlangDir"

# --- CUDA Toolkit (newest installed) -------------------------------------
$cudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
$cuda = $null
if ($env:CUDA_PATH -and (Test-Path (Join-Path $env:CUDA_PATH "bin\nvcc.exe"))) {
    $cuda = $env:CUDA_PATH
} elseif (Test-Path $cudaRoot) {
    $cuda = Get-ChildItem $cudaRoot -Directory |
        Where-Object { Test-Path (Join-Path $_.FullName "bin\nvcc.exe") } |
        Sort-Object Name -Descending |
        Select-Object -First 1 -ExpandProperty FullName
}
if (-not $cuda) { Die "CUDA Toolkit (nvcc.exe) not found under '$cudaRoot'. Install the CUDA Toolkit." }
$env:CUDA_PATH = $cuda
# nvcc lives in bin\; the nvrtc + cudart runtime DLLs live in bin\x64 on CUDA 13+.
# Slang's CUDA->PTX pass-through dynamically loads nvrtc, so bin\x64 MUST be on PATH
# or every kernel fails with `error[E52002]: pass-through compiler not found`.
$cudaBin = Join-Path $cuda "bin"
$cudaBinX64 = Join-Path $cudaBin "x64"
Write-Step "CUDA:     $cuda"

# --- libclang (bindgen) via VS LLVM --------------------------------------
$libclangDir = $env:LIBCLANG_PATH
if (-not ($libclangDir -and (Test-Path (Join-Path $libclangDir "libclang.dll")))) {
    $libclangDir = $null
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vswhere) {
        $vsRoot = & $vswhere -latest -products * -property installationPath 2>$null
        if ($vsRoot) {
            $cand = Join-Path $vsRoot "VC\Tools\Llvm\x64\bin"
            if (Test-Path (Join-Path $cand "libclang.dll")) { $libclangDir = $cand }
        }
    }
}
if (-not $libclangDir) {
    Die "libclang.dll not found. Install the 'C++ Clang tools for Windows' VS component, or set `$env:LIBCLANG_PATH."
}
$env:LIBCLANG_PATH = $libclangDir
Write-Step "libclang: $libclangDir"

# --- Compose PATH + backend ----------------------------------------------
# Order matters: slang\bin and CUDA\bin must precede the rest so Slang's PTX
# pass-through finds nvcc and the slang DLLs resolve at link/runtime.
$pathParts = @((Join-Path $SlangDir "bin"), $cudaBin)
if (Test-Path $cudaBinX64) { $pathParts += $cudaBinX64 }
$pathParts += (Join-Path $env:USERPROFILE ".cargo\bin")
$env:PATH = ($pathParts -join ";") + ";" + $env:PATH
$env:SPECTRA_BACKEND = "cuda"
Write-Step "Backend:  cuda (NVIDIA)"

# --- Build ----------------------------------------------------------------
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot
$cargo = Join-Path $env:USERPROFILE ".cargo\bin\cargo.exe"
# Building vox_app binaries (editor) needs the app's own spectra-native feature;
# vox_render alone uses vox_render/spectra-native.
$feature = if ($Package -eq "vox_render") { "spectra-native" } else { "spectra-native" }
$cargoArgs = @("build", "-p", $Package, "--features", $feature)
if ($Bin) { $cargoArgs += @("--bin", $Bin) }
if ($Release) { $cargoArgs += "--release" }

Write-Step ("cargo " + ($cargoArgs -join " "))
if ($Log) {
    & $cargo @cargoArgs *>&1 | Tee-Object -FilePath $Log
} else {
    & $cargo @cargoArgs
}
$code = $LASTEXITCODE
Write-Host ("[build-gpu] BUILD_EXIT=" + $code) -ForegroundColor ($(if ($code -eq 0) { "Green" } else { "Red" }))
if ($code -eq 0 -and $Run -and $Bin) {
    $profileDir = if ($Release) { "release" } else { "debug" }
    $exe = Join-Path $repoRoot ("target\" + $profileDir + "\" + $Bin + ".exe")
    Write-Step ("Launching " + $exe)
    & $exe
    Write-Host ("[build-gpu] RUN_EXIT=" + $LASTEXITCODE)
}
exit $code
