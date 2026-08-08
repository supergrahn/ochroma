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
    # Optional example target to build (mutually exclusive with -Bin).
    [string]$Example,
    # After building, run the binary (requires -Bin).
    [switch]$Run,
    # Cargo features. If omitted: "spectra-native" for vox_render/vox_app, none otherwise.
    # Pass "" explicitly to force no features (e.g. the wgpu game `play` binary).
    [string]$Features,
    # Pass --no-default-features.
    [switch]$NoDefaultFeatures,
    # Repo to build from (cd here before cargo). Default: this script's repo (ochroma).
    # Set to the urban_horizon repo to build the game (its path-deps reach back to ochroma).
    [string]$WorkDir
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
$env:SPECTRA_NVCC = Join-Path $cuda "bin\nvcc.exe"
# nvcc lives in bin\; the nvrtc + cudart runtime DLLs live in bin\x64 on CUDA 13+.
# Slang's CUDA->PTX pass-through dynamically loads nvrtc, so bin\x64 MUST be on PATH
# or every kernel fails with `error[E52002]: pass-through compiler not found`.
$cudaBin = Join-Path $cuda "bin"
$cudaBinX64 = Join-Path $cudaBin "x64"
Write-Step "CUDA:     $cuda"

# --- Vulkan SDK (headers + loader import library for native FSR) ----------
$vulkanSdk = $env:VULKAN_SDK
if (-not ($vulkanSdk -and
           (Test-Path (Join-Path $vulkanSdk "Include\vulkan\vulkan.h")) -and
           (Test-Path (Join-Path $vulkanSdk "Lib\vulkan-1.lib")))) {
    $vulkanSdk = $null
    $vulkanRoot = "C:\VulkanSDK"
    if (Test-Path $vulkanRoot) {
        $vulkanSdk = Get-ChildItem $vulkanRoot -Directory |
            Where-Object {
                (Test-Path (Join-Path $_.FullName "Include\vulkan\vulkan.h")) -and
                (Test-Path (Join-Path $_.FullName "Lib\vulkan-1.lib"))
            } |
            Sort-Object Name -Descending |
            Select-Object -First 1 -ExpandProperty FullName
    }
}
if (-not $vulkanSdk) {
    Die "Vulkan SDK not found. The all-vendor Windows build requires Vulkan headers and vulkan-1.lib; set `$env:VULKAN_SDK or install KhronosGroup.VulkanSDK."
}
$env:VULKAN_SDK = $vulkanSdk
Write-Step "Vulkan:   $vulkanSdk"

# --- glslang (FSR permutation compiler) ----------------------------------
# ffx-perm-gen must compile its Vulkan shader permutations even when this
# Windows build will normally select CUDA/DLSS.  Its upstream default is the
# Linux-only path `/usr/bin/glslangValidator`; the vendored ffx-sys bridge
# accepts the platform-correct executable through FFX_GLSLANG.  Resolve it once
# here, where every other build tool is discovered, and fail before a long
# cargo build if the all-vendor binary requested `fsr` without it.
$glslang = $env:FFX_GLSLANG
if (-not ($glslang -and (Test-Path $glslang))) {
    $glslang = $null
    $glslangCommand = Get-Command glslangValidator.exe -ErrorAction SilentlyContinue
    if ($glslangCommand) { $glslang = $glslangCommand.Source }
}
if (-not $glslang) {
    $glslangCandidates = @(
        (Join-Path $env:USERPROFILE "glslang\bin\glslangValidator.exe")
    )
    if ($env:VULKAN_SDK) {
        $glslangCandidates += (Join-Path $env:VULKAN_SDK "Bin\glslangValidator.exe")
    }
    foreach ($candidate in $glslangCandidates) {
        if (Test-Path $candidate) {
            $glslang = (Resolve-Path $candidate).Path
            break
        }
    }
}
if ($glslang) {
    $env:FFX_GLSLANG = $glslang
    Write-Step "glslang:  $glslang"
} elseif ($Features -and (($Features -split ',') -contains 'fsr')) {
    Die "glslangValidator.exe not found. The all-vendor Windows build includes FSR; set `$env:FFX_GLSLANG or install glslang under `$env:USERPROFILE\glslang."
}

# --- CMake (native FidelityFX libraries) ---------------------------------
$cmake = $env:CMAKE
if (-not ($cmake -and (Test-Path $cmake))) {
    $cmake = $null
    $cmakeCommand = Get-Command cmake.exe -ErrorAction SilentlyContinue
    if ($cmakeCommand) { $cmake = $cmakeCommand.Source }
}
if (-not $cmake) {
    $cmakeCandidates = @((Join-Path $env:ProgramFiles "CMake\bin\cmake.exe"))
    $vswhereCmake = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vswhereCmake) {
        $vsCmakeRoot = & $vswhereCmake -latest -products * -property installationPath 2>$null
        if ($vsCmakeRoot) {
            $cmakeCandidates += (Join-Path $vsCmakeRoot "Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe")
        }
    }
    foreach ($candidate in $cmakeCandidates) {
        if (Test-Path $candidate) {
            $cmake = (Resolve-Path $candidate).Path
            break
        }
    }
}
if (-not $cmake) {
    Die "cmake.exe not found. The FSR native libraries require CMake; set `$env:CMAKE or install the Visual Studio CMake component."
}
$env:CMAKE = $cmake
Write-Step "CMake:    $cmake"

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

# --- MSVC host compiler (cl.exe) for nvcc --ptx --------------------------
# spectra-optix/build.rs compiles the OptiX device programs with `nvcc --ptx`.
# Even for PTX-only output, nvcc invokes the MSVC host compiler `cl.exe` for its
# preprocessing pass — so cl.exe MUST be on PATH or nvcc fails and build.rs
# falls back to the committed device_programs.ptx (still correct, but stale if
# the .cu changed). Import the VS dev environment here so nvcc compiles FRESH.
$vswhere2 = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (Test-Path $vswhere2) {
    $vsRoot2 = & $vswhere2 -latest -products * -property installationPath 2>$null
    if ($vsRoot2) {
        $vcvars = Join-Path $vsRoot2 "VC\Auxiliary\Build\vcvars64.bat"
        if (Test-Path $vcvars) {
            # Run vcvars in cmd and import the resulting env into this session so
            # cl.exe (and the MSVC INCLUDE/LIB) are visible to cargo -> build.rs.
            cmd /c "`"$vcvars`" >nul 2>&1 && set" | ForEach-Object {
                if ($_ -match '^([^=]+)=(.*)$') {
                    Set-Item -Path "Env:$($matches[1])" -Value $matches[2]
                }
            }
            Write-Step "MSVC:     vcvars64 imported (cl.exe on PATH for nvcc --ptx)"
        }
    }
}

# --- Compose PATH + backend ----------------------------------------------
# Order matters: slang\bin and CUDA\bin must precede the rest so Slang's PTX
# pass-through finds nvcc and the slang DLLs resolve at link/runtime.
$pathParts = @((Join-Path $SlangDir "bin"), $cudaBin, (Join-Path $vulkanSdk "Bin"))
if (Test-Path $cudaBinX64) { $pathParts += $cudaBinX64 }
$pathParts += (Join-Path $env:USERPROFILE ".cargo\bin")
$env:PATH = ($pathParts -join ";") + ";" + $env:PATH
$env:SPECTRA_BACKEND = "cuda"
Write-Step "Backend:  cuda (NVIDIA)"

# --- AOT kernel precompilation -------------------------------------------
# spectra-renderer's build.rs precompiles every .slang kernel to PTX at build
# time (-> get_ptx()), so the SHIPPED game loads compiled kernels and never runs
# nvrtc/Slang at runtime. Prefer the Ochroma runtime bundle, then explicit
# engine overrides, then support the dev Spectra source layouts used on the box:
#   %OCHROMA_RUNTIME_DIR%\kernels
#   %OCHROMA_RUNTIME_DIR%\spectra\slang
#   C:\Users\<you>\src\spectra\slang
#   C:\Users\<you>\spectra\slang
$kernelCandidates = @()
if ($env:OCHROMA_RUNTIME_DIR) {
    $kernelCandidates += @(
        (Join-Path $env:OCHROMA_RUNTIME_DIR "kernels"),
        (Join-Path $env:OCHROMA_RUNTIME_DIR "spectra\slang"),
        (Join-Path $env:OCHROMA_RUNTIME_DIR "spectra\kernels"),
        (Join-Path $env:OCHROMA_RUNTIME_DIR "runtime\kernels"),
        (Join-Path $env:OCHROMA_RUNTIME_DIR "runtime\spectra\slang")
    )
}
if ($env:OCHROMA_SLANG_KERNEL_DIR) { $kernelCandidates += $env:OCHROMA_SLANG_KERNEL_DIR }
if ($env:SLANG_KERNEL_DIR) { $kernelCandidates += $env:SLANG_KERNEL_DIR }
if ($env:SPECTRA_SLANG_DIR) { $kernelCandidates += $env:SPECTRA_SLANG_DIR }
$kernelCandidates += @(
    (Join-Path $PSScriptRoot "..\..\src\spectra\slang"),
    (Join-Path $PSScriptRoot "..\..\spectra\slang")
)
$spectraKernels = $null
foreach ($candidate in $kernelCandidates) {
    if ($candidate -and (Test-Path $candidate)) {
        $spectraKernels = Resolve-Path $candidate -ErrorAction Stop
        break
    }
}
if ($spectraKernels) {
    $env:SLANG_KERNEL_DIR = $spectraKernels.Path
    $env:SPECTRA_SLANG_DIR = $spectraKernels.Path
    $env:OCHROMA_SLANG_KERNEL_DIR = $spectraKernels.Path
    Write-Step "Kernels:  $($spectraKernels.Path)  (AOT PTX precompile)"
} else {
    Write-Host "[build-gpu] WARN: spectra/slang kernel dir not found; PTX precompile will be empty." -ForegroundColor Yellow
    Write-Host "[build-gpu] Checked: $($kernelCandidates -join '; ')" -ForegroundColor Yellow
}

# --- Build ----------------------------------------------------------------
$repoRoot = if ($WorkDir) { $WorkDir } else { Split-Path -Parent $PSScriptRoot }
if (-not (Test-Path (Join-Path $repoRoot "Cargo.toml"))) {
    Die "No Cargo.toml at WorkDir '$repoRoot'."
}
Set-Location $repoRoot
Write-Step "WorkDir:  $repoRoot"
$cargo = Join-Path $env:USERPROFILE ".cargo\bin\cargo.exe"
# Default features: the GPU renderer crates need spectra-native; everything else
# (e.g. the wgpu game `play` binary) builds with its own defaults.
if (-not $PSBoundParameters.ContainsKey('Features')) {
    $Features = if ($Package -in @('vox_render', 'vox_app')) {
        'spectra-native,spectra-native-optix'
    } else { '' }
}
$cargoArgs = @("build", "-p", $Package)
if ($Bin) { $cargoArgs += @("--bin", $Bin) }
if ($Example) { $cargoArgs += @("--example", $Example) }
if ($Features) { $cargoArgs += @("--features", $Features) }
if ($NoDefaultFeatures) { $cargoArgs += "--no-default-features" }
if ($Release) { $cargoArgs += "--release" }

Write-Step ("cargo " + ($cargoArgs -join " "))
# cargo prints compiler warnings on STDERR. This script sets
# $ErrorActionPreference = "Stop" (top), under which a NATIVE command's stderr
# write is surfaced as a terminating NativeCommandError — so the FIRST cargo
# warning (e.g. "unused import") aborted the whole build at this line before a
# single crate finished (reproduced over non-interactive ssh). Drop to Continue
# around the cargo call so stderr is captured, not fatal; the authoritative
# result is $LASTEXITCODE, checked below.
$prevEAP = $ErrorActionPreference
$ErrorActionPreference = "Continue"
if ($Log) {
    & $cargo @cargoArgs *>&1 | Tee-Object -FilePath $Log
} else {
    & $cargo @cargoArgs
}
$code = $LASTEXITCODE
$ErrorActionPreference = $prevEAP
Write-Host ("[build-gpu] BUILD_EXIT=" + $code) -ForegroundColor ($(if ($code -eq 0) { "Green" } else { "Red" }))
if ($code -eq 0 -and $Run -and $Bin) {
    $profileDir = if ($Release) { "release" } else { "debug" }
    $exe = Join-Path $repoRoot ("target\" + $profileDir + "\" + $Bin + ".exe")
    Write-Step ("Launching " + $exe)
    & $exe
    Write-Host ("[build-gpu] RUN_EXIT=" + $LASTEXITCODE)
}
exit $code
