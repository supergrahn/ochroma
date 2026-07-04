@echo off
cd /d C:\Users\tom_e\ochroma\projects\urban_horizon
set SLANG_DIR=C:\Users\tom_e\slang
set SLANG_KERNEL_DIR=C:\Users\tom_e\src\spectra\slang
set SPECTRA_BACKEND=cuda
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3
set PATH=%SLANG_DIR%\bin;%CUDA_PATH%\bin;%CUDA_PATH%\bin\x64;%PATH%
set OCHROMA_FORCE_MAP=forge_river_valley
set OCHROMA_FIDELITY_TIER=performance
set OCHROMA_RETAINED_SCENE=1
set SPECTRA_DLSS_RR=1
set SPECTRA_NGX_DLSSD_DIR=C:\Users\tom_e\ngx-dlssd
set OCHROMA_HERO_DIST=3000
set OCHROMA_HERO_PITCH=0.50
set OCHROMA_HERO_HOUR=13.0
set OCHROMA_PRESENT_BENCH_DISPLAY=3840x2160
set OCHROMA_PRESENT_SHOT=C:\Users\tom_e\dlss-shots\map30_rel_overview.png
set OCHROMA_PRESENT_BENCH_FRAMES=20
set OCHROMA_PRESENT_BENCH_WARMUP=5
target\release\play.exe > C:\Users\tom_e\map30_rel_overview.log 2>&1
echo PLAY_EXIT=%errorlevel%
