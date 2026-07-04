@echo off
cd /d C:\Users\tom_e\ochroma\projects\urban_horizon
set SLANG_DIR=C:\Users\tom_e\slang
set SLANG_KERNEL_DIR=C:\Users\tom_e\src\spectra\slang
set SPECTRA_BACKEND=cuda
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3
set PATH=%SLANG_DIR%\bin;%CUDA_PATH%\bin;%CUDA_PATH%\bin\x64;%PATH%
rem ==== EXPOSURE RE-ANCHOR: render.ron exposure_ev -2.3 -> -6.0 (config-driven, NO SPECTRA_EXPOSURE_EV env).
rem Same map (forge_grand_delta), single hero shot. Expect: sunlit GREEN terrain, not blown white.
set OCHROMA_FORCE_MAP=forge_grand_delta
set OCHROMA_FIDELITY_TIER=beauty
set OCHROMA_RETAINED_SCENE=1
set SPECTRA_OPTIX_TRACE=1
set SPECTRA_DLSS_RR=1
set SPECTRA_NGX_DLSSD_DIR=C:\Users\tom_e\ngx-dlssd
set OCHROMA_TIME_OF_DAY=12
set OCHROMA_DAY_OF_YEAR=172
set SPECTRA_RENDER_OVERRIDE=C:\Users\tom_e\src\spectra\config\render-override.ron
set OCHROMA_PRESENT_BENCH_DISPLAY=3840x2160
set OCHROMA_PRESENT_BENCH_FRAMES=4
set OCHROMA_PRESENT_BENCH_WARMUP=14
set OCHROMA_RES_IN=1280x720
set SPECTRA_FIREFLY_CLAMP=12
set SPECTRA_FIREFLY_THRESHOLD=100
set OCHROMA_SKY_SCALE=0.1
set OCHROMA_SUN_SCALE=-1.0
set SPECTRA_ROCK_PRIOR=-100.0
set SPECTRA_ROCK_SLOPE_LO=2.0
set SPECTRA_ROCK_SLOPE_HI=3.0
set SPECTRA_DIRT_AMP=0.0
set SPECTRA_TERRAIN_ENABLED=1
set OCHROMA_HERO_CX=3379
set OCHROMA_HERO_CZ=2200
set OCHROMA_HERO_DIST=1200
set OCHROMA_HERO_PITCH=0.34
set OCHROMA_HERO_YAW=0.06

taskkill /F /IM play.exe >nul 2>&1
set OCHROMA_PRESENT_SHOT=C:\Users\tom_e\dlss-shots\recipe_norock.png
target\release\play.exe > C:\Users\tom_e\recipe_norock.log 2>&1
echo NOROCK_EXIT=%errorlevel%
echo NOROCK_DONE
