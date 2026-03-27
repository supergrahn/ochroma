# Phase 18 — Photorealism

**Goal:** Achieve photorealistic rendering quality that surpasses Unreal Engine 5's Nanite/Lumen through spectral accuracy.

## 18.1 Full Spectral Path Tracing
- Integrate Spectra's wavefront path tracer for offline renders
- Hero wavelength sampling (380-780nm continuous, not just 8 bands)
- Physically accurate caustics via photon mapping
- Polarisation tracking for reflections

## 18.2 Neural Denoiser (Production)
- OptiX AI denoiser integration via Spectra
- Temporal stability: denoised frames don't flicker
- Spectral-aware: denoise per wavelength, not just RGB
- Real-time at 1440p with 4 samples per pixel

## 18.3 Material Scanning
- Photograph a real material → extract spectral reflectance
- Phone camera + calibration target → 8-band SPD estimation
- Build a library of real-world scanned materials

## 18.4 HDR Output
- 10-bit colour depth for compatible displays
- Dolby Vision / HDR10 metadata
- Wide colour gamut (Rec. 2020)
- Spectral-to-display mapping preserving out-of-gamut highlights

## 18.5 Cinematic Camera
- Depth of field with bokeh shapes
- Motion blur from camera and object movement
- Lens flare from bright lights
- Film grain for artistic effect
- Chromatic aberration at wide angles

## Exit Criteria
- [ ] Path-traced render indistinguishable from photograph at 4K
- [ ] Denoised real-time output at 1440p 60fps
- [ ] Scanned material matches real-world appearance
- [ ] HDR output on compatible display
- [ ] Cinematic camera produces film-quality screenshots
