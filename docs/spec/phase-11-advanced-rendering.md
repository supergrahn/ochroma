# Phase 11 — Advanced Rendering

**Goal:** Push Ochroma's rendering beyond what Unreal offers — leveraging spectral Gaussian splatting for effects that mesh-based engines physically cannot do.

## 11.1 Global Illumination via Spectral Bounce

Unlike Unreal's Lumen (which approximates GI on meshes), Ochroma can do physically correct spectral GI because every surface stores its reflectance spectrum. Light bouncing off a red brick wall actually shifts the spectrum of nearby surfaces toward red — not an approximation, but real physics.

- Implement spectral irradiance cache: each spatial cell stores incoming spectral radiance from all directions
- First bounce: direct sun/light → surface SPD → reflected spectrum
- Second bounce: reflected spectrum → nearby surfaces → colour bleeding
- Cache results in the SVO for real-time lookup

## 11.2 Volumetric Atmosphere

- Rayleigh scattering for blue sky and red sunsets
- Mie scattering for haze and fog
- God rays (volumetric light shafts through gaps in buildings)
- Spectral fog: fog attenuates different wavelengths differently

## 11.3 Water Rendering

- River and lake surfaces with spectral reflection/transmission
- Fresnel effect: more reflective at grazing angles
- Caustics from light through water onto riverbed
- Rain ripple effects on water surfaces
- Flowing water animation via UV offset

## 11.4 Neural Radiance Caching

- Small MLP network per spatial region learns indirect lighting
- Train online during gameplay (first few seconds per viewpoint)
- After convergence: real-time GI at minimal cost
- This is what Spectra already does — integrate their NRC system

## 11.5 Spectral Subsurface Scattering

- Light penetrates skin, leaves, wax, marble
- Different wavelengths penetrate to different depths
- Red light travels further through skin than blue
- Gaussian splats naturally model this: overlapping translucent volumes

## Exit Criteria

- [ ] Colour bleeding visible: red wall tints nearby white wall pink
- [ ] Volumetric god rays through building gaps at sunset
- [ ] River with reflections and caustics
- [ ] Neural radiance cache converges in <5 seconds
- [ ] Subsurface scattering visible on vegetation
