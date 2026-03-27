# Phase 23 — Spectral Audio Engine

**Goal:** Audio as physically accurate as rendering. Sound propagation through Gaussian splat geometry with spectral frequency absorption — a world first.

## 23.1 Acoustic Ray Tracing
- Cast audio rays from sources through the SVO
- Gaussian splat materials absorb frequencies based on spectral properties
- Brick absorbs high frequencies → muffled sound behind walls
- Glass transmits high frequencies → clear sound through windows

## 23.2 Reverb from Geometry
- Narrow streets: short reverb, many reflections
- Open plazas: long reverb, few reflections
- Indoor spaces: room-specific impulse response
- Computed from SVO density around listener

## 23.3 Doppler and Distance
- Moving sources: frequency shift based on relative velocity
- Distance: inverse-square attenuation with air absorption
- Obstruction: Gaussians between source and listener attenuate sound

## 23.4 Procedural Sound Generation
- Traffic noise generated from vehicle count and speed
- Construction sounds from building growth state
- Nature: procedural bird calls, wind through trees
- Weather: rain intensity → droplet density → sound

## Exit Criteria
- [ ] Sound behind a brick wall is audibly muffled
- [ ] Reverb differs between narrow street and open plaza
- [ ] Moving vehicles produce Doppler shift
- [ ] Rain sound intensity matches visual precipitation
