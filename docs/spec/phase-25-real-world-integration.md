# Phase 25 — Real-World Data Integration

**Goal:** Feed real-world data into the engine — GIS maps, OpenStreetMap, census data, weather APIs — to create digital twins of real cities.

## 25.1 OpenStreetMap Import
- Parse OSM .pbf files into road networks and building footprints
- Convert building footprints to Proc-GS rules
- Road types map to Ochroma road types
- Amenity tags → service building placement

## 25.2 GIS Elevation Data
- Import SRTM/ASTER DEM heightmaps
- Convert to terrain splats with correct elevation
- River/water body detection from elevation contours

## 25.3 Census Data Integration
- Import population density per district
- Map demographics to citizen generation (age distribution, education levels)
- Real-world economic data → starting budget parameters

## 25.4 Live Weather API
- Connect to OpenWeatherMap or similar
- Real-time weather in the digital twin matches the real city
- Spectral rendering adjusts to real-world illumination conditions

## 25.5 Satellite Imagery Draping
- Overlay satellite photos as spectral textures on terrain
- Colour-match Gaussian splats to satellite imagery
- Transition between satellite view and procedural detail

## Exit Criteria
- [ ] Import a real city from OpenStreetMap and render it
- [ ] Terrain elevation matches real-world heights
- [ ] Population distribution matches census data
- [ ] Live weather updates affect rendering in real-time
