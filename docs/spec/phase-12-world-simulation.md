# Phase 12 — World-Scale Simulation

**Goal:** Scale the simulation to 100km² with millions of entities, matching Cities: Skylines 2's scale while surpassing its depth.

## 12.1 Agent-Based Traffic (Replace LWR)

- Individual vehicle agents on road network (not just density)
- Lane-changing behaviour
- Intersection signal logic (traffic lights, roundabout priority)
- Emergency vehicle routing (fire trucks, ambulances)
- Parking simulation (vehicles park near destination)

## 12.2 Dynamic Weather System

- Weather state machine: Clear → Cloudy → Rain → Storm → Clear
- Regional weather (different districts can have different weather)
- Temperature affecting citizen behaviour (stay indoors in extreme weather)
- Wind affecting smoke/particle direction
- Lightning strikes during storms (visual + possible fire trigger)

## 12.3 Ecosystem Simulation

- Trees grow over game-time (splat count increases)
- Vegetation spreads in parks and unmaintained areas
- Wildlife in parks (birds, squirrels as small splat clusters)
- Seasons affect vegetation colour (green → orange → bare → green)

## 12.4 Advanced Economy

- Supply and demand curves per commodity
- Import/export with external trade partners
- Dynamic pricing: scarce goods cost more
- Real estate market: property values affect citizen behaviour
- Tourism: visitors generate revenue

## 12.5 Multi-Tile Streaming

- Async tile load/unload as camera moves
- Predictive pre-loading based on camera velocity
- NVMe-optimised I/O via io_uring (Linux)
- VRAM budget management: evict least-recently-viewed tiles

## Exit Criteria

- [ ] 100km² world navigable without loading screens
- [ ] Individual vehicles visible on roads with lane behaviour
- [ ] Weather changes affect gameplay and visuals
- [ ] Trees grow and seasons change vegetation
- [ ] Dynamic market prices respond to supply/demand
