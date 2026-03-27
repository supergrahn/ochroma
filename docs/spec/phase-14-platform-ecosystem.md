# Phase 14 — Platform & Ecosystem

**Goal:** Transform Ochroma from a game engine into a platform with marketplace, community tools, and cross-platform support.

## 14.1 Asset Marketplace

- Online asset library where creators share .vxm assets and Proc-GS rules
- Revenue sharing for paid assets
- Rating and review system
- Automatic spectral validation (ensures assets work with the spectral pipeline)

## 14.2 Console Ports

- PlayStation 5 (Vulkan via MoltenVK-equivalent, or native GNM)
- Xbox Series X (DirectX 12 backend for wgpu)
- Nintendo Switch 2 (Vulkan)
- Each platform: controller mapping, performance targets, certification requirements

## 14.3 Mobile / Web

- WebGPU backend for browser-based play
- Mobile: reduced LOD budgets, simplified simulation
- Cloud streaming option: render on server, stream to thin client

## 14.4 Cross-Platform Multiplayer

- Account system linking Steam, PlayStation, Xbox, Epic
- Cross-platform lobbies
- Shared save format across platforms
- Voice chat integration

## 14.5 Engine Marketplace

- Plugin system: third-party engine extensions
- Template projects: city builder, RPG, horror, exploration
- Tutorial ecosystem with interactive examples
- Certification program for Ochroma developers

## Exit Criteria

- [ ] Game runs on PlayStation 5
- [ ] Asset marketplace serves 100+ community assets
- [ ] WebGPU build runs in Chrome
- [ ] Cross-platform multiplayer works between PC and console
