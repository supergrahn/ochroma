# Ochroma Editor — Complete Specification

The editor runs inside the engine, toggled with Tab. It uses egui for UI panels and renders gizmos/overlays on top of the 3D viewport.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Engine Window                              │
│  ┌──────────┬──────────────────────────────┬───────────────┐ │
│  │ Scene    │                              │  Details      │ │
│  │ Outliner │      3D Viewport             │  Panel        │ │
│  │          │                              │               │ │
│  │ [Tree]   │  [Scene rendered here]       │  Transform    │ │
│  │          │  [Gizmos on top]             │  Components   │ │
│  │          │  [Grid on ground]            │  Scripts      │ │
│  │          │                              │  Custom Data  │ │
│  ├──────────┤                              ├───────────────┤ │
│  │ Content  │                              │  World        │ │
│  │ Browser  │                              │  Settings     │ │
│  │          │                              │               │ │
│  │ [Files]  ├──────────────────────────────┤  Time of Day  │ │
│  │ [Search] │  Toolbar: Play|Pause|Stop    │  Gravity      │ │
│  │          │  Gizmo: Move|Rotate|Scale    │  Fog          │ │
│  │          │  Snap: Grid|Surface          │  Skybox       │ │
│  └──────────┴──────────────────────────────┴───────────────┘ │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │  Output Log / Console                                    │ │
│  └──────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Feature List — Prioritized

### Critical (must have for first release)

**1. 3D Viewport with Navigation**
- egui `CentralPanel` renders the 3D scene
- Orbit camera: middle-mouse drag to orbit, scroll to zoom
- Pan camera: shift+middle-mouse to pan
- Fly mode: right-click+WASD for FPS-style navigation
- Grid rendered on XZ plane (lines burned into framebuffer or drawn as splats)
- Frame selected: press F to zoom to selected entity

Implementation: The viewport is the scene rendered by the engine. Editor overlays (gizmos, grid, selection highlight) are drawn on top after the scene render.

**2. Scene Outliner (hierarchy tree)**
Already partially exists. Enhance:
- Collapsible tree with parent/child hierarchy
- Drag to reparent entities
- Right-click context menu: Duplicate, Delete, Rename, Add Child
- Icons per entity type (mesh, light, audio, script)
- Visibility toggle per entity (eye icon)
- Lock toggle (padlock icon)
- Search/filter bar

**3. Details Panel (property inspector)**
Already partially exists. Enhance:
- Auto-generates UI for ALL components on the selected entity
- Transform: position/rotation/scale with drag-value widgets
- AssetRef: file path with browse button
- Collider: shape selector dropdown + dimension inputs
- Scripts: list with + button to add, - to remove
- Audio: clip path, volume slider, looping toggle
- Light: colour picker, intensity, radius
- Custom Data: key-value editor (add/remove/edit)
- Tags: tag list with + button

**4. Transform Gizmos (world/local space)**
Already have 2D line gizmos. Enhance:
- Toggle world-space vs local-space (press X)
- Proper axis constraint: click+drag on one axis
- Plane constraint: click+drag on plane between two axes
- Screen-space uniform scale: click+drag center
- Visual feedback: highlight hovered axis in yellow
- Numeric input: type exact values in toolbar

**5. Play/Stop/Pause Controls**
- Toolbar buttons: Play (runs game scripts + physics), Pause, Stop (reset to editor state)
- Play saves the current editor state, enters play mode
- Stop restores the saved state (undo all play-mode changes)
- Pause freezes physics/scripts but allows camera movement
- Keyboard: F5 = Play, Shift+F5 = Stop, F6 = Pause

**6. Undo/Redo with History Panel**
Already have UndoStack. Enhance:
- Every editor action (move, rotate, add, delete, property change) pushed to undo stack
- Ctrl+Z undo, Ctrl+Y redo
- History panel shows last N actions with descriptions
- Click an action in history to jump to that state

**7. Output Log / Console**
- Bottom panel showing engine log messages
- Filter by severity: Info, Warning, Error
- Search/filter text
- Clear button
- Console input: type Rhai expressions, evaluate live
- Color-coded: green=info, yellow=warning, red=error

**8. Grid and Snapping**
- World grid visible on XZ plane (drawn as lines in framebuffer)
- Grid snap: entities snap to grid when moving (toggle with G)
- Configurable grid size: 0.25, 0.5, 1.0, 2.0, 5.0 metres
- Surface snap: snap to surface of other entities (for placing objects on terrain)
- Rotation snap: snap to 15° increments (toggle with R)

### High Priority (should have)

**9. Content Browser (enhanced)**
Already exists. Add:
- Thumbnail preview for .ply files (render small preview image)
- Drag from browser to viewport to place
- Right-click: Import, Delete, Rename, Show in Explorer
- Filter by type tabs: All | Models | Audio | Scripts | Maps

**10. Copy/Paste/Duplicate**
- Ctrl+C: copy selected entity (serialize to clipboard)
- Ctrl+V: paste at camera position
- Ctrl+D: duplicate in-place (offset by 1 unit)
- Multi-select: shift+click to select multiple, copy/paste all

**11. World Settings Panel**
- Time of day slider (0-24h)
- Gravity value
- Fog enabled/density/colour
- Ambient light colour/intensity
- Sky settings (skybox path or procedural sky)
- Physics settings (timestep, gravity)

**12. Prefab System**
- Select entities → right-click → Create Prefab
- Prefab saved as .ochroma_prefab (JSON: entity hierarchy + components)
- Drag prefab from browser → instantiates all entities
- Edit prefab → propagates to all instances (optional)

### Medium Priority (nice to have)

**13. Multi-Viewport**
- Split view: top/front/side orthographic views alongside perspective
- Toggle with Ctrl+1,2,3,4

**14. Grouping and Layers**
- Select multiple → Ctrl+G to group
- Layers panel: create layers, assign entities, toggle visibility per layer
- Layer uses: "Environment", "Gameplay", "Lighting", "Debug"

**15. Stats/Profiler Overlay**
- FPS, frame time, splat count, entity count
- Draw call count, VRAM usage
- Physics step time
- Toggleable with ` key

**16. Terrain Sculpting Tools**
- Raise/lower brush
- Smooth brush
- Flatten brush
- Paint material brush (assign material zones to terrain)
- All operating on the volumetric SDF

**17. Foliage Painting**
- Select a foliage type (tree, bush, grass .ply)
- Paint mode: click-drag to scatter instances
- Density, scale randomness, alignment to surface normal
- Erase mode to remove foliage

### Low Priority (later)

**18. Blueprint/Visual Scripting** — defer to Rhai text scripting
**19. Material Editor** — defer to TOML hot-reload
**20. Sequencer** — defer to CinematicCamera keyframes
**21. Level Streaming** — defer to tile manager
**22. Plugin System** — defer to mod manager

## Implementation Plan

### Batch A: Editor Core (3 agents)

**Agent A1: Editor State Machine + Play/Pause/Stop**
- EditorMode enum: Editing, Playing, Paused
- Play: serialize world state → run scripts/physics
- Stop: deserialize saved state → restore
- Keyboard shortcuts: F5, Shift+F5, F6
- Toolbar buttons in egui

**Agent A2: Enhanced Outliner + Details Panel**
- Tree widget with collapse/expand
- Context menu (right-click)
- Auto-generated property UI for all component types
- Color picker for lights
- File browser button for asset paths

**Agent A3: Grid, Snapping, Copy/Paste**
- Grid rendering (lines on XZ plane)
- Snap-to-grid in gizmo drag
- Copy/paste/duplicate entities
- Multi-select with shift+click

### Batch B: Editor Polish (3 agents)

**Agent B1: Output Log + Console**
- Log panel with severity filtering
- Rhai console input
- Colour-coded messages

**Agent B2: World Settings + Prefabs**
- World settings panel
- Prefab save/load/instantiate

**Agent B3: Terrain + Foliage Tools**
- SDF sculpt brushes in editor
- Foliage scatter paint mode
