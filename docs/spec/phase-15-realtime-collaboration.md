# Phase 15 — Real-Time Collaboration & Live Editing

**Goal:** Multiple developers and players can edit the same world simultaneously with conflict-free resolution — something Unreal's workflow cannot match.

## 15.1 CRDT World State
- Every entity change is a CRDT operation
- Concurrent edits merge automatically (no locks, no conflicts)
- Undo/redo per-user (not global)
- Operation log for time-travel debugging

## 15.2 Live Asset Editing
- Modify a Proc-GS rule → all instances update in real-time
- Change a spectral material → every surface using it re-renders immediately
- Hot-swap .vxm assets without restarting

## 15.3 Collaborative City Building
- Multiple mayors edit different districts simultaneously
- See other users' cursors and actions in real-time
- Permission zones: each user can only edit their assigned district
- Merge sessions: combine independently built districts

## Exit Criteria
- [ ] Two users edit the same city without conflicts
- [ ] Rule change propagates to all instances within 1 frame
- [ ] Operation log can replay the last 1000 edits
