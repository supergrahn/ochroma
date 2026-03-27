# Phase 17 — Machine Learning Engine

**Goal:** Embed ML inference directly into the engine for real-time AI that learns and adapts.

## 17.1 Neural Traffic Prediction
- ML model predicts traffic congestion 10 minutes ahead
- Suggests road improvements to the player
- Trained on the city's actual traffic patterns

## 17.2 Citizen Behaviour Learning
- Citizens learn preferred routes over time (not just shortest path)
- Neighbourhood preferences emerge from satisfaction history
- Economic agents learn market dynamics

## 17.3 Procedural Detail Enhancement
- Neural network adds micro-detail to Proc-GS buildings in real-time
- Learns from player feedback (which buildings they modify vs leave)
- Style transfer: apply one district's aesthetic to another

## 17.4 Smart City Advisor
- ML-powered advisor that learns what works for this specific city
- "Last time you increased industrial zones, pollution rose 20% — consider adding parks first"
- Personalised to player's style

## 17.5 Neural Compression
- Train a tiny neural network to compress .vxm assets 10x
- Decompress on GPU in real-time
- Enables larger worlds in less VRAM

## Exit Criteria
- [ ] Traffic prediction model runs at <1ms per frame
- [ ] Citizens adapt routes based on experience
- [ ] Neural detail enhancement visibly improves building quality
- [ ] Advisor references player's actual history
