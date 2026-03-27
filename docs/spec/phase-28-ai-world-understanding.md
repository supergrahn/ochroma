# Phase 28 — AI World Understanding

**Goal:** The engine understands the world it renders. AI systems can answer questions about the scene, generate captions, and enable natural language interaction.

## 28.1 Scene Understanding
- "What is that building?" → "A 3-story Victorian terraced house, built during the Industrial Era, housing 6 citizens"
- Semantic queries on the entity graph
- Natural language scene description generation

## 28.2 Natural Language Commands
- "Build a park near the school" → AI finds school, selects nearby empty land, zones as park
- "Make the traffic flow better on Main Street" → AI analyses congestion, suggests road improvements
- "Show me where crime is highest" → AI activates crime overlay, moves camera to worst area

## 28.3 Conversational AI Advisor
- "Why are citizens unhappy?" → "Most complaints are about long commute times. Consider adding a bus route between Northgate and the industrial district."
- Context-aware: references actual city data
- Remembers previous conversations within a session

## Exit Criteria
- [ ] Clicking a building shows AI-generated description
- [ ] "Build a park near the school" executes correctly
- [ ] AI advisor explains city problems with specific data
