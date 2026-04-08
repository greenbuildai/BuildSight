# Leon — Handoff from Toni
**Date:** 2026-04-05
**From:** Toni
**Status:** ACTIVE — Refactor Monitoring Workspace

---

## Status

Leon, Phase 2.5 was completed successfully. We are now moving into **Phase 3: Unified Monitoring Hub**.

The goal is to elevate the dashboard from a functional prototype to a polished industrial command center by unifying the live surveillance feed and PPE detection into a single workspace.

## Active Assignment: Monitoring Workspace Consolidation

### Objective
Unify the "Live Surveillance" feed and "PPE Detection" panel into a single, integrated monitoring workspace. Move the "Video" and "Live" mode selectors from the sidebar/bottom-panel to the top navigation header to serve as primary global controls.

### Technical Implementation

1.  **Lift Monitoring State**:
    - In `App.tsx`, introduce a `dashboardMode` state (`'LIVE' | 'VIDEO' | 'IMAGE'`).
    - This state will control what is rendered in the `hero-grid__main` area.

2.  **Top Navigation Mode Selector**:
    - Build a professional mode switcher component in the `header` (inside `main-body` in `App.tsx`).
    - This will serve as the primary global control for the dashboard's operational state.

3.  **Unified Workspace Hub**:
    - Update `hero-grid__main` to render the appropriate component dynamically based on `dashboardMode`.
    - `LIVE` will show the interactive surveillance map (`LiveFeed.tsx`).
    - `VIDEO/IMAGE` will show the detection logic currently in `DetectionPanel.tsx`.

4.  **Refactor DetectionPanel.tsx**:
    - Refactor the component to take `mode` as a prop and remove its internal tabs.
    - Remove the bottom-panel implementation from `App.tsx`, consolidating its functionality into the central hub.

### References & Resources
- [Implementation Plan](file:///C:/Users/brigh/.gemini/antigravity/brain/30e548f9-0a7c-4e4b-995c-71d7fd173690/implementation_plan.md) (Detailed breakdown by Toni)
- [App.tsx](file:///e:/Company/Green%20Build%20AI/Prototypes/BuildSight/dashboard/src/App.tsx) (Main layout)
- [DetectionPanel.tsx](file:///e:/Company/Green%20Build%20AI/Prototypes/BuildSight/dashboard/src/components/DetectionPanel.tsx) (Detection logic sources)
- [App.css](file:///e:/Company/Green%20Build%20AI/Prototypes/BuildSight/dashboard/src/App.css) (Layout styling)

---

**Next Steps**: Execute the refactor by updating `App.tsx`, `DetectionPanel.tsx`, and `App.css` in accordance with the established industrial aesthetic.
