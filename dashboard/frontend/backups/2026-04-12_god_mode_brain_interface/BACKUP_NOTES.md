# Backup Notes: GOD MODE Brain Interface

**Date:** 2026-04-12
**Reason for Backup:** Complete redesign of the GOD MODE interface as requested to replace the Spline-based 3D core with a native React-based BuildSight Brain interface.

## Current GOD MODE Layout
- **Top Command Strip:** Contains the branding, a status block, KPI intelligence strip, and an exit button.
- **Left HUD (Tactical & Escalation):** Panels for active escalation protocols (alerts feed) and safety intelligence (metrics and recommended actions).
- **Center Stage:** Features a pulsing ring and a full-screen Spline iframe (`https://app.spline.design/file/9df9601f-9999-4c76-b44f-8328b500f7b8`), functioning as the "Neural Twin". It handles loading states and errors.
- **Right HUD (Analytics & Turner AI):** Tabbed interface for GEO AI (worker density heatmap, zone status), TURNER AI (chat oversight), and CAMERAS (site health monitoring).
- **Footer Controls:** A drawer containing the Core Matrix Override for detection thresholds and AI routing parameters.

## Component Behaviors
- **Spline Integration Logic:** Rendered via an `iframe`. Uses an `onLoad` handler to update the `splineLoaded` state and hides the fallback spinner. Uses `ErrorBoundary` to catch mounting issues.
- **GeoAI Module:** Displays a static mock heatmap (SVG background) and zone status (Alpha/Beta).
- **Turner AI Module:** Displays hardcoded AI chat bubbles indicating oversight status and alerts. Input is disabled.
- **Escalation Workflow:** Driven by `liveAlerts` from `DetectionStatsContext`, rendering animated alert items based on severity. The overall state (`stable`, `elevated`, `critical`) is computed based on proximity violations and helmet compliance.
- **Risk Visualization:** A `riskIntensity` score drives the color and animation speed of a pulsing ring behind the Spline iframe.

## Goal
Replace the Spline `iframe` completely with a native React component depicting the BuildSight Brain (circular neural-core visualization, pulsing rings, orbiting nodes, dynamic status indicators). Ensure full BuildSight customization features, native intelligence modules, an advanced GeoAI layer, and a Turner AI command layer are integrated into a futuristic, premium, enterprise-grade dark-mode native UI.
