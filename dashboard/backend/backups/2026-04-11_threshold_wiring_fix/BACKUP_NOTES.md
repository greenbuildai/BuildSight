# Backup Notes - Threshold Wiring Fix
Date: 2026-04-11

## Problems Being Fixed
1. Sidebar MIN CONFIDENCE slider and Detection Panel class sliders not synchronized
2. Construction materials (cement bags, bricks, buckets) detected as workers at low thresholds
3. Auto scene classifier not re-evaluating when threshold changes
4. Auto scene classifier using raw detection count instead of validated worker count

## Current Known State
- Sidebar slider: 25% (global)
- Worker slider: 25%
- Helmet slider: 50%
- Vest slider: 35%
- Scene classifier: does not respond to threshold changes
- No valid worker validation pipeline exists yet
- No material suppression gates exist yet

## Actual Live Frontend Paths
- Detection panel: dashboard/src/components/DetectionPanel.tsx
- Detection panel styles: dashboard/src/components/DetectionPanel.css
- Settings context: dashboard/src/SettingsContext.tsx
- Sidebar threshold control: dashboard/src/App.tsx
- Settings panel: dashboard/src/components/SettingsPanel.tsx
