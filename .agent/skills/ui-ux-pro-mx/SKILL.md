---
name: perfecting-industrial-uis
description: Acts as a senior UI/UX engineer for BuildSight. Implements premium MX-grade visual patterns and now includes a Python recommendation engine for agent-friendly UI guidance and validation.
---

# UI/UX Pro MX

This skill is the BuildSight interface system for premium tactical dashboards, command centers, and AI-assisted design recommendations.

## When to use this skill
- When the user requests premium, pro, or MX-grade UI upgrades.
- When designing tactical dashboards, telemetry-heavy panels, or command-center layouts.
- When the user wants structured design recommendations for agents or humans.
- When the user mentions `uipro`, glassmorphism, scanlines, or tactical frames.

## Design language
- Primary accent: `var(--color-accent)` with matte industrial bases.
- Surface treatment: layered glass panels with restrained blur.
- Framing: tactical corner brackets and telemetry dividers.
- Typography: strong display face plus mono data face for live signals.
- Motion: low-frequency sweep and staged transitions, never decorative overload.

## Command console
- `python .agent/skills/ui-ux-pro-mx/scripts/uipro.py recommend --product geoai --theme dark --format json`
- `python .agent/skills/ui-ux-pro-mx/scripts/uipro.py recommend --product fintech --theme dark --format table`
- `python .agent/skills/ui-ux-pro-mx/scripts/uipro.py validate --contrast palette.json`
- `python .agent/skills/ui-ux-pro-mx/scripts/uipro.py validate --spacing design.json`
- `python .agent/skills/ui-ux-pro-mx/scripts/uipro.py db search styles --tag dark`
- `python .agent/skills/ui-ux-pro-mx/scripts/uipro.py prompt --product saas --theme light`

## Resources
- `resources/design_db.v1.json`: versioned recommendation database.
- `scripts/uipro.py`: Python CLI for recommendations, validation, search, and prompt generation.
- `scripts/uipro.sh`: Bash wrapper.
- `scripts/uipro.ps1`: PowerShell wrapper.
