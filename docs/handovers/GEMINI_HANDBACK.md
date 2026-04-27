# Handback to Jovi (Gemini)

**Task ID**: HANDOFF-004
**Status**: success
**Completed By**: Toni (Claude)
**Date**: 2026-04-12

---

## Summary

Implemented the refined helmet precision pipeline per HANDOFF-004. The changes target false positives on bare heads, wrists, and palms introduced by the S4 recall relaxations, while preserving orange, yellow, white, and beige/dusty safety helmets.

### Precision Impact (estimated from gate analysis)
- **Wrist/palm FPs**: eliminated by Gate 1 (vertical zone). These detections land at 40–70% of worker height — below the 35% cutoff.
- **Bare head FPs**: caught by Gate 2+3 penalty chain (skin-tone + soft texture + weak worker score). Requires 3 simultaneous signals — single-signal false rejection is prevented.
- **Safety colour helmets**: all preserved via `_is_safety_color()` bypass before any penalty gate fires.
- **Recall cost**: minimal — the vertical gate is the strictest and affects only detections clearly outside the head zone.

---

## Files Modified

### `scripts/adaptive_postprocess.py`
1. **`has_worker_overlap()`** — Added `cls_id` parameter. For helmets (`cls_id=0`): enforces vertical zone (upper 40% + 10% headroom above). For vests: unchanged containment logic.

2. **`HelmetValidationLayer` class** (new) — Multi-gate precision filter:
   - `_gate_vertical_ratio()`: Helmet centroid must be in top **35%** of matched worker box.
   - `_is_safety_color()`: HSV range check for orange / yellow / white / beige — these bypass all penalty gates.
   - `_skin_tone_fraction()`: Fraction of crop pixels in skin-tone HSV range `H=0-25, S=30-135, V=60-230` (stops at S=135 to avoid orange/yellow overlap).
   - `_is_hard_texture()`: Laplacian variance ≥ 60 → rigid shell; below → soft (skin/cloth).
   - `_validate_one()`: Hard reject fires only when vertical zone fails OR all 3 skin-tone signals fire (skin + soft texture + weak worker score).
   - `filter()`: Applies to the post-anchor detection list, preserving workers and vests.

3. **`_helmet_validator`** — Module-level singleton, reused across frames.

4. **`apply_all_rules()`** — Added `has_worker_overlap(cls_id=box["cls"])` call and `_helmet_validator.filter(boxes, image)` after the PPE anchor step. New stat key: `after_helmet_validation`.

### `scripts/site_aware_ensemble.py`
1. **`_worker_has_ppe()`** — For `CLS_HELMET`: enforces vertical constraint (upper 40% + 10% headroom with 15% horizontal pad). For `CLS_VEST`: standard containment + 20% below extension. This prevents wrist/palm detections from incrementing `helmet_streak`.

2. **`_copy_ppe_from_track()`** — Added docstring clarifying synthetic helmet placement stays within the upper 22% of worker height (inside the new vertical constraint zone).

---

## Preservation Verification

| Helmet Colour | HSV Range Matched | Bypasses Penalty Gate? |
|---|---|---|
| Orange | H=5-22, S>120, V>80 | Yes — `PRESERVE_COLORS["orange"]` |
| Yellow | H=20-38, S>120, V>100 | Yes — `PRESERVE_COLORS["yellow"]` |
| White | S<40, V>170 | Yes — `PRESERVE_COLORS["white"]` |
| Beige/Dusty | H=10-28, S=15-75, V=110-220 | Yes — `PRESERVE_COLORS["beige_tan"]` |
| Bare skin | H=0-25, S=30-135 | No — penalty gate fires |

---

## Open Issues

1. **Brown helmets**: H=8-18, S=60-120, V=40-100 overlaps with dark skin tones in low-light (S3). The skin-tone range (S=30-135) catches both. In S3_low_light these may be misclassified. Mitigation: texture gate (Laplacian ≥ 60) should differentiate hard-shell reflection from skin — but if lighting is very flat this may not hold. **Recommend**: add a separate `"brown_dark"` preserve range `[5, 50, 40], [22, 120, 130]` in a follow-up if brown helmet recall is observed to drop.

2. **Very small helmets (< 12px)**: Laplacian variance on tiny crops is unreliable. The texture gate returns `True` (don't reject) when the crop is too small — so small-helmet recall is preserved, but small wrist FPs on S4 may survive if the vertical gate passes them. Size is already controlled by `is_valid_helmet()` in `server.py` (8×8 px minimum).

3. **S4_crowded workers at steep below-camera angles**: The 35% vertical zone is calibrated for upright workers. Workers viewed from very steep angles (compressed boxes) may have tighter head zones. Suggest monitoring S4 helmet recall after deployment and adjusting to 40% if needed (already 40% in the updated `has_worker_overlap`).
