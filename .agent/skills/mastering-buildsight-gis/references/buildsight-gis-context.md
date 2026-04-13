# BuildSight GIS System Context
## Reference for GIS Expert Skill — Phase 2 Spatial Safety Intelligence

---

## System Overview

**Project:** BuildSight — AI-powered Construction Site Safety Monitoring  
**Company:** Green Build AI (IGBC AP certified, NBC 2016 aligned)  
**Deployment Site:** G+1 Residential Building, Maran Constructions, Thanjavur, Tamil Nadu  
**Academic Context:** SASTRA Deemed University — B.Tech Civil Engg (Construction Management), 12-credit major project

---

## Delivered Phase 2 Assets (Current State)

### PostGIS Schema (12 tables, 4 views, 3 functions)

```sql
-- Core Tables
zones                  -- 16 spatial zones (4×4 grid, SRID 32644)
cameras                -- Camera metadata + frustum polygons
detection_events       -- CV detections with spatial location
worker_tracks          -- ByteTrack IDs → spatial trajectories (Phase 5)
construction_schedule  -- Floor/Activity/Expected workers per zone-time
alert_log              -- Generated alerts with tier + zone + timestamp
engineer_feedback      -- Manual corrections for adaptive trust calibration
risk_scores            -- Aggregated zone risk (updated every 5 min)
ppe_violations         -- Per-class violation counts per zone
shift_summary          -- Daily/shift risk summaries
zone_occupancy         -- Real-time worker count per zone
anchor_points          -- Homography calibration anchors per camera

-- Real-time Views
v_current_risk         -- Live risk scores joined to zone geometries
v_active_violations    -- Unresolved PPE violations last 30 min
v_zone_occupancy       -- Current worker count per zone
v_schedule_mismatch    -- Zones where actual ≠ expected workers

-- Spatial Functions
assign_zone(geom)      -- Returns zone_id for a given geometry
compute_risk(zone_id)  -- Recomputes risk score for a zone
check_exclusion(geom, radius) -- Checks BOCW 15m exclusion zone
```

### Delivered Shapefiles (7 files, UTM Zone 44N / EPSG:32644)

| File | Contents |
|------|----------|
| `site_boundary.shp` | Full site perimeter |
| `zone_grid.shp` | 16 zones (4×4 default, to be updated with actual plan) |
| `camera_coverage.shp` | Camera frustum polygons (estimated) |
| `high_risk_areas.shp` | Pre-marked scaffold/machinery zones |
| `exclusion_zones.shp` | BOCW 15m machinery exclusion buffers |
| `entry_exit_points.shp` | Site gates and worker flow paths |
| `nbc_clearances.shp` | NBC 2016 minimum clearance boundaries |

### Python PostGIS Connector
- Connection pool (ThreadedConnectionPool, min=2, max=10)
- Async-ready wrapper for real-time detection inserts
- Spatial query helpers: `get_zone_for_point()`, `get_risk_snapshot()`, `log_alert()`

---

## Zone Grid Specification

**Current state:** Default 4×4 grid (16 equal zones across site extent)  
**Pending:** Maran Constructions site plan upload → redigitize to actual zone layout

**Zone naming convention:**
```
Z01 Z02 Z03 Z04   ← North
Z05 Z06 Z07 Z08
Z09 Z10 Z11 Z12
Z13 Z14 Z15 Z16   ← South (entry/exit)
```

**Zone attributes:**
- `zone_id` (integer, 1–16)
- `zone_name` (text, e.g., "Scaffold East", "Entry Gate")
- `risk_baseline` (float, 0–1, NBC-based pre-assessment)
- `floor_level` (integer, ground=0, first=1)
- `activity_type` (text, e.g., "masonry", "formwork", "finishing")
- `max_workers` (integer, NBC-compliant safe occupancy)
- `geom` (POLYGON, SRID 32644)

---

## CV-to-GIS Pipeline (Detection → Spatial Event)

### Input (from Ensemble)
```json
{
  "frame_id": 1042,
  "camera_id": "cam_01",
  "timestamp": "2025-11-15T09:32:11.423Z",
  "detections": [
    {
      "class_id": 3,
      "class_name": "worker",
      "confidence": 0.87,
      "bbox": [312, 445, 89, 201],  // [x, y, w, h] pixels
      "ppe_status": {
        "helmet": true,
        "vest": false,
        "boots": true
      }
    }
  ]
}
```

### Processing Pipeline
```
1. Extract bbox centre → (cx, cy) = (x + w/2, y + h*0.7)  [torso-centre]
2. Apply confidence gate: skip if confidence < zone_gate[scene_type]
3. H = load_homography(camera_id)
4. site_xy = pixel_to_site_coords((cx, cy), H)
5. geom = ST_SetSRID(ST_MakePoint(site_xy[0], site_xy[1]), 32644)
6. zone_id = assign_zone(geom)
7. INSERT INTO detection_events (frame_id, camera_id, class_id, confidence,
                                  ppe_status, location, zone_id, timestamp)
8. IF ppe_status has violations → INSERT INTO ppe_violations
9. CALL compute_risk(zone_id)
10. CALL check_alert_thresholds(zone_id)
```

---

## Alert System (3-Tier, Anti-Fatigue Design)

| Tier | Trigger | Aggregation | Channel |
|------|---------|-------------|---------|
| CRITICAL | Worker in exclusion zone / scaffold without helmet in >3.5m zone | Immediate | SMS + Dashboard |
| HIGH | >30% of zone workers missing PPE for >2 min | 1 per zone per 5 min | Dashboard + Log |
| PERIODIC | Zone risk score >0.7 for >15 min | 1 per zone per shift | Daily report |

**Anti-fatigue rules:**
- Max 3 CRITICAL alerts per zone per hour before escalating to supervisor mode
- Mass violation: 1 aggregated alert (not N per-worker alerts)
- Auto-suppress duplicate alerts within 60-second window

---

## Adaptive Trust Calibration

**Concept:** Dynamically adjust weight of CV detections vs. schedule data based on recent accuracy.

```python
trust_weight_cv = 0.7       # Initial CV trust
trust_weight_schedule = 0.3  # Initial schedule trust

# After each engineer feedback correction:
if correction_source == 'cv_false_positive':
    trust_weight_cv *= 0.95      # Decay CV trust for this zone/condition
    trust_weight_schedule *= 1.05 # Increase schedule trust
elif correction_source == 'schedule_error':
    trust_weight_schedule *= 0.95
    trust_weight_cv *= 1.05

# Normalize
total = trust_weight_cv + trust_weight_schedule
trust_weight_cv /= total
trust_weight_schedule /= total
```

**Log to:** `engineer_feedback` table → drives quarterly model recalibration

---

## Scene Conditions → GIS Implications

| Condition | Scene | GIS Impact |
|-----------|-------|------------|
| Normal | S1 | Standard confidence gates; all zones active |
| Dusty | S2 | Reduce confidence gate by 0.05; flag dust-prone zones |
| Low-light | S3 | Apply night-shift zone schema; disable color-check violations |
| Crowded | S4 | Enable crowd-density routing; zone occupancy cap alerts |

---

## Environment Variables Required

```bash
PG_HOST=localhost           # or sastra-node1 PostGIS host
PG_DB=buildsight
PG_USER=joseva
PG_PASS=<from .env>
PG_PORT=5432
CAMERA_CONFIG=/nfsshare/joseva/config/cameras.json
HOMOGRAPHY_DIR=/nfsshare/joseva/config/homography/
SHAPEFILE_DIR=/nfsshare/joseva/spatial/shapefiles/
```
