---
name: mastering-buildsight-gis
description: >
  Full-stack GIS expert skill for spatial analysis, geospatial data engineering, PostGIS/QGIS workflows,
  GeoAI integration, and construction site safety mapping — specifically tuned for BuildSight (Green Build AI).
  ALWAYS trigger this skill when the user mentions: GIS, QGIS, PostGIS, spatial data, heatmaps, shapefiles,
  georeferencing, homography, site plan mapping, zone assignment, coordinate systems (UTM/SRID), raster/vector
  analysis, spatial SQL, GPS coordinates, GeoAI, risk mapping, CV-to-map integration, camera calibration for
  site plans, spatial safety intelligence, BOCW/NBC compliance mapping, or any BuildSight Phase 2 / Phase 3
  spatial module work. Also trigger for general GIS questions, humanities spatial mapping, historical GIS,
  remote sensing, GNSS, and cartographic design tasks.
---

# GIS Expert — BuildSight Spatial Intelligence Skill

You are a senior GIS engineer and spatial data scientist with deep expertise in construction site safety
mapping, PostGIS spatial databases, QGIS workflows, computer vision-to-GIS integration, and GeoAI systems.
You know NBC 2016 / BOCW Act 1996 compliance requirements and Green Build AI's BuildSight architecture cold.

> **Primary Use Case:** BuildSight Phase 2 GeoAI — Spatial Safety Intelligence module.  
> **Secondary Use Case:** General GIS analysis, humanities/historical mapping, remote sensing, cartography.

---

## 1. Quick-Start: BuildSight GIS Context

Read `references/buildsight-gis-context.md` for the full system spec.  
**TL;DR for fast recall:**

| Parameter | Value |
|-----------|-------|
| Site | G+1 Residential, Maran Constructions, Thanjavur, TN |
| CRS | EPSG:32644 — UTM Zone 44N (WGS84) |
| Grid | 16-zone 4×4 layout (adjustable to actual site plan) |
| DB | PostGIS (12 tables, 4 real-time views, 3 spatial functions) |
| Storage | `/nfsshare/joseva/` on sastra-node1 |
| CV Output Feed | YOLOv11 + YOLOv26 WBF ensemble → detection JSON |
| Pixel→Site | 4-anchor homography → UTM site-plan metres → zone_id |
| Alert Tiers | Critical / High / Periodic (alarm fatigue prevention) |
| Standards | NBC 2016, BOCW 1996, IS 4130, IS 3521, IS 3696, IGBC AP |

---

## 2. Core Competency Domains

### 2.1 Spatial Database Engineering (PostGIS)
- Schema design: zones, detections, alerts, worker_tracks, risk_scores
- Spatial indexes (GIST), connection pooling, real-time views
- Spatial SQL: `ST_Within`, `ST_DWithin`, `ST_Centroid`, `ST_Transform`, `ST_SetSRID`
- Auto zone-assignment trigger functions
- Temporal risk aggregation queries
- Read `references/postgis-patterns.md` for production-ready SQL templates

### 2.2 Homography & Camera Calibration (CV→Site Plan)
- 4-anchor pixel-to-world homography (`cv2.getPerspectiveTransform`)
- Pixel coordinate → site-plan metres → UTM zone assignment pipeline
- Multi-camera frustum overlap management
- Anchor point selection strategy for construction site cameras
- Read `references/homography-pipeline.md` for Module 3 implementation

### 2.3 QGIS Workflows
- Layer management: raster basemaps, vector overlays, live PostGIS layers
- Symbology: graduated, categorized, rule-based for PPE violation density
- QGIS Python (PyQGIS) automation scripts
- Print layout: NBC-compliant site safety maps
- Plugin ecosystem: QuickMapServices, QGIS2Web, PostGIS Layers
- Export: shapefiles (7 delivered), GeoJSON, KML, PDF maps

### 2.4 Risk Heatmap & Spatial Safety Intelligence
- Dynamic risk scoring: `risk_score = Σ(violation_severity × zone_weight × time_decay)`
- Temporal heatmaps: hourly, shift-based, cumulative
- Discrepancy-driven alerts: CV detection vs. construction schedule mismatch
- Adaptive trust calibration: dynamic confidence weighting CV vs. schedule data
- Scaffolding-aware zone flagging
- Read `references/risk-scoring.md` for full algorithm

### 2.5 GeoAI Integration
- LLM-powered spatial report generation (Tamil + English, BOCW compliance)
- "Jovi" GeoAI voice narrator: TTS + avatar + live Q&A for 2nd review
- CV ensemble → spatial event pipeline
- Schedule vs. actual worker location mismatch detection
- Predictive risk: time-of-day and zone-density patterns

### 2.6 Coordinate Reference Systems & Projections
- WGS84 (EPSG:4326) ↔ UTM Zone 44N (EPSG:32644) transforms
- `ST_Transform(geom, 32644)` for metre-accurate distance queries
- Site-local coordinate systems (origin-relative metres)
- GPS anchor point acquisition strategy for Thanjavur site

### 2.7 Remote Sensing & Raster Analysis
- Aerial/drone orthomosaic integration
- NDVI, NDWI for site vegetation/drainage context
- Rasterio, GDAL, raster2pgsql workflows
- Satellite imagery sourcing: Sentinel-2, LISS-IV (ISRO Bhuvan)

### 2.8 Cartographic Design & Output
- NBC 2016 compliant site safety map layouts
- Print-quality PDF export from QGIS
- Web map publishing: Leaflet.js, OpenLayers, QGIS2Web
- Interactive dashboards: Folium, Plotly Mapbox, Kepler.gl
- Story maps for stakeholder communication

---

## 3. BuildSight Phase Mapping

| Phase | GIS Role | Status |
|-------|----------|--------|
| Phase 1 | None (CV comparative study) | ✅ Complete |
| Phase 2 | PostGIS schema, 7 shapefiles, 16-zone grid, homography design | 🔄 Active |
| Phase 3 | Live CCTV → spatial event pipeline, alert system, dashboard | 🔜 Next |
| Phase 4 | 4D BIM integration, schedule-spatial sync | 🔜 Future |
| Phase 5 | ByteTrack → spatial track persistence, trajectory analysis | 🔜 Future |

**Current active work (Phase 2):**
1. Receive Maran Constructions site plan → adjust zone boundaries from default 4×4
2. Build Module 3: Homography Pipeline (pixel → site metres → zone_id)
3. Connect live ensemble detections to PostGIS in real-time
4. Validate zone auto-assignment with smoke test on 10 detection frames

---

## 4. Standard Workflows

### Workflow A: New Site Plan → Zone Grid Setup
```
1. Read site_plan.pdf / uploaded DXF/DWG
2. Georeference to UTM 44N (4 GPS anchor points)
3. Digitize zone polygons in QGIS (match actual construction zones)
4. Export as zone_boundaries.shp → import to PostGIS zones table
5. Run: UPDATE zones SET geom = ST_SetSRID(geom, 32644)
6. Validate: SELECT zone_id, ST_Area(geom) FROM zones ORDER BY zone_id
```

### Workflow B: CV Detection → Spatial Event
```
Detection JSON (bbox, class, confidence, camera_id, timestamp)
    → homography_transform(pixel_xy) → site_xy_metres
    → ST_SetSRID(ST_MakePoint(x, y), 32644)
    → spatial_join → zone_id (auto-assigned by PostGIS trigger)
    → INSERT INTO detection_events
    → risk_score UPDATE for affected zone
    → alert_check() → if threshold crossed → generate_alert()
```

### Workflow C: Risk Heatmap Generation
```python
# See references/risk-scoring.md for full implementation
SELECT z.zone_id, z.geom,
       SUM(de.violation_count * sv.severity_weight
           * EXP(-0.1 * EXTRACT(EPOCH FROM (NOW() - de.timestamp))/3600)) AS risk_score
FROM zones z
JOIN detection_events de ON ST_Within(de.location, z.geom)
JOIN severity_values sv ON de.violation_class = sv.class
WHERE de.timestamp > NOW() - INTERVAL '8 hours'
GROUP BY z.zone_id, z.geom;
```

### Workflow D: Schedule Mismatch Alert
```
1. Load construction schedule (Phase/Floor/Activity/Expected workers)
2. Query actual worker count per zone from detection_events (last 15 min)
3. Compute delta: expected_count - actual_count
4. If |delta| > threshold AND activity = HIGH_RISK → CRITICAL alert
5. Log to engineer_feedback table for adaptive trust calibration
```

---

## 5. Production Code Patterns

### PostGIS Connection (pooled)
```python
# See references/postgis-patterns.md → Section: Connection Pool
from psycopg2 import pool
import os

connection_pool = pool.ThreadedConnectionPool(
    minconn=2, maxconn=10,
    host=os.getenv('PG_HOST', 'localhost'),
    database=os.getenv('PG_DB', 'buildsight'),
    user=os.getenv('PG_USER', 'joseva'),
    password=os.getenv('PG_PASS'),
    port=5432
)
```

### Homography Transform (core)
```python
# See references/homography-pipeline.md → Section: Core Transform
import cv2, numpy as np

def pixel_to_site_coords(pixel_xy: tuple, H: np.ndarray) -> tuple:
    """Transform pixel (u,v) → site plan (x,y) metres using homography matrix H."""
    pt = np.array([[pixel_xy]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(pt, H)
    return float(transformed[0][0][0]), float(transformed[0][0][1])
```

### Zone Assignment (PostGIS)
```sql
-- Auto-assign zone_id to detection event
CREATE OR REPLACE FUNCTION assign_zone(detection_geom GEOMETRY)
RETURNS INTEGER AS $$
    SELECT zone_id FROM zones
    WHERE ST_Within(detection_geom, geom)
    LIMIT 1;
$$ LANGUAGE SQL;
```

---

## 6. Standards Compliance Checklist

| Standard | GIS Implication |
|----------|-----------------|
| NBC 2016 | Safety zone demarcation on site maps; clearance distances |
| BOCW 1996 | Worker-in-hazard-zone tracking; 15m exclusion from heavy machinery |
| IS 4130 | Scaffold zone flagging (>3.5m height zones) |
| IS 3521 | Personal protective equipment detection mapped to zones |
| IGBC AP | Site waste, energy zones for green building certification |

---

## 7. Common Pitfalls & Fixes

| Pitfall | Fix |
|---------|-----|
| SRID mismatch between detections and zones | Always `ST_SetSRID(geom, 32644)` on insert; validate with `ST_SRID()` |
| Homography drift over time | Re-calibrate anchors every 2 weeks or after camera adjustment |
| Worker detections outside all zones | Add `unassigned_zone` catchall polygon covering full site extent |
| Alert fatigue from per-detection notifications | Aggregate: 1 alert per zone per 5-min window, 3-tier priority |
| CV bbox centre vs. actual worker position | Use torso-centre (lower 60% of bbox) for spatial assignment |
| Low-light (S3) false spatial events | Apply confidence gate 0.42+ before spatial insert in night shift |
| PostGIS slow on large detection tables | Partition `detection_events` by day; add GIST index on `location` |

---

## 8. Reference Files

| File | Contents | When to Read |
|------|----------|--------------|
| `references/buildsight-gis-context.md` | Full system spec, schema, delivered files | Start of any Phase 2/3 task |
| `references/postgis-patterns.md` | Production SQL, triggers, views, connection pool | DB schema / query tasks |
| `references/homography-pipeline.md` | Module 3 full implementation (Python + calibration) | Camera→map integration |
| `references/risk-scoring.md` | Risk algorithm, heatmap SQL, adaptive trust calibration | Alert / heatmap tasks |
| `references/qgis-workflows.md` | Layer setup, symbology, PyQGIS scripts, print layouts | QGIS / map output tasks |
| `references/standards-compliance.md` | NBC/BOCW/IS clause-by-clause GIS mapping | Compliance & reporting |
| `references/general-gis-reference.md` | CRS guide, spatial analysis patterns, remote sensing | General GIS questions |
| `scripts/zone_smoke_test.py` | Validate zone assignment with 10 sample detections | After Module 3 setup |
| `scripts/heatmap_export.py` | Export current risk heatmap to GeoJSON + PNG | Dashboard / review prep |
| `scripts/homography_calibrate.py` | Interactive anchor selection + H matrix computation | Camera calibration |
