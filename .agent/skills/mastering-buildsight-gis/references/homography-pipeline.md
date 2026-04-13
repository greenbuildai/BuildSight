# Module 3: Homography Pipeline — Pixel → Site Plan → Zone
## BuildSight GIS Expert Skill Reference

---

## Overview

Module 3 bridges the CV detection layer (pixel space) with the GIS spatial layer (site plan metres → UTM zone assignment).  
It is a 3-stage pipeline:

```
Pixel (u, v)
    ↓ [H matrix — perspectiveTransform]
Site Plan Metres (x, y)  ← local origin at site SW corner
    ↓ [Affine transform from anchor GPS points]
UTM Zone 44N (easting, northing)  ← EPSG:32644
    ↓ [PostGIS ST_Within]
zone_id
```

---

## Stage 1: Homography Calibration

### Anchor Point Strategy (4 points per camera)

Select anchors that are:
- Coplanar with the working floor plane (not elevated objects)
- Spread across the camera frustum (not clustered)
- Permanently identifiable: column bases, painted marks, kerb corners
- GPS-measurable: accessible with GNSS receiver or total station

**Minimum anchor layout:**
```
[Anchor A]----[Anchor B]
    |               |
[Anchor C]----[Anchor D]
```

### Calibration Script (scripts/homography_calibrate.py)

```python
#!/usr/bin/env python3
"""
BuildSight Module 3 — Homography Calibration Tool
Usage: python homography_calibrate.py --camera cam_01 --image frame_sample.jpg
"""

import cv2
import numpy as np
import json
import argparse
import os
from pathlib import Path

class HomographyCalibrator:
    def __init__(self, camera_id: str, output_dir: str):
        self.camera_id = camera_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pixel_points = []
        self.site_points = []   # In site-plan metres (local origin)
        self.anchor_labels = ['A', 'B', 'C', 'D']
        self.H = None

    def collect_anchors_interactive(self, image_path: str):
        """Interactive anchor selection via mouse clicks on sample frame."""
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        clone = img.copy()
        self.pixel_points = []
        idx = [0]

        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and idx[0] < 4:
                label = self.anchor_labels[idx[0]]
                self.pixel_points.append([x, y])
                cv2.circle(clone, (x, y), 6, (0, 255, 0), -1)
                cv2.putText(clone, label, (x+8, y-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow("Select 4 Anchors (A→B→C→D)", clone)
                print(f"Anchor {label}: pixel ({x}, {y})")
                idx[0] += 1
                if idx[0] == 4:
                    print("All 4 anchors selected. Press any key to continue.")

        cv2.imshow("Select 4 Anchors (A→B→C→D)", clone)
        cv2.setMouseCallback("Select 4 Anchors (A→B→C→D)", click_event)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        if len(self.pixel_points) < 4:
            raise ValueError("Need exactly 4 anchor points.")

    def input_site_coordinates(self):
        """Prompt for site-plan coordinates (metres from SW origin) for each anchor."""
        print("\nEnter site-plan coordinates in METRES (from SW corner origin):")
        self.site_points = []
        for label in self.anchor_labels:
            x = float(input(f"  Anchor {label} — X (East, metres): "))
            y = float(input(f"  Anchor {label} — Y (North, metres): "))
            self.site_points.append([x, y])

    def compute_homography(self) -> np.ndarray:
        """Compute H matrix from pixel → site-plan metres."""
        src = np.array(self.pixel_points, dtype=np.float32)
        dst = np.array(self.site_points, dtype=np.float32)
        self.H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if self.H is None:
            raise RuntimeError("Homography computation failed. Check anchor quality.")
        reproj_err = self._reprojection_error(src, dst)
        print(f"\nReprojection error: {reproj_err:.4f} metres (target: <0.3m)")
        return self.H

    def _reprojection_error(self, src: np.ndarray, dst: np.ndarray) -> float:
        projected = cv2.perspectiveTransform(src.reshape(-1, 1, 2), self.H)
        projected = projected.reshape(-1, 2)
        return float(np.mean(np.linalg.norm(projected - dst, axis=1)))

    def save(self, site_origin_utm: dict):
        """
        Save H matrix + metadata to JSON.
        site_origin_utm: {'easting': float, 'northing': float, 'srid': 32644}
        """
        config = {
            'camera_id': self.camera_id,
            'H': self.H.tolist(),
            'pixel_anchors': self.pixel_points,
            'site_anchors': self.site_points,
            'site_origin_utm': site_origin_utm,
            'srid': 32644,
            'calibration_notes': 'Anchors: column base NW, NE, SE, SW'
        }
        out_path = self.output_dir / f"{self.camera_id}_homography.json"
        with open(out_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"\nHomography saved: {out_path}")
        return out_path


def load_homography(camera_id: str, homography_dir: str) -> tuple:
    """Load H matrix and site origin for a given camera. Returns (H, origin_utm)."""
    path = Path(homography_dir) / f"{camera_id}_homography.json"
    with open(path) as f:
        config = json.load(f)
    H = np.array(config['H'], dtype=np.float64)
    origin = config['site_origin_utm']
    return H, origin


def pixel_to_site_metres(pixel_xy: tuple, H: np.ndarray) -> tuple:
    """Transform pixel (u,v) → site-plan (x,y) in metres."""
    pt = np.array([[list(pixel_xy)]], dtype=np.float32)
    result = cv2.perspectiveTransform(pt, H)
    x, y = float(result[0][0][0]), float(result[0][0][1])
    return x, y


def site_metres_to_utm(site_xy: tuple, origin_utm: dict) -> tuple:
    """Convert site-local metres to UTM easting/northing."""
    easting = origin_utm['easting'] + site_xy[0]
    northing = origin_utm['northing'] + site_xy[1]
    return easting, northing


def pixel_to_postgis_point(pixel_xy: tuple, H: np.ndarray,
                            origin_utm: dict) -> str:
    """Full pipeline: pixel → PostGIS ST_GeomFromText POINT string."""
    site_xy = pixel_to_site_metres(pixel_xy, H)
    easting, northing = site_metres_to_utm(site_xy, origin_utm)
    return f"ST_SetSRID(ST_MakePoint({easting:.4f}, {northing:.4f}), 32644)"


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BuildSight Homography Calibration')
    parser.add_argument('--camera', required=True, help='Camera ID (e.g. cam_01)')
    parser.add_argument('--image', required=True, help='Sample frame image path')
    parser.add_argument('--output', default=os.getenv('HOMOGRAPHY_DIR', '/tmp/homography'),
                        help='Output directory for JSON config')
    parser.add_argument('--origin-easting', type=float, required=True,
                        help='Site SW corner UTM easting (EPSG:32644)')
    parser.add_argument('--origin-northing', type=float, required=True,
                        help='Site SW corner UTM northing (EPSG:32644)')
    args = parser.parse_args()

    cal = HomographyCalibrator(args.camera, args.output)
    cal.collect_anchors_interactive(args.image)
    cal.input_site_coordinates()
    cal.compute_homography()
    origin = {
        'easting': args.origin_easting,
        'northing': args.origin_northing,
        'srid': 32644
    }
    cal.save(origin)
```

---

## Stage 2: Real-Time Detection → Site Coordinates

```python
# Integrates with ensemble output stream
# Called from Phase 3 real-time orchestrator

import os
from homography_calibrate import load_homography, pixel_to_postgis_point

HOMOGRAPHY_DIR = os.getenv('HOMOGRAPHY_DIR', '/nfsshare/joseva/config/homography')

# Cache H matrices per camera (load once, reuse)
_H_CACHE = {}

def get_homography(camera_id: str):
    if camera_id not in _H_CACHE:
        H, origin = load_homography(camera_id, HOMOGRAPHY_DIR)
        _H_CACHE[camera_id] = (H, origin)
    return _H_CACHE[camera_id]


def detection_to_spatial(detection: dict, camera_id: str) -> dict:
    """
    Convert a single detection dict to a spatially-enriched dict.
    Uses torso-centre (y offset 70% down bbox) for accurate ground-plane mapping.
    """
    H, origin = get_homography(camera_id)
    x, y, w, h = detection['bbox']
    cx = x + w / 2
    cy = y + h * 0.70   # torso-centre heuristic

    site_xy = pixel_to_site_metres((cx, cy), H)
    easting, northing = site_metres_to_utm(site_xy, origin)

    return {
        **detection,
        'site_x': site_xy[0],
        'site_y': site_xy[1],
        'utm_easting': easting,
        'utm_northing': northing,
        'postgis_point': f"ST_SetSRID(ST_MakePoint({easting:.4f}, {northing:.4f}), 32644)"
    }
```

---

## Stage 3: Zone Assignment (PostGIS)

```sql
-- Fast zone lookup (uses GIST spatial index)
-- Expects detection location in EPSG:32644

CREATE INDEX IF NOT EXISTS idx_zones_geom ON zones USING GIST(geom);

-- Zone assignment function (already in schema)
CREATE OR REPLACE FUNCTION assign_zone(detection_geom GEOMETRY)
RETURNS INTEGER AS $$
DECLARE
    result_zone_id INTEGER;
BEGIN
    SELECT zone_id INTO result_zone_id
    FROM zones
    WHERE ST_Within(detection_geom, geom)
      AND ST_SRID(detection_geom) = 32644
    LIMIT 1;

    -- Fallback: nearest zone if point is outside all zones
    IF result_zone_id IS NULL THEN
        SELECT zone_id INTO result_zone_id
        FROM zones
        ORDER BY ST_Distance(detection_geom, geom) ASC
        LIMIT 1;
    END IF;

    RETURN result_zone_id;
END;
$$ LANGUAGE plpgsql;
```

---

## Multi-Camera Calibration Plan (Phase 3)

| Camera | Position | Coverage | Priority |
|--------|----------|----------|----------|
| cam_01 | North entry | Zones Z01-Z04 entry area | P1 |
| cam_02 | Scaffold East | Zones Z03-Z07 scaffold | P1 |
| cam_03 | Ground floor central | Zones Z05-Z12 | P2 |
| cam_04 | South exit | Zones Z13-Z16 | P2 |

**Each camera needs separate H matrix.** Overlap zones (Z03, Z07) use multi-camera fusion:
```
final_location = weighted_average(cam_01_location, cam_02_location,
                                   weights=[conf_01, conf_02])
```

---

## Validation: Smoke Test

Run `scripts/zone_smoke_test.py` after calibration to verify end-to-end mapping.

**Pass criteria:**
- All 10 test detections assigned a zone_id (no NULL)
- Mean reprojection error < 0.3 metres
- No detection mapped to a physically impossible zone (e.g., cam_01 → Z16)
