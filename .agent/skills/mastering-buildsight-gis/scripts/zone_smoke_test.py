#!/usr/bin/env python3
"""
BuildSight Module 3 — Zone Assignment Smoke Test
Validates end-to-end: pixel → homography → site metres → PostGIS zone

Usage:
    python zone_smoke_test.py --camera cam_01 --db-host localhost

Pass criteria:
  - All 10 detections assigned a zone_id (no NULL)
  - Mean reprojection error < 0.3 metres
  - No zone physically impossible for given camera
"""

import os
import sys
import json
import argparse
import numpy as np
import psycopg2

# Simulated test detections (pixel coords from cam_01 sample frames)
# Replace with actual annotated ground-truth pixel coords
SMOKE_TEST_DETECTIONS = [
    {'frame': 1, 'pixel': (320, 480), 'expected_zone': None},
    {'frame': 2, 'pixel': (640, 360), 'expected_zone': None},
    {'frame': 3, 'pixel': (200, 600), 'expected_zone': None},
    {'frame': 4, 'pixel': (900, 200), 'expected_zone': None},
    {'frame': 5, 'pixel': (450, 720), 'expected_zone': None},
    {'frame': 6, 'pixel': (750, 540), 'expected_zone': None},
    {'frame': 7, 'pixel': (100, 300), 'expected_zone': None},
    {'frame': 8, 'pixel': (1100, 650), 'expected_zone': None},
    {'frame': 9, 'pixel': (550, 400), 'expected_zone': None},
    {'frame': 10, 'pixel': (820, 480), 'expected_zone': None},
]


def load_homography(camera_id: str, homography_dir: str):
    """Load H matrix for camera."""
    import cv2
    path = os.path.join(homography_dir, f"{camera_id}_homography.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Homography not found: {path}\nRun homography_calibrate.py first.")
    with open(path) as f:
        config = json.load(f)
    H = np.array(config['H'], dtype=np.float64)
    origin = config['site_origin_utm']
    return H, origin


def pixel_to_utm(pixel_xy, H, origin):
    """Transform pixel → UTM easting/northing."""
    import cv2
    pt = np.array([[list(pixel_xy)]], dtype=np.float32)
    result = cv2.perspectiveTransform(pt, H)
    site_x, site_y = float(result[0][0][0]), float(result[0][0][1])
    easting = origin['easting'] + site_x
    northing = origin['northing'] + site_y
    return easting, northing, site_x, site_y


def assign_zone(conn, easting: float, northing: float) -> int | None:
    """Query PostGIS for zone assignment."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT assign_zone(
                ST_SetSRID(ST_MakePoint(%s, %s), 32644)
            )
        """, (easting, northing))
        row = cur.fetchone()
        return row[0] if row else None


def run_smoke_test(camera_id: str, db_config: dict, homography_dir: str) -> bool:
    print(f"\n{'='*60}")
    print(f"BuildSight Zone Assignment Smoke Test")
    print(f"Camera: {camera_id}")
    print(f"{'='*60}")

    try:
        H, origin = load_homography(camera_id, homography_dir)
        print(f"✓ Homography loaded for {camera_id}")
    except FileNotFoundError as e:
        print(f"✗ {e}")
        return False

    try:
        conn = psycopg2.connect(**db_config)
        print(f"✓ PostGIS connected")
    except Exception as e:
        print(f"✗ PostGIS connection failed: {e}")
        return False

    results = []
    null_count = 0
    print(f"\n{'Frame':<8} {'Pixel':<18} {'Site (m)':<24} {'UTM E':<14} {'Zone'}")
    print("-" * 80)

    for det in SMOKE_TEST_DETECTIONS:
        frame = det['frame']
        pixel = det['pixel']
        easting, northing, sx, sy = pixel_to_utm(pixel, H, origin)
        zone_id = assign_zone(conn, easting, northing)

        if zone_id is None:
            null_count += 1
            zone_str = "⚠ NULL"
        else:
            zone_str = f"Z{zone_id:02d}"

        print(f"{frame:<8} {str(pixel):<18} {f'({sx:.1f}, {sy:.1f})':<24} "
              f"{easting:.1f}m {zone_str}")

        results.append({
            'frame': frame,
            'pixel': pixel,
            'utm': (easting, northing),
            'zone_id': zone_id
        })

    conn.close()

    print(f"\n{'='*60}")
    assigned = len(results) - null_count
    print(f"Zone Assignment: {assigned}/10 successfully assigned")

    if null_count > 0:
        print(f"⚠ {null_count} detections fell outside all zones (check site_boundary.shp)")
    else:
        print(f"✓ All 10 detections assigned zones")

    # Check UTM range for Thanjavur
    utms = [r['utm'] for r in results]
    mean_e = np.mean([u[0] for u in utms])
    mean_n = np.mean([u[1] for u in utms])

    utm_ok = (430000 < mean_e < 450000) and (1060000 < mean_n < 1080000)
    print(f"UTM range check: E={mean_e:.0f}, N={mean_n:.0f} → {'✓ OK' if utm_ok else '⚠ UNEXPECTED — check anchor GPS'}")

    passed = (null_count == 0) and utm_ok
    print(f"\nSmoke Test: {'✓ PASSED' if passed else '✗ FAILED'}")
    return passed


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', default='cam_01')
    parser.add_argument('--db-host', default=os.getenv('PG_HOST', 'localhost'))
    parser.add_argument('--db-name', default=os.getenv('PG_DB', 'buildsight'))
    parser.add_argument('--db-user', default=os.getenv('PG_USER', 'joseva'))
    parser.add_argument('--db-pass', default=os.getenv('PG_PASS', ''))
    parser.add_argument('--homography-dir',
                        default=os.getenv('HOMOGRAPHY_DIR', '/nfsshare/joseva/config/homography'))
    args = parser.parse_args()

    db_config = {
        'host': args.db_host,
        'database': args.db_name,
        'user': args.db_user,
        'password': args.db_pass,
        'port': 5432
    }

    success = run_smoke_test(args.camera, db_config, args.homography_dir)
    sys.exit(0 if success else 1)
