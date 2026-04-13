# Risk Scoring, Heatmaps & Adaptive Trust Calibration
## BuildSight GIS Expert Skill Reference

---

## Risk Score Formula

```
risk_score(zone, t) = baseline(zone)
    + Σ_i [ severity(class_i) × confidence_i × time_decay(t - t_i) ]

Where:
  severity(missing_helmet) = 0.40
  severity(missing_vest)   = 0.25
  severity(missing_boots)  = 0.15
  time_decay(Δt_hours)     = exp(-0.1 × Δt)   [half-life ≈ 7 hours]
  
  score is capped at 1.0
  baseline from zones.risk_baseline (NBC pre-assessment)
```

---

## Heatmap Export Script (scripts/heatmap_export.py)

```python
#!/usr/bin/env python3
"""
BuildSight — Risk Heatmap Export
Exports current risk state as GeoJSON + PNG for dashboard and 2nd review.
Usage: python heatmap_export.py --output /nfsshare/joseva/outputs/
"""

import json
import os
import argparse
import psycopg2
from pathlib import Path


def get_risk_geojson(conn) -> dict:
    """Fetch current risk scores from PostGIS as GeoJSON."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT json_build_object(
                'type', 'FeatureCollection',
                'features', json_agg(
                    json_build_object(
                        'type', 'Feature',
                        'geometry', ST_AsGeoJSON(geom)::json,
                        'properties', json_build_object(
                            'zone_id', zone_id,
                            'zone_name', zone_name,
                            'risk_score', ROUND(risk_score::NUMERIC, 4),
                            'risk_level', risk_level,
                            'violations', contributing_violations,
                            'last_updated', last_updated
                        )
                    )
                )
            ) FROM v_current_risk;
        """)
        row = cur.fetchone()
        return row[0] if row else {}


def get_violation_points(conn) -> list:
    """Fetch active violation locations as GeoJSON points."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT json_build_object(
                'type', 'Feature',
                'geometry', ST_AsGeoJSON(location)::json,
                'properties', json_build_object(
                    'zone_id', zone_id,
                    'camera_id', camera_id,
                    'missing_helmet', missing_helmet,
                    'missing_vest', missing_vest,
                    'missing_boots', missing_boots,
                    'confidence', confidence,
                    'timestamp', timestamp
                )
            )
            FROM v_active_violations;
        """)
        return [row[0] for row in cur.fetchall()]


def export_heatmap_png(geojson: dict, output_path: str):
    """Generate PNG heatmap using matplotlib + geopandas."""
    try:
        import geopandas as gpd
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        from shapely.geometry import shape
        import pandas as pd

        features = geojson.get('features', [])
        if not features:
            print("No risk data to plot.")
            return

        gdf = gpd.GeoDataFrame([
            {
                'geometry': shape(f['geometry']),
                **f['properties']
            }
            for f in features
        ], crs='EPSG:32644')

        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        cmap = mcolors.LinearSegmentedColormap.from_list(
            'risk', ['#00CC44', '#FFCC00', '#FF4400'], N=256
        )

        gdf.plot(
            column='risk_score',
            cmap=cmap,
            vmin=0, vmax=1,
            ax=ax,
            legend=True,
            legend_kwds={'label': 'Risk Score', 'shrink': 0.7},
            edgecolor='black',
            linewidth=0.5,
            alpha=0.8
        )

        # Zone labels
        for _, row in gdf.iterrows():
            cx, cy = row.geometry.centroid.x, row.geometry.centroid.y
            ax.annotate(
                f"Z{row['zone_id']}\n{row['risk_score']:.2f}",
                xy=(cx, cy), ha='center', va='center',
                fontsize=8, fontweight='bold', color='black'
            )

        ax.set_title('BuildSight — Zone Risk Heatmap\n(Green Build AI / NBC 2016 Compliant)',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('UTM Easting (m)', fontsize=10)
        ax.set_ylabel('UTM Northing (m)', fontsize=10)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Heatmap PNG saved: {output_path}")

    except ImportError as e:
        print(f"Missing dependency for PNG export: {e}")
        print("Install: pip install geopandas matplotlib shapely --break-system-packages")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='/nfsshare/joseva/outputs',
                        help='Output directory')
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    conn = psycopg2.connect(
        host=os.environ['PG_HOST'],
        database=os.environ['PG_DB'],
        user=os.environ['PG_USER'],
        password=os.environ['PG_PASS'],
        port=int(os.getenv('PG_PORT', 5432))
    )

    try:
        # Export zone risk GeoJSON
        risk_geojson = get_risk_geojson(conn)
        violation_points = get_violation_points(conn)

        # Merge into combined GeoJSON
        combined = {
            'type': 'FeatureCollection',
            'features': risk_geojson.get('features', []) + violation_points
        }

        json_path = output_dir / 'current_risk_heatmap.geojson'
        with open(json_path, 'w') as f:
            json.dump(combined, f, indent=2, default=str)
        print(f"GeoJSON saved: {json_path}")

        # Export PNG
        png_path = output_dir / 'current_risk_heatmap.png'
        export_heatmap_png(risk_geojson, str(png_path))

    finally:
        conn.close()


if __name__ == '__main__':
    main()
```

---

## Adaptive Trust Calibration (Full Implementation)

```python
"""
Adaptive Trust Calibration for BuildSight GeoAI.
Adjusts confidence weights between CV detections and construction schedule
based on engineer feedback over time.
"""

import os
import psycopg2
from dataclasses import dataclass, field
from typing import Literal
import logging

logger = logging.getLogger('buildsight.trust')

FeedbackType = Literal[
    'cv_false_positive', 'cv_false_negative',
    'schedule_error', 'zone_error', 'correct'
]


@dataclass
class TrustState:
    cv_weight: float = 0.70
    schedule_weight: float = 0.30
    total_feedback: int = 0
    cv_errors: int = 0
    schedule_errors: int = 0
    decay_rate: float = 0.05
    min_weight: float = 0.20
    max_weight: float = 0.90

    def update(self, feedback_type: FeedbackType) -> None:
        self.total_feedback += 1

        if feedback_type in ('cv_false_positive', 'cv_false_negative'):
            self.cv_errors += 1
            self.cv_weight = max(self.min_weight,
                                  self.cv_weight * (1 - self.decay_rate))
            self.schedule_weight = min(self.max_weight,
                                        self.schedule_weight * (1 + self.decay_rate))
        elif feedback_type == 'schedule_error':
            self.schedule_errors += 1
            self.schedule_weight = max(self.min_weight,
                                        self.schedule_weight * (1 - self.decay_rate))
            self.cv_weight = min(self.max_weight,
                                  self.cv_weight * (1 + self.decay_rate))

        # Normalize
        total = self.cv_weight + self.schedule_weight
        self.cv_weight /= total
        self.schedule_weight /= total

        logger.info(f"Trust updated: CV={self.cv_weight:.3f} Sched={self.schedule_weight:.3f}")

    def weighted_worker_count(self, cv_count: int, schedule_count: int) -> float:
        return self.cv_weight * cv_count + self.schedule_weight * schedule_count

    def summary(self) -> dict:
        return {
            'cv_weight': round(self.cv_weight, 4),
            'schedule_weight': round(self.schedule_weight, 4),
            'total_feedback': self.total_feedback,
            'cv_error_rate': round(self.cv_errors / max(1, self.total_feedback), 4),
            'schedule_error_rate': round(self.schedule_errors / max(1, self.total_feedback), 4)
        }
```

---

## Alert Threshold Configuration

```python
# config/alert_thresholds.yaml (load with PyYAML)

ALERT_CONFIG = {
    'CRITICAL': {
        'triggers': [
            {'condition': 'worker_in_exclusion_zone', 'immediate': True},
            {'condition': 'missing_helmet_height_gt_3.5m', 'immediate': True},
            {'condition': 'zone_occupancy_gt_max_workers', 'delay_sec': 0}
        ],
        'aggregation_window_sec': 60,
        'max_per_zone_per_hour': 3,
        'channels': ['sms', 'dashboard', 'alert_log']
    },
    'HIGH': {
        'triggers': [
            {'condition': 'ppe_violation_pct_gt_30', 'duration_sec': 120},
            {'condition': 'schedule_mismatch_gt_3_workers', 'duration_sec': 300}
        ],
        'aggregation_window_sec': 300,
        'channels': ['dashboard', 'alert_log']
    },
    'PERIODIC': {
        'triggers': [
            {'condition': 'risk_score_gt_0.7', 'duration_sec': 900}
        ],
        'aggregation_window_sec': 3600,
        'channels': ['alert_log', 'daily_report']
    }
}
```
