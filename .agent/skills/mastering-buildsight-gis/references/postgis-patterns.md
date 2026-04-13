# PostGIS Production Patterns
## BuildSight GIS Expert Skill Reference

---

## Connection Pool (Production)

```python
import os
import psycopg2
from psycopg2 import pool
from contextlib import contextmanager
import logging

logger = logging.getLogger('buildsight.gis')

_pool: pool.ThreadedConnectionPool = None

def init_pool():
    global _pool
    _pool = pool.ThreadedConnectionPool(
        minconn=2,
        maxconn=10,
        host=os.environ['PG_HOST'],
        database=os.environ['PG_DB'],
        user=os.environ['PG_USER'],
        password=os.environ['PG_PASS'],
        port=int(os.getenv('PG_PORT', 5432)),
        connect_timeout=10
    )
    logger.info("PostGIS connection pool initialized")

@contextmanager
def get_conn():
    conn = _pool.getconn()
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"DB error: {e}")
        raise
    finally:
        _pool.putconn(conn)
```

---

## Core Schema DDL

```sql
-- Enable PostGIS
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS postgis_topology;

-- Zones table
CREATE TABLE zones (
    zone_id         SERIAL PRIMARY KEY,
    zone_name       TEXT NOT NULL,
    floor_level     INTEGER DEFAULT 0,
    activity_type   TEXT,
    risk_baseline   FLOAT CHECK (risk_baseline BETWEEN 0 AND 1) DEFAULT 0.3,
    max_workers     INTEGER DEFAULT 20,
    geom            GEOMETRY(POLYGON, 32644) NOT NULL
);
CREATE INDEX idx_zones_geom ON zones USING GIST(geom);

-- Detection events
CREATE TABLE detection_events (
    id              BIGSERIAL PRIMARY KEY,
    frame_id        BIGINT,
    camera_id       TEXT NOT NULL,
    class_id        INTEGER NOT NULL,    -- 0=helmet,1=vest,2=boots,3=worker
    class_name      TEXT,
    confidence      FLOAT CHECK (confidence BETWEEN 0 AND 1),
    has_helmet      BOOLEAN DEFAULT FALSE,
    has_vest        BOOLEAN DEFAULT FALSE,
    has_boots       BOOLEAN DEFAULT FALSE,
    location        GEOMETRY(POINT, 32644),
    zone_id         INTEGER REFERENCES zones(zone_id),
    scene_condition TEXT DEFAULT 'S1',   -- S1/S2/S3/S4
    timestamp       TIMESTAMPTZ NOT NULL DEFAULT NOW()
) PARTITION BY RANGE (timestamp);

-- Partition by day for performance
CREATE TABLE detection_events_2025_11
    PARTITION OF detection_events
    FOR VALUES FROM ('2025-11-01') TO ('2025-12-01');

CREATE INDEX idx_det_events_location ON detection_events USING GIST(location);
CREATE INDEX idx_det_events_zone_ts  ON detection_events(zone_id, timestamp DESC);
CREATE INDEX idx_det_events_camera   ON detection_events(camera_id, timestamp DESC);

-- Alerts
CREATE TABLE alert_log (
    id              BIGSERIAL PRIMARY KEY,
    alert_tier      TEXT CHECK (alert_tier IN ('CRITICAL','HIGH','PERIODIC')),
    zone_id         INTEGER REFERENCES zones(zone_id),
    trigger_reason  TEXT,
    worker_count    INTEGER,
    violation_count INTEGER,
    resolved        BOOLEAN DEFAULT FALSE,
    resolved_by     TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    resolved_at     TIMESTAMPTZ
);

-- Risk scores (updated every 5 min)
CREATE TABLE risk_scores (
    zone_id         INTEGER PRIMARY KEY REFERENCES zones(zone_id),
    current_score   FLOAT CHECK (current_score BETWEEN 0 AND 1),
    last_updated    TIMESTAMPTZ DEFAULT NOW(),
    contributing_violations INTEGER DEFAULT 0
);

-- Engineer feedback (adaptive trust calibration)
CREATE TABLE engineer_feedback (
    id              BIGSERIAL PRIMARY KEY,
    detection_id    BIGINT REFERENCES detection_events(id),
    feedback_type   TEXT CHECK (feedback_type IN
                    ('cv_false_positive','cv_false_negative',
                     'schedule_error','zone_error','correct')),
    corrected_zone  INTEGER REFERENCES zones(zone_id),
    notes           TEXT,
    submitted_by    TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
```

---

## Real-Time Views

```sql
-- Live risk scores joined to zone geometries (for QGIS live layer)
CREATE OR REPLACE VIEW v_current_risk AS
SELECT
    z.zone_id,
    z.zone_name,
    z.geom,
    COALESCE(rs.current_score, z.risk_baseline) AS risk_score,
    rs.contributing_violations,
    rs.last_updated,
    CASE
        WHEN COALESCE(rs.current_score, z.risk_baseline) >= 0.8 THEN 'CRITICAL'
        WHEN COALESCE(rs.current_score, z.risk_baseline) >= 0.5 THEN 'HIGH'
        ELSE 'NORMAL'
    END AS risk_level
FROM zones z
LEFT JOIN risk_scores rs ON z.zone_id = rs.zone_id;

-- Active PPE violations (last 30 min)
CREATE OR REPLACE VIEW v_active_violations AS
SELECT
    de.id,
    de.zone_id,
    z.zone_name,
    de.camera_id,
    de.class_name,
    de.confidence,
    NOT de.has_helmet AS missing_helmet,
    NOT de.has_vest   AS missing_vest,
    NOT de.has_boots  AS missing_boots,
    de.location,
    de.timestamp
FROM detection_events de
JOIN zones z ON de.zone_id = z.zone_id
WHERE de.class_id = 3  -- worker
  AND de.timestamp > NOW() - INTERVAL '30 minutes'
  AND (NOT de.has_helmet OR NOT de.has_vest OR NOT de.has_boots);

-- Schedule mismatch (requires construction_schedule table)
CREATE OR REPLACE VIEW v_schedule_mismatch AS
SELECT
    z.zone_id,
    z.zone_name,
    cs.expected_workers,
    COALESCE(actual.worker_count, 0) AS actual_workers,
    ABS(cs.expected_workers - COALESCE(actual.worker_count, 0)) AS delta,
    cs.activity_type,
    cs.shift_start
FROM zones z
JOIN construction_schedule cs
    ON z.zone_id = cs.zone_id
    AND cs.shift_start <= NOW()
    AND cs.shift_end > NOW()
LEFT JOIN (
    SELECT zone_id, COUNT(*) AS worker_count
    FROM detection_events
    WHERE class_id = 3
      AND timestamp > NOW() - INTERVAL '15 minutes'
    GROUP BY zone_id
) actual ON z.zone_id = actual.zone_id
WHERE ABS(cs.expected_workers - COALESCE(actual.worker_count, 0)) > 2;
```

---

## Spatial Functions

```sql
-- Risk score computation (time-decay weighted)
CREATE OR REPLACE FUNCTION compute_risk(p_zone_id INTEGER)
RETURNS FLOAT AS $$
DECLARE
    v_score FLOAT;
    v_baseline FLOAT;
BEGIN
    SELECT risk_baseline INTO v_baseline FROM zones WHERE zone_id = p_zone_id;

    SELECT
        v_baseline + LEAST(0.9, SUM(
            CASE
                WHEN NOT has_helmet THEN 0.4
                WHEN NOT has_vest   THEN 0.25
                WHEN NOT has_boots  THEN 0.15
                ELSE 0
            END
            * confidence
            * EXP(-0.1 * EXTRACT(EPOCH FROM (NOW() - timestamp)) / 3600.0)
        ))
    INTO v_score
    FROM detection_events
    WHERE zone_id = p_zone_id
      AND class_id = 3
      AND timestamp > NOW() - INTERVAL '8 hours';

    UPDATE risk_scores
    SET current_score = COALESCE(v_score, v_baseline),
        last_updated = NOW(),
        contributing_violations = (
            SELECT COUNT(*) FROM detection_events
            WHERE zone_id = p_zone_id
              AND (NOT has_helmet OR NOT has_vest OR NOT has_boots)
              AND timestamp > NOW() - INTERVAL '8 hours'
        )
    WHERE zone_id = p_zone_id;

    IF NOT FOUND THEN
        INSERT INTO risk_scores (zone_id, current_score, last_updated)
        VALUES (p_zone_id, COALESCE(v_score, v_baseline), NOW());
    END IF;

    RETURN COALESCE(v_score, v_baseline);
END;
$$ LANGUAGE plpgsql;

-- BOCW exclusion zone check (15m buffer around machinery)
CREATE OR REPLACE FUNCTION check_exclusion(
    detection_geom GEOMETRY,
    exclusion_radius_m FLOAT DEFAULT 15.0
) RETURNS BOOLEAN AS $$
    SELECT EXISTS (
        SELECT 1 FROM zones
        WHERE activity_type = 'heavy_machinery'
          AND ST_DWithin(detection_geom, geom, exclusion_radius_m)
    );
$$ LANGUAGE SQL;
```

---

## Common Query Patterns

```sql
-- Zone worker density (last 15 min)
SELECT z.zone_id, z.zone_name, COUNT(de.id) AS workers,
       z.max_workers,
       ROUND(COUNT(de.id)::NUMERIC / z.max_workers * 100, 1) AS occupancy_pct
FROM zones z
LEFT JOIN detection_events de
    ON de.zone_id = z.zone_id
    AND de.class_id = 3
    AND de.timestamp > NOW() - INTERVAL '15 minutes'
GROUP BY z.zone_id, z.zone_name, z.max_workers
ORDER BY occupancy_pct DESC;

-- PPE compliance rate per zone (today)
SELECT
    zone_id,
    COUNT(*) AS total_workers,
    ROUND(100.0 * SUM(CASE WHEN has_helmet AND has_vest AND has_boots THEN 1 ELSE 0 END) / COUNT(*), 1)
        AS compliance_pct
FROM detection_events
WHERE class_id = 3
  AND timestamp > CURRENT_DATE
GROUP BY zone_id;

-- Export heatmap data for GeoJSON
SELECT json_build_object(
    'type', 'FeatureCollection',
    'features', json_agg(
        json_build_object(
            'type', 'Feature',
            'geometry', ST_AsGeoJSON(geom)::json,
            'properties', json_build_object(
                'zone_id', zone_id,
                'risk_score', risk_score,
                'risk_level', risk_level
            )
        )
    )
) FROM v_current_risk;
```

---

## Performance Tuning

```sql
-- Analyze query performance
EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT)
SELECT * FROM detection_events
WHERE zone_id = 5 AND timestamp > NOW() - INTERVAL '1 hour';

-- Vacuum and analyze after large ingestion
VACUUM ANALYZE detection_events;

-- Check index usage
SELECT indexname, idx_scan, idx_tup_read
FROM pg_stat_user_indexes
WHERE tablename = 'detection_events';
```
