-- ============================================================
-- BuildSight GeoAI — Module 4: PostGIS Schema
-- Maran Constructions, Thanjavur | Green Build AI | IGBC AP
-- ============================================================
-- Setup:
--   1. Install PostgreSQL 15+ with PostGIS 3.x extension
--   2. createdb buildsight
--   3. psql -d buildsight -f module4_postgis_schema.sql
-- ============================================================

-- Enable PostGIS
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS postgis_topology;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================
-- TABLE 1: cameras
-- One row per physical camera installed on site
-- ============================================================
CREATE TABLE cameras (
    camera_id       TEXT PRIMARY KEY,              -- 'CAM-01'
    site_id         TEXT NOT NULL DEFAULT 'MARAN-TJ-001',
    description     TEXT,
    location_note   TEXT,                          -- 'Neighbour bldg south, 25ft'
    mount_height_m  FLOAT,                         -- 7.62
    mount_lat       FLOAT,                         -- 10.81651098
    mount_lon       FLOAT,                         -- 78.66891976
    rtsp_url        TEXT,
    h_matrix        FLOAT[3][3],                   -- 3x3 homography stored in DB
    h_calibrated_at TIMESTAMPTZ,
    h_error_m       FLOAT,                         -- reprojection error
    is_active       BOOLEAN DEFAULT TRUE,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================
-- TABLE 2: risk_zones
-- GeoJSON zone polygons stored as PostGIS geometry
-- ============================================================
CREATE TABLE risk_zones (
    zone_id         SERIAL PRIMARY KEY,
    zone_name       TEXT NOT NULL UNIQUE,          -- 'high_risk_scaffolding'
    risk_level      TEXT NOT NULL,                 -- HIGH/MODERATE/LOW/CRITICAL
    alert_level     TEXT,                          -- CRITICAL/WARNING/ADVISORY
    alert_enabled   BOOLEAN DEFAULT TRUE,
    description     TEXT,
    z_base_m        FLOAT DEFAULT 0,               -- bottom elevation
    z_top_m         FLOAT,                         -- top elevation
    geom            GEOMETRY(POLYGONZ, 4326),      -- 3D polygon in WGS84
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================
-- TABLE 3: detections
-- Every detected worker per frame — primary fact table
-- ============================================================
CREATE TABLE detections (
    detection_id    UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    camera_id       TEXT REFERENCES cameras(camera_id),
    timestamp       TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- YOLO output
    class_name      TEXT NOT NULL,                 -- 'worker'
    confidence      FLOAT CHECK (confidence BETWEEN 0 AND 1),
    bbox_x1         INT, bbox_y1 INT,
    bbox_x2         INT, bbox_y2 INT,
    bbox_height_px  INT,

    -- Homography output
    world_x         FLOAT,                         -- meters from SW corner
    world_y         FLOAT,
    world_z         FLOAT DEFAULT 0,               -- estimated height (m)

    -- Zone classification
    risk_zone       TEXT,                          -- CRITICAL/HIGH/MODERATE/LOW
    zone_name       TEXT REFERENCES risk_zones(zone_name),
    is_outside_site BOOLEAN DEFAULT FALSE,

    -- PPE status
    has_helmet      BOOLEAN DEFAULT FALSE,
    has_vest        BOOLEAN DEFAULT FALSE,
    ppe_compliant   BOOLEAN GENERATED ALWAYS AS (has_helmet AND has_vest) STORED,

    -- GPS geometry (for QGIS live layer)
    geom            GEOMETRY(POINTZ, 4326),

    -- Indexes for fast queries
    CONSTRAINT valid_world CHECK (world_x IS NULL OR (world_x BETWEEN -5 AND 30))
);

-- ============================================================
-- TABLE 4: risk_scores
-- Computed every 2 seconds per zone — drives heatmap
-- ============================================================
CREATE TABLE risk_scores (
    score_id        SERIAL PRIMARY KEY,
    computed_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    zone_name       TEXT REFERENCES risk_zones(zone_name),
    time_window_min INT DEFAULT 5,

    -- BOCW multi-factor risk formula components
    -- Risk = 0.35*density + 0.30*ppe + 0.20*proximity + 0.15*movement
    worker_count    INT DEFAULT 0,
    density_factor  FLOAT,                         -- workers/m²
    ppe_factor      FLOAT,                         -- violation ratio 0-1
    proximity_factor FLOAT,                        -- hazard proximity score
    movement_factor FLOAT,                         -- position std deviation

    -- Final score
    risk_score      FLOAT CHECK (risk_score BETWEEN 0 AND 1),
    risk_level      TEXT,                          -- LOW/MODERATE/HIGH/CRITICAL

    -- Heatmap grid cell
    grid_x          INT,                           -- 2m grid col
    grid_y          INT,                           -- 2m grid row
    geom            GEOMETRY(POINT, 4326)
);

-- ============================================================
-- TABLE 5: alerts
-- Zone-level alerts with acknowledgment tracking
-- ============================================================
CREATE TABLE alerts (
    alert_id        UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    triggered_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    zone_name       TEXT,
    risk_level      TEXT NOT NULL,                 -- CRITICAL/HIGH/MODERATE
    alert_type      TEXT NOT NULL,                 -- PPE_VIOLATION/ZONE_BREACH/DENSITY
    message         TEXT NOT NULL,
    worker_count    INT DEFAULT 0,
    ppe_violations  INT DEFAULT 0,

    -- BOCW compliance reference
    bocw_clause     TEXT,                          -- e.g. 'BOCW §40 — Fall protection'
    nbc_clause      TEXT,                          -- e.g. 'NBC 2016 Pt.7 §3.2'

    -- Acknowledgment
    acknowledged    BOOLEAN DEFAULT FALSE,
    acknowledged_by TEXT,
    acknowledged_at TIMESTAMPTZ,
    action_taken    TEXT,

    -- Detection snapshot
    detection_ids   UUID[],                        -- which detections triggered this
    geom            GEOMETRY(POINT, 4326)
);

-- ============================================================
-- TABLE 6: heatmap_snapshots
-- Pre-computed heatmap grids saved every 30 seconds
-- ============================================================
CREATE TABLE heatmap_snapshots (
    snapshot_id     SERIAL PRIMARY KEY,
    captured_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    time_window_min INT DEFAULT 5,
    grid_data       JSONB NOT NULL,               -- {cell_x: {cell_y: risk_score}}
    max_risk_score  FLOAT,
    total_workers   INT,
    total_violations INT,
    geom_bbox       GEOMETRY(POLYGON, 4326)
);

-- ============================================================
-- TABLE 7: zone_events
-- Time-series of zone-level occupancy for BOCW reporting
-- ============================================================
CREATE TABLE zone_events (
    event_id        SERIAL PRIMARY KEY,
    event_time      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    zone_name       TEXT REFERENCES risk_zones(zone_name),
    worker_count    INT,
    ppe_violation_count INT,
    avg_time_in_zone_sec FLOAT,
    risk_score      FLOAT,
    risk_level      TEXT
);

-- ============================================================
-- SPATIAL INDEXES (PostGIS R-Tree — critical for performance)
-- ============================================================
CREATE INDEX idx_detections_geom
    ON detections USING GIST(geom);

CREATE INDEX idx_detections_time
    ON detections(timestamp DESC);

CREATE INDEX idx_detections_zone_time
    ON detections(zone_name, timestamp DESC);

CREATE INDEX idx_detections_camera_time
    ON detections(camera_id, timestamp DESC);

CREATE INDEX idx_risk_zones_geom
    ON risk_zones USING GIST(geom);

CREATE INDEX idx_risk_scores_time
    ON risk_scores(computed_at DESC);

CREATE INDEX idx_risk_scores_zone_time
    ON risk_scores(zone_name, computed_at DESC);

CREATE INDEX idx_alerts_time
    ON alerts(triggered_at DESC);

CREATE INDEX idx_alerts_unacked
    ON alerts(triggered_at DESC)
    WHERE acknowledged = FALSE;

-- ============================================================
-- VIEWS: pre-built queries for dashboard
-- ============================================================

-- Live worker positions (last 10 seconds)
CREATE VIEW v_live_workers AS
SELECT
    d.detection_id,
    d.camera_id,
    d.timestamp,
    d.world_x, d.world_y, d.world_z,
    d.risk_zone,
    d.has_helmet, d.has_vest, d.ppe_compliant,
    d.geom,
    ST_AsGeoJSON(d.geom) AS geojson_point
FROM detections d
WHERE d.timestamp > NOW() - INTERVAL '10 seconds'
  AND d.is_outside_site = FALSE
  AND d.class_name = 'worker';

-- Zone occupancy summary (last 5 minutes)
CREATE VIEW v_zone_summary AS
SELECT
    z.zone_name,
    z.risk_level,
    COUNT(d.detection_id)                                         AS worker_count,
    SUM(CASE WHEN NOT d.ppe_compliant THEN 1 ELSE 0 END)        AS ppe_violations,
    ROUND(AVG(d.world_z)::numeric, 2)                           AS avg_height_m,
    MAX(d.timestamp)                                             AS last_seen
FROM risk_zones z
LEFT JOIN detections d
    ON d.zone_name = z.zone_name
    AND d.timestamp > NOW() - INTERVAL '5 minutes'
GROUP BY z.zone_name, z.risk_level
ORDER BY
    CASE z.risk_level
        WHEN 'CRITICAL' THEN 1 WHEN 'HIGH' THEN 2
        WHEN 'MODERATE' THEN 3 WHEN 'LOW'  THEN 4
        ELSE 5
    END;

-- Unacknowledged alerts
CREATE VIEW v_active_alerts AS
SELECT
    alert_id, triggered_at, zone_name, risk_level,
    alert_type, message, worker_count, ppe_violations,
    bocw_clause, nbc_clause,
    EXTRACT(EPOCH FROM (NOW() - triggered_at)) AS age_seconds
FROM alerts
WHERE acknowledged = FALSE
ORDER BY triggered_at DESC;

-- BOCW compliance report (daily)
CREATE VIEW v_bocw_daily_report AS
SELECT
    DATE(timestamp)                             AS report_date,
    zone_name,
    COUNT(*)                                    AS total_detections,
    SUM(CASE WHEN NOT ppe_compliant THEN 1 ELSE 0 END) AS violations,
    ROUND(100.0 * SUM(CASE WHEN ppe_compliant THEN 1 ELSE 0 END)
          / NULLIF(COUNT(*),0), 1)              AS compliance_pct,
    MAX(world_z)                                AS max_detected_height_m,
    COUNT(DISTINCT DATE_TRUNC('hour',timestamp)) AS active_hours
FROM detections
WHERE class_name = 'worker'
  AND is_outside_site = FALSE
GROUP BY DATE(timestamp), zone_name
ORDER BY report_date DESC, zone_name;

-- ============================================================
-- SEED DATA: Insert camera and zones
-- ============================================================

INSERT INTO cameras
    (camera_id, description, location_note, mount_height_m,
     mount_lat, mount_lon)
VALUES
    ('CAM-01', 'Main site camera',
     'Neighbour building south face, 25ft elevation, looking north',
     7.620, 10.81651098, 78.66891976)
ON CONFLICT DO NOTHING;

-- Insert zone geometries (matches GeoJSON)
INSERT INTO risk_zones (zone_name, risk_level, alert_level, description, z_base_m, z_top_m)
VALUES
    ('site_boundary',        'none',     NULL,       'Full site boundary (3m buffer)', 0, 6.096),
    ('high_risk_scaffolding','HIGH',     'CRITICAL', 'Scaffolding perimeter band',     0, 6.096),
    ('high_risk_staircase',  'CRITICAL', 'CRITICAL', 'Staircase NW corner — fall risk',0, 6.096),
    ('moderate_risk_interior','MODERATE','WARNING',  'Interior active work zone',      0, 4.877),
    ('low_risk_parking',     'LOW',      'ADVISORY', 'Car parking and support area',   0, 6.096)
ON CONFLICT DO NOTHING;

-- ============================================================
-- STORED PROCEDURE: compute_risk_score()
-- Called every 2 seconds by the heatmap engine
-- BOCW Formula: Risk = 0.35*D + 0.30*P + 0.20*H + 0.15*M
-- ============================================================
CREATE OR REPLACE FUNCTION compute_risk_score(
    p_zone_name TEXT,
    p_window_min INT DEFAULT 5
) RETURNS FLOAT AS $$
DECLARE
    v_worker_count    INT;
    v_zone_area_m2    FLOAT;
    v_density         FLOAT;
    v_ppe_violations  INT;
    v_ppe_factor      FLOAT;
    v_avg_height      FLOAT;
    v_proximity       FLOAT;
    v_movement        FLOAT;
    v_risk_score      FLOAT;
BEGIN
    -- Density factor (workers / m²)
    SELECT COUNT(*),
           COALESCE(SUM(CASE WHEN NOT ppe_compliant THEN 1 ELSE 0 END), 0)
    INTO v_worker_count, v_ppe_violations
    FROM detections
    WHERE zone_name = p_zone_name
      AND timestamp > NOW() - (p_window_min || ' minutes')::INTERVAL
      AND class_name = 'worker';

    -- Zone area (approximate)
    SELECT COALESCE(ST_Area(ST_Transform(geom, 32644)), 100)
    INTO v_zone_area_m2
    FROM risk_zones WHERE zone_name = p_zone_name;

    -- Normalise density: 1 worker/m² = max (score=1)
    v_density := LEAST(v_worker_count::FLOAT / GREATEST(v_zone_area_m2, 1), 1.0);

    -- PPE factor: ratio of violators
    v_ppe_factor := CASE WHEN v_worker_count > 0
                         THEN v_ppe_violations::FLOAT / v_worker_count
                         ELSE 0 END;

    -- Height proximity factor: higher workers → higher risk
    SELECT COALESCE(AVG(world_z), 0) INTO v_avg_height
    FROM detections
    WHERE zone_name = p_zone_name
      AND timestamp > NOW() - (p_window_min || ' minutes')::INTERVAL;
    v_proximity := LEAST(v_avg_height / 6.096, 1.0);  -- normalise to building height

    -- Movement factor: position std deviation (normalised to site width)
    SELECT COALESCE(STDDEV(world_x) + STDDEV(world_y), 0) / 18.9
    INTO v_movement
    FROM detections
    WHERE zone_name = p_zone_name
      AND timestamp > NOW() - (p_window_min || ' minutes')::INTERVAL;
    v_movement := LEAST(COALESCE(v_movement, 0), 1.0);

    -- BOCW multi-factor formula
    v_risk_score := (0.35 * v_density)
                  + (0.30 * v_ppe_factor)
                  + (0.20 * v_proximity)
                  + (0.15 * v_movement);

    RETURN LEAST(v_risk_score, 1.0);
END;
$$ LANGUAGE plpgsql;

-- ============================================================
-- GRANT permissions for BuildSight app user
-- ============================================================
-- CREATE USER buildsight_app WITH PASSWORD 'change_this_password';
-- GRANT CONNECT ON DATABASE buildsight TO buildsight_app;
-- GRANT USAGE ON SCHEMA public TO buildsight_app;
-- GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA public TO buildsight_app;
-- GRANT EXECUTE ON FUNCTION compute_risk_score TO buildsight_app;

-- ============================================================
-- QUICK HEALTH CHECK
-- ============================================================
SELECT 'Schema created successfully' AS status,
       COUNT(*) AS zone_count FROM risk_zones;
