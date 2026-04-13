# General GIS Reference
## Coordinate Systems, Spatial Analysis, Remote Sensing, QGIS

---

## Coordinate Reference Systems

### Common CRS for India / South Asia

| CRS | EPSG | Use Case |
|-----|------|----------|
| WGS84 Geographic | 4326 | GPS coordinates, global datasets |
| UTM Zone 43N | 32643 | West India (Mumbai, Gujarat) |
| UTM Zone 44N | 32644 | Central/South India — **BuildSight (Thanjavur)** |
| UTM Zone 45N | 32645 | East India (Kolkata, Odisha) |
| India-specific (Everest) | 24378 | Survey of India legacy data |
| Web Mercator | 3857 | Web tiles (OpenStreetMap, Google Maps) |

### CRS Conversion (Python)

```python
from pyproj import Transformer

# WGS84 lat/lon → UTM 44N
transformer = Transformer.from_crs(4326, 32644, always_xy=True)
easting, northing = transformer.transform(lon, lat)

# UTM 44N → WGS84
transformer_inv = Transformer.from_crs(32644, 4326, always_xy=True)
lon, lat = transformer_inv.transform(easting, northing)
```

### PostGIS CRS Transforms

```sql
-- Reproject geometry
SELECT ST_Transform(geom, 32644) FROM raw_points WHERE ST_SRID(geom) = 4326;

-- Force SRID without reprojection (when CRS is known but unset)
SELECT ST_SetSRID(geom, 32644) FROM zones WHERE ST_SRID(geom) = 0;

-- Check SRID of all geometries in a table
SELECT DISTINCT ST_SRID(geom) FROM detection_events;
```

---

## Spatial Analysis Patterns

### Proximity Analysis

```sql
-- Workers within 15m of machinery (BOCW exclusion)
SELECT de.id, de.zone_id, ST_Distance(de.location, mz.geom) AS dist_m
FROM detection_events de
CROSS JOIN (SELECT geom FROM zones WHERE activity_type = 'heavy_machinery') mz
WHERE ST_DWithin(de.location, mz.geom, 15.0)
  AND de.class_id = 3
  AND de.timestamp > NOW() - INTERVAL '5 minutes';

-- Nearest zone to a point
SELECT zone_id, zone_name,
       ST_Distance(geom, ST_SetSRID(ST_MakePoint(438000, 1071000), 32644)) AS dist_m
FROM zones
ORDER BY dist_m ASC LIMIT 1;
```

### Density Mapping

```sql
-- Worker density grid (50m × 50m cells)
WITH grid AS (
    SELECT ST_SetSRID(
        ST_Translate(
            ST_GeomFromText('POLYGON((0 0, 50 0, 50 50, 0 50, 0 0))'),
            50 * x_idx + ST_XMin(ext.bounds),
            50 * y_idx + ST_YMin(ext.bounds)
        ), 32644
    ) AS cell,
    x_idx, y_idx
    FROM (SELECT ST_Extent(geom)::geometry AS bounds FROM zones) ext,
    generate_series(0, 10) AS x_idx,
    generate_series(0, 10) AS y_idx
)
SELECT g.x_idx, g.y_idx, g.cell,
       COUNT(de.id) AS worker_count
FROM grid g
LEFT JOIN detection_events de
    ON ST_Within(de.location, g.cell)
    AND de.class_id = 3
    AND de.timestamp > NOW() - INTERVAL '1 hour'
GROUP BY g.x_idx, g.y_idx, g.cell;
```

### Temporal Patterns

```sql
-- Hourly violation rate per zone
SELECT
    zone_id,
    DATE_TRUNC('hour', timestamp) AS hour,
    COUNT(*) FILTER (WHERE NOT has_helmet OR NOT has_vest) AS violations,
    COUNT(*) AS total_workers
FROM detection_events
WHERE class_id = 3
  AND timestamp > NOW() - INTERVAL '7 days'
GROUP BY zone_id, hour
ORDER BY zone_id, hour;
```

---

## QGIS Workflows

### Connect PostGIS Live Layer

```python
# PyQGIS — Add PostGIS layer
from qgis.core import QgsDataSourceUri, QgsVectorLayer, QgsProject

uri = QgsDataSourceUri()
uri.setConnection('localhost', '5432', 'buildsight', 'joseva', '')
uri.setDataSource('public', 'v_current_risk', 'geom', '', 'zone_id')
layer = QgsVectorLayer(uri.uri(False), 'Risk Heatmap', 'postgres')
QgsProject.instance().addMapLayer(layer)
```

### Graduated Symbology (Risk Score)

```python
from qgis.core import (QgsGraduatedSymbolRenderer, QgsRendererRange,
                        QgsFillSymbol, QgsClassificationQuantile)
import qgis.core as qgis

ranges = [
    (0.0, 0.3, '#00CC44', 'Low Risk'),
    (0.3, 0.6, '#FFCC00', 'Medium Risk'),
    (0.6, 0.8, '#FF8800', 'High Risk'),
    (0.8, 1.0, '#CC0000', 'Critical Risk')
]

renderer_ranges = []
for low, high, color, label in ranges:
    symbol = QgsFillSymbol.createSimple({'color': color, 'outline_color': '#333333'})
    renderer_ranges.append(QgsRendererRange(low, high, symbol, label))

renderer = QgsGraduatedSymbolRenderer('risk_score', renderer_ranges)
layer.setRenderer(renderer)
layer.triggerRepaint()
```

### Export Print Layout (NBC Compliant)

```python
from qgis.core import QgsLayoutExporter, QgsPrintLayout, QgsProject

layout = QgsProject.instance().layoutManager().layoutByName('BuildSight Safety Map')
exporter = QgsLayoutExporter(layout)
settings = QgsLayoutExporter.PdfExportSettings()
settings.dpi = 300
exporter.exportToPdf('/tmp/buildsight_safety_map.pdf', settings)
```

---

## Remote Sensing Fundamentals

### NDVI (Vegetation Index)

```python
import rasterio
import numpy as np

with rasterio.open('sentinel2_band4.tif') as red_src:
    red = red_src.read(1).astype(float)
with rasterio.open('sentinel2_band8.tif') as nir_src:
    nir = nir_src.read(1).astype(float)

ndvi = (nir - red) / (nir + red + 1e-8)
ndvi = np.clip(ndvi, -1, 1)
```

### Raster to PostGIS

```bash
# Import raster to PostGIS
raster2pgsql -s 32644 -I -C -M site_basemap.tif public.site_basemap | psql -d buildsight

# Create raster overview (pyramid) for performance
psql -d buildsight -c "SELECT AddRasterConstraints('public', 'site_basemap', 'rast'::name);"
```

### ISRO Bhuvan Data Sources (India)

| Dataset | URL | Use |
|---------|-----|-----|
| Cartosat-1 DEM | bhuvan.nrsc.gov.in | 2.5m elevation, site grading |
| LISS-IV Ortho | bhuvan.nrsc.gov.in | 5.8m optical satellite |
| Sentinel-2 | sentinel.esa.int | 10m multispectral, vegetation |
| OpenTopography DEM | opentopography.org | SRTM 30m, regional context |

---

## GeoJSON / Shapefile Interchange

```python
import geopandas as gpd

# Shapefile → PostGIS
gdf = gpd.read_file('zone_grid.shp')
gdf = gdf.set_crs(32644)  # ensure CRS set
gdf.to_postgis('zones', engine, if_exists='replace', index=False)

# PostGIS → GeoJSON (for web dashboard)
gdf_risk = gpd.read_postgis('SELECT * FROM v_current_risk', conn, geom_col='geom')
gdf_risk.to_file('current_risk.geojson', driver='GeoJSON')

# GeoJSON → Leaflet (Node.js / inline HTML)
# Load current_risk.geojson → L.geoJSON(data, {style: riskStyle}).addTo(map)
```

---

## Coordinate QA Checklist

Before inserting any geometry into PostGIS, verify:
- [ ] SRID set correctly (should be 32644 for BuildSight)
- [ ] Coordinates are in metres (not degrees) — easting ~430,000–450,000 for Thanjavur area
- [ ] No NULL geometries in batch
- [ ] Geometries are valid: `SELECT ST_IsValid(geom) FROM zones`
- [ ] Spatial index exists: `SELECT * FROM pg_indexes WHERE tablename = 'zones'`
- [ ] Extent is reasonable: `SELECT ST_Extent(geom) FROM zones`

Expected Thanjavur UTM 44N extent:
```
Easting:  ~433,000 – 437,000 m
Northing: ~1,068,000 – 1,073,000 m
```
