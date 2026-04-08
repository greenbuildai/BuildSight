import json
import uuid
from typing import Dict, Any, List
from .models import GeoData, GeoZone

def parse_geojson(geojson_str: str) -> GeoData:
    try:
        data = json.loads(geojson_str)
    except json.JSONDecodeError:
        return GeoData()

    zones = []
    boundary = None
    
    # Simple GeoJSON FeatureCollection parsing
    features = data.get("features", [])
    if not features and data.get("type") == "Feature":
        features = [data]
        
    for feature in features:
        geom = feature.get("geometry", {})
        props = feature.get("properties", {})
        
        g_type = geom.get("type")
        coords = geom.get("coordinates")
        
        if not coords:
            continue
            
        # Treat Polygons and MultiPolygons as zones
        if g_type in ["Polygon", "MultiPolygon"]:
            zone_id = props.get("id") or str(uuid.uuid4())[:8]
            name = props.get("name") or props.get("label") or f"Zone-{zone_id}"
            z_type = props.get("type") or "generic"
            risk = props.get("risk") or props.get("risk_level") or "low"
            
            # Simplified normalization for different Polygon types
            if g_type == "Polygon":
                poly_coords = [coords] if isinstance(coords[0][0], float) else coords
            else: # MultiPolygon
                poly_coords = coords[0] # Just take the first polygon for now
                
            zones.append(GeoZone(
                id=zone_id,
                name=name,
                type=z_type,
                coordinates=poly_coords,
                risk_level=risk,
                properties=props
            ))
        
        # Check for site boundary specific naming
        if "boundary" in props.get("name", "").lower() or props.get("class") == "boundary":
            boundary = coords[0] if g_type == "Polygon" else coords

    return GeoData(zones=zones, boundary=boundary, metadata=data.get("metadata", {}))
