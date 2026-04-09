import os
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from .models import GeoData
from .parser import parse_geojson
from typing import Optional

router = APIRouter()

# In-memory store for now
current_geo_data: Optional[GeoData] = None

@router.post("/upload")
async def upload_geojson(file: UploadFile = File(...)):
    global current_geo_data
    content = await file.read()
    try:
        current_geo_data = parse_geojson(content.decode("utf-8"))
        return {"status": "success", "zones_found": len(current_geo_data.zones)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse GeoJSON: {str(e)}")

@router.get("/zones")
async def get_zones():
    if not current_geo_data:
        return {"zones": []}
    return {"zones": current_geo_data.zones}

@router.get("/boundaries")
async def get_boundaries():
    if not current_geo_data:
        return {"boundary": None}
    return {"boundary": current_geo_data.boundary}

@router.delete("/clear")
async def clear_geo_data():
    global current_geo_data
    current_geo_data = None
    return {"status": "cleared"}

@router.get("/geojson")
async def get_geojson_file():
    # Attempt to locate the geojson in various project-relative paths
    possible_paths = [
        "../../buildsight_zones_3d.geojson",
        "../../../buildsight_zones_3d.geojson",
        "buildsight_zones_3d.geojson"
    ]
    
    current_dir = os.path.dirname(__file__)
    for p in possible_paths:
        abs_p = os.path.abspath(os.path.join(current_dir, p))
        if os.path.exists(abs_p):
            return FileResponse(abs_p, media_type="application/json")
    
    raise HTTPException(status_code=404, detail="GeoJSON file not found in build path")

@router.get("/site-config")
async def get_site_config():
    return {
        "site_id": "CH-SITE-01-TIRUCHIRAPPALLI",
        "center": [10.81662, 78.66891],
        "bounds": {
            "sw": [10.81658333, 78.66883333],
            "ne": [10.81666, 78.669] 
        },
        "dimensions": {
            "width_m": 18.90,
            "depth_m": 9.75
        },
        "cell_size_m": 2.0,
        "ws_endpoint": "ws://localhost:8765"
    }
