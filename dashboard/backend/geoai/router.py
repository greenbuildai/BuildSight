from fastapi import APIRouter, UploadFile, File, HTTPException
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
