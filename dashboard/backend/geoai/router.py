"""
BuildSight GeoAI — FastAPI Router
===================================
Mounts under /api/geoai (see server.py include_router call).

Endpoints:
  Static zones / GeoJSON
    GET  /zones              — parsed zones from last uploaded GeoJSON
    GET  /boundaries         — site boundary polygon
    POST /upload             — upload a GeoJSON file
    GET  /geojson            — serve raw buildsight_zones_3d.geojson
    GET  /site-config        — site metadata (centre, dims, WS url)
    DELETE /clear            — wipe in-memory zone state

  Dynamic zones (operator-drawn geofences)
    GET    /dynamic-zones            — list all dynamic zones
    POST   /dynamic-zones            — create a new zone
    PUT    /dynamic-zones/{zone_id}  — update an existing zone
    DELETE /dynamic-zones/{zone_id}  — remove a zone

  VLM (Moondream2 site-description)
    GET  /vlm/latest         — return cached description + metadata
    POST /vlm/query          — run VLM with a custom question

  SAM (Segment Anything zone auto-detect)
    POST /sam/detect         — run SAM on the latest frame, return zone features
    GET  /sam/status         — whether SAM is loaded and ready
"""

import json
import os
import time
import uuid
import logging
from pathlib import Path
from typing import List, Optional, Any

logger = logging.getLogger("buildsight")

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import FileResponse

from .models import (
    GeoData,
    DynamicZone,
    DynamicZoneCreate,
    DynamicZoneUpdate,
    VLMEntry,
    VLMQueryRequest,
    SAMStatusResponse,
    SAMPromptRequest,
    SAMZoneResult,
    SpatialVLMRequest,
    SpatialVLMResponse,
)
from .parser import parse_geojson

log = logging.getLogger("BuildSight.GeoAI.Router")

router = APIRouter()

# ── In-memory GeoJSON state ───────────────────────────────────────────────────
current_geo_data: Optional[GeoData] = None

# ── Dynamic zones — persisted to JSON ────────────────────────────────────────
_ZONES_FILE = Path(__file__).parent.parent / "dynamic_zones.json"

_RISK_COLORS = {
    "low":      "#00e676",
    "moderate": "#ffd600",
    "high":     "#ff7b00",
    "critical": "#ff3b3b",
    "none":     "#00b4d8",
}

def auto_load_geojson():
    """
    Automatically loads the site GeoJSON into memory at startup.
    Ensures the /api/geoai/zones endpoint is immediately available.
    """
    global current_geo_data
    
    # Path to the source-of-truth GeoJSON
    geojson_path = os.path.join(os.path.dirname(__file__), "..", "..", "public", "buildsight_zones_complete.geojson")
    geojson_path = os.path.abspath(geojson_path)
    
    logger.info(f"Checking for GeoAI zone file at: {geojson_path}")
    
    if not os.path.exists(geojson_path):
        logger.warning(f"GeoAI zone file NOT FOUND: {geojson_path}. Starting with empty zones.")
        current_geo_data = GeoData()
        return

    try:
        with open(geojson_path, "r", encoding="utf-8") as f:
            content = f.read()
            if not content.strip():
                raise ValueError("GeoJSON file is empty")
                
            data = json.loads(content)
            
            # Simple validation
            features = data.get("features", [])
            logger.info(f"Successfully loaded {len(features)} GeoAI zones from disk")
            
            # Convert to our internal GeoData model
            from .parser import parse_geojson
            current_geo_data = parse_geojson(content)
            
    except Exception as e:
        logger.error(f"FATAL: Failed to auto-load GeoAI zones: {e}")
        # Initialize with empty data to prevent downstream crashes
        current_geo_data = GeoData()

# Trigger auto-load on module import
auto_load_geojson()

# ── Zone templates — coordinates match zones.geojson so they overlay correctly ──
# Each polygon is [lng, lat] pairs identical to the static GeoJSON zones.
_TEMPLATES: dict = {
    "high_risk_scaffolding": [
        [78.66881504, 10.81656524], [78.66902447, 10.81656524],
        [78.66902447, 10.81668959], [78.66881504, 10.81668959],
    ],
    "high_risk_staircase": [
        [78.66883333, 10.81665071], [78.66887796, 10.81665071],
        [78.66887796, 10.81667151], [78.66883333, 10.81667151],
    ],
    "moderate_risk_interior": [
        [78.66884248, 10.81659237], [78.66897655, 10.81659237],
        [78.66897655, 10.81666246], [78.66884248, 10.81666246],
    ],
    "low_risk_parking": [
        [78.66898112, 10.81658333], [78.66900618, 10.81658333],
        [78.66900618, 10.81667151], [78.66898112, 10.81667151],
    ],
    "site_boundary": [
        [78.66880589, 10.81655620], [78.66903362, 10.81655620],
        [78.66903362, 10.81669864], [78.66880589, 10.81669864],
    ],
}


def _load_zones() -> List[DynamicZone]:
    if not _ZONES_FILE.exists():
        return []
    try:
        data = json.loads(_ZONES_FILE.read_text())
        return [DynamicZone(**z) for z in data]
    except Exception as exc:
        log.warning("Could not load dynamic_zones.json: %s", exc)
        return []


def _save_zones(zones: List[DynamicZone]) -> None:
    try:
        _ZONES_FILE.write_text(
            json.dumps([z.model_dump() for z in zones], indent=2)
        )
        # Bust spatial mapper zone cache so next inference picks up new zones immediately
        try:
            import server as _srv  # type: ignore
            _srv.spatial_mapper.last_refresh = 0
        except Exception:
            pass
    except Exception as exc:
        log.error("Could not persist dynamic_zones.json: %s", exc)


# ── Helpers to pull current detection stats from server.py ──────────────────
def _get_fallback_stats() -> dict:
    """Pull live stats from the detection server's shared state if available."""
    try:
        # server.py exposes `detection_stats` at module level
        import server as srv  # type: ignore
        s = getattr(srv, "detection_stats", {})
        return dict(s) if s else {}
    except Exception:
        return {}


def _get_latest_frame_jpeg() -> Optional[bytes]:
    """Get the most recent JPEG frame from the detection server if available."""
    try:
        import server as srv  # type: ignore
        return getattr(srv, "latest_frame_jpeg", None)
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# Static GeoJSON endpoints
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/upload")
async def upload_geojson(file: UploadFile = File(...)):
    global current_geo_data
    content = await file.read()
    try:
        current_geo_data = parse_geojson(content.decode("utf-8"))
        return {"status": "success", "zones_found": len(current_geo_data.zones)}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to parse GeoJSON: {exc}")


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
    candidates = [
        "../../buildsight_zones_complete.geojson",
        "../../buildsight_zones_3d.geojson",
        "../../../buildsight_zones_complete.geojson",
        "../../../buildsight_zones_3d.geojson",
        "buildsight_zones_complete.geojson",
    ]
    current_dir = os.path.dirname(__file__)
    for p in candidates:
        abs_p = os.path.abspath(os.path.join(current_dir, p))
        if os.path.exists(abs_p):
            return FileResponse(abs_p, media_type="application/json")
    raise HTTPException(status_code=404, detail="GeoJSON file not found")


@router.get("/site-config")
async def get_site_config():
    return {
        "site_id": "CH-SITE-01-TIRUCHIRAPPALLI",
        "center": [10.81662, 78.66891],
        "bounds": {
            "sw": [10.81658333, 78.66883333],
            "ne": [10.81666, 78.669],
        },
        "dimensions": {"width_m": 18.90, "depth_m": 9.75},
        "cell_size_m": 2.0,
        "ws_endpoint": "ws://localhost:8765",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Dynamic zone CRUD
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/dynamic-zones", response_model=List[DynamicZone])
async def list_dynamic_zones():
    return _load_zones()


@router.get("/dynamic-zones/templates")
async def get_zone_templates():
    """Return preset polygon coordinate templates keyed by zone name."""
    return {"templates": _TEMPLATES}


@router.post("/dynamic-zones", response_model=DynamicZone)
async def create_dynamic_zone(payload: DynamicZoneCreate):
    zones = _load_zones()

    # Auto-close the ring if caller didn't
    coords = list(payload.coordinates)
    if coords and coords[0] != coords[-1]:
        coords.append(coords[0])

    if len(coords) < 4:
        raise HTTPException(
            status_code=400,
            detail="Zone polygon must have at least 3 vertices (4 coords with closing point).",
        )

    color = payload.color or _RISK_COLORS.get(payload.risk_level.lower(), "#00b4d8")
    zone = DynamicZone(
        id=str(uuid.uuid4()),
        name=payload.name,
        risk_level=payload.risk_level.lower(),
        zone_type=payload.zone_type,
        color=color,
        coordinates=coords,
        description=payload.description or "",
        created_at=time.time(),
        is_active=True,
    )
    zones.append(zone)
    _save_zones(zones)
    log.info("Dynamic zone created: %s (%s)", zone.name, zone.id)
    return zone


@router.put("/dynamic-zones/{zone_id}", response_model=DynamicZone)
async def update_dynamic_zone(zone_id: str, payload: DynamicZoneUpdate):
    zones = _load_zones()
    idx = next((i for i, z in enumerate(zones) if z.id == zone_id), None)
    if idx is None:
        raise HTTPException(status_code=404, detail=f"Zone {zone_id} not found")

    z = zones[idx]
    updated = z.model_copy(
        update={k: v for k, v in payload.model_dump(exclude_none=True).items()}
    )
    # Re-derive color if risk_level changed and no explicit color given
    if payload.risk_level and not payload.color:
        updated = updated.model_copy(
            update={"color": _RISK_COLORS.get(updated.risk_level, updated.color)}
        )
    zones[idx] = updated
    _save_zones(zones)
    return updated


@router.delete("/dynamic-zones/{zone_id}")
async def delete_dynamic_zone(zone_id: str):
    zones = _load_zones()
    new_zones = [z for z in zones if z.id != zone_id]
    if len(new_zones) == len(zones):
        raise HTTPException(status_code=404, detail=f"Zone {zone_id} not found")
    _save_zones(new_zones)
    return {"status": "deleted", "id": zone_id}


# ═══════════════════════════════════════════════════════════════════════════════
# VLM endpoints
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/vlm/latest", response_model=VLMEntry)
async def vlm_latest():
    """Return the cached VLM description. Triggers a refresh if stale."""
    from geoai_vlm_util import get_cached_entry, describe_frame_async, is_available  # type: ignore

    entry, stale = get_cached_entry()
    if stale or not entry:
        stats = _get_fallback_stats()
        jpeg = _get_latest_frame_jpeg()
        entry = await describe_frame_async(jpeg_bytes=jpeg, fallback_stats=stats)

    return VLMEntry(
        description=entry.get("description", "No description available."),
        timestamp=entry.get("timestamp", time.time()),
        source=entry.get("source", "rule_based"),
        question=entry.get("question", ""),
        vlm_available=is_available(),
    )


@router.post("/vlm/query", response_model=VLMEntry)
async def vlm_query(body: VLMQueryRequest):
    """Run the VLM with a custom question on the latest frame."""
    from geoai_vlm_util import describe_frame_async, is_available  # type: ignore

    stats = _get_fallback_stats()
    jpeg = _get_latest_frame_jpeg()
    entry = await describe_frame_async(
        jpeg_bytes=jpeg,
        question=body.question,
        fallback_stats=stats,
        force_refresh=body.force_refresh,
    )
    return VLMEntry(
        description=entry.get("description", ""),
        timestamp=entry.get("timestamp", time.time()),
        source=entry.get("source", "rule_based"),
        question=entry.get("question", body.question),
        vlm_available=is_available(),
    )


@router.get("/vlm/status")
async def vlm_status():
    from geoai_vlm_util import is_available  # type: ignore
    return {"available": is_available()}


# ── SAM Lifecycle & Enrichment ───────────────────────────────────────────────
_SAM_ANALYZER: Optional[Any] = None
_MAPPER: Optional[Any] = None

def _get_mapper():
    """Lazy-load the SpatialMapper."""
    global _MAPPER
    if _MAPPER is not None:
        return _MAPPER
    from .utils.spatial_mapper import SpatialMapper
    _MAPPER = SpatialMapper()
    return _MAPPER

def _get_sam_analyzer():
    """Lazy-load the SAMZoneAnalyzer enriched with spatial intelligence."""
    global _SAM_ANALYZER
    if _SAM_ANALYZER is not None:
        return _SAM_ANALYZER
    
    try:
        # Import from root buildsight_intelligence
        import sys
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
        if root_dir not in sys.path:
            sys.path.append(root_dir)
            
        from buildsight_intelligence import SAMZoneAnalyzer
        mapper = _get_mapper()
        _SAM_ANALYZER = SAMZoneAnalyzer(mapper)
        log.info("SAM: SAMZoneAnalyzer initialized and cached.")
        return _SAM_ANALYZER
    except Exception as exc:
        log.error("SAM: Failed to initialize SAMZoneAnalyzer: %s", exc)
        return None

# ═══════════════════════════════════════════════════════════════════════════════
# Spatial Expert System (VLM + SAM)
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/vlm/spatial-query", response_model=SpatialVLMResponse)
async def spatial_vlm_query(request: SpatialVLMRequest):
    """
    Expert System: Query the VLM about a specific spatial coordinate.
    """
    start_time = time.time()
    try:
        from geoai_vlm_util import describe_frame_async
        mapper = _get_mapper()
        
        # 1. Coordinate conversion: GPS -> Local Meters -> Pixels
        wx, wy = mapper.gps_to_world(request.lat, request.lon)
        # Convert world meters back to normalized pixels then to absolute pixels
        px_norm = wx / mapper.site["width_m"]
        py_norm = 1.0 - (wy / mapper.site["depth_m"])
        
        px = int(px_norm * mapper.frame_w)
        py = int(py_norm * mapper.frame_h)
        
        # 2. Extract latest frame from main process
        try:
            import sys
            root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
            if root_dir not in sys.path: sys.path.append(root_dir)
            from main import global_intelligence  # type: ignore[import]
            frame = None
            if global_intelligence and global_intelligence.last_frame is not None:
                 frame = global_intelligence.last_frame.copy()
        except ImportError:
            # Fallback if importing main is problematic in this env
            frame = None
        
        if frame is None:
            # Try getting raw jpeg from server fallback
            jpeg = _get_latest_frame_jpeg()
            if jpeg:
                import cv2
                import numpy as np
                arr = np.frombuffer(jpeg, np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(status_code=503, detail="CCTV feed unavailable for spatial query")

        # 3. Call VLM with spatial crop
        result = await describe_frame_async(
            frame_bgr=frame,
            question=request.question,
            point_prompt=(px, py)
        )
        
        inference_ms = (time.time() - start_time) * 1000
        
        return SpatialVLMResponse(
            description=result["description"],
            lat=request.lat,
            lon=request.lon,
            timestamp=result["timestamp"],
            source=result["source"],
            avg_inference_ms=inference_ms,
            status="ready"
        )
        
    except Exception as e:
        log.error(f"GeoAI Router: Spatial VLM Query failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sam/status", response_model=SAMStatusResponse)
async def sam_status():
    """Check health and device state of the SAM inference engine."""
    from geoai_sam_util import is_available, _SAM_MODEL
    import torch
    
    is_ready = is_available()
    device = "cpu"
    vram = 0.0
    
    if is_ready and _SAM_MODEL is not None:
        device = str(next(_SAM_MODEL.parameters()).device)
        if "cuda" in device:
            vram = torch.cuda.memory_allocated() / 1024**3
            
    return SAMStatusResponse(
        loaded=is_ready,
        model_type="vit_b",
        device=device,
        vram_allocated_gb=round(vram, 3) if "cuda" in device else None,
        avg_inference_ms=450.0,
        status="ready" if is_ready else "initializing"
    )


@router.post("/sam/prompt", response_model=List[SAMZoneResult])
async def sam_prompt(body: SAMPromptRequest):
    """
    Interactive SAM segmentation based on frontend click prompts.
    Converts polygons to rich GeoJSON with world-space metadata.
    """
    from geoai_sam_util import segment_frame_async, is_available
    import numpy as np
    import cv2

    if not is_available():
        raise HTTPException(status_code=503, detail="SAM engine is not loaded or missing weights.")

    jpeg = _get_latest_frame_jpeg()
    if not jpeg:
        raise HTTPException(status_code=404, detail="No active CCTV frame available for analysis.")

    analyzer = _get_sam_analyzer()
    if not analyzer:
         raise HTTPException(status_code=500, detail="Spatial intelligence engine failed to initialize.")

    start_time = time.time()
    try:
        # 1. Decode frame
        arr = np.frombuffer(jpeg, np.uint8)
        frame_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        
        # 2. Convert pixel prompts (x, y)
        point_prompts = [(int(p[0]), int(p[1])) for p in body.prompts]
        
        # 3. Core SAM Segmentation (Returns normalized GeoJSON features)
        raw_features = await segment_frame_async(
            frame_bgr, 
            point_prompts=point_prompts,
            min_score=body.min_score
        )

        # 4. Spatial Enrichment using SAMZoneAnalyzer
        enriched_results = []
        for feat in raw_features:
            norm_polygon = feat["geometry"]["coordinates"][0]
            confidence = feat["properties"].get("confidence", 1.0)
            
            # Enrich with world metrics
            enrichment = analyzer.enrich_segment(norm_polygon)
            
            result = SAMZoneResult(
                id=f"sam-{uuid.uuid4().hex[:6]}",
                geojson={
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [enrichment["gps_polygon"]]
                    },
                    "properties": {
                        "area_m2": enrichment["area_m2"],
                        "confidence": confidence,
                        "type": "auto_segment"
                    }
                },
                confidence=confidence,
                area_m2=enrichment["area_m2"],
                centroid_gps=enrichment["centroid_gps"],
                bbox_gps=enrichment["bbox_gps"],
                inference_ms=(time.time() - start_time) * 1000,
                device=str(analyzer.mapper.status)
            )
            enriched_results.append(result)

        return enriched_results

    except Exception as exc:
        log.error("SAM prompt error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
