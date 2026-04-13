from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class GeoPoint(BaseModel):
    lat: float
    lng: float


class GeoZone(BaseModel):
    id: str
    name: str
    type: str  # e.g., "restricted", "work", "hazard"
    coordinates: List[List[List[float]]]  # GeoJSON format [lng, lat]
    risk_level: str = "low"
    properties: Dict[str, Any] = Field(default_factory=dict)


class GeoData(BaseModel):
    zones: List[GeoZone] = Field(default_factory=list)
    boundary: Optional[List[List[float]]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ── Dynamic Zone models ───────────────────────────────────────────────────────

class DynamicZoneCreate(BaseModel):
    """Payload for creating a new operator-defined zone."""
    name: str
    risk_level: str = "moderate"          # low / moderate / high / critical
    zone_type: str = "restricted"         # restricted / work / hazard / safe / custom
    color: Optional[str] = None           # hex override, e.g. "#ff3b30"
    # GeoJSON ring: list of [lng, lat] pairs, auto-closed by the server
    coordinates: List[List[float]]
    description: Optional[str] = None


class DynamicZone(BaseModel):
    """A persisted operator-defined geofence zone."""
    id: str
    name: str
    risk_level: str
    zone_type: str
    color: str
    coordinates: List[List[float]]        # ring of [lng, lat] pairs
    description: str
    created_at: float                     # Unix timestamp
    is_active: bool = True


class DynamicZoneUpdate(BaseModel):
    """Partial update payload for an existing zone."""
    name: Optional[str] = None
    risk_level: Optional[str] = None
    zone_type: Optional[str] = None
    color: Optional[str] = None
    coordinates: Optional[List[List[float]]] = None
    description: Optional[str] = None
    is_active: Optional[bool] = None


# ── VLM models ────────────────────────────────────────────────────────────────

class VLMQueryRequest(BaseModel):
    """Custom question to send to the VLM."""
    question: str
    force_refresh: bool = False


class VLMEntry(BaseModel):
    """A single VLM site-description entry."""
    description: str
    timestamp: float
    source: str          # "moondream2" | "rule_based"
    question: str
    vlm_available: bool


# ── SAM models ────────────────────────────────────────────────────────────────

class SAMPromptRequest(BaseModel):
    """Payload for point-based SAM prompts."""
    # List of [x, y] pixel coordinates
    prompts: List[List[float]]
    # Optional labels: 1 for foreground, 0 for background
    labels: Optional[List[int]] = None
    min_score: float = 0.70


class SAMZoneResult(BaseModel):
    """Refined segmentation result for frontend display."""
    id: str
    geojson: Dict[str, Any]      # Polygon Feature
    confidence: float
    area_m2: float               # Calculated world-area
    centroid_gps: List[float]    # [lat, lon]
    bbox_gps: List[List[float]]  # [[minLat, minLon], [maxLat, maxLon]]
    inference_ms: float
    device: str                  # "cuda" | "cpu"


class SAMStatusResponse(BaseModel):
    """Health check for the SAM inference engine."""
    loaded: bool
    model_type: str
    device: str
    vram_allocated_gb: Optional[float] = None
    avg_inference_ms: float
    status: str                  # "ready" | "initializing" | "error"


class SpatialVLMRequest(BaseModel):
    lat: float
    lon: float
    question: Optional[str] = "Describe this specific area and identify any safety issues."


class SpatialVLMResponse(BaseModel):
    description: str
    lat: float
    lon: float
    timestamp: float
    source: str
    avg_inference_ms: float
    status: str                  # "ready" | "initializing" | "error"
