from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class GeoPoint(BaseModel):
    lat: float
    lng: float

class GeoZone(BaseModel):
    id: str
    name: str
    type: str  # e.g., "restricted", "work", "hazard"
    coordinates: List[List[List[float]]] # GeoJSON format [lng, lat]
    risk_level: str = "low"
    properties: Dict[str, Any] = Field(default_factory=dict)

class GeoData(BaseModel):
    zones: List[GeoZone] = Field(default_factory=list)
    boundary: Optional[List[List[float]]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
