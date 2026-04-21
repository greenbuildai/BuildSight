from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class ComplianceStatus(BaseModel):
    helmet: bool
    vest: bool
    helmet_state: Optional[str] = None
    vest_state: Optional[str] = None
    overall_state: Optional[str] = None

class DetectedObject(BaseModel):
    track_id: int
    class_name: str  # 'class' is a reserved keyword in Python, using class_name and aliasing if needed
    bbox: List[float] # [x1, y1, x2, y2]
    compliance: ComplianceStatus
    occluded: bool = False

class DetectionResult(BaseModel):
    frame_id: int
    timestamp: datetime
    objects: List[DetectedObject]

class AlertEvent(BaseModel):
    type: str = "violation"
    severity: str # "high", "medium", "low"
    message: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    snapshot_url: Optional[str] = None
    zone_id: Optional[str] = None
    worker_id: Optional[int] = None
    violation: Optional[str] = None
