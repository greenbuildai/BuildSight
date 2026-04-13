import numpy as np
import cv2
import math
from typing import List, Dict, Any, Optional

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Assume `SpatialWorker` comes from the buildsight_intelligence 
try:
    from buildsight_intelligence import SpatialWorker, SITE_CONFIG
except ImportError:
    # Fallback configs if not directly accessible
    SITE_CONFIG = {
        "width_m":  18.90,
        "depth_m":   9.75,
        "sw_lat":   10.81658333,
        "sw_lon":   78.66873333,
        "rotation_deg": 85.0,
    }
    class SpatialWorker: pass


class HeatmapEngine:
    """
    Dynamic Heatmap Intelligence System for BuildSight.
    Maintains a 2D density grid mapping worker spatial presence, 
    risk factors, and temporal decay.
    """

    def __init__(
        self,
        site_width: float = SITE_CONFIG.get("width_m", 18.90),
        site_depth: float = SITE_CONFIG.get("depth_m", 9.75),
        resolution_m: float = 0.5,
        decay_rate: float = 0.90,
        enable_smoothing: bool = True
    ):
        self.site_width = site_width
        self.site_depth = site_depth
        self.resolution = resolution_m
        self.decay_rate = decay_rate
        self.enable_smoothing = enable_smoothing

        self.cols = math.ceil(self.site_width / self.resolution)
        self.rows = math.ceil(self.site_depth / self.resolution)

        # The core density grid (raw intensity)
        self.grid = np.zeros((self.rows, self.cols), dtype=np.float32)
        
        # Max intensity tracking for normalization
        self.peak_intensity = 1.0 

    def _world_to_grid(self, wx: float, wy: float) -> tuple[int, int]:
        """Convert world coordinates (meters) to grid indices."""
        c = int(wx / self.resolution)
        r = int(wy / self.resolution)
        
        c = max(0, min(self.cols - 1, c))
        r = max(0, min(self.rows - 1, r))
        
        return r, c

    def _calculate_risk_weight(self, worker: Any) -> float:
        """
        Calculate risk-weighted intensity based on:
        - Missing PPE
        - Unsafe behavior (Risk/Critical status)
        """
        weight = 1.0  # Base presence weight
        
        # PPE penalties
        if getattr(worker, 'has_helmet', True) is False:
            weight += 1.5
        if getattr(worker, 'has_vest', True) is False:
            weight += 1.5
            
        # Behavior / Stationarity (Dwell mapping)
        dwell_sev = getattr(worker, 'dwell_severity', 'NONE')
        if dwell_sev == 'HIGH':
            weight += 2.0
        elif dwell_sev == 'MEDIUM':
            weight += 1.0
            
        # Overall Risk Status
        status = getattr(worker, 'status', 'SAFE')
        if status == 'CRITICAL':
            weight += 3.0
        elif status == 'AT_RISK':
            weight += 1.5
            
        # Restricted Area Violation (Zone Risk)
        zone = getattr(worker, 'zone', '')
        if "restricted" in zone.lower() or "high_risk" in zone.lower():
            weight += 2.0
            
        return weight

    def update(self, workers: List[Any]) -> None:
        """
        Update the heatmap grid with a new frame of workers.
        Applies decay, adds new worker weights, and bounds the values.
        """
        # Apply temporal decay
        self.grid *= self.decay_rate
        
        # Process new worker positions
        for w in workers:
            wx = getattr(w, 'world_x', 0.0)
            wy = getattr(w, 'world_y', 0.0)
            
            r, c = self._world_to_grid(wx, wy)
            
            # Get intensity based on behavioral risk
            intensity = self._calculate_risk_weight(w)
            
            # Inject into grid (we use a small radius of 1 cell for base injection)
            self.grid[r, c] += intensity
            
            # Simple soft falloff to immediate neighbors
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0: continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.rows and 0 <= nc < self.cols:
                        self.grid[nr, nc] += (intensity * 0.25)
                        
        # Track peak for normalization to avoid capping out completely if heavy crowding
        current_max = np.max(self.grid)
        if current_max > self.peak_intensity:
            self.peak_intensity = current_max
        else:
            # slowly decay peak intensity to allow the map to cool down
            self.peak_intensity = max(1.0, self.peak_intensity * 0.99)

    def get_normalized_grid(self) -> np.ndarray:
        """
        Return the grid normalized between 0.0 and 1.0.
        Applies optional Gaussian smoothing.
        """
        if self.peak_intensity > 0:
            norm_grid = self.grid / self.peak_intensity
        else:
            norm_grid = self.grid.copy()
            
        # Hard cap at 1.0 just in case
        np.clip(norm_grid, 0.0, 1.0, out=norm_grid)
        
        if self.enable_smoothing:
            # Apply a 3x3 or 5x5 Gaussian blur for cleaner transitions
            norm_grid = cv2.GaussianBlur(norm_grid, (5, 5), sigmaX=1.5, sigmaY=1.5)
            
        return norm_grid

    def get_frontend_payload(self) -> Dict[str, Any]:
        """
        Package the heatmap into a format suitable for JSON serialization
        over WebSocket.
        """
        norm_grid = self.get_normalized_grid()
        
        # For bandwidth efficiency, we flatten to a 1D array of rounded floats
        # Multiply by 100 to send as integers to save bytes, or keep as float.
        # Float rounded to 2 decimals is good.
        flat_grid = [round(float(v), 2) for v in norm_grid.flatten()]
        
        return {
            "cols": self.cols,
            "rows": self.rows,
            "resolution_m": self.resolution,
            "data": flat_grid
        }
