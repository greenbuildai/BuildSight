import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional
from .. import config

logger = logging.getLogger(__name__)

class GISService:
    def __init__(self):
        # Coordinates should be loaded from config/DB
        # Example: Red Zone as a list of (x, y) tuples
        self.red_zone = np.array(config.RED_ZONE_COORDS, dtype=np.int32) if config.RED_ZONE_COORDS else None

    def is_in_red_zone(self, point: Tuple[int, int]) -> bool:
        """
        Check if a pixel point (x, y) is inside the Red Zone polygon.
        """
        if self.red_zone is None or len(self.red_zone) < 3:
            return False
        
        # pointPolygonTest returns >0 if inside, 0 if on edge, <0 if outside
        result = cv2.pointPolygonTest(self.red_zone, point, False)
        return result >= 0

    def get_zone_name(self, point: Tuple[int, int]) -> Optional[str]:
        if self.is_in_red_zone(point):
            return "Red Zone"
        return None

# Global instance
gis_service = GISService()
