import pytest
import numpy as np
from ..services.gis import gis_service, GISService

def test_red_zone_check():
    # Setup simple square polygon 0,0 to 100,100
    gis = GISService()
    gis.red_zone = np.array([[0,0], [100,0], [100,100], [0,100]], dtype=np.int32)
    
    # Point inside
    assert gis.is_in_red_zone((50, 50)) == True
    
    # Point outside
    assert gis.is_in_red_zone((150, 50)) == False
    
    # Point on edge (cv2.pointPolygonTest handles this based on flag, we check >=0)
    assert gis.is_in_red_zone((0, 50)) == True
