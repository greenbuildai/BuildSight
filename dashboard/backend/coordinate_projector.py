"""
BuildSight Pixel-to-World Coordinate Projector
Converts CCTV bounding-box pixel coordinates to real-world
UTM and lat/lng coordinates for GeoAI map rendering.

Uses homographic projection from Ground Control Points (GCPs).
Default config uses approximate coordinates for the construction
site visible in the satellite view (site centre near Tiruchirappalli).
"""

import numpy as np
import cv2
from typing import Optional

# Default camera configuration.
# GCPs: [pixel_x, pixel_y, utm_e_offset_m, utm_n_offset_m]
# utm offsets are metres from site_origin_lat/lng.
# These should be re-calibrated per deployment by clicking known
# landmarks on both the CCTV frame and the satellite map.
DEFAULT_CAMERA_CONFIG: dict = {
    "utm_zone": "44N",
    "site_origin_lat": 10.81662,   # GeoAIMap SITE_CENTER lat
    "site_origin_lng": 78.66891,   # GeoAIMap SITE_CENTER lng
    "site_utm_e_base": 430000.0,
    "site_utm_n_base": 1196500.0,

    # [pixel_x, pixel_y, utm_e_offset_m, utm_n_offset_m]
    # Calibrated so the full 1280×720 CCTV frame maps onto a
    # roughly 30 m (E) × 20 m (N) site footprint.
    "ground_control_points": [
        [0,    0,    0.0,   0.0],
        [1280, 0,    30.0,  0.0],
        [1280, 720,  30.0, -20.0],
        [0,    720,  0.0,  -20.0],
        [640,  360,  15.0, -10.0],
    ],

    "frame_width": 1280,
    "frame_height": 720,

    # Simplified bounding boxes in UTM-offset space (metres from origin).
    # These mirror the GeoAI zone layout in the satellite view.
    "zone_bounds": {
        "zone_excavation_01":     {"min_e":  0, "max_e":  8, "min_n": -6,  "max_n":  0},
        "zone_formwork_01":       {"min_e":  8, "max_e": 18, "min_n": -8,  "max_n":  0},
        "zone_concrete_pour_01":  {"min_e": -8, "max_e":  0, "min_n": -12, "max_n": -6},
        "zone_steel_erection_01": {"min_e":  0, "max_e": 12, "min_n":  0,  "max_n": 10},
    },

    "zone_names": {
        "zone_excavation_01":     "Excavation Zone",
        "zone_formwork_01":       "Formwork Zone",
        "zone_concrete_pour_01":  "Concrete Pour Zone",
        "zone_steel_erection_01": "Steel Erection Area",
    },
}


class PixelToWorldProjector:
    """
    Projects CCTV pixel coordinates to real-world geographic coordinates
    using a homographic transformation built from calibrated GCPs.
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or DEFAULT_CAMERA_CONFIG
        self._homography: Optional[np.ndarray] = None
        self._build_homography()

    def _build_homography(self) -> None:
        gcps = self.config["ground_control_points"]
        if len(gcps) < 4:
            return

        src = np.float32([[g[0], g[1]] for g in gcps]).reshape(-1, 1, 2)
        dst = np.float32([[g[2], g[3]] for g in gcps]).reshape(-1, 1, 2)
        H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        self._homography = H

    def pixel_to_world(
        self,
        pixel_x: float,
        pixel_y: float,
        frame_shape: tuple,
    ) -> dict:
        """
        Convert pixel (x, y) to geographic coordinates.

        Args:
            pixel_x: x coordinate in the CCTV frame
            pixel_y: y coordinate in the CCTV frame
            frame_shape: (height, width[, channels]) of the frame

        Returns:
            dict with lat, lng, utm_e, utm_n, zone_id, zone_name
        """
        fw = self.config["frame_width"]
        fh = self.config["frame_height"]
        scale_x = fw / frame_shape[1]
        scale_y = fh / frame_shape[0]
        sx = pixel_x * scale_x
        sy = pixel_y * scale_y

        if self._homography is not None:
            pt = np.float32([[sx, sy]]).reshape(-1, 1, 2)
            world = cv2.perspectiveTransform(pt, self._homography)
            utm_e_off = float(world[0][0][0])
            utm_n_off = float(world[0][0][1])
        else:
            utm_e_off = (sx / fw) * 30.0
            utm_n_off = -(sy / fh) * 20.0

        origin_lat = self.config["site_origin_lat"]
        origin_lng = self.config["site_origin_lng"]
        cos_lat = np.cos(np.radians(origin_lat))

        lat = origin_lat + (utm_n_off / 111139.0)
        lng = origin_lng + (utm_e_off / (111139.0 * cos_lat))

        zone_id, zone_name = self._containing_zone(utm_e_off, utm_n_off)

        return {
            "lat":           round(lat, 7),
            "lng":           round(lng, 7),
            "utm_e":         round(self.config.get("site_utm_e_base", 430000) + utm_e_off, 2),
            "utm_n":         round(self.config.get("site_utm_n_base", 1196500) + utm_n_off, 2),
            "utm_e_offset":  round(utm_e_off, 2),
            "utm_n_offset":  round(utm_n_off, 2),
            "zone_id":       zone_id,
            "zone_name":     zone_name,
        }

    def _containing_zone(self, utm_e_off: float, utm_n_off: float):
        bounds = self.config.get("zone_bounds", {})
        names  = self.config.get("zone_names", {})
        for zid, b in bounds.items():
            if b["min_e"] <= utm_e_off <= b["max_e"] and b["min_n"] <= utm_n_off <= b["max_n"]:
                return zid, names.get(zid, zid)
        return None, None
