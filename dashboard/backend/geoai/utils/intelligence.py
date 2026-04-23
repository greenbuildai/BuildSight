"""
BuildSight GeoAI — Intelligence Layer
======================================
Handles Florence-2 (via geoai_vlm_util) for site narration and SAM for segmentation.

NOTE (2026-04-22): The internal Moondream2 VLM loading has been REMOVED.
All VLM narration is now delegated to `geoai_vlm_util.describe_frame_sync()`
which uses Florence-2-base. This prevents the "Double VLM Load" crash that
occurred when both intelligence.py and geoai_vlm_util.py tried to load
separate VLM models simultaneously, exhausting VRAM.
"""

import os
import torch
import cv2
import numpy as np
import logging
from PIL import Image
from typing import Dict, List, Optional, Any

# Configure Logger
logger = logging.getLogger("BuildSight.Intelligence")


class BuildSightIntelligence:
    """
    High-end AI Layer for BuildSight GeoAI.
    Handles Florence-2 (VLM) for site narration via geoai_vlm_util
    and SAM for segmentation.
    """
    
    def __init__(self, weights_dir: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.weights_dir = weights_dir
        
        # SAM Configuration
        self.sam_path = os.path.join(weights_dir, "sam_vit_b.pth")
        self.sam_model = None
        self.sam_predictor = None
        
        # VLM is now handled by geoai_vlm_util (Florence-2-base)
        # No internal VLM loading — prevents double VRAM allocation
        self._vlm_util_available = False
        try:
            import geoai_vlm_util
            self._vlm_util_available = True
            logger.info("🧠 VLM Intelligence: Delegated to geoai_vlm_util (Florence-2-base)")
        except ImportError:
            logger.warning("⚠️ geoai_vlm_util not importable — VLM narration disabled")
        
        # SAM will be lazy-loaded to save VRAM unless needed
        
    def _load_sam(self):
        """Lazy load SAM."""
        if self.sam_predictor:
            return
            
        if not os.path.exists(self.sam_path):
            logger.warning(f"⚠️ SAM weights not found at {self.sam_path}. Segmentation disabled.")
            return
            
        try:
            from segment_anything import sam_model_registry, SamPredictor
            logger.info(f"🧠 Loading SAM (vit_b) on {self.device}...")
            
            self.sam_model = sam_model_registry["vit_b"](checkpoint=self.sam_path)
            self.sam_model.to(self.device)
            self.sam_predictor = SamPredictor(self.sam_model)
            logger.info("✅ SAM Loaded Successfully.")
        except Exception as e:
            logger.error(f"❌ Failed to load SAM: {e}")

    def segment_frame(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """
        Generate semantic segmentation polygons for the site.
        Extracts masks using SAM, converts to contours, and simplifies into polygons.
        """
        self._load_sam()
        if not self.sam_predictor:
            return []
            
        try:
            logger.info("🧠 Running SAM Segmentation...")
            # Convert to RGB for SAM
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            self.sam_predictor.set_image(frame_rgb)
            
            # Predict roughly across the center area to get key structural masks
            # For a real site, we could use a grid of points or bounding boxes
            # Here, we use a few placeholder center points for 'site boundaries'
            h, w = frame_rgb.shape[:2]
            input_point = np.array([[w//2, h//2], [w//3, h//3], [2*w//3, 2*h//3]])
            input_label = np.array([1, 1, 1])
            
            masks, scores, _ = self.sam_predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
            )
            
            # Use the mask with the highest score
            best_mask_idx = np.argmax(scores)
            mask = masks[best_mask_idx].astype(np.uint8) * 255
            
            # Convert mask to contours and then simplified polygons
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            polygons = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 1000: # Filter small noise
                    # Simplify the polygon
                    epsilon = 0.01 * cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, epsilon, True)
                    
                    # Convert to flat list or list of [x, y] format appropriate for frontend
                    poly_points = approx.reshape(-1, 2).tolist()
                    polygons.append({
                        "name": "Segmented Region",
                        "score": float(scores[best_mask_idx]),
                        "geometry": poly_points
                    })
            
            return polygons
            
        except Exception as e:
            logger.error(f"⚠️ Segmentation error: {e}")
            return []
        finally:
            # VRAM Safeguard: Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def narrate_frame(self, frame: np.ndarray) -> str:
        """
        Generate a spatial narration of the current frame using Florence-2
        via geoai_vlm_util (single VLM instance shared across the app).
        """
        if not self._vlm_util_available:
            return "VLM Intelligence Off — geoai_vlm_util not available"
            
        try:
            import geoai_vlm_util
            
            result = geoai_vlm_util.describe_frame_sync(
                frame_bgr=frame,
                question=(
                    "Describe this construction site layout, focus on safety hazards, "
                    "worker activity, and structural progress. Keep it professional and concise."
                ),
            )
            
            description = result.get("description", "")
            source = result.get("source", "rule_based")
            
            if description:
                logger.debug(f"🎙️ Narration via {source}: {description[:60]}...")
                return description
            else:
                return "Site activity observed. Processing vision telemetry..."
                
        except Exception as e:
            logger.error(f"⚠️ Narration error: {e}")
            return "Narration stream interrupted."
