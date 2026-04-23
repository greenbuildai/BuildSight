"""
BuildSight GeoAI — Production Pipeline Bridge
==============================================
Maran Constructions, Thanjavur | Green Build AI

PURPOSE:
  The "Heart" of the GeoAI system. This script:
  1. Pulls detections (YOLOv8/PPE).
  2. Maps pixels to GPS/World coordinates via SpatialMapper.
  3. Classifies detection into PostGIS zones (Scaffolding, Staircase, etc.).
  4. Runs BOCW safety logic (§38, §40, §41).
  5. Pushes to PostGIS & Broadcasts to Heatmap Engine.

USAGE:
  python geoai_pipeline.py --source rd4.mp4 --db
"""

import os
import sys
import cv2
import json
import logging
import time
import asyncio
import websockets
from datetime import datetime
from typing import List, Dict, Any, Optional
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
from dotenv import load_dotenv

# Path management
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(PROJECT_ROOT, "dashboard", "backend")
sys.path.append(BACKEND_DIR)

try:
    from geoai.utils.spatial_mapper import SpatialMapper       # type: ignore[import]
    from heatmap_engine import HeatmapEngine                   # type: ignore[import]
    from buildsight_ensemble import EnsemblePipeline           # type: ignore[import]
    from geoai.utils.intelligence import BuildSightIntelligence  # type: ignore[import]
except ImportError as e:
    print(f"Warning: Module imports failed: {e}. Ensure dashboard/backend is in path.")

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("BuildSight.GeoAI")

load_dotenv(os.path.join(PROJECT_ROOT, "dashboard", "backend", ".env"))

class GeoAIPipeline:
    def __init__(self, source_path: str, use_db: bool = False, use_ws: bool = False, model_mode: str = "ensemble"):
        self.source = source_path
        self.use_db = use_db
        self.use_ws = use_ws
        self.model_mode = model_mode
        self.camera_id = "CAM-01"
        self.ws_url = "ws://localhost:8765"
        self.ws_conn = None
        
        # Intelligence Layer (VLM / SAM)
        self.intel = BuildSightIntelligence(weights_dir=os.path.join(BACKEND_DIR, "weights"))
        self.last_narration_time = 0
        self.last_segmentation_time = 0
        self.latest_frame = None
        self.narration_interval = 17.0 # Seconds (staggered from SAM)
        self.segmentation_interval = 25.0 # Seconds
        self.is_spatial_expert_mode = False
        
        # Load Components
        self.mapper = SpatialMapper(
            homography_path=os.path.join(PROJECT_ROOT, f"camera_{self.camera_id.lower().replace('-','')}_H.npy")
        )
        self.heatmap = HeatmapEngine()
        
        if self.model_mode == "ensemble":
            logger.info("🧠 Initializing BuildSight Ensemble Pipeline...")
            self.detector = EnsemblePipeline()
        else:
            logger.info("⚠️ Ensemble mode not selected. Using placeholder detector.")
            self.detector = None
        
        if self.use_db:
            self._init_db()

    def _init_db(self):
        """Initialize PostgreSQL Connection."""
        try:
            self.db_conn = psycopg2.connect(
                host=os.getenv("POSTGRES_HOST", "localhost"),
                port=os.getenv("POSTGRES_PORT", "5432"),
                dbname=os.getenv("POSTGRES_DB", "buildsight_geoai"),
                user=os.getenv("POSTGRES_USER", "postgres"),
                password=os.getenv("POSTGRES_PASSWORD", "password")
            )
            logger.info("✅ Connected to PostGIS: buildsight_geoai")
        except Exception as e:
            logger.error(f"❌ DB Connection Failed: {e}")
            self.use_db = False

    def process_frame(self, frame_id: int, detections: List[Dict]):
        """
        Primary Bridge Logic:
        Pixel -> Spatial -> Zone -> DB
        """
        processed_detections = []
        timestamp = datetime.now()

        for det in detections:
            # 1. Coordinate Transform (Pixel -> Local Meters)
            # Support both internal formats (Ensemble results or raw box)
            if "box" in det:
                x1, y1, x2, y2 = det["box"]
                conf = det["confidence"]
                cls_name = det["class"]
                ppe_comp = det.get("has_helmet", True) and det.get("has_vest", True)
            else:
                x1, y1, x2, y2 = det['x1'], det['y1'], det['x2'], det['y2']
                conf = det['conf']
                cls_name = det['class']
                ppe_comp = det.get('ppe_compliant', True)

            px = (x1 + x2) / 2
            py = y2 # Use bottom of box for feet
            
            world_x, world_y = self.mapper.pixel_to_world(px, py)
            lat, lon = self.mapper.world_to_gps(world_x, world_y)
            
            # 2. Zone Classification & Risk Scoring
            zone_info = self._get_zone_at_point(world_x, world_y)
            
            p_det = {
                "id": f"{cls_name}_{int(time.time()*100)}",
                "camera_id": self.camera_id,
                "timestamp": timestamp.isoformat(),
                "class": cls_name,
                "conf": conf,
                "world_x": round(world_x, 3),
                "world_y": round(world_y, 3),
                "lat": lat,
                "lon": lon,
                "zone": zone_info['name'],
                "risk_level": zone_info['risk'],
                "has_helmet": det.get("has_helmet", ppe_comp),
                "has_vest": det.get("has_vest", ppe_comp),
                "ppe_compliant": ppe_comp
            }
            processed_detections.append(p_det)

        # 3. Update Heatmap
        self.heatmap.update(processed_detections)
        
        # 4. Persistence
        if self.use_db:
            self._save_to_db(processed_detections)
            
        return processed_detections

    def _get_zone_at_point(self, x: float, y: float) -> Dict:
        """Fallback zone logic if DB query is not used."""
        # Matches the risk_zones table names
        if 16.16 <= x <= 18.9: return {"name": "low_risk_parking", "risk": "LOW"}
        if 0 <= x <= 18.9 and 0 <= y <= 9.75: return {"name": "moderate_risk_interior", "risk": "MODERATE"}
        return {"name": "outside", "risk": "NONE"}

    def _save_to_db(self, detections: List[Dict]):
        """Push batch detections to PostGIS."""
        if not self.db_conn: return
        
        query = """
            INSERT INTO detections (
                camera_id, timestamp, class_name, confidence, 
                world_x, world_y, zone_name, has_helmet, has_vest, geom
            ) VALUES %s
        """
        
        data = [
            (
                d['camera_id'], d['timestamp'], d['class'], d['conf'],
                d['world_x'], d['world_y'], d['zone'], 
                d['has_helmet'], d['has_vest'],
                f"SRID=4326;POINTZ({d['lon']} {d['lat']} 0)"
            ) for d in detections
        ]
        
        try:
            with self.db_conn.cursor() as cur:
                from psycopg2.extras import execute_values
                execute_values(cur, query, data)
                self.db_conn.commit()
        except Exception as e:
            logger.error(f"❌ DB Write Error: {e}")
            self.db_conn.rollback()

    async def _push_to_websocket(self, detections: List[Dict]):
        """Transmit telemetry to GeoAI HUD using persistent connection."""
        if not self.use_ws: return
        
        try:
            if not self.ws_conn:
                self.ws_conn = await websockets.connect(self.ws_url, max_size=8 * 1024 * 1024)
            
            payload = {
                "type": "update_detections",
                "camera_id": self.camera_id,
                "detections": detections,
                "fps": detections[0].get("_fps", 0) if detections else 0,
                "latency_ms": detections[0].get("_latency", 0) if detections else 0,
                "scene_condition": detections[0].get("_condition", "S1_normal") if detections else "S1_normal"
            }
            await self.ws_conn.send(json.dumps(payload))
        except Exception as e:
            logger.warning(f"📡 WebSocket push failed: {e}. Reconnecting...")
            self.ws_conn = None 

    async def _narration_loop(self):
        """Background loop for VLM site narration via Florence-2 (geoai_vlm_util)."""
        logger.info("🎙️ Narration Intelligence Loop Started (Florence-2).")
        while True:
            try:
                if self.latest_frame is not None and (time.time() - self.last_narration_time > self.narration_interval):
                    logger.info("🧠 Generating Spatial Narration via Florence-2...")
                    
                    # Use geoai_vlm_util for single-instance VLM (no double VRAM load)
                    try:
                        from geoai_vlm_util import describe_frame_sync
                        result = describe_frame_sync(
                            frame_bgr=self.latest_frame,
                            question=(
                                "Describe this construction site layout, focus on safety hazards, "
                                "worker activity, and structural progress."
                            ),
                        )
                        narration = result.get("description", "Site activity under analysis.")
                        source = result.get("source", "rule_based")
                    except Exception as vlm_err:
                        logger.warning(f"⚠️ VLM narration failed ({vlm_err}), using intelligence fallback.")
                        narration = self.intel.narrate_frame(self.latest_frame)
                        source = "fallback"
                    
                    if self.use_ws and self.ws_conn:
                        payload = {
                            "type": "spatial_narration",
                            "text": narration,
                            "source": source,
                            "timestamp": datetime.now().isoformat()
                        }
                        await self.ws_conn.send(json.dumps(payload))
                        logger.info(f"🎙️ Narration Broadcasted ({source}): {narration[:50]}...")
                    
                    self.last_narration_time = time.time()
            except Exception as e:
                logger.error(f"⚠️ Narration loop error: {e}")
            
            await asyncio.sleep(1) # Check every second

    async def _segmentation_loop(self):
        """Background loop for SAM segmentation based on motion checks."""
        logger.info("🧩 Segmentation Loop Started.")
        prev_gray = None
        MOTION_THRESHOLD = 20
        first_frame_done = False
        
        while True:
            try:
                if self.latest_frame is not None:
                    current_gray = cv2.cvtColor(self.latest_frame, cv2.COLOR_BGR2GRAY)
                    
                    run_seg = False
                    if not first_frame_done:
                        run_seg = True
                        first_frame_done = True
                    elif time.time() - self.last_segmentation_time > self.segmentation_interval:
                        if prev_gray is not None:
                            frame_diff = cv2.absdiff(prev_gray, current_gray)
                            motion_score = np.mean(frame_diff)
                            if motion_score > MOTION_THRESHOLD:
                                run_seg = True
                                logger.info(f"motion_score {motion_score:.1f} > {MOTION_THRESHOLD}. Rerunning SAM.")
                    
                    if run_seg:
                        logger.info("🧠 Generating SAM Segmentation Polygons...")
                        polygons = self.intel.segment_frame(self.latest_frame)
                        
                        if self.use_ws and self.ws_conn and polygons:
                            payload = {
                                "type": "site_segmentation",
                                "polygons": polygons,
                                "timestamp": datetime.now().isoformat()
                            }
                            await self.ws_conn.send(json.dumps(payload))
                            logger.info(f"🧩 Sent {len(polygons)} polygons via WS.")
                        
                        self.last_segmentation_time = time.time()
                        prev_gray = current_gray
            except Exception as e:
                logger.error(f"⚠️ Segmentation loop error: {e}")
            
            await asyncio.sleep(2) # Check every 2 seconds

    async def run_live(self):
        """BuildSight GeoAI Live Pipeline entry point."""
        logger.info(f"🚀 Starting GeoAI Production Pipeline: {self.source}")
        
        # 1. Start Intelligence Tasks in background
        narration_task = asyncio.create_task(self._narration_loop())
        segmentation_task = asyncio.create_task(self._segmentation_loop())
        
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            logger.error(f"❌ Failed to open video source: {self.source}")
            narration_task.cancel()
            segmentation_task.cancel()
            return

        # 2. Run Pipeline Loop
        try:
            frame_idx = 0
            while True:
                # 1. Read frame
                ret, frame = cap.read()
                self.latest_frame = frame # Store for intelligence layer
                
                # Looping logic: reset if end of video reached
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                frame_idx += 1
                start_time = time.time()

                # 2-6. Ensemble Detection (Inference, Merge, PPE, Tracking)
                if self.detector:
                    results = self.detector.run(frame)
                    detections = results.get("detections", [])
                else:
                    detections = []

                # 7-8. Spatial Mapping & Zone Assignment
                processed_data = self.process_frame(frame_idx, detections)
                
                # Performance metrics
                latency_ms = (time.time() - start_time) * 1000
                fps = 1.0 / (time.time() - start_time)
                
                # Tag telemetry metadata into the first detection object for the WS pusher to extract
                if processed_data:
                    processed_data[0]["_fps"] = round(fps, 1)
                    processed_data[0]["_latency"] = int(latency_ms)
                    processed_data[0]["_condition"] = "S1_normal" # Placeholder for future dynamic classification
                
                # 9. Save to DB
                if self.use_db:
                    self._save_to_db(processed_data)
                
                # 10. Push to WebSocket
                if self.use_ws:
                    await self._push_to_websocket(processed_data)

                # Performance reporting
                fps = 1.0 / (time.time() - start_time)
                if frame_idx % 30 == 0:
                    logger.info(f"Pipeline Active | Frame: {frame_idx} | FPS: {fps:.2f} | Workers: {len([d for d in processed_data if d['class'] == 'worker'])}")

                # Controlled execution speed
                await asyncio.sleep(0.01)
        finally:
            narration_task.cancel()
            segmentation_task.cancel()
            cap.release()

    def run_demo(self):
        """Run a test loop using mock data."""
        logger.info(f"🚀 Starting GeoAI Bridge Demo for {self.source}")
        
        mock_detections = [
            {"box": [500, 400, 600, 800], "class": "worker", "confidence": 0.92, "has_helmet": False, "has_vest": True},
            {"box": [1200, 500, 1300, 900], "class": "worker", "confidence": 0.88, "has_helmet": True, "has_vest": True}
        ]
        
        for i in range(5):
            logger.info(f"Processing Frame {i}...")
            # Manual conversion of mock format to internal process_frame format
            formatted = []
            for m in mock_detections:
                formatted.append({
                    "x1": m["box"][0], "y1": m["box"][1], 
                    "x2": m["box"][2], "y2": m["box"][3],
                    "class": m["class"], "conf": m["confidence"],
                    "ppe_compliant": m["has_helmet"] and m["has_vest"]
                })
            results = self.process_frame(i, formatted)
            for r in results:
                logger.info(f"  -> Worker at ({r['world_x']}m, {r['world_y']}m) | Zone: {r['zone']} | Risk: {r['risk_level']}")
            time.sleep(1)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=r"E:\Company\Green Build AI\Prototypes\BuildSight\buildsight-base\gis\inputs\rd4.mp4")
    parser.add_argument("--db", action="store_true")
    parser.add_argument("--ws", action="store_true")
    parser.add_argument("--model-mode", default="ensemble")
    args = parser.parse_args()
    
    pipeline = GeoAIPipeline(args.source, use_db=args.db, use_ws=args.ws, model_mode=args.model_mode)
    
    if args.ws:
        asyncio.run(pipeline.run_live())
    else:
        pipeline.run_demo()
