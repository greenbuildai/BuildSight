"""
BuildSight — Adaptive Multi-Worker Detection Pipeline
=====================================================
Automatically switches between single-pass and two-stage detection
based on worker density. When multiple workers are clustered together,
the system crops each worker region and runs PPE detection per-crop
to eliminate cross-worker PPE misassociation.

Architecture:
  ┌─────────────────────────────────────────────────────┐
  │              Frame Input                            │
  │                  │                                  │
  │      ┌───────────▼────────────┐                     │
  │      │  Stage 0: Worker Count │                     │
  │      │  & Proximity Analysis  │                     │
  │      └───────────┬────────────┘                     │
  │           ┌──────┴──────┐                           │
  │     Sparse│             │Crowded                    │
  │      (<3 workers or     │(≥3 workers AND            │
  │       no overlap)       │ proximity < threshold)    │
  │           │             │                           │
  │   ┌───────▼──────┐  ┌──▼──────────────────────┐    │
  │   │ Single-Pass  │  │ Two-Stage Detection     │    │
  │   │ YOLOv8 Full  │  │ Stage 1: Worker detect  │    │
  │   │ Frame Detect │  │ Stage 2: Crop+PPE per   │    │
  │   └───────┬──────┘  │         worker          │    │
  │           │         └──────────┬──────────────┘    │
  │           └────────┬───────────┘                    │
  │                    ▼                                │
  │         Unified Results + Per-Worker PPE Map        │
  └─────────────────────────────────────────────────────┘

Author: BuildSight / Green Build AI
Target: YOLOv8n PPE Detection for Construction Sites
"""

import cv2
import numpy as np
from ultralytics import YOLO
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from enum import Enum
import time
import logging

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("BuildSight.MultiWorker")


class DetectionMode(Enum):
    SINGLE_PASS = "single_pass"       # Normal: full-frame detection
    MULTI_WORKER = "multi_worker"     # Crowded: two-stage crop+detect


@dataclass
class Detection:
    """Single detection result with worker-PPE association."""
    bbox: Tuple[int, int, int, int]   # (x1, y1, x2, y2)
    class_id: int
    class_name: str
    confidence: float
    worker_id: Optional[int] = None   # Which worker this PPE belongs to


@dataclass
class WorkerProfile:
    """Complete PPE profile for a single worker."""
    worker_id: int
    worker_bbox: Tuple[int, int, int, int]
    detections: List[Detection] = field(default_factory=list)
    has_helmet: bool = False
    has_vest: bool = False
    has_gloves: bool = False
    has_boots: bool = False
    compliance_score: float = 0.0

    def compute_compliance(self):
        """Compute PPE compliance based on detected items."""
        required = {'helmet': self.has_helmet, 'safety_vest': self.has_vest}
        met = sum(1 for v in required.values() if v)
        self.compliance_score = met / len(required) if required else 0.0


@dataclass
class FrameResult:
    """Complete detection result for one frame."""
    mode_used: DetectionMode
    worker_count: int
    worker_profiles: List[WorkerProfile]
    all_detections: List[Detection]
    crowding_score: float              # 0.0 = isolated, 1.0 = highly crowded
    processing_time_ms: float
    frame_shape: Tuple[int, int, int]


# ─────────────────────────────────────────────────────────────
# Core: Crowd Analyzer
# ─────────────────────────────────────────────────────────────

class CrowdAnalyzer:
    """
    Analyzes worker spatial distribution to decide detection mode.

    Triggers multi-worker mode when:
      1. Worker count ≥ min_workers_for_crowd  (default: 3)
      2. Average pairwise centroid distance < proximity_threshold_px
         OR any pair has IoU > iou_overlap_threshold
    """

    def __init__(
        self,
        min_workers_for_crowd: int = 3,
        proximity_threshold_px: int = 150,
        iou_overlap_threshold: float = 0.15,
    ):
        self.min_workers = min_workers_for_crowd
        self.prox_thresh = proximity_threshold_px
        self.iou_thresh = iou_overlap_threshold

    def analyze(self, worker_boxes: List[Tuple[int, int, int, int]]) -> Tuple[bool, float]:
        """
        Returns (should_use_multi_worker: bool, crowding_score: float).

        crowding_score ∈ [0, 1]:
          0 = no workers or fully isolated
          1 = extremely crowded / heavily overlapping
        """
        n = len(worker_boxes)
        if n < self.min_workers:
            return False, 0.0

        # Compute pairwise centroid distances and IoUs
        centroids = []
        for (x1, y1, x2, y2) in worker_boxes:
            centroids.append(((x1 + x2) / 2, (y1 + y2) / 2))

        distances = []
        max_iou = 0.0
        overlap_pairs = 0

        for i in range(n):
            for j in range(i + 1, n):
                # Centroid distance
                dx = centroids[i][0] - centroids[j][0]
                dy = centroids[i][1] - centroids[j][1]
                dist = (dx**2 + dy**2) ** 0.5
                distances.append(dist)

                # IoU
                iou = self._compute_iou(worker_boxes[i], worker_boxes[j])
                max_iou = max(max_iou, iou)
                if iou > self.iou_thresh:
                    overlap_pairs += 1

        avg_distance = np.mean(distances) if distances else float('inf')
        total_pairs = n * (n - 1) / 2

        # Crowding score: blend of proximity + overlap metrics
        proximity_score = max(0.0, 1.0 - (avg_distance / (self.prox_thresh * 3)))
        overlap_score = overlap_pairs / total_pairs if total_pairs > 0 else 0.0
        crowding_score = min(1.0, 0.6 * proximity_score + 0.4 * overlap_score)

        # Decision: trigger multi-worker if proximity OR overlap exceeds threshold
        is_crowded = (avg_distance < self.prox_thresh) or (max_iou > self.iou_thresh)

        return is_crowded, crowding_score

    @staticmethod
    def _compute_iou(box_a, box_b) -> float:
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0


# ─────────────────────────────────────────────────────────────
# Core: Adaptive Multi-Worker Detector
# ─────────────────────────────────────────────────────────────

class AdaptiveMultiWorkerDetector:
    """
    Main detection pipeline that automatically switches between:
      - Single-pass: standard YOLOv8 full-frame detection
      - Multi-worker: two-stage crop-and-classify per worker

    Usage:
        detector = AdaptiveMultiWorkerDetector(
            model_path="best.pt",  # your trained YOLOv8 model
        )
        result = detector.detect(frame)

        for wp in result.worker_profiles:
            print(f"Worker {wp.worker_id}: helmet={wp.has_helmet}, vest={wp.has_vest}")
    """

    # ── Class name mapping (adjust to match YOUR trained model) ──
    # Common PPE dataset classes:
    PPE_CLASSES = {
        'helmet': 'helmet',
        'hard-hat': 'helmet',
        'head': 'helmet',
        'safety_vest': 'safety_vest',
        'vest': 'safety_vest',
        'safety vest': 'safety_vest',
        'safety_vest': 'safety_vest',
        'worker': 'worker',
        'person': 'worker',
        'gloves': 'gloves',
        'boots': 'boots',
        'no-helmet': 'no_helmet',
        'no_helmet': 'no_helmet',
        'no-vest': 'no_vest',
        'no_vest': 'no_vest',
        'no-safety vest': 'no_vest',
    }

    WORKER_CLASSES = {'worker', 'person'}
    PPE_ITEM_CLASSES = {'helmet', 'hard-hat', 'head', 'safety_vest', 'vest',
                        'safety vest', 'safety_vest', 'gloves', 'boots',
                        'no-helmet', 'no_helmet', 'no-vest', 'no_vest',
                        'no-safety vest'}

    def __init__(
        self,
        model_path: str = "best.pt",
        conf_threshold: float = 0.35,
        crowd_conf_threshold: float = 0.25,       # Lower threshold for crowded crops
        iou_threshold: float = 0.45,
        crowd_iou_threshold: float = 0.55,         # Higher NMS IoU for crowded (less suppression)
        crop_padding_ratio: float = 0.25,           # Expand crop by 25% around worker
        device: str = "",                           # "" = auto (GPU if available)
        img_size: int = 640,
        crowd_crop_size: int = 320,                 # Resize crops to this for Stage 2
    ):
        self.model = YOLO(model_path)
        self.conf = conf_threshold
        self.crowd_conf = crowd_conf_threshold
        self.iou = iou_threshold
        self.crowd_iou = crowd_iou_threshold
        self.padding = crop_padding_ratio
        self.device = device
        self.img_size = img_size
        self.crowd_crop_size = crowd_crop_size

        self.crowd_analyzer = CrowdAnalyzer()

        # Extract class names from model
        self.class_names = self.model.names  # {0: 'helmet', 1: 'worker', ...}
        logger.info(f"Model loaded: {model_path}")
        logger.info(f"Classes: {self.class_names}")

    # ───────── PUBLIC API ─────────

    def detect(self, frame: np.ndarray) -> FrameResult:
        """
        Run adaptive detection on a single frame.

        Returns FrameResult with per-worker PPE profiles.
        """
        t0 = time.perf_counter()
        h, w = frame.shape[:2]

        # ── Stage 0: Quick worker-only pass to assess crowding ──
        worker_boxes, worker_confs = self._detect_workers(frame)

        is_crowded, crowding_score = self.crowd_analyzer.analyze(worker_boxes)
        mode = DetectionMode.MULTI_WORKER if is_crowded else DetectionMode.SINGLE_PASS

        logger.info(
            f"Workers={len(worker_boxes)} | Crowding={crowding_score:.2f} | Mode={mode.value}"
        )

        # ── Route to appropriate pipeline ──
        if mode == DetectionMode.SINGLE_PASS:
            all_detections = self._single_pass_detect(frame)
            worker_profiles = self._associate_ppe_spatial(all_detections, worker_boxes)
        else:
            all_detections, worker_profiles = self._multi_worker_detect(
                frame, worker_boxes, worker_confs
            )

        # ── Compile result ──
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return FrameResult(
            mode_used=mode,
            worker_count=len(worker_boxes),
            worker_profiles=worker_profiles,
            all_detections=all_detections,
            crowding_score=crowding_score,
            processing_time_ms=elapsed_ms,
            frame_shape=frame.shape,
        )

    # ───────── STAGE 0: Worker Detection ─────────

    def _detect_workers(self, frame: np.ndarray) -> Tuple[List, List]:
        """Quick pass: detect only worker/person class."""
        results = self.model.predict(
            frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.img_size,
            device=self.device,
            verbose=False,
        )

        worker_boxes = []
        worker_confs = []

        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0])
                cls_name = self.class_names.get(cls_id, "unknown")
                if cls_name.lower() in self.WORKER_CLASSES:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    worker_boxes.append((x1, y1, x2, y2))
                    worker_confs.append(float(box.conf[0]))

        return worker_boxes, worker_confs

    # ───────── MODE A: Single-Pass Detection ─────────

    def _single_pass_detect(self, frame: np.ndarray) -> List[Detection]:
        """Standard full-frame YOLOv8 detection."""
        results = self.model.predict(
            frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.img_size,
            device=self.device,
            verbose=False,
        )

        detections = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0])
                cls_name = self.class_names.get(cls_id, "unknown")
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                detections.append(Detection(
                    bbox=(x1, y1, x2, y2),
                    class_id=cls_id,
                    class_name=cls_name,
                    confidence=float(box.conf[0]),
                ))

        return detections

    # ───────── MODE B: Multi-Worker Two-Stage Detection ─────────

    def _multi_worker_detect(
        self,
        frame: np.ndarray,
        worker_boxes: List[Tuple[int, int, int, int]],
        worker_confs: List[float],
    ) -> Tuple[List[Detection], List[WorkerProfile]]:
        """
        Two-stage detection:
          Stage 1: Use pre-detected worker boxes (already done in Stage 0)
          Stage 2: Crop each worker region → run PPE detection on the crop
                   → map detections back to original frame coordinates
        """
        h, w = frame.shape[:2]
        all_detections = []
        worker_profiles = []

        for wid, (wb, wc) in enumerate(zip(worker_boxes, worker_confs)):
            # ── Create padded crop region ──
            crop_box = self._expand_box(wb, w, h, self.padding)
            cx1, cy1, cx2, cy2 = crop_box
            crop = frame[cy1:cy2, cx1:cx2]

            if crop.size == 0:
                continue

            # ── Stage 2: Run detection on crop with relaxed thresholds ──
            crop_results = self.model.predict(
                crop,
                conf=self.crowd_conf,          # Lower confidence → catch more PPE
                iou=self.crowd_iou,            # Higher NMS IoU → less suppression
                imgsz=self.crowd_crop_size,
                device=self.device,
                verbose=False,
            )

            # ── Build worker profile ──
            profile = WorkerProfile(
                worker_id=wid,
                worker_bbox=wb,
            )

            # Add the worker detection itself
            worker_det = Detection(
                bbox=wb,
                class_id=-1,
                class_name="worker",
                confidence=wc,
                worker_id=wid,
            )
            all_detections.append(worker_det)
            profile.detections.append(worker_det)

            # ── Map crop detections back to frame coordinates ──
            for r in crop_results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = self.class_names.get(cls_id, "unknown")

                    # Skip worker detections within crop (we already have them)
                    if cls_name.lower() in self.WORKER_CLASSES:
                        continue

                    # Map crop coords → frame coords
                    bx1, by1, bx2, by2 = map(int, box.xyxy[0].tolist())
                    frame_x1 = cx1 + bx1
                    frame_y1 = cy1 + by1
                    frame_x2 = cx1 + bx2
                    frame_y2 = cy1 + by2

                    det = Detection(
                        bbox=(frame_x1, frame_y1, frame_x2, frame_y2),
                        class_id=cls_id,
                        class_name=cls_name,
                        confidence=float(box.conf[0]),
                        worker_id=wid,
                    )
                    all_detections.append(det)
                    profile.detections.append(det)

                    # ── Update PPE flags ──
                    normalized = self.PPE_CLASSES.get(cls_name.lower(), cls_name.lower())
                    if normalized == 'helmet':
                        profile.has_helmet = True
                    elif normalized == 'safety_vest':
                        profile.has_vest = True
                    elif normalized == 'gloves':
                        profile.has_gloves = True
                    elif normalized == 'boots':
                        profile.has_boots = True

            profile.compute_compliance()
            worker_profiles.append(profile)

        return all_detections, worker_profiles

    # ───────── Spatial PPE Association (for single-pass mode) ─────────

    def _associate_ppe_spatial(
        self,
        detections: List[Detection],
        worker_boxes: List[Tuple[int, int, int, int]],
    ) -> List[WorkerProfile]:
        """
        Associate detected PPE items with nearest worker using spatial proximity.
        Used only in SINGLE_PASS mode.
        """
        profiles = []
        for wid, wb in enumerate(worker_boxes):
            profile = WorkerProfile(worker_id=wid, worker_bbox=wb)
            profiles.append(profile)

        # Separate PPE detections from worker detections
        ppe_dets = [d for d in detections if d.class_name.lower() not in self.WORKER_CLASSES]

        for det in ppe_dets:
            # Find which worker's bounding box contains or is closest to this PPE
            best_worker = self._find_nearest_worker(det.bbox, worker_boxes)
            if best_worker is not None:
                det.worker_id = best_worker
                profiles[best_worker].detections.append(det)

                normalized = self.PPE_CLASSES.get(det.class_name.lower(), det.class_name.lower())
                if normalized == 'helmet':
                    profiles[best_worker].has_helmet = True
                elif normalized == 'safety_vest':
                    profiles[best_worker].has_vest = True
                elif normalized == 'gloves':
                    profiles[best_worker].has_gloves = True
                elif normalized == 'boots':
                    profiles[best_worker].has_boots = True

        for p in profiles:
            p.compute_compliance()

        return profiles

    def _find_nearest_worker(
        self, ppe_box: Tuple[int, int, int, int],
        worker_boxes: List[Tuple[int, int, int, int]],
    ) -> Optional[int]:
        """Find which worker a PPE item belongs to (containment first, then distance)."""
        ppe_cx = (ppe_box[0] + ppe_box[2]) / 2
        ppe_cy = (ppe_box[1] + ppe_box[3]) / 2

        # First: check containment (PPE centroid inside worker box)
        for i, wb in enumerate(worker_boxes):
            if wb[0] <= ppe_cx <= wb[2] and wb[1] <= ppe_cy <= wb[3]:
                return i

        # Fallback: nearest centroid
        best_dist = float('inf')
        best_idx = None
        for i, wb in enumerate(worker_boxes):
            wcx = (wb[0] + wb[2]) / 2
            wcy = (wb[1] + wb[3]) / 2
            dist = ((ppe_cx - wcx)**2 + (ppe_cy - wcy)**2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_idx = i

        # Only associate if within reasonable distance (1.5x worker box diagonal)
        if best_idx is not None:
            wb = worker_boxes[best_idx]
            diag = ((wb[2] - wb[0])**2 + (wb[3] - wb[1])**2) ** 0.5
            if best_dist < diag * 1.5:
                return best_idx

        return None

    # ───────── Utility ─────────

    @staticmethod
    def _expand_box(
        box: Tuple[int, int, int, int],
        frame_w: int, frame_h: int,
        padding_ratio: float,
    ) -> Tuple[int, int, int, int]:
        """Expand a bounding box by padding_ratio, clamped to frame bounds."""
        x1, y1, x2, y2 = box
        bw = x2 - x1
        bh = y2 - y1
        px = int(bw * padding_ratio)
        py = int(bh * padding_ratio)
        return (
            max(0, x1 - px),
            max(0, y1 - py),
            min(frame_w, x2 + px),
            min(frame_h, y2 + py),
        )


# ─────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────

class DetectionVisualizer:
    """Draw results with per-worker color coding and PPE status badges."""

    # Distinct colors per worker (BGR)
    WORKER_COLORS = [
        (0, 255, 0),     # Green
        (255, 128, 0),   # Blue-ish
        (0, 128, 255),   # Orange
        (255, 0, 255),   # Magenta
        (255, 255, 0),   # Cyan
        (0, 255, 255),   # Yellow
        (128, 0, 255),   # Purple
        (255, 0, 128),   # Pink
    ]

    PPE_COLOR_OK = (0, 200, 0)       # Green
    PPE_COLOR_MISSING = (0, 0, 220)  # Red

    @classmethod
    def draw(cls, frame: np.ndarray, result: FrameResult) -> np.ndarray:
        """Draw all detections with worker-PPE color coding."""
        vis = frame.copy()

        # ── Draw per-worker profiles ──
        for wp in result.worker_profiles:
            color = cls.WORKER_COLORS[wp.worker_id % len(cls.WORKER_COLORS)]
            x1, y1, x2, y2 = wp.worker_bbox

            # Worker bounding box (thick)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 3)

            # Worker ID + compliance label
            compliance_pct = int(wp.compliance_score * 100)
            label = f"W{wp.worker_id} [{compliance_pct}%]"
            cls._draw_label(vis, label, (x1, y1 - 10), color)

            # PPE status badges (top-right of worker box)
            badges = []
            if wp.has_helmet:
                badges.append(("H", cls.PPE_COLOR_OK))
            else:
                badges.append(("H", cls.PPE_COLOR_MISSING))
            if wp.has_vest:
                badges.append(("V", cls.PPE_COLOR_OK))
            else:
                badges.append(("V", cls.PPE_COLOR_MISSING))

            badge_x = x2 + 5
            for i, (badge_text, badge_color) in enumerate(badges):
                by = y1 + i * 25
                cv2.rectangle(vis, (badge_x, by), (badge_x + 25, by + 22), badge_color, -1)
                cv2.putText(vis, badge_text, (badge_x + 4, by + 17),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Draw PPE detections associated with this worker
            for det in wp.detections:
                if det.class_name.lower() in AdaptiveMultiWorkerDetector.WORKER_CLASSES:
                    continue
                dx1, dy1, dx2, dy2 = det.bbox
                cv2.rectangle(vis, (dx1, dy1), (dx2, dy2), color, 2)
                ppe_label = f"{det.class_name} {det.confidence:.2f}"
                cls._draw_label(vis, ppe_label, (dx1, dy1 - 5), color, font_scale=0.45)

        # ── HUD: Mode indicator ──
        mode_text = f"Mode: {result.mode_used.value.upper()}"
        crowd_text = f"Crowd: {result.crowding_score:.2f}"
        fps_text = f"{result.processing_time_ms:.0f}ms"

        hud_color = (0, 0, 200) if result.mode_used == DetectionMode.MULTI_WORKER else (0, 180, 0)
        cv2.rectangle(vis, (10, 10), (280, 95), (0, 0, 0), -1)
        cv2.rectangle(vis, (10, 10), (280, 95), hud_color, 2)
        cv2.putText(vis, mode_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, hud_color, 2)
        cv2.putText(vis, crowd_text, (20, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(vis, fps_text, (20, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(vis, f"Workers: {result.worker_count}", (150, 58),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        return vis

    @staticmethod
    def _draw_label(img, text, org, color, font_scale=0.55, thickness=2):
        """Draw text with background rectangle."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        x, y = org
        cv2.rectangle(img, (x, y - th - 8), (x + tw + 4, y + 2), color, -1)
        cv2.putText(img, text, (x + 2, y - 4), font, font_scale, (255, 255, 255), thickness)


# ─────────────────────────────────────────────────────────────
# Runner: Process image / video / webcam
# ─────────────────────────────────────────────────────────────

def process_image(
    image_path: str,
    model_path: str = "best.pt",
    output_path: str = "output_multi_worker.jpg",
    show: bool = False,
):
    """Process a single image through the adaptive pipeline."""
    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    detector = AdaptiveMultiWorkerDetector(model_path=model_path)
    result = detector.detect(frame)

    # Print per-worker report
    print("\n" + "=" * 60)
    print(f"  BuildSight Multi-Worker Detection Report")
    print(f"  Mode: {result.mode_used.value} | Workers: {result.worker_count}")
    print(f"  Crowding Score: {result.crowding_score:.2f}")
    print(f"  Processing Time: {result.processing_time_ms:.1f} ms")
    print("=" * 60)

    for wp in result.worker_profiles:
        status = "COMPLIANT" if wp.compliance_score >= 1.0 else "NON-COMPLIANT"
        print(f"\n  Worker {wp.worker_id} — {status} ({wp.compliance_score*100:.0f}%)")
        print(f"    Helmet: {'YES' if wp.has_helmet else 'MISSING'}")
        print(f"    Vest:   {'YES' if wp.has_vest else 'MISSING'}")
        ppe_dets = [d for d in wp.detections if d.class_name not in ('worker', 'person')]
        for d in ppe_dets:
            print(f"    → {d.class_name}: {d.confidence:.2f}")

    print("\n" + "=" * 60)

    # Visualize
    vis = DetectionVisualizer.draw(frame, result)
    cv2.imwrite(output_path, vis)
    print(f"\n  Output saved: {output_path}")

    if show:
        cv2.imshow("BuildSight Multi-Worker Detection", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return result


def process_video(
    source,  # video path, RTSP URL, or 0 for webcam
    model_path: str = "best.pt",
    output_path: Optional[str] = None,
    show: bool = True,
):
    """Process video stream with adaptive detection."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")

    detector = AdaptiveMultiWorkerDetector(model_path=model_path)

    # Video writer setup
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result = detector.detect(frame)
            vis = DetectionVisualizer.draw(frame, result)

            if writer:
                writer.write(vis)
            if show:
                cv2.imshow("BuildSight Multi-Worker Detection", vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_count += 1
            if frame_count % 30 == 0:
                logger.info(
                    f"Frame {frame_count}: {result.mode_used.value} | "
                    f"Workers={result.worker_count} | "
                    f"{result.processing_time_ms:.0f}ms"
                )

    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        logger.info(f"Processed {frame_count} frames")


# ─────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BuildSight Adaptive Multi-Worker Detection")
    parser.add_argument("--source", type=str, required=True,
                       help="Image path, video path, RTSP URL, or '0' for webcam")
    parser.add_argument("--model", type=str, default="best.pt",
                       help="Path to trained YOLOv8 model weights")
    parser.add_argument("--output", type=str, default=None,
                       help="Output path for annotated result")
    parser.add_argument("--show", action="store_true",
                       help="Display result in window")

    args = parser.parse_args()

    source = args.source

    # Detect if source is image or video
    if source.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
        out = args.output or "output_multi_worker.jpg"
        process_image(source, model_path=args.model, output_path=out, show=args.show)
    else:
        if source == '0':
            source = 0
        out = args.output or "output_multi_worker.mp4"
        process_video(source, model_path=args.model, output_path=out, show=args.show)
