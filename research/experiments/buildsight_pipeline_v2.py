"""
BuildSight — Complete PPE Pipeline v2 (5-Stage)
================================================
Fixes ALL detected issues across Normal, Dusty, Crowded scenarios:

  1. Duplicate vest fragments      → Cluster-Merge NMS
  2. Partial vest detections       → Coverage Validation
  3. Cross-worker vest bleeding    → Torso-Center Alignment
  4. Duplicate worker boxes        → Worker Dedup NMS        ← NEW
  5. False PPE (head→helmet,       → PPE Authenticity Check  ← NEW
     shirt→vest)

Problem Analysis for v2 fixes:
  
  DUPLICATE WORKERS (Images 1 & 2):
    - Same worker detected 2x with slightly different boxes
    - Old pipeline skipped worker merging to avoid merging different people
    - Fix: Apply IoU-based dedup specifically for workers with higher threshold (0.40)
  
  FALSE HELMET (head detected as helmet):
    - Model sees round head shape and predicts "helmet"
    - Real helmets are ABOVE the head, have distinct color (yellow/white/orange)
    - Fix: Validate helmet position (must be at top of worker) + size ratio check
  
  FALSE VEST (regular shirt detected as safety vest):
    - Model sees colored clothing and predicts "safety_vest"  
    - Real safety vests have high-visibility fluorescent colors + reflective properties
    - Fix: HSV color validation — real vests have saturated fluorescent yellow/orange/green
           + size ratio validation (vest covers specific torso proportion)

Author: BuildSight / Green Build AI
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("BuildSight.v2")


# ─────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────

@dataclass
class Detection:
    bbox: Tuple[float, float, float, float]
    class_id: int
    class_name: str
    confidence: float
    worker_id: Optional[int] = None
    merged_from: int = 1
    authentic: bool = True      # Set to False if fails authenticity check
    rejection_reason: str = ""


@dataclass
class WorkerPPE:
    worker_id: int
    worker_box: Tuple[float, float, float, float]
    helmet: Optional[Detection] = None
    vest: Optional[Detection] = None

    @property
    def has_helmet(self):
        return self.helmet is not None

    @property
    def has_vest(self):
        return self.vest is not None

    @property
    def compliance(self):
        return (int(self.has_helmet) + int(self.has_vest)) / 2.0


# ─────────────────────────────────────────────────────────────
# Stage 1: Worker Dedup NMS (NEW)
# ─────────────────────────────────────────────────────────────

class WorkerDedupNMS:
    """
    Deduplicate worker detections specifically.

    Why separate from PPE merge:
      - Workers need HIGHER IoU threshold (0.40) to avoid merging
        two people standing close together
      - PPE fragments need LOWER threshold (0.10) because fragments
        of the same vest barely overlap
      - Different merge strategy: for workers, keep highest confidence;
        for PPE, take union box

    Handles the "same worker detected twice" problem in dusty/low-vis conditions.
    """

    def __init__(
        self,
        worker_iou_threshold: float = 0.40,
        worker_containment_threshold: float = 0.65,
    ):
        self.iou_thresh = worker_iou_threshold
        self.containment_thresh = worker_containment_threshold

    def dedup(self, detections: List[Detection], worker_classes: set) -> List[Detection]:
        """Deduplicate worker detections, pass through everything else."""
        workers = [d for d in detections if d.class_name.lower() in worker_classes]
        non_workers = [d for d in detections if d.class_name.lower() not in worker_classes]

        if len(workers) <= 1:
            return detections

        # Sort by confidence (highest first)
        workers.sort(key=lambda d: d.confidence, reverse=True)

        keep = []
        suppressed = set()

        for i, w in enumerate(workers):
            if i in suppressed:
                continue
            keep.append(w)

            for j in range(i + 1, len(workers)):
                if j in suppressed:
                    continue
                iou = self._iou(w.bbox, workers[j].bbox)
                cont = self._containment(w.bbox, workers[j].bbox)

                if iou > self.iou_thresh or cont > self.containment_thresh:
                    suppressed.add(j)

        n_removed = len(workers) - len(keep)
        if n_removed > 0:
            logger.info(f"Worker dedup: {len(workers)} → {len(keep)} (-{n_removed} duplicates)")

        return keep + non_workers

    @staticmethod
    def _iou(a, b):
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        aa = (a[2] - a[0]) * (a[3] - a[1])
        ab = (b[2] - b[0]) * (b[3] - b[1])
        union = aa + ab - inter
        return inter / union if union > 0 else 0

    @staticmethod
    def _containment(a, b):
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        smaller = min((a[2]-a[0])*(a[3]-a[1]), (b[2]-b[0])*(b[3]-b[1]))
        return inter / smaller if smaller > 0 else 0


# ─────────────────────────────────────────────────────────────
# Stage 2: PPE Cluster-Merge (from v1)
# ─────────────────────────────────────────────────────────────

class PPEClusterMerge:
    """Merge overlapping same-class PPE detections into single boxes."""

    def __init__(self, merge_iou=0.10, containment=0.50, proximity_ratio=0.3):
        self.merge_iou = merge_iou
        self.containment = containment
        self.prox_ratio = proximity_ratio

    def merge(self, detections: List[Detection], worker_classes: set) -> List[Detection]:
        by_class: Dict[str, List[Detection]] = {}
        for d in detections:
            by_class.setdefault(d.class_name.lower(), []).append(d)

        result = []
        for cls, dets in by_class.items():
            if cls in worker_classes:
                result.extend(dets)
            else:
                result.extend(self._merge_group(dets))
        return result

    def _merge_group(self, dets):
        n = len(dets)
        if n <= 1:
            return dets

        parent = list(range(n))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for i in range(n):
            for j in range(i+1, n):
                if self._should_merge(dets[i].bbox, dets[j].bbox):
                    union(i, j)

        clusters: Dict[int, List[int]] = {}
        for i in range(n):
            clusters.setdefault(find(i), []).append(i)

        result = []
        for indices in clusters.values():
            group = [dets[i] for i in indices]
            if len(group) == 1:
                result.append(group[0])
            else:
                x1 = min(d.bbox[0] for d in group)
                y1 = min(d.bbox[1] for d in group)
                x2 = max(d.bbox[2] for d in group)
                y2 = max(d.bbox[3] for d in group)
                best = max(group, key=lambda d: d.confidence)
                result.append(Detection(
                    bbox=(x1, y1, x2, y2),
                    class_id=best.class_id,
                    class_name=best.class_name,
                    confidence=min(0.99, best.confidence + 0.03 * (len(group)-1)),
                    merged_from=len(group),
                ))
        return result

    def _should_merge(self, a, b):
        if self._iou(a, b) > self.merge_iou:
            return True
        if self._containment(a, b) > self.containment:
            return True
        # Vertical stacking check
        h_overlap = max(0, min(a[2], b[2]) - max(a[0], b[0]))
        min_w = min(a[2]-a[0], b[2]-b[0])
        if min_w > 0 and h_overlap / min_w >= 0.5:
            top, bot = (a, b) if a[1] < b[1] else (b, a)
            gap = max(0, bot[1] - top[3])
            max_h = max(a[3]-a[1], b[3]-b[1])
            if max_h > 0 and gap / max_h < self.prox_ratio:
                return True
        return False

    @staticmethod
    def _iou(a, b):
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, ix2-ix1) * max(0, iy2-iy1)
        aa = (a[2]-a[0])*(a[3]-a[1])
        ab = (b[2]-b[0])*(b[3]-b[1])
        return inter/(aa+ab-inter) if (aa+ab-inter) > 0 else 0

    @staticmethod
    def _containment(a, b):
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, ix2-ix1) * max(0, iy2-iy1)
        smaller = min((a[2]-a[0])*(a[3]-a[1]), (b[2]-b[0])*(b[3]-b[1]))
        return inter/smaller if smaller > 0 else 0


# ─────────────────────────────────────────────────────────────
# Stage 3: PPE Authenticity Validator (NEW)
# ─────────────────────────────────────────────────────────────

class PPEAuthenticityValidator:
    """
    Validates that detected PPE items are REAL PPE, not regular clothing.

    FALSE HELMET detection (head → helmet):
      A bare head or hair gets detected as a helmet because:
      - Round shape matches helmet training data
      - Dark hair color can look like a dark helmet
      
      Real helmet indicators:
      - Positioned at/above top of head (top 20% of worker)
      - Has distinct solid color: bright yellow, white, orange, red, blue
      - Size ratio: helmet width ≈ 20-45% of worker width
      - Helmet has more uniform color than hair
    
    FALSE VEST detection (shirt → vest):
      A colored shirt/jacket gets detected as a safety vest because:
      - Similar color tone (green, orange shirts)
      - Similar position on body
      
      Real safety vest indicators:
      - HIGH saturation fluorescent colors (S > 100 in HSV)
      - Specific hue ranges: fluorescent yellow (20-35°), orange (5-20°),
        fluorescent green (35-75°)
      - High brightness (V > 120)
      - Large area of consistent high-vis color (>30% of vest bbox)
    """

    # HSV ranges for genuine PPE colors
    # Fluorescent yellow: H=20-40, S=80-255, V=150-255
    # Fluorescent orange: H=5-25, S=100-255, V=150-255
    # Fluorescent green:  H=35-80, S=80-255, V=120-255
    # White helmet:       H=0-180, S=0-40, V=200-255
    # Yellow helmet:      H=15-40, S=100-255, V=150-255
    # Orange helmet:      H=5-25, S=100-255, V=150-255
    # Red helmet:         H=0-10 or 170-180, S=100-255, V=100-255
    # Blue helmet:        H=100-130, S=80-255, V=80-255

    VEST_HSV_RANGES = [
        # Fluorescent Yellow-Green (most common safety vest color)
        {"name": "fl_yellow", "h_min": 18, "h_max": 45, "s_min": 80, "s_max": 255, "v_min": 140, "v_max": 255},
        # Fluorescent Orange
        {"name": "fl_orange", "h_min": 5, "h_max": 22, "s_min": 100, "s_max": 255, "v_min": 140, "v_max": 255},
        # Fluorescent Green
        {"name": "fl_green", "h_min": 35, "h_max": 80, "s_min": 70, "s_max": 255, "v_min": 120, "v_max": 255},
    ]

    HELMET_HSV_RANGES = [
        # White
        {"name": "white", "h_min": 0, "h_max": 180, "s_min": 0, "s_max": 50, "v_min": 190, "v_max": 255},
        # Yellow
        {"name": "yellow", "h_min": 15, "h_max": 42, "s_min": 80, "s_max": 255, "v_min": 140, "v_max": 255},
        # Orange
        {"name": "orange", "h_min": 5, "h_max": 25, "s_min": 100, "s_max": 255, "v_min": 130, "v_max": 255},
        # Red (wraps around H=0)
        {"name": "red_low", "h_min": 0, "h_max": 10, "s_min": 100, "s_max": 255, "v_min": 100, "v_max": 255},
        {"name": "red_high", "h_min": 165, "h_max": 180, "s_min": 100, "s_max": 255, "v_min": 100, "v_max": 255},
        # Blue
        {"name": "blue", "h_min": 95, "h_max": 135, "s_min": 70, "s_max": 255, "v_min": 70, "v_max": 255},
    ]

    def __init__(
        self,
        # Vest color: minimum fraction of vest bbox pixels that must be hi-vis
        vest_min_hivis_ratio: float = 0.15,
        # Helmet color: minimum fraction of helmet bbox that must be helmet color
        helmet_min_color_ratio: float = 0.20,
        # Helmet position: must be in top X% of worker height
        helmet_max_y_frac: float = 0.30,
        # Helmet size ratio relative to worker
        helmet_min_width_ratio: float = 0.12,
        helmet_max_width_ratio: float = 0.55,
        helmet_min_height_ratio: float = 0.05,
        helmet_max_height_ratio: float = 0.25,
        # Vest size ratio relative to worker
        vest_min_width_ratio: float = 0.30,
        vest_min_height_ratio: float = 0.10,
        # Enable/disable color check (can disable for night/IR cameras)
        enable_color_check: bool = True,
    ):
        self.vest_hivis_ratio = vest_min_hivis_ratio
        self.helmet_color_ratio = helmet_min_color_ratio
        self.helmet_max_y = helmet_max_y_frac
        self.helmet_w_range = (helmet_min_width_ratio, helmet_max_width_ratio)
        self.helmet_h_range = (helmet_min_height_ratio, helmet_max_height_ratio)
        self.vest_min_w = vest_min_width_ratio
        self.vest_min_h = vest_min_height_ratio
        self.color_check = enable_color_check

    def validate_helmet(
        self,
        helmet: Detection,
        worker_box: Tuple[float, float, float, float],
        frame: np.ndarray,
        debug: bool = False,
    ) -> Tuple[bool, str]:
        """
        Validate helmet authenticity.
        Returns (is_authentic, reason).
        """
        wx1, wy1, wx2, wy2 = worker_box
        ww = wx2 - wx1
        wh = wy2 - wy1
        hx1, hy1, hx2, hy2 = helmet.bbox
        hw = hx2 - hx1
        hh = hy2 - hy1

        if ww <= 0 or wh <= 0 or hw <= 0 or hh <= 0:
            return True, "skip_no_dims"

        # ── Check 1: Position — helmet must be at top of worker ──
        helmet_center_y = (hy1 + hy2) / 2
        rel_y = (helmet_center_y - wy1) / wh

        if rel_y > self.helmet_max_y:
            reason = f"position_too_low (y={rel_y:.2f} > {self.helmet_max_y})"
            if debug:
                logger.info(f"  Helmet REJECTED: {reason}")
            return False, reason

        # ── Check 2: Size ratio ──
        w_ratio = hw / ww
        h_ratio = hh / wh

        if w_ratio < self.helmet_w_range[0] or w_ratio > self.helmet_w_range[1]:
            reason = f"width_ratio_bad ({w_ratio:.2f} not in {self.helmet_w_range})"
            if debug:
                logger.info(f"  Helmet REJECTED: {reason}")
            return False, reason

        if h_ratio < self.helmet_h_range[0] or h_ratio > self.helmet_h_range[1]:
            reason = f"height_ratio_bad ({h_ratio:.2f} not in {self.helmet_h_range})"
            if debug:
                logger.info(f"  Helmet REJECTED: {reason}")
            return False, reason

        # ── Check 3: Color validation ──
        if self.color_check:
            color_ratio = self._check_helmet_color(helmet.bbox, frame)
            if color_ratio < self.helmet_color_ratio:
                reason = (
                    f"color_not_helmet ({color_ratio:.2f} < {self.helmet_color_ratio}) "
                    f"— likely bare head or hair"
                )
                if debug:
                    logger.info(f"  Helmet REJECTED: {reason}")
                return False, reason

        if debug:
            logger.info(f"  Helmet PASSED all checks")
        return True, "authentic"

    def validate_vest(
        self,
        vest: Detection,
        worker_box: Tuple[float, float, float, float],
        frame: np.ndarray,
        debug: bool = False,
    ) -> Tuple[bool, str]:
        """
        Validate vest authenticity.
        Returns (is_authentic, reason).
        """
        wx1, wy1, wx2, wy2 = worker_box
        ww = wx2 - wx1
        wh = wy2 - wy1
        vx1, vy1, vx2, vy2 = vest.bbox
        vw = vx2 - vx1
        vh = vy2 - vy1

        if ww <= 0 or wh <= 0 or vw <= 0 or vh <= 0:
            return True, "skip_no_dims"

        # ── Check 1: Size ratio ──
        w_ratio = vw / ww
        h_ratio = vh / wh

        if w_ratio < self.vest_min_w:
            reason = f"vest_too_narrow ({w_ratio:.2f} < {self.vest_min_w})"
            if debug:
                logger.info(f"  Vest REJECTED: {reason}")
            return False, reason

        if h_ratio < self.vest_min_h:
            reason = f"vest_too_short ({h_ratio:.2f} < {self.vest_min_h})"
            if debug:
                logger.info(f"  Vest REJECTED: {reason}")
            return False, reason

        # ── Check 2: Color validation — must have hi-vis fluorescent colors ──
        if self.color_check:
            hivis_ratio = self._check_vest_color(vest.bbox, frame)
            if hivis_ratio < self.vest_hivis_ratio:
                reason = (
                    f"no_hivis_color ({hivis_ratio:.2f} < {self.vest_hivis_ratio}) "
                    f"— likely regular clothing"
                )
                if debug:
                    logger.info(f"  Vest REJECTED: {reason}")
                return False, reason

        if debug:
            logger.info(f"  Vest PASSED all checks (hivis={hivis_ratio:.2f})" if self.color_check
                       else f"  Vest PASSED (color check disabled)")
        return True, "authentic"

    def _check_vest_color(self, bbox, frame) -> float:
        """
        Check what fraction of the vest bbox contains hi-vis fluorescent colors.
        Returns ratio 0.0 to 1.0.
        """
        x1, y1, x2, y2 = map(int, bbox)
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return 0.0

        crop = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        total_pixels = hsv.shape[0] * hsv.shape[1]

        if total_pixels == 0:
            return 0.0

        # Check each hi-vis color range
        hivis_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for rng in self.VEST_HSV_RANGES:
            lower = np.array([rng["h_min"], rng["s_min"], rng["v_min"]])
            upper = np.array([rng["h_max"], rng["s_max"], rng["v_max"]])
            mask = cv2.inRange(hsv, lower, upper)
            hivis_mask = cv2.bitwise_or(hivis_mask, mask)

        hivis_pixels = cv2.countNonZero(hivis_mask)
        return hivis_pixels / total_pixels

    def _check_helmet_color(self, bbox, frame) -> float:
        """
        Check what fraction of the helmet bbox contains helmet-like colors.
        """
        x1, y1, x2, y2 = map(int, bbox)
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return 0.0

        crop = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        total_pixels = hsv.shape[0] * hsv.shape[1]

        if total_pixels == 0:
            return 0.0

        helmet_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for rng in self.HELMET_HSV_RANGES:
            lower = np.array([rng["h_min"], rng["s_min"], rng["v_min"]])
            upper = np.array([rng["h_max"], rng["s_max"], rng["v_max"]])
            mask = cv2.inRange(hsv, lower, upper)
            helmet_mask = cv2.bitwise_or(helmet_mask, mask)

        helmet_pixels = cv2.countNonZero(helmet_mask)
        return helmet_pixels / total_pixels


# ─────────────────────────────────────────────────────────────
# Stage 4: Torso-Center PPE Assignment (from v1)
# ─────────────────────────────────────────────────────────────

class TorsoCenterAssigner:
    """Assign PPE to workers using strict torso-center alignment."""

    VEST_CLASSES = {'safety_vest', 'vest', 'safety vest'}
    HELMET_CLASSES = {'helmet', 'hard-hat', 'head', 'hard_hat'}
    WORKER_CLASSES = {'worker', 'person'}

    def __init__(self, h_center_tol=0.30, strip_inset=0.20, min_overlap=0.25):
        self.h_tol = h_center_tol
        self.inset = strip_inset
        self.min_overlap = min_overlap

    def assign(
        self,
        detections: List[Detection],
        frame: np.ndarray,
        authenticity: PPEAuthenticityValidator,
        debug: bool = False,
    ) -> Tuple[List[WorkerPPE], List[Detection]]:
        """Assign PPE with torso-center + authenticity validation."""
        workers = [d for d in detections if d.class_name.lower() in self.WORKER_CLASSES]
        vests = [d for d in detections if d.class_name.lower() in self.VEST_CLASSES]
        helmets = [d for d in detections if d.class_name.lower() in self.HELMET_CLASSES]
        other = [d for d in detections if d.class_name.lower() not in
                 (self.WORKER_CLASSES | self.VEST_CLASSES | self.HELMET_CLASSES)]

        profiles = [WorkerPPE(worker_id=i, worker_box=w.bbox) for i, w in enumerate(workers)]

        # ── Assign vests ──
        vest_map: Dict[int, Tuple[Detection, float]] = {}

        for vest in vests:
            best_wid = None
            best_score = 0.0

            for wp in profiles:
                belongs, score = self._vest_belongs(vest, wp.worker_box)
                if not belongs:
                    continue

                # Authenticity check
                is_real, reason = authenticity.validate_vest(vest, wp.worker_box, frame, debug)
                if not is_real:
                    vest.authentic = False
                    vest.rejection_reason = reason
                    if debug:
                        logger.info(f"  Vest for W{wp.worker_id} failed authenticity: {reason}")
                    continue

                if score > best_score:
                    best_score = score
                    best_wid = wp.worker_id

            if best_wid is not None:
                if best_wid not in vest_map or best_score > vest_map[best_wid][1]:
                    vest_map[best_wid] = (vest, best_score)

        for wid, (vest, _) in vest_map.items():
            profiles[wid].vest = vest
            vest.worker_id = wid

        # ── Assign helmets ──
        helmet_map: Dict[int, Tuple[Detection, float]] = {}

        for helmet in helmets:
            best_wid = None
            best_score = 0.0

            for wp in profiles:
                belongs, score = self._helmet_belongs(helmet, wp.worker_box)
                if not belongs:
                    continue

                # Authenticity check
                is_real, reason = authenticity.validate_helmet(helmet, wp.worker_box, frame, debug)
                if not is_real:
                    helmet.authentic = False
                    helmet.rejection_reason = reason
                    if debug:
                        logger.info(f"  Helmet for W{wp.worker_id} failed authenticity: {reason}")
                    continue

                if score > best_score:
                    best_score = score
                    best_wid = wp.worker_id

            if best_wid is not None:
                if best_wid not in helmet_map or best_score > helmet_map[best_wid][1]:
                    helmet_map[best_wid] = (helmet, best_score)

        for wid, (helmet, _) in helmet_map.items():
            profiles[wid].helmet = helmet
            helmet.worker_id = wid

        # Build clean detections
        clean = list(workers) + other
        for wp in profiles:
            if wp.vest:
                clean.append(wp.vest)
            if wp.helmet:
                clean.append(wp.helmet)

        # Summary
        real_vests = sum(1 for wp in profiles if wp.has_vest)
        real_helmets = sum(1 for wp in profiles if wp.has_helmet)
        fake_vests = sum(1 for v in vests if not v.authentic)
        fake_helmets = sum(1 for h in helmets if not h.authentic)

        logger.info(
            f"Assignment: {len(workers)} workers | "
            f"Vests: {real_vests} real, {fake_vests} fake rejected | "
            f"Helmets: {real_helmets} real, {fake_helmets} fake rejected"
        )

        return profiles, clean

    def _vest_belongs(self, vest, worker_box) -> Tuple[bool, float]:
        wx1, wy1, wx2, wy2 = worker_box
        ww, wh = wx2 - wx1, wy2 - wy1
        if ww <= 0 or wh <= 0:
            return False, 0

        vest_cx = (vest.bbox[0] + vest.bbox[2]) / 2
        worker_cx = (wx1 + wx2) / 2
        h_offset = abs(vest_cx - worker_cx) / ww
        if h_offset > self.h_tol:
            return False, 0

        # Strip overlap
        sx1 = wx1 + ww * self.inset
        sx2 = wx2 - ww * self.inset
        sy1 = wy1 + wh * 0.15
        sy2 = wy1 + wh * 0.75
        s_area = (sx2 - sx1) * (sy2 - sy1)
        if s_area <= 0:
            return False, 0

        ix1 = max(vest.bbox[0], sx1)
        iy1 = max(vest.bbox[1], sy1)
        ix2 = min(vest.bbox[2], sx2)
        iy2 = min(vest.bbox[3], sy2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        overlap = inter / s_area

        if overlap < self.min_overlap:
            return False, 0

        score = (1 - h_offset / self.h_tol) * 0.4 + min(overlap / 0.5, 1) * 0.4 + vest.confidence * 0.2
        return True, score

    def _helmet_belongs(self, helmet, worker_box) -> Tuple[bool, float]:
        wx1, wy1, wx2, wy2 = worker_box
        ww, wh = wx2 - wx1, wy2 - wy1
        if ww <= 0 or wh <= 0:
            return False, 0

        hcx = (helmet.bbox[0] + helmet.bbox[2]) / 2
        hcy = (helmet.bbox[1] + helmet.bbox[3]) / 2
        wcx = (wx1 + wx2) / 2

        h_offset = abs(hcx - wcx) / ww
        if h_offset > 0.35:
            return False, 0

        rel_y = (hcy - wy1) / wh
        if rel_y > 0.40:
            return False, 0

        score = (1 - h_offset / 0.35) * 0.4 + (1 - rel_y / 0.4) * 0.4 + helmet.confidence * 0.2
        return True, score


# ─────────────────────────────────────────────────────────────
# Complete 5-Stage Pipeline
# ─────────────────────────────────────────────────────────────

class BuildSightPipelineV2:
    """
    Complete 5-stage PPE detection post-processing:

      Stage 1: Worker Dedup NMS        → fix duplicate worker boxes
      Stage 2: PPE Cluster-Merge       → fix duplicate vest/helmet fragments
      Stage 3: PPE Authenticity Check   → reject head→helmet, shirt→vest
      Stage 4: Torso-Center Assignment  → fix cross-worker PPE bleeding
      Stage 5: Compliance Scoring       → per-worker report

    Usage:
        pipeline = BuildSightPipelineV2()
        profiles, clean = pipeline.process(raw_detections, frame)
    """

    WORKER_CLASSES = {'worker', 'person'}

    def __init__(
        self,
        # Stage 1
        worker_iou_thresh: float = 0.40,
        # Stage 2
        merge_iou: float = 0.10,
        # Stage 3
        vest_min_hivis: float = 0.15,
        helmet_min_color: float = 0.20,
        enable_color_check: bool = True,
        # Stage 4
        h_center_tol: float = 0.30,
    ):
        self.worker_dedup = WorkerDedupNMS(worker_iou_threshold=worker_iou_thresh)
        self.ppe_merge = PPEClusterMerge(merge_iou=merge_iou)
        self.authenticity = PPEAuthenticityValidator(
            vest_min_hivis_ratio=vest_min_hivis,
            helmet_min_color_ratio=helmet_min_color,
            enable_color_check=enable_color_check,
        )
        self.assigner = TorsoCenterAssigner(h_center_tol=h_center_tol)

    def process(
        self,
        detections: List[Detection],
        frame: np.ndarray,
        debug: bool = False,
    ) -> Tuple[List[WorkerPPE], List[Detection]]:
        n0 = len(detections)

        # Stage 1: Worker dedup
        s1 = self.worker_dedup.dedup(detections, self.WORKER_CLASSES)
        if debug:
            logger.info(f"Stage 1 (worker dedup): {n0} → {len(s1)}")

        # Stage 2: PPE merge
        s2 = self.ppe_merge.merge(s1, self.WORKER_CLASSES)
        if debug:
            logger.info(f"Stage 2 (PPE merge): {len(s1)} → {len(s2)}")

        # Stage 3 + 4: Authenticity + Torso-center assignment (combined)
        profiles, clean = self.assigner.assign(s2, frame, self.authenticity, debug)

        if debug:
            logger.info(f"Stage 3-4 (auth+assign): {len(s2)} → {len(clean)} final")
            for wp in profiles:
                s = "OK" if wp.compliance >= 1.0 else "NON-COMPLIANT"
                logger.info(f"  W{wp.worker_id}: {s} vest={wp.has_vest} helmet={wp.has_helmet}")

        return profiles, clean


# ─────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────

def draw_v2(frame, profiles, show_strips=True):
    vis = frame.copy()
    COLORS = [(50,205,50),(255,140,0),(30,144,255),(255,0,255),(0,255,255),(255,255,0)]

    for wp in profiles:
        c = COLORS[wp.worker_id % len(COLORS)]
        x1, y1, x2, y2 = map(int, wp.worker_box)
        ww, wh = x2-x1, y2-y1

        cv2.rectangle(vis, (x1,y1), (x2,y2), c, 2)

        if show_strips:
            sx1 = int(x1 + ww*0.20)
            sx2 = int(x2 - ww*0.20)
            sy1 = int(y1 + wh*0.15)
            sy2 = int(y1 + wh*0.75)
            ov = vis.copy()
            cv2.rectangle(ov, (sx1,sy1), (sx2,sy2), c, -1)
            cv2.addWeighted(ov, 0.12, vis, 0.88, 0, vis)

        comp = int(wp.compliance * 100)
        _label(vis, f"W{wp.worker_id} [{comp}%]", (x1, y1-8), c)

        if wp.vest:
            vb = tuple(map(int, wp.vest.bbox))
            cv2.rectangle(vis, vb[:2], vb[2:], (0,220,0), 2)
            _label(vis, f"vest {wp.vest.confidence:.2f}", (vb[0], vb[1]-5), (0,220,0), 0.45)

        if wp.helmet:
            hb = tuple(map(int, wp.helmet.bbox))
            cv2.rectangle(vis, hb[:2], hb[2:], (0,200,255), 2)
            _label(vis, f"helmet {wp.helmet.confidence:.2f}", (hb[0], hb[1]-5), (0,200,255), 0.45)

        bx = x2 + 5
        _badge(vis, "H", bx, y1, (0,200,255) if wp.has_helmet else (0,0,220))
        _badge(vis, "V", bx, y1+25, (0,220,0) if wp.has_vest else (0,0,220))

    return vis

def _label(img, text, org, color, scale=0.55, thick=2):
    f = cv2.FONT_HERSHEY_SIMPLEX
    (tw,th),_ = cv2.getTextSize(text,f,scale,thick)
    x,y = int(org[0]),int(org[1])
    cv2.rectangle(img,(x,y-th-6),(x+tw+4,y+2),color,-1)
    cv2.putText(img,text,(x+2,y-2),f,scale,(255,255,255),thick)

def _badge(img,text,x,y,color):
    cv2.rectangle(img,(x,y),(x+25,y+22),color,-1)
    cv2.putText(img,text,(x+5,y+17),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),2)


# ─────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────

def run(
    image_path: str,
    model_path: str = "best.pt",
    output_path: str = "output_v2.jpg",
    conf: float = 0.25,
    iou: float = 0.45,
    debug: bool = True,
    show: bool = False,
):
    from ultralytics import YOLO

    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"Cannot read: {image_path}")

    model = YOLO(model_path)
    pipeline = BuildSightPipelineV2()

    results = model.predict(frame, conf=conf, iou=iou, verbose=False)
    raw = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            cid = int(box.cls[0])
            x1,y1,x2,y2 = box.xyxy[0].tolist()
            raw.append(Detection(
                bbox=(x1,y1,x2,y2), class_id=cid,
                class_name=model.names.get(cid,"unknown"),
                confidence=float(box.conf[0]),
            ))

    print(f"\nRaw: {len(raw)} detections")
    profiles, clean = pipeline.process(raw, frame, debug=debug)

    print(f"\n{'='*55}")
    print(f"  BUILDSIGHT v2 PPE REPORT (5-Stage)")
    print(f"{'='*55}")
    for wp in profiles:
        s = "COMPLIANT" if wp.compliance >= 1.0 else "NON-COMPLIANT"
        print(f"\n  Worker {wp.worker_id} — {s} ({wp.compliance*100:.0f}%)")
        print(f"    Vest:   {'YES' if wp.has_vest else 'MISSING'}")
        print(f"    Helmet: {'YES' if wp.has_helmet else 'MISSING'}")

    # Show rejected fakes
    fakes = [d for d in raw if not d.authentic]
    if fakes:
        print(f"\n  Rejected false PPE ({len(fakes)}):")
        for f in fakes:
            print(f"    {f.class_name} at {tuple(map(int,f.bbox))}: {f.rejection_reason}")
    print(f"\n{'='*55}")

    vis = draw_v2(frame, profiles)
    cv2.imwrite(output_path, vis)
    print(f"Saved: {output_path}")

    if show:
        cv2.imshow("BuildSight v2", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return profiles


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--model", default="best.pt")
    p.add_argument("--output", default="output_v2.jpg")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--show", action="store_true")
    p.add_argument("--no-color-check", action="store_true",
                   help="Disable HSV color validation (for night/IR cameras)")
    a = p.parse_args()
    run(a.source, a.model, a.output, a.conf, a.iou, a.debug, a.show)
