#!/usr/bin/env python3
"""
adaptive_postprocess.py
=======================
Full adaptive post-processing for BuildSight PPE detection.
Applies condition-aware filtering rules to YOLOv11 and YOLOv26 outputs.
"""

import csv
import json
from pathlib import Path

import cv2
import numpy as np


class MaterialSuppressionLayer:
    """
    Suppresses construction material false positives regardless of confidence threshold.
    Uses a cheapest-first gate pipeline. A detection is suppressed when 3+ gates
    confirm non-human / material characteristics.
    """

    MATERIAL_COLOR_RANGES = {
        "yellow_cement_bag": ([20, 100, 100], [35, 255, 255]),
        "brick_red": ([5, 80, 60], [18, 255, 255]),
        "blue_bucket": ([100, 80, 60], [130, 255, 255]),
        "cement_pile_grey": ([0, 0, 180], [180, 40, 255]),
    }

    def __init__(self, min_worker_height_px=40, max_worker_area_ratio=0.15,
                 static_frame_threshold=60):
        self.min_worker_height_px = min_worker_height_px
        self.max_worker_area_ratio = max_worker_area_ratio
        self.static_frame_threshold = static_frame_threshold
        self.static_counters = {}
        self.clutter_mask = None
        self._mask_shape = None

    def build_clutter_mask(self, frame):
        """Build a static clutter mask from a reference frame."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        for lower, upper in self.MATERIAL_COLOR_RANGES.values():
            color_mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            mask = cv2.bitwise_or(mask, color_mask)
        kernel = np.ones((15, 15), np.uint8)
        self.clutter_mask = cv2.dilate(mask, kernel, iterations=1)
        self._mask_shape = frame.shape[:2]
        return self.clutter_mask

    def ensure_clutter_mask(self, frame):
        if self.clutter_mask is None or self._mask_shape != frame.shape[:2]:
            self.build_clutter_mask(frame)

    def _gate1_geometry(self, box, frame_area):
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1
        if h < self.min_worker_height_px:
            return True
        if w > 0 and (h / w) < 0.5:
            return True
        area = w * h
        if frame_area > 0 and (area / frame_area) > self.max_worker_area_ratio:
            return True
        return False

    def _gate2_clutter_mask(self, box):
        if self.clutter_mask is None:
            return False
        x1, y1, x2, y2 = [int(v) for v in box]
        cx = max(0, min((x1 + x2) // 2, self.clutter_mask.shape[1] - 1))
        cy = max(0, min((y1 + y2) // 2, self.clutter_mask.shape[0] - 1))
        return self.clutter_mask[cy, cx] > 0

    def _gate3_color(self, frame, box):
        x1, y1, x2, y2 = [int(v) for v in box]
        x1 = max(0, min(x1, frame.shape[1] - 1))
        x2 = max(x1 + 1, min(x2, frame.shape[1]))
        y1 = max(0, min(y1, frame.shape[0] - 1))
        y2 = max(y1 + 1, min(y2, frame.shape[0]))
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return False
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        roi_area = max(1, roi.shape[0] * roi.shape[1])
        for lower, upper in self.MATERIAL_COLOR_RANGES.values():
            mask = cv2.inRange(hsv_roi, np.array(lower), np.array(upper))
            coverage = cv2.countNonZero(mask) / roi_area
            if coverage > 0.45:
                return True
        return False

    def _gate4_motion(self, track_id, is_static_this_frame):
        # No track ID means we can't determine staticness — do not penalise
        if track_id == -1:
            return False
        if track_id not in self.static_counters:
            self.static_counters[track_id] = 0
        if is_static_this_frame:
            self.static_counters[track_id] += 1
        else:
            self.static_counters[track_id] = 0  # reset immediately on movement
        return self.static_counters[track_id] >= self.static_frame_threshold

    def _gate5_ppe(self, has_ppe_nearby, box):
        if has_ppe_nearby:
            return False
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1
        if h > 0 and (w / h) > 1.5:
            return True
        return False

    def should_suppress(self, detection, frame, frame_area,
                        is_static, has_ppe_nearby):
        """
        Suppress only when 4 or more gates confirm material-like characteristics.
        Raised from 3→4 so a single borderline gate signal (e.g. is_static
        defaulting True early in a video) cannot combine with one other weak
        signal to eliminate a real worker.
        """
        self.ensure_clutter_mask(frame)
        box = detection["box"]
        track_id = detection.get("track_id", -1)

        gate_failures = 0
        if self._gate1_geometry(box, frame_area):
            gate_failures += 1
        if self._gate2_clutter_mask(box):
            gate_failures += 1
        if self._gate3_color(frame, box):
            gate_failures += 1
        if self._gate4_motion(track_id, is_static):
            gate_failures += 1
        if self._gate5_ppe(has_ppe_nearby, box):
            gate_failures += 1

        return gate_failures >= 4

    def cleanup_stale_counters(self, active_track_ids: set) -> None:
        """Remove counters for tracks that no longer exist. Call periodically."""
        stale = [tid for tid in self.static_counters if tid not in active_track_ids]
        for tid in stale:
            del self.static_counters[tid]


class ValidWorkerValidator:
    """
    Multi-gate validator that determines whether a detection is a genuine worker.
    The valid_worker_count output is the ONLY number passed to the scene classifier.
    Raw detection counts are never used for scene classification.
    """

    def __init__(self, min_human_score=0.35, min_height_px=40,
                 min_aspect_ratio=0.40, max_aspect_ratio=4.5,
                 static_disqualify_frames=90, temporal_persistence_frames=2):
        self.min_human_score = min_human_score
        self.min_height_px = min_height_px
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.static_disqualify_frames = static_disqualify_frames
        self.temporal_persistence_frames = temporal_persistence_frames
        self.appearance_counts = {}
        self.static_counts = {}

    def _gate_A_confidence(self, detection, worker_threshold):
        return detection.get("confidence", detection.get("score", 0.0)) >= worker_threshold

    def _gate_B_human_score(self, detection):
        human_score = detection.get("human_score", None)
        if human_score is not None:
            return human_score >= self.min_human_score
        box = detection["box"]
        w = box[2] - box[0]
        h = box[3] - box[1]
        if w <= 0:
            return False
        ratio = h / w
        return 1.2 <= ratio <= 5.0

    def _gate_C_size_aspect(self, detection):
        box = detection["box"]
        w = box[2] - box[0]
        h = box[3] - box[1]
        if h < self.min_height_px or w <= 0:
            return False
        ratio = h / w
        return self.min_aspect_ratio <= ratio <= self.max_aspect_ratio

    def _gate_D_motion_temporal(self, detection, is_static):
        track_id = detection.get("track_id", -1)

        if track_id not in self.static_counts:
            self.static_counts[track_id] = 0
        if track_id not in self.appearance_counts:
            self.appearance_counts[track_id] = 0

        if is_static:
            self.static_counts[track_id] += 1
        else:
            self.static_counts[track_id] = 0  # reset immediately on any movement

        self.appearance_counts[track_id] += 1

        # A long-established track is almost certainly a real worker — never disqualify
        if self.appearance_counts[track_id] >= 30:
            return True

        # Disqualify only after very long unbroken static period AND track is new
        if (self.static_counts[track_id] >= self.static_disqualify_frames
                and self.appearance_counts[track_id] < 30):
            return False

        # Accept if seen consistently for temporal_persistence_frames
        if self.appearance_counts[track_id] >= self.temporal_persistence_frames:
            return True

        # Accept if moving
        return not is_static

    def _gate_E_ppe_or_geometry(self, detection):
        has_ppe = detection.get("has_ppe_nearby", False)
        track_id = detection.get("track_id")
        # Accept after 2 frames of consistent tracking (was 3)
        has_tracking = track_id is not None and self.appearance_counts.get(track_id, 0) >= 2
        strong_geometry = detection.get("human_score", 0) >= 0.5  # was 0.6
        # Geometry proxy: human-shaped box is taller than wide (h/w 1.0–5.0)
        box = detection.get("box", [0, 0, 1, 1])
        bw = max(box[2] - box[0], 1)
        bh = max(box[3] - box[1], 1)
        reasonable_geometry = 1.0 <= (bh / bw) <= 5.0
        return has_ppe or has_tracking or strong_geometry or reasonable_geometry

    def get_valid_workers(self, detections, worker_threshold):
        candidates = []

        for det in detections:
            is_static = det.get("is_static", False)

            if not self._gate_A_confidence(det, worker_threshold):
                det["rejection_gate"] = "A_confidence"
                continue
            if not self._gate_B_human_score(det):
                det["rejection_gate"] = "B_human_score"
                continue
            if not self._gate_C_size_aspect(det):
                det["rejection_gate"] = "C_size_aspect"
                continue
            if not self._gate_D_motion_temporal(det, is_static):
                det["rejection_gate"] = "D_motion"
                continue
            if not self._gate_E_ppe_or_geometry(det):
                det["rejection_gate"] = "E_ppe_geometry"
                continue

            candidates.append(det)

        valid_workers = self._gate_F_duplicate_suppression(candidates)
        return valid_workers, len(valid_workers)

    def _gate_F_duplicate_suppression(self, detections):
        if len(detections) <= 1:
            return detections

        kept = []
        suppressed_indices = set()

        for i, det_a in enumerate(detections):
            if i in suppressed_indices:
                continue
            for j, det_b in enumerate(detections):
                if i >= j or j in suppressed_indices:
                    continue
                overlap = self._compute_iou(det_a["box"], det_b["box"])
                if overlap > 0.45:
                    conf_a = det_a.get("confidence", det_a.get("score", 0.0))
                    conf_b = det_b.get("confidence", det_b.get("score", 0.0))
                    if conf_a >= conf_b:
                        suppressed_indices.add(j)
                    else:
                        suppressed_indices.add(i)
            if i not in suppressed_indices:
                kept.append(det_a)

        return kept

    def cleanup_stale_tracks(self, active_track_ids: set) -> None:
        """Remove state for track IDs no longer active. Call every ~150 frames."""
        for d in (self.static_counts, self.appearance_counts):
            stale = [tid for tid in d if tid not in active_track_ids]
            for tid in stale:
                del d[tid]

    def _compute_iou(self, box_a, box_b):
        xa = max(box_a[0], box_b[0])
        ya = max(box_a[1], box_b[1])
        xb = min(box_a[2], box_b[2])
        yb = min(box_a[3], box_b[3])
        inter = max(0, xb - xa) * max(0, yb - ya)
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

CONF_THRESHOLDS = {
    # Tightened S1 worker (0.42→0.45) and S4 worker (0.38→0.42) to reduce
    # clutter/material false positives in normal and crowded scenes.
    "S1_normal": {"worker": 0.45, "helmet": 0.32, "safety_vest": 0.50},
    "S2_dusty": {"worker": 0.32, "helmet": 0.18, "safety_vest": 0.30},
    "S3_low_light": {"worker": 0.32, "helmet": 0.18, "safety_vest": 0.30},
    # RECALL FIX 2026-04-11: S4 thresholds lowered for worker recall.
    # worker 0.42→0.28 — ValidWorkerValidator gate A; distant/elevated workers
    #   score 0.28-0.38 after WBF and were killed here.
    # helmet 0.24→0.18 — small helmets on distant workers have lower confidence.
    # safety_vest 0.36→0.22 — partially-occluded vests score lower in crowded scenes.
    "S4_crowded": {"worker": 0.28, "helmet": 0.18, "safety_vest": 0.22},
}

NMS_IOU = {
    "S1_normal": 0.40,
    "S2_dusty": 0.50,
    "S3_low_light": 0.50,
    "S4_crowded": 0.35,
}

CLS_NAMES = {0: "helmet", 1: "safety_vest", 2: "worker"}
CLS_COLORS = {0: (0, 255, 0), 1: (0, 165, 255), 2: (255, 100, 0)}
MAX_BOX_AREA_FRACTION = 0.20
MAX_BOX_AREA_FRACTION_BY_CONDITION = {"S4_crowded": 0.30}
WORKER_MIN_HUMAN_SCORE = {
    # Raised S1 (0.43→0.46), S2 (0.38→0.41), S4 (0.30→0.36) to reject
    # material sacks, cement bags, and squatting-object detections.
    "S1_normal": 0.46,
    "S2_dusty": 0.41,
    "S3_low_light": 0.36,
    # RECALL FIX 2026-04-11: S4 0.36→0.24 — elevated/distant workers have
    # genuinely lower human-score proxies due to smaller apparent size and
    # partial occlusion by walls/scaffolding.
    "S4_crowded": 0.24,
}
WORKER_MIN_PIXEL_HEIGHT = {
    # Raised minimums — small blobs below these are almost always clutter.
    # S1: 24→30, S2: 22→26, S3: 20→22, S4: 18→22
    "S1_normal": 30,
    "S2_dusty": 26,
    "S3_low_light": 22,
    # RECALL FIX 2026-04-11: S4 22→14px — workers on elevated walls/platforms
    # appear smaller from below-camera angles; real clutter at 14-21px is
    # still suppressed by aspect ratio, human-score, and multi-gate checks.
    "S4_crowded": 14,
}
CONDITIONS = ["S1_normal", "S2_dusty", "S3_low_light", "S4_crowded"]
GPU = 0

CONDITION_SPLIT_JSON = Path("/nfsshare/joseva/condition_eval_results/val_condition_splits.json")
VAL_IMG_DIR = Path("/nfsshare/joseva/buildsight/dataset/Backup_Dataset(annotated)/YOLOv11/images/val")
OUT_DIR = Path("/nfsshare/joseva/val_annotated_adaptive_v2")
LOG_DIR = Path("/nfsshare/joseva/logs")

MODEL_PATHS = {
    "yolo11": "/nfsshare/joseva/buildsight/dataset/Backup_Dataset(annotated)/YOLOv11/runs/detect/runs/train/buildsight/weights/best.pt",
    "yolo26": "/nfsshare/joseva/buildsight/dataset/Backup_Dataset(annotated)/YOLOv26/runs/detect/runs/train/buildsight/weights/best.pt",
}


def iou(box_a, box_b):
    xi1 = max(box_a[0], box_b[0])
    yi1 = max(box_a[1], box_b[1])
    xi2 = min(box_a[2], box_b[2])
    yi2 = min(box_a[3], box_b[3])
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    if inter == 0:
        return 0.0
    union = (
        (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        + (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        - inter
    )
    return inter / max(union, 1e-6)


def is_valid_aspect(box, cls_name):
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    if height == 0:
        return False
    ratio = width / height

    if cls_name == "worker":
        # Tightened upper bound: 1.25→0.95. Real standing/crouching workers are
        # taller than wide. Cement bags, buckets, and scaffolding cross-sections
        # are typically wider than tall (ratio > 0.95) and are rejected here.
        return 0.15 <= ratio <= 0.95
    if cls_name == "helmet":
        return 0.40 <= ratio <= 2.0
    if cls_name == "safety_vest":
        return 0.25 <= ratio <= 1.80
    return True


def has_worker_overlap(ppe_box, worker_boxes, min_iou=0.08, cls_id=None):
    """
    Return True if a PPE box has a plausible spatial association with at least
    one worker box.

    For helmets (cls_id=0): enforces vertical constraint — helmet centroid must
    be within the upper 40% of the worker box height (±10% headroom above).
    This rejects wrist/palm false positives that are spatially near a worker
    but clearly below the head zone.

    For other PPE (vest, cls_id=1): standard centroid containment with 20% expand.
    """
    px_c = (ppe_box[0] + ppe_box[2]) / 2
    py_c = (ppe_box[1] + ppe_box[3]) / 2
    for worker_box in worker_boxes:
        wx1, wy1, wx2, wy2 = worker_box
        worker_h = max(wy2 - wy1, 1)

        if cls_id == 0:  # helmet — enforce vertical zone
            w_pad = (wx2 - wx1) * 0.15
            # Upper 40% of worker + 10% headroom above the box top
            hy_top = wy1 - worker_h * 0.10
            hy_bot = wy1 + worker_h * 0.40
            if wx1 - w_pad <= px_c <= wx2 + w_pad and hy_top <= py_c <= hy_bot:
                return True
        else:
            if iou(ppe_box, worker_box) >= min_iou:
                return True
            w_expand = (wx2 - wx1) * 0.20
            h_expand = worker_h * 0.20
            if (
                wx1 - w_expand <= px_c <= wx2 + w_expand
                and wy1 - h_expand <= py_c <= wy2 + h_expand
            ):
                return True
    return False


class HelmetValidationLayer:
    """
    Multi-gate validation to eliminate false-positive helmet detections on:
      • Bare heads / wrists / palms (skin-toned, smooth texture)
      • Cloth wraps / kerchiefs (elongated, outside head zone)

    Design principles:
      • Skin-tone check is a PENALTY, NOT a hard reject.  Hard rejection occurs
        only when 3+ signals fire together — this preserves orange, yellow, white,
        beige, and dusty safety helmets that share colour overlap with skin tones.
      • Safety colours (high-saturation orange/yellow, low-saturation white/beige)
        bypass the skin-tone gate entirely.
      • Vertical position is Gate 1: helmet centroid must be in top 35% of the
        matched worker box — wrist and palm FPs are eliminated here alone.
      • Texture (Laplacian variance) distinguishes rigid hard-hat shells from
        the soft surface of hair, cloth, and bare skin.
    """

    # Safety colour HSV ranges — detections with these colours are always kept
    PRESERVE_COLORS: dict = {
        "orange":     ([5,  120,  80], [22, 255, 255]),   # high-sat orange hard hat
        "yellow":     ([20, 120, 100], [38, 255, 255]),   # high-sat yellow hard hat
        "white":      ([0,    0, 170], [180,  40, 255]),  # low-sat, bright white
        "beige_tan":  ([10,  15, 110], [28,   75, 220]),  # dusty / faded tan helmets
    }

    # Skin-tone HSV range (Indian skin tones, CCTV lighting)
    # Deliberately stops at S=135 to avoid confusing low-saturation orange helmets
    SKIN_LO = np.array([0,  30,  60], dtype=np.uint8)
    SKIN_HI = np.array([25, 135, 230], dtype=np.uint8)

    # Laplacian variance floor — rigid shells score higher than skin/cloth
    TEXTURE_HARD_HAT_VAR = 60.0

    def _crop(self, frame: np.ndarray, box: list) -> np.ndarray | None:
        if frame is None:
            return None
        h_img, w_img = frame.shape[:2]
        x1, y1, x2, y2 = (max(0, int(round(v))) for v in box)
        x2 = min(max(x1 + 1, x2), w_img)
        y2 = min(max(y1 + 1, y2), h_img)
        crop = frame[y1:y2, x1:x2]
        return crop if crop.size > 0 else None

    def _is_safety_color(self, crop: np.ndarray | None) -> bool:
        """True if the crop contains a known safety-helmet colour at ≥ 20 % coverage."""
        if crop is None or crop.size == 0:
            return False
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        area = max(crop.shape[0] * crop.shape[1], 1)
        for lo, hi in self.PRESERVE_COLORS.values():
            mask = cv2.inRange(hsv, np.array(lo, dtype=np.uint8),
                                    np.array(hi, dtype=np.uint8))
            if cv2.countNonZero(mask) / area > 0.20:
                return True
        return False

    def _skin_tone_fraction(self, crop: np.ndarray | None) -> float:
        """Fraction of crop pixels in skin-tone HSV range."""
        if crop is None or crop.size == 0:
            return 0.0
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.SKIN_LO, self.SKIN_HI)
        return float(cv2.countNonZero(mask)) / max(crop.shape[0] * crop.shape[1], 1)

    def _is_hard_texture(self, crop: np.ndarray | None) -> bool:
        """True if texture looks rigid (high Laplacian variance), False if soft."""
        if crop is None or crop.size == 0:
            return True   # unknown → don't penalise
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var()) >= self.TEXTURE_HARD_HAT_VAR

    def _in_vertical_zone(self, hbox: list, worker_box: list) -> bool:
        """True if helmet centroid is within top 35% of the worker box height."""
        _hx1, hy1, _hx2, hy2 = hbox
        _wx1, wy1, _wx2, wy2 = worker_box
        worker_h = max(wy2 - wy1, 1)
        hcy = (hy1 + hy2) / 2.0
        return hcy <= wy1 + worker_h * 0.35

    def _validate_one(self, hdet: dict, worker_det: dict,
                      frame: np.ndarray | None) -> bool:
        """Return True (keep) / False (reject) for a single helmet + worker pair."""
        hbox = hdet["box"]
        wbox = worker_det["box"]
        crop = self._crop(frame, hbox)

        # Safety colours always pass — never suppress orange/yellow/white helmets
        if self._is_safety_color(crop):
            return True

        # Gate 1 — vertical position (head zone check)
        in_zone = self._in_vertical_zone(hbox, wbox)
        if not in_zone:
            # Below the head zone with no safety colour → reject immediately
            return False

        # Gate 2+3 — skin-tone penalty (needs 3 signals to hard-reject)
        penalties = 0
        skin_frac = self._skin_tone_fraction(crop)
        if skin_frac > 0.35:
            penalties += 1
        if not self._is_hard_texture(crop):
            penalties += 1
        worker_score = float(worker_det.get("score", 0.5))
        if worker_score < 0.32:
            penalties += 1
        # Hard reject only when all three skin-tone signals fire
        if penalties >= 3:
            return False

        return True

    def filter(self, boxes: list, frame: np.ndarray | None) -> list:
        """
        Filter a combined list of detections (workers + helmets + vests).
        Helmet boxes (cls=0) are validated against their nearest worker;
        workers and vests pass through unchanged.

        Returns the filtered list.
        """
        worker_boxes = [d for d in boxes if d["cls"] == 2]
        if not worker_boxes:
            # No workers — all helmets are orphans; remove them
            return [d for d in boxes if d["cls"] != 0]

        kept = []
        for det in boxes:
            if det["cls"] != 0:  # worker or vest — pass through
                kept.append(det)
                continue

            hbox = det["box"]
            hcx = (hbox[0] + hbox[2]) / 2
            hcy = (hbox[1] + hbox[3]) / 2

            # Find the best-matched worker (nearest centroid within search zone)
            best_worker: dict | None = None
            best_dist = float("inf")
            for wd in worker_boxes:
                wx1, wy1, wx2, wy2 = wd["box"]
                bw = wx2 - wx1
                pad = bw * 0.15
                bh = wy2 - wy1
                # Helmet search zone: upper 75% + 10% headroom above
                if (wx1 - pad <= hcx <= wx2 + pad
                        and wy1 - bh * 0.10 <= hcy <= wy1 + bh * 0.75):
                    wcx = (wx1 + wx2) / 2
                    wcy = (wy1 + wy2) / 2
                    dist = ((hcx - wcx) ** 2 + (hcy - wcy) ** 2) ** 0.5
                    if dist < best_dist:
                        best_dist = dist
                        best_worker = wd

            if best_worker is None:
                # No worker in proximity — orphan, skip (filter_orphan_ppe handles these)
                kept.append(det)
                continue

            if self._validate_one(det, best_worker, frame):
                kept.append(det)

        return kept


# Module-level singleton — reused across frames (stateless, safe to share)
_helmet_validator = HelmetValidationLayer()


def suppress_large_by_small(boxes, overlap_thresh=0.30):
    if not boxes:
        return []

    areas = [
        (box["box"][2] - box["box"][0]) * (box["box"][3] - box["box"][1])
        for box in boxes
    ]
    median_area = sorted(areas)[len(areas) // 2]
    small_boxes = [box for box, area in zip(boxes, areas) if area <= median_area]
    result = []

    for box, area in zip(boxes, areas):
        if area > median_area * 3:
            overlapping_small = [small for small in small_boxes if iou(box["box"], small["box"]) > overlap_thresh]
            if overlapping_small:
                if box["score"] > max(small["score"] for small in overlapping_small) + 0.15:
                    result.append(box)
                continue
        result.append(box)

    return result


def cross_class_nms(boxes, iou_thresh=0.38):
    """
    Same-class NMS — cross-class suppression intentionally disabled because
    worker/PPE overlap is expected and removing one class harms association.

    IoU threshold tightened 0.45→0.38 for worker class:
      Two boxes for the same worker that survived WBF typically have IoU 0.38–0.55.
      The old 0.45 threshold missed many of these, leaving duplicate worker boxes.
      Non-worker classes keep an effective threshold of 0.45 to avoid over-suppressing
      genuine co-located helmets and vests on adjacent workers.
    """
    boxes = sorted(boxes, key=lambda item: -item["score"])
    keep = [True] * len(boxes)
    for i in range(len(boxes)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(boxes)):
            if not keep[j] or boxes[i]["cls"] != boxes[j]["cls"]:
                continue
            # Tighter IoU threshold for workers (cls == 2)
            effective_thresh = iou_thresh if boxes[i]["cls"] == 2 else 0.45
            if iou(boxes[i]["box"], boxes[j]["box"]) > effective_thresh:
                keep[j] = False
                continue
            # Centroid-distance suppression for workers: two worker boxes whose
            # centroids are within 20 px of each other are almost certainly the
            # same person even if their IoU is slightly below the threshold.
            if boxes[i]["cls"] == 2:
                ax = (boxes[i]["box"][0] + boxes[i]["box"][2]) / 2.0
                ay = (boxes[i]["box"][1] + boxes[i]["box"][3]) / 2.0
                bx = (boxes[j]["box"][0] + boxes[j]["box"][2]) / 2.0
                by = (boxes[j]["box"][1] + boxes[j]["box"][3]) / 2.0
                if ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5 < 20.0:
                    keep[j] = False
    return [box for box, is_kept in zip(boxes, keep) if is_kept]


def _clamp_box(box, img_w, img_h):
    x1, y1, x2, y2 = box
    return [
        max(0, min(img_w - 1, int(round(x1)))),
        max(0, min(img_h - 1, int(round(y1)))),
        max(0, min(img_w - 1, int(round(x2)))),
        max(0, min(img_h - 1, int(round(y2)))),
    ]


def _worker_support_score(box, all_boxes):
    if not all_boxes:
        return 0.0
    support = 0.0
    for other in all_boxes:
        if other["box"] == box["box"]:
            continue
        ov = iou(box["box"], other["box"])
        if other["cls"] == 2:
            support = max(support, min(1.0, ov * 2.0))
        elif other["cls"] in (0, 1):
            px1, py1, px2, py2 = other["box"]
            wx1, wy1, wx2, wy2 = box["box"]
            cx = (px1 + px2) / 2
            cy = (py1 + py2) / 2
            if wx1 <= cx <= wx2 and wy1 <= cy <= wy2:
                support = max(support, 0.75 if other["cls"] == 0 else 0.65)
    return support


def compute_worker_human_score(box, image):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = _clamp_box(box, w, h)
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    texture_std = float(np.std(gray)) / 64.0
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_x, grad_y)
    grad_score = float(np.mean(grad_mag > 18.0))

    edges = cv2.Canny(gray, 40, 120)
    edge_density = float(np.mean(edges > 0))
    edge_score = 1.0 - min(1.0, abs(edge_density - 0.10) / 0.10)

    edge_mask = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1) > 0
    row_profile = edge_mask.mean(axis=1) if edge_mask.size else np.zeros((1,), dtype=np.float32)
    col_profile = edge_mask.mean(axis=0) if edge_mask.size else np.zeros((1,), dtype=np.float32)

    top = float(np.mean(row_profile[: max(1, bh // 4)]))
    middle = float(np.mean(row_profile[bh // 4 : max(bh // 4 + 1, (3 * bh) // 4)]))
    bottom = float(np.mean(row_profile[max(0, (3 * bh) // 4) :]))
    lower_mid = float(np.mean(row_profile[max(0, bh // 2) : max(bh // 2 + 1, (7 * bh) // 8)]))
    col_mean = float(np.mean(col_profile))
    col_std = float(np.std(col_profile))

    aspect = bw / max(bh, 1.0)
    aspect_score = 1.0 - min(1.0, abs(aspect - 0.42) / 0.55)

    shape_score = 0.0
    if middle > 0:
        if top < middle * 0.95:
            shape_score += 0.45
        if bottom < middle * 1.10:
            shape_score += 0.30
        if col_std > col_mean * 0.35:
            shape_score += 0.25

    # Low-contrast clothing can still form a human silhouette even if texture is weak.
    silhouette_bonus = 0.0
    if aspect <= 0.70 and bh >= 40:
        if 0.03 <= edge_density <= 0.18:
            silhouette_bonus += 0.18
        if grad_score >= 0.08:
            silhouette_bonus += 0.12

    # Material sacks/buckets tend to be squat, bottom-heavy, horizontally uniform,
    # and weak in vertical edge structure. Penalties are cumulative.
    bag_penalty = 0.0
    if aspect >= 0.60 and bh <= 95:
        if bottom > max(top, middle) * 1.10:
            bag_penalty += 0.14
        if lower_mid > middle * 1.05:
            bag_penalty += 0.10
        if col_std < max(0.02, col_mean * 0.18):
            bag_penalty += 0.12
        if grad_score < 0.07 and edge_density < 0.06:
            bag_penalty += 0.14
        if top < 0.015 and middle < 0.04:
            bag_penalty += 0.10
    if aspect >= 0.80:
        bag_penalty += 0.10
    # Very uniform color in the crop → likely flat material surface, not clothing
    if texture_std < 0.08 and grad_score < 0.10:
        bag_penalty += 0.12

    score = (
        0.34 * np.clip(aspect_score, 0.0, 1.0)
        + 0.14 * np.clip(texture_std, 0.0, 1.0)
        + 0.18 * np.clip(edge_score, 0.0, 1.0)
        + 0.22 * np.clip(shape_score, 0.0, 1.0)
        + 0.12 * np.clip(grad_score, 0.0, 1.0)
    )
    score += silhouette_bonus
    score -= bag_penalty
    return float(np.clip(score, 0.0, 1.0))


def _is_blue_bucket(crop_bgr) -> float:
    """
    Returns a blue-bucket probability [0, 1] based on HSV blue-hue dominance.
    Blue buckets on Indian construction sites typically cluster in H=100-130 (HSV),
    high saturation (>90), mid-high value. Returns > 0.5 when blue dominates.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return 0.0
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    blue_lo = np.array([95, 80, 60], dtype=np.uint8)
    blue_hi = np.array([135, 255, 255], dtype=np.uint8)
    blue_mask = cv2.inRange(hsv, blue_lo, blue_hi)
    blue_frac = float(np.mean(blue_mask > 0))
    return min(1.0, blue_frac * 2.5)


def _is_cement_bag(crop_bgr) -> float:
    """
    Returns a cement/sand-bag probability [0, 1].
    Cement bags: gray/white/tan tones, low saturation, horizontally wide,
    uniform texture (low gradient variance). Score > 0.55 → likely bag.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return 0.0
    h, w = crop_bgr.shape[:2]
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    sat = float(np.mean(hsv[:, :, 1]))
    val = float(np.mean(hsv[:, :, 2]))
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_x, grad_y)
    grad_std = float(np.std(grad_mag))

    score = 0.0
    # Low saturation = not vivid = plausible bag/material
    if sat < 55:
        score += 0.30
    elif sat < 80:
        score += 0.15
    # Typical bag brightness range
    if 60 < val < 200:
        score += 0.15
    # Uniform surface = low gradient variation
    if grad_std < 18:
        score += 0.30
    elif grad_std < 28:
        score += 0.15
    # Wide horizontal blob = bag shape
    if w > 0 and h > 0 and (w / h) > 0.85:
        score += 0.15
    return min(1.0, score)


def _is_scaffolding(crop_bgr) -> float:
    """
    Returns a scaffolding probability [0, 1].
    Scaffolding: strong regular vertical/horizontal line structure, metallic gray,
    high edge density in a grid-like pattern. Score > 0.55 → likely scaffolding.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.mean(edges > 0))

    # Check for strong horizontal/vertical line dominance via Hough
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=15,
                             minLineLength=12, maxLineGap=4)
    line_count = len(lines) if lines is not None else 0

    score = 0.0
    # Very high edge density in a structured pattern = scaffolding
    if edge_density > 0.20:
        score += 0.35
    elif edge_density > 0.14:
        score += 0.20
    # Many short structural lines = scaffolding grid
    h, w = crop_bgr.shape[:2]
    crop_area = max(h * w, 1)
    line_density = line_count / (crop_area / 400)  # lines per 400px²
    if line_density > 3.0:
        score += 0.35
    elif line_density > 1.5:
        score += 0.20

    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    sat = float(np.mean(hsv[:, :, 1]))
    # Low saturation metallic color
    if sat < 40:
        score += 0.15
    return min(1.0, score)


def suppress_hard_negatives(boxes, image, condition):
    """
    Hard negative suppression for known Indian construction-site clutter:
      - Blue buckets / barrels  (strong blue HSV dominance)
      - Cement / sand bags      (gray/white, wide, uniform texture)
      - Scaffolding             (high-edge metallic grid structure)

    Only applied to worker (cls=2) boxes. PPE boxes are not suppressed here
    because the PPE-to-worker anchor check handles floating detections.

    Returns (filtered_boxes, suppressed_count).
    """
    if image is None:
        return boxes, 0

    h_img, w_img = image.shape[:2]
    kept = []
    suppressed = 0

    # Thresholds — slightly looser in degraded conditions to avoid recall loss
    blue_thresh = 0.55 if condition == "S1_normal" else 0.65
    bag_thresh = 0.62 if condition == "S1_normal" else 0.70
    scaffold_thresh = 0.60 if condition == "S1_normal" else 0.68

    for box in boxes:
        if box["cls"] != 2:
            kept.append(box)
            continue

        x1, y1, x2, y2 = [int(round(v)) for v in box["box"]]
        x1 = max(0, min(x1, w_img - 1))
        y1 = max(0, min(y1, h_img - 1))
        x2 = max(x1 + 1, min(x2, w_img))
        y2 = max(y1 + 1, min(y2, h_img))
        crop = image[y1:y2, x1:x2]

        if crop.size == 0:
            kept.append(box)
            continue

        bucket_score = _is_blue_bucket(crop)
        bag_score = _is_cement_bag(crop)
        scaffold_score = _is_scaffolding(crop)

        # High-confidence individual classifier suppresses the detection
        if bucket_score >= blue_thresh:
            suppressed += 1
            continue
        if bag_score >= bag_thresh:
            suppressed += 1
            continue
        if scaffold_score >= scaffold_thresh:
            suppressed += 1
            continue

        # Combined — two medium-confidence signals together are enough to suppress
        combined = max(
            0.5 * bucket_score + 0.5 * bag_score,
            0.5 * bucket_score + 0.5 * scaffold_score,
            0.5 * bag_score + 0.5 * scaffold_score,
        )
        if combined >= 0.58 and box["score"] < 0.75:
            suppressed += 1
            continue

        kept.append(box)

    return kept, suppressed


def suppress_material_workers(boxes, image, condition):
    if image is None:
        return boxes, 0

    kept = []
    rejected = 0
    min_score = WORKER_MIN_HUMAN_SCORE.get(condition, 0.50)
    min_height = WORKER_MIN_PIXEL_HEIGHT.get(condition, 20)

    for box in boxes:
        if box["cls"] != 2:
            kept.append(box)
            continue

        x1, y1, x2, y2 = box["box"]
        height = y2 - y1
        if height < min_height:
            kept.append(box)
            continue

        human_score = compute_worker_human_score(box["box"], image)
        support_score = _worker_support_score(box, boxes)
        combined = max(human_score, 0.65 * human_score + 0.35 * support_score)
        box["human_score"] = round(combined, 3)
        aspect = (x2 - x1) / max((y2 - y1), 1.0)
        short_wide_material = height <= 112 and aspect >= 0.60 and support_score < 0.45
        weak_human_shape = aspect >= 0.56 and height <= 120 and human_score < (min_score - 0.06)

        slender_worker = aspect <= 0.72 and (y2 - y1) >= (min_height + 8)
        if short_wide_material and combined < (min_score + 0.08):
            rejected += 1
            continue
        if weak_human_shape and combined < min_score and box["score"] < 0.78 and not slender_worker:
            rejected += 1
            continue
        if combined < min_score and support_score < 0.60 and box["score"] < 0.85 and not (slender_worker and combined >= (min_score - 0.08)):
            rejected += 1
            continue
        kept.append(box)

    return kept, rejected


def vertical_position_filter(boxes, img_h, condition):
    if condition != "S1_normal":
        return boxes

    filtered = []
    for box in boxes:
        if box["cls"] == 2:
            x1, y1, x2, y2 = box["box"]
            cy = (y1 + y2) / 2
            box_h = y2 - y1
            cy_frac = cy / img_h
            box_h_frac = box_h / img_h
            if cy_frac < 0.45 and box_h_frac > 0.08 and box["score"] < 0.90:
                continue
        filtered.append(box)
    return filtered


def area_filtered_boxes(boxes, img_area, area_fraction):
    kept = []
    worker_boxes = [box for box in boxes if CLS_NAMES[box["cls"]] == "worker"]
    for box in boxes:
        x1, y1, x2, y2 = box["box"]
        area = (x2 - x1) * (y2 - y1)
        if area / img_area <= area_fraction:
            kept.append(box)
            continue
        if CLS_NAMES[box["cls"]] == "worker":
            overlapping_workers = [
                worker
                for worker in worker_boxes
                if worker is not box and iou(box["box"], worker["box"]) > 0.30
            ]
            if overlapping_workers:
                kept.append(box)
    return kept


def apply_all_rules(raw_boxes, condition, img_w, img_h, image=None, track_context=None):
    img_area = img_w * img_h
    cond_conf = CONF_THRESHOLDS[condition]
    area_fraction = MAX_BOX_AREA_FRACTION_BY_CONDITION.get(condition, MAX_BOX_AREA_FRACTION)
    stats = {"raw": len(raw_boxes)}

    boxes = [box for box in raw_boxes if box["score"] >= cond_conf[CLS_NAMES[box["cls"]]]]
    stats["after_conf"] = len(boxes)

    boxes = area_filtered_boxes(boxes, img_area, area_fraction)
    stats["after_area"] = len(boxes)

    boxes = [box for box in boxes if is_valid_aspect(box["box"], CLS_NAMES[box["cls"]])]
    stats["after_aspect"] = len(boxes)

    boxes = suppress_large_by_small(boxes)
    stats["after_large_suppress"] = len(boxes)

    boxes = cross_class_nms(boxes, NMS_IOU[condition])
    stats["after_cross_nms"] = len(boxes)

    boxes = vertical_position_filter(boxes, img_h, condition)
    stats["after_vertical"] = len(boxes)

    boxes, suppressed_material = suppress_material_workers(boxes, image, condition)
    stats["after_worker_validation"] = len(boxes)
    stats["suppressed_material_workers"] = suppressed_material

    # Hard negative suppression: blue buckets, cement bags, scaffolding
    boxes, suppressed_hard = suppress_hard_negatives(boxes, image, condition)
    stats["after_hard_negative"] = len(boxes)
    stats["suppressed_hard_negatives"] = suppressed_hard

    worker_boxes = [box["box"] for box in boxes if box["cls"] == 2]
    if worker_boxes:
        anchored = []
        for box in boxes:
            if box["cls"] in (0, 1) and not has_worker_overlap(box["box"], worker_boxes,
                                                                cls_id=box["cls"]):
                continue
            anchored.append(box)
        boxes = anchored
    else:
        ppe_boxes = [box for box in boxes if box["cls"] in (0, 1)]
        worker_free = [box for box in boxes if box["cls"] not in (0, 1)]
        ppe_boxes = sorted(ppe_boxes, key=lambda item: -item["score"])[:3]
        boxes = worker_free + ppe_boxes

    stats["after_ppe_anchor"] = len(boxes)

    # ── Helmet precision gate ────────────────────────────────────────────────
    # Run HelmetValidationLayer AFTER worker anchoring so every helmet already
    # has a candidate worker nearby. Skin-tone / texture / vertical checks here
    # eliminate bare head, wrist, and palm false positives that survive aspect-
    # ratio and confidence filters.
    boxes = _helmet_validator.filter(boxes, image)
    stats["after_helmet_validation"] = len(boxes)
    stats["final"] = len(boxes)
    return boxes, stats


def draw_boxes(image, boxes, condition, model_label, stats):
    for box in boxes:
        x1, y1, x2, y2 = [int(v) for v in box["box"]]
        color = CLS_COLORS[box["cls"]]
        label = f"{CLS_NAMES[box['cls']]} {box['score']:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(image, (x1, y1 - text_h - 4), (x1 + text_w + 2, y1), color, -1)
        cv2.putText(image, label, (x1 + 1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    banner = (
        f"[ADAPTIVE] {model_label}|{condition} "
        f"raw:{stats['raw']}->final:{stats['final']} "
        f"(-{stats['raw'] - stats['final']})"
    )
    cv2.rectangle(image, (0, 0), (image.shape[1], 20), (0, 0, 0), -1)
    cv2.putText(image, banner, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 100), 1)
    return image


def run():
    from ultralytics import YOLO

    splits = json.loads(CONDITION_SPLIT_JSON.read_text())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    csv_rows = []

    for model_key, model_path in MODEL_PATHS.items():
        model_label = "YOLOv11" if model_key == "yolo11" else "YOLOv26"
        print(f"\n=== {model_label} ===")
        model = YOLO(model_path)

        for condition in CONDITIONS:
            file_names = splits.get(condition, [])
            if not file_names:
                continue

            dst_dir = OUT_DIR / condition / model_label
            dst_dir.mkdir(parents=True, exist_ok=True)
            print(f"  {condition}: {len(file_names)} images")

            for file_name in file_names:
                img_path = VAL_IMG_DIR / file_name
                if not img_path.exists():
                    continue

                image = cv2.imread(str(img_path))
                if image is None:
                    continue
                height, width = image.shape[:2]

                result = model.predict(str(img_path), device=GPU, verbose=False, conf=0.07, iou=0.35)[0]
                raw_boxes = []
                for box in (result.boxes or []):
                    raw_boxes.append(
                        {
                            "box": box.xyxy[0].cpu().numpy().tolist(),
                            "cls": int(box.cls[0]),
                            "score": float(box.conf[0]),
                        }
                    )

                final_boxes, stats = apply_all_rules(raw_boxes, condition, width, height)
                out_img = draw_boxes(image.copy(), final_boxes, condition, model_label, stats)
                cv2.imwrite(str(dst_dir / file_name), out_img)

                csv_rows.append(
                    {
                        "model": model_label,
                        "condition": condition,
                        "file": file_name,
                        "raw": stats["raw"],
                        "after_conf": stats["after_conf"],
                        "after_area": stats["after_area"],
                        "after_aspect": stats["after_aspect"],
                        "after_large_suppress": stats["after_large_suppress"],
                        "after_cross_nms": stats["after_cross_nms"],
                        "after_vertical": stats["after_vertical"],
                        "after_ppe_anchor": stats["after_ppe_anchor"],
                        "final": stats["final"],
                    }
                )

    csv_path = LOG_DIR / "adaptive_postprocess_v2_summary.csv"
    if csv_rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(csv_rows)

    print(f"\nDone. Output: {OUT_DIR}")
    print(f"Summary CSV: {csv_path}")


if __name__ == "__main__":
    run()
