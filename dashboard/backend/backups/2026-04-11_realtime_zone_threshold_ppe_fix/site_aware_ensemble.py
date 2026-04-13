#!/usr/bin/env python3
"""
Runtime utilities for BuildSight's site-aware ensemble.

This module adds four pieces missing from the original runtime path:
1. Automatic site-condition estimation from each frame
2. Condition-specific preprocessing
3. Condition-specific WBF tuning with crowded-scene worker recovery
4. Lightweight temporal smoothing for worker/PPE consistency
"""

from __future__ import annotations

from dataclasses import dataclass, field  # field used for position_history default
from typing import Dict, List, Tuple

import cv2
import numpy as np

CLS_HELMET = 0
CLS_VEST = 1
CLS_WORKER = 2


@dataclass
class ConditionDecision:
    key: str
    brightness: float
    contrast: float
    saturation: float
    rough_worker_count: int
    crowd_overlap: float


@dataclass
class WorkerTrack:
    track_id: int
    bbox: List[float]
    hits: int = 0
    misses: int = 0
    human_score_ema: float = 0.0
    helmet_streak: int = 0
    vest_streak: int = 0
    last_frame: int = -1
    # Motion tracking: accumulate recent centre-point positions to detect
    # static objects masquerading as workers (cement bags, machinery etc.)
    position_history: List[List[float]] = field(default_factory=list)

    def centre(self) -> List[float]:
        return [(self.bbox[0] + self.bbox[2]) / 2.0,
                (self.bbox[1] + self.bbox[3]) / 2.0]

    def is_static(self, min_hits: int = 6, max_variance_px: float = 3.5) -> bool:
        """True when the track has been seen enough times but barely moved."""
        if self.hits < min_hits or len(self.position_history) < min_hits:
            return False
        recent = self.position_history[-min_hits:]
        xs = [p[0] for p in recent]
        ys = [p[1] for p in recent]
        variance = float(np.var(xs) + np.var(ys))
        return variance < (max_variance_px ** 2)


def iou(box_a: List[float], box_b: List[float]) -> float:
    xi1 = max(box_a[0], box_b[0])
    yi1 = max(box_a[1], box_b[1])
    xi2 = min(box_a[2], box_b[2])
    yi2 = min(box_a[3], box_b[3])
    inter = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)
    if inter <= 0.0:
        return 0.0
    union = (
        (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        + (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        - inter
    )
    return inter / max(union, 1e-6)


def mean_pairwise_worker_overlap(worker_boxes: List[List[float]]) -> float:
    if len(worker_boxes) < 2:
        return 0.0
    overlaps = []
    for i in range(len(worker_boxes)):
        for j in range(i + 1, len(worker_boxes)):
            overlaps.append(iou(worker_boxes[i], worker_boxes[j]))
    return float(np.mean(overlaps)) if overlaps else 0.0


def detect_condition(frame: np.ndarray, rough_worker_boxes: List[List[float]]) -> ConditionDecision:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    saturation = float(np.mean(hsv[:, :, 1]))
    worker_count = len(rough_worker_boxes)
    crowd_overlap = mean_pairwise_worker_overlap(rough_worker_boxes)

    if brightness < 72:
        key = "S3_low_light"
    elif contrast < 52 and saturation < 78:
        key = "S2_dusty"
    elif worker_count >= 3 or crowd_overlap >= 0.05:
        key = "S4_crowded"
    else:
        key = "S1_normal"

    return ConditionDecision(
        key=key,
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        rough_worker_count=worker_count,
        crowd_overlap=crowd_overlap,
    )


def preprocess_frame(frame: np.ndarray, condition: str) -> np.ndarray:
    out = frame.copy()

    if condition in {"S2_dusty", "S3_low_light"}:
        clip_limit = 3.0 if condition == "S2_dusty" else 4.0
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        l = clahe.apply(l)
        out = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    if condition == "S3_low_light":
        gamma = 1.7
        table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)], dtype=np.uint8)
        out = cv2.LUT(out, table)
    elif condition == "S2_dusty":
        blur = cv2.GaussianBlur(out, (0, 0), 1.2)
        out = cv2.addWeighted(out, 1.35, blur, -0.35, 0)

    return out


def profile_for_condition(condition: str) -> Dict[str, object]:
    profiles = {
        "S1_normal": {
            "pre_conf": 0.10,
            "wbf_iou": {CLS_HELMET: 0.45, CLS_VEST: 0.50, CLS_WORKER: 0.55},
            # Raised worker post_gate 0.24→0.28 to reject low-confidence material FPs.
            "post_gate": {CLS_HELMET: 0.22, CLS_VEST: 0.24, CLS_WORKER: 0.28},
            "recover_worker": 0.0,
        },
        "S2_dusty": {
            "pre_conf": 0.08,
            "wbf_iou": {CLS_HELMET: 0.42, CLS_VEST: 0.48, CLS_WORKER: 0.52},
            "post_gate": {CLS_HELMET: 0.18, CLS_VEST: 0.20, CLS_WORKER: 0.20},
            "recover_worker": 0.0,
        },
        "S3_low_light": {
            "pre_conf": 0.08,
            "wbf_iou": {CLS_HELMET: 0.40, CLS_VEST: 0.46, CLS_WORKER: 0.50},
            "post_gate": {CLS_HELMET: 0.16, CLS_VEST: 0.18, CLS_WORKER: 0.18},
            "recover_worker": 0.0,
        },
        "S4_crowded": {
            # RECALL FIX 2026-04-11:
            # wbf_iou[WORKER] 0.40→0.28 — prevent adjacent workers merging into one box.
            # post_gate[WORKER] 0.20→0.14 — allow weakly-agreed-on distant workers through.
            # post_gate[HELMET] 0.18→0.13 — small helmet recall for distant workers.
            # post_gate[VEST]   0.22→0.14 — partially-occluded vest recall.
            # recover_worker 0.32→0.20    — more worker candidates from raw outputs.
            "pre_conf": 0.07,
            "wbf_iou": {CLS_HELMET: 0.35, CLS_VEST: 0.42, CLS_WORKER: 0.28},
            "post_gate": {CLS_HELMET: 0.13, CLS_VEST: 0.14, CLS_WORKER: 0.14},
            "recover_worker": 0.20,
        },
    }
    return profiles[condition]


def wbf_fuse_condition(
    all_preds: List[Tuple[List[List[float]], List[float], List[int]]],
    img_w: int,
    img_h: int,
    condition: str,
    model_weights: List[float],
) -> List[Dict[str, object]]:
    profile = profile_for_condition(condition)
    flat = []
    for model_idx, (boxes, scores, labels) in enumerate(all_preds):
        for box, score, label in zip(boxes, scores, labels):
            norm = [box[0] / img_w, box[1] / img_h, box[2] / img_w, box[3] / img_h]
            flat.append((norm, float(score), int(label), model_idx))

    fused = []
    for cls_id in (CLS_HELMET, CLS_VEST, CLS_WORKER):
        cls_flat = [(b, s, mi) for b, s, l, mi in flat if l == cls_id]
        if not cls_flat:
            continue
        cls_flat.sort(key=lambda item: -item[1])
        used = [False] * len(cls_flat)
        iou_thresh = profile["wbf_iou"][cls_id]
        for i in range(len(cls_flat)):
            if used[i]:
                continue
            cluster_boxes = [cls_flat[i][0]]
            cluster_scores = [cls_flat[i][1]]
            cluster_models = [cls_flat[i][2]]
            used[i] = True

            for j in range(i + 1, len(cls_flat)):
                if used[j]:
                    continue
                if iou(cls_flat[i][0], cls_flat[j][0]) >= iou_thresh:
                    cluster_boxes.append(cls_flat[j][0])
                    cluster_scores.append(cls_flat[j][1])
                    cluster_models.append(cls_flat[j][2])
                    used[j] = True

            raw_weights = np.array(
                [model_weights[mi] * score for mi, score in zip(cluster_models, cluster_scores)],
                dtype=np.float32,
            )
            norm_weights = raw_weights / max(raw_weights.sum(), 1e-6)
            fused_box = np.average(cluster_boxes, axis=0, weights=norm_weights)
            source_bonus = 0.03 * max(0, len(set(cluster_models)) - 1)
            fused_score = float(np.average(cluster_scores, weights=norm_weights) + source_bonus)
            if fused_score < profile["post_gate"][cls_id]:
                continue

            fused.append(
                {
                    "box": [
                        fused_box[0] * img_w,
                        fused_box[1] * img_h,
                        fused_box[2] * img_w,
                        fused_box[3] * img_h,
                    ],
                    "score": min(0.99, fused_score),
                    "cls": cls_id,
                    "sources": len(set(cluster_models)),
                }
            )

    if condition == "S4_crowded":
        advance_recovery_frame()
        fused = recover_crowded_workers(fused, all_preds, profile["recover_worker"], img_h=img_h)

    return fused


# ── Per-region recovery cooldown ──────────────────────────────────────────────
# Maps a coarse spatial grid cell → frame index at which recovery last fired.
# Prevents the recovery pass from repeatedly re-triggering in the same spatial
# region across consecutive frames, which was generating wave-like duplicate
# boxes in dense areas of the frame.
_recovery_cooldown: Dict[str, int] = {}
_recovery_global_frame: int = 0
RECOVERY_COOLDOWN_FRAMES = 2   # lowered 4→2: recover newly-arrived workers faster


def _grid_cell(box: List[float], cell_size: int = 80) -> str:
    """Map a bounding box to a coarse grid cell string key."""
    cx = int((box[0] + box[2]) / 2.0 / cell_size)
    cy = int((box[1] + box[3]) / 2.0 / cell_size)
    return f"{cx},{cy}"


def advance_recovery_frame() -> None:
    """Increment the recovery frame counter.  Called once per inference cycle."""
    global _recovery_global_frame
    _recovery_global_frame += 1


def recover_crowded_workers(
    fused_boxes: List[Dict[str, object]],
    all_preds: List[Tuple[List[List[float]], List[float], List[int]]],
    min_score: float,
    img_h: int = 0,
) -> List[Dict[str, object]]:
    """
    Recover workers missed by WBF in crowded scenes.

    RECALL FIX 2026-04-11:
    - Duplicate IoU lowered 0.65→0.50 (adjacent workers have IoU 0.10-0.45)
    - Height floor lowered 24→14px (elevated wall workers appear smaller)
    - Aspect filter widened 0.90→1.30 (below-camera angle workers look wider)
    - Survival score thresholds lowered: 0.56→0.36, neighbor+0.52→neighbor+0.28
    - New elevated-zone path: top 40% of frame, score ≥ 0.25
    - PPE search expanded by ±25% horizontal margin
    """
    workers = [box for box in fused_boxes if box["cls"] == CLS_WORKER]
    ppe = [box for box in fused_boxes if box["cls"] in (CLS_HELMET, CLS_VEST)]
    recovered: List[Dict[str, object]] = []

    candidates = []
    for boxes, scores, labels in all_preds:
        for box, score, label in zip(boxes, scores, labels):
            if int(label) == CLS_WORKER and float(score) >= min_score:
                candidates.append({"box": list(box), "score": float(score), "cls": CLS_WORKER, "sources": 1})

    for cand in sorted(candidates, key=lambda item: -item["score"]):
        if any(iou(cand["box"], existing["box"]) >= 0.50 for existing in workers + recovered):
            continue

        width  = cand["box"][2] - cand["box"][0]
        height = cand["box"][3] - cand["box"][1]
        aspect = width / max(height, 1.0)

        if height < 14 or aspect > 1.30:
            continue

        # PPE search with ±25% horizontal margin
        cx1_exp = cand["box"][0] - width * 0.25
        cx2_exp = cand["box"][2] + width * 0.25
        ppe_support = 0
        for det in ppe:
            cx = (det["box"][0] + det["box"][2]) / 2.0
            cy = (det["box"][1] + det["box"][3]) / 2.0
            if cx1_exp <= cx <= cx2_exp and cand["box"][1] <= cy <= cand["box"][3]:
                ppe_support += 1

        neighbor_support = sum(1 for other in workers if iou(cand["box"], other["box"]) >= 0.02)

        # Elevated-zone: workers on walls/platforms are in top 40% of frame
        is_elevated = (img_h > 0 and cand["box"][1] < img_h * 0.40)

        survives = (
            ppe_support > 0
            or (neighbor_support > 0 and cand["score"] >= 0.28)
            or cand["score"] >= 0.36
            or (is_elevated and cand["score"] >= 0.25)
        )

        if survives:
            cell = _grid_cell(cand["box"])
            last_fired = _recovery_cooldown.get(cell, -999)
            if (_recovery_global_frame - last_fired) < RECOVERY_COOLDOWN_FRAMES:
                continue
            _recovery_cooldown[cell] = _recovery_global_frame
            recovered.append(cand)

    return fused_boxes + recovered


class TemporalPPEFilter:
    def __init__(self, max_misses: int = 6):
        self.max_misses = max_misses
        self._next_track_id = 1
        self.tracks: Dict[int, WorkerTrack] = {}

    def update(self, boxes: List[Dict[str, object]], frame_index: int) -> List[Dict[str, object]]:
        workers = [box for box in boxes if box["cls"] == CLS_WORKER]
        ppe = [box for box in boxes if box["cls"] in (CLS_HELMET, CLS_VEST)]
        assignments: Dict[int, int] = {}

        unmatched_track_ids = set(self.tracks.keys())
        for worker_idx, worker in enumerate(workers):
            best_track = None
            best_iou = 0.0
            for track_id, track in self.tracks.items():
                ov = iou(worker["box"], track.bbox)
                if ov > 0.30 and ov > best_iou:
                    best_iou = ov
                    best_track = track_id
            if best_track is None:
                best_track = self._next_track_id
                self._next_track_id += 1
                self.tracks[best_track] = WorkerTrack(track_id=best_track, bbox=list(worker["box"]))
            assignments[worker_idx] = best_track
            unmatched_track_ids.discard(best_track)

        for track_id in unmatched_track_ids:
            self.tracks[track_id].misses += 1

        for worker_idx, worker in enumerate(workers):
            track = self.tracks[assignments[worker_idx]]
            track.hits += 1
            track.misses = 0
            track.last_frame = frame_index
            track.bbox = list(worker["box"])
            human_score = float(worker.get("human_score", worker["score"]))
            track.human_score_ema = 0.6 * track.human_score_ema + 0.4 * human_score if track.hits > 1 else human_score
            worker["track_id"] = track.track_id
            # Record position for motion-based static suppression
            cx = (track.bbox[0] + track.bbox[2]) / 2.0
            cy = (track.bbox[1] + track.bbox[3]) / 2.0
            track.position_history.append([cx, cy])
            if len(track.position_history) > 20:  # rolling window
                track.position_history.pop(0)

            has_helmet = _worker_has_ppe(worker["box"], ppe, CLS_HELMET)
            has_vest = _worker_has_ppe(worker["box"], ppe, CLS_VEST)
            track.helmet_streak = min(6, track.helmet_streak + 1) if has_helmet else max(0, track.helmet_streak - 1)
            track.vest_streak = min(6, track.vest_streak + 1) if has_vest else max(0, track.vest_streak - 1)

            if not has_helmet and track.helmet_streak >= 2:
                inferred = _copy_ppe_from_track(worker["box"], CLS_HELMET, 0.16)
                if inferred is not None:
                    ppe.append(inferred)
            if not has_vest and track.vest_streak >= 2:
                inferred = _copy_ppe_from_track(worker["box"], CLS_VEST, 0.18)
                if inferred is not None:
                    ppe.append(inferred)

        stale = [track_id for track_id, track in self.tracks.items() if track.misses > self.max_misses]
        for track_id in stale:
            del self.tracks[track_id]

        static_filtered = []
        for worker in workers:
            track = self.tracks.get(worker.get("track_id"))
            if track is None:
                static_filtered.append(worker)
                continue

            # Ghost suppression: require a new detection to survive ≥ 2 consecutive
            # frames before it is displayed.
            # RECALL FIX: lowered score bypass 0.72→0.45 so workers that both
            # models agree on (fused ~0.35-0.44) are not invisible on frame 1.
            has_ppe = track.helmet_streak > 0 or track.vest_streak > 0
            if track.hits < 2 and not has_ppe and worker["score"] < 0.45:
                continue

            # Human-score gate: relaxed EMA floor 0.40→0.28 for S4 —
            # distant/elevated workers have genuinely lower human-score EMAs.
            if (track.hits >= 3
                    and track.human_score_ema < 0.28
                    and track.helmet_streak == 0
                    and track.vest_streak == 0):
                continue

            # Motion-based gate: objects that don't move across 6+ frames are
            # almost certainly static clutter (bags, machinery, scaffolding).
            # RECALL FIX: raised score bypass 0.80→0.60 so static workers
            # (workers standing still on walls) are not suppressed.
            if (track.is_static(min_hits=6, max_variance_px=3.0)
                    and worker["score"] < 0.60
                    and track.helmet_streak == 0
                    and track.vest_streak == 0):
                continue

            static_filtered.append(worker)

        output = static_filtered + ppe
        output.sort(key=lambda item: (item["cls"], -item["score"]))
        return output


def _worker_has_ppe(worker_box: List[float], ppe_boxes: List[Dict[str, object]], cls_id: int) -> bool:
    for det in ppe_boxes:
        if det["cls"] != cls_id:
            continue
        cx = (det["box"][0] + det["box"][2]) / 2.0
        cy = (det["box"][1] + det["box"][3]) / 2.0
        if worker_box[0] <= cx <= worker_box[2] and worker_box[1] <= cy <= worker_box[3]:
            return True
    return False


def _copy_ppe_from_track(worker_box: List[float], cls_id: int, score: float) -> Dict[str, object] | None:
    x1, y1, x2, y2 = worker_box
    w = x2 - x1
    h = y2 - y1
    if cls_id == CLS_HELMET:
        return {
            "box": [x1 + 0.28 * w, y1 - 0.04 * h, x1 + 0.72 * w, y1 + 0.22 * h],
            "score": score,
            "cls": cls_id,
            "synthetic": True,
        }
    if cls_id == CLS_VEST:
        return {
            "box": [x1 + 0.18 * w, y1 + 0.18 * h, x1 + 0.82 * w, y1 + 0.62 * h],
            "score": score,
            "cls": cls_id,
            "synthetic": True,
        }
    return None
