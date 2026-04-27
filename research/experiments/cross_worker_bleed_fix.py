"""
BuildSight — Cross-Worker PPE Bleeding Fix
===========================================
Problem:
  In crowded scenes, Worker A's vest detection box extends into Worker B's
  body region. Naive spatial association (overlap / nearest centroid) assigns
  the vest to BOTH workers — giving Worker B a false "vest detected" when
  they're actually non-compliant.

Root Cause:
  - Bounding boxes in crowded scenes overlap significantly
  - A vest bbox might overlap 40% with Worker A and 60% with Worker B
  - Centroid-based assignment picks the closest worker, but in overlap zones
    the centroid might fall inside the wrong worker's box
  - No check for whether the vest actually covers the worker's OWN torso

Solution — Torso-Center Alignment:
  Instead of "does the vest overlap with the worker's box?", ask:
  "does the vest cover the CENTER of THIS worker's torso?"

  A vest genuinely worn by a worker will:
    1. Have its horizontal center within ±25% of the worker's horizontal center
    2. Cover the worker's torso region (25-70% down from head)
    3. Have significant overlap with the worker's CENTRAL torso strip
       (not just the edges of their bounding box)

  A vest bleeding from an adjacent worker will:
    1. Be horizontally offset — its center is closer to the other worker
    2. Only overlap with the EDGE of this worker's box, not the center
    3. Fail the central-strip overlap test

Author: BuildSight / Green Build AI
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import cv2
import logging

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("BuildSight.BleedFix")


# ─────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────

@dataclass
class Detection:
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    class_id: int
    class_name: str
    confidence: float
    worker_id: Optional[int] = None
    merged_from: int = 1


@dataclass
class WorkerPPE:
    worker_id: int
    worker_box: Tuple[float, float, float, float]
    helmet: Optional[Detection] = None
    vest: Optional[Detection] = None

    @property
    def has_helmet(self) -> bool:
        return self.helmet is not None

    @property
    def has_vest(self) -> bool:
        return self.vest is not None

    @property
    def compliance(self) -> float:
        return (int(self.has_helmet) + int(self.has_vest)) / 2.0


# ─────────────────────────────────────────────────────────────
# Core Fix: Torso-Center Alignment Validator
# ─────────────────────────────────────────────────────────────

class TorsoCenterValidator:
    """
    Validates that a PPE detection genuinely belongs to a specific worker
    by checking alignment with the worker's CENTRAL torso strip.

    The key insight:
      ┌──────────────┐        ┌──────────────┐
      │   Worker 1   │        │   Worker 2   │
      │              │        │  ┌────────┐  │
      │    No vest   │←bleed──│  │  VEST  │  │
      │              │        │  └────────┘  │
      └──────────────┘        └──────────────┘

      The vest's horizontal center is aligned with Worker 2's center,
      NOT Worker 1's center. Even though the vest box overlaps with
      Worker 1's bounding box edge, it fails the torso-center test.

    Central torso strip:
      - Horizontal: middle 50% of worker box (25% inset from each side)
      - Vertical: 20% to 70% down from top of worker box
    """

    def __init__(
        self,
        # Horizontal alignment: vest center must be within this fraction
        # of worker center (relative to worker width)
        h_center_tolerance: float = 0.30,
        # Central torso strip: horizontal inset from each side
        torso_strip_inset: float = 0.20,
        # Minimum overlap between vest and central torso strip
        min_strip_overlap_ratio: float = 0.25,
        # Vertical torso range (fraction from top of worker)
        torso_top_frac: float = 0.15,
        torso_bot_frac: float = 0.75,
        # Minimum vest coverage of torso strip width
        min_width_coverage: float = 0.30,
    ):
        self.h_center_tol = h_center_tolerance
        self.strip_inset = torso_strip_inset
        self.min_strip_overlap = min_strip_overlap_ratio
        self.torso_top = torso_top_frac
        self.torso_bot = torso_bot_frac
        self.min_width_cov = min_width_coverage

    def vest_belongs_to_worker(
        self,
        vest: Detection,
        worker_box: Tuple[float, float, float, float],
        debug: bool = False,
    ) -> Tuple[bool, float]:
        """
        Check if a vest detection genuinely belongs to this worker.

        Returns:
            (belongs: bool, score: float)
            - belongs: True if the vest is actually on this worker
            - score: 0.0 to 1.0 quality score (used for picking best assignment)
        """
        wx1, wy1, wx2, wy2 = worker_box
        ww = wx2 - wx1
        wh = wy2 - wy1

        vx1, vy1, vx2, vy2 = vest.bbox
        vw = vx2 - vx1
        vh = vy2 - vy1

        if ww <= 0 or wh <= 0 or vw <= 0 or vh <= 0:
            return False, 0.0

        # ── Check 1: Horizontal Center Alignment ──
        # The vest's horizontal center must be close to the worker's center
        worker_cx = (wx1 + wx2) / 2
        vest_cx = (vx1 + vx2) / 2
        h_offset = abs(vest_cx - worker_cx) / ww

        h_aligned = h_offset <= self.h_center_tol

        if debug:
            logger.info(
                f"  H-center check: vest_cx={vest_cx:.0f}, "
                f"worker_cx={worker_cx:.0f}, offset={h_offset:.2f} "
                f"(tol={self.h_center_tol}) → {'PASS' if h_aligned else 'FAIL'}"
            )

        if not h_aligned:
            return False, 0.0

        # ── Check 2: Central Torso Strip Overlap ──
        # Define the worker's central torso strip
        strip_x1 = wx1 + ww * self.strip_inset
        strip_x2 = wx2 - ww * self.strip_inset
        strip_y1 = wy1 + wh * self.torso_top
        strip_y2 = wy1 + wh * self.torso_bot
        strip_w = strip_x2 - strip_x1
        strip_h = strip_y2 - strip_y1
        strip_area = strip_w * strip_h

        if strip_area <= 0:
            return False, 0.0

        # Intersection of vest with torso strip
        ix1 = max(vx1, strip_x1)
        iy1 = max(vy1, strip_y1)
        ix2 = min(vx2, strip_x2)
        iy2 = min(vy2, strip_y2)
        inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)

        strip_overlap = inter_area / strip_area
        passes_strip = strip_overlap >= self.min_strip_overlap

        if debug:
            logger.info(
                f"  Strip overlap: {strip_overlap:.2f} "
                f"(min={self.min_strip_overlap}) → {'PASS' if passes_strip else 'FAIL'}"
            )

        if not passes_strip:
            return False, 0.0

        # ── Check 3: Width Coverage ──
        # How much of the torso strip width does the vest actually cover?
        vest_in_strip_w = max(0, min(vx2, strip_x2) - max(vx1, strip_x1))
        width_coverage = vest_in_strip_w / strip_w if strip_w > 0 else 0
        passes_width = width_coverage >= self.min_width_cov

        if debug:
            logger.info(
                f"  Width coverage: {width_coverage:.2f} "
                f"(min={self.min_width_cov}) → {'PASS' if passes_width else 'FAIL'}"
            )

        if not passes_width:
            return False, 0.0

        # ── Compute quality score ──
        # Higher = better alignment with this worker's torso
        h_align_score = max(0, 1.0 - h_offset / self.h_center_tol)
        coverage_score = min(strip_overlap / 0.5, 1.0)  # Saturates at 50% overlap
        width_score = min(width_coverage / 0.6, 1.0)     # Saturates at 60% coverage

        quality = (
            0.35 * h_align_score +
            0.30 * coverage_score +
            0.20 * width_score +
            0.15 * vest.confidence
        )

        if debug:
            logger.info(
                f"  Quality score: {quality:.3f} "
                f"(h_align={h_align_score:.2f}, coverage={coverage_score:.2f}, "
                f"width={width_score:.2f}, conf={vest.confidence:.2f})"
            )

        return True, quality

    def helmet_belongs_to_worker(
        self,
        helmet: Detection,
        worker_box: Tuple[float, float, float, float],
    ) -> Tuple[bool, float]:
        """Same concept but for helmets — must be above the worker's head region."""
        wx1, wy1, wx2, wy2 = worker_box
        ww = wx2 - wx1
        wh = wy2 - wy1

        hx1, hy1, hx2, hy2 = helmet.bbox
        helmet_cx = (hx1 + hx2) / 2
        helmet_cy = (hy1 + hy2) / 2
        worker_cx = (wx1 + wx2) / 2

        if ww <= 0 or wh <= 0:
            return False, 0.0

        # Horizontal alignment (tighter for helmets — must be near head center)
        h_offset = abs(helmet_cx - worker_cx) / ww
        if h_offset > 0.35:
            return False, 0.0

        # Vertical position: helmet must be in top 35% of worker
        rel_y = (helmet_cy - wy1) / wh
        if rel_y > 0.40:
            return False, 0.0

        # Score
        h_score = max(0, 1.0 - h_offset / 0.35)
        v_score = max(0, 1.0 - rel_y / 0.40)
        quality = 0.4 * h_score + 0.4 * v_score + 0.2 * helmet.confidence

        return True, quality


# ─────────────────────────────────────────────────────────────
# Enhanced Worker-PPE Assigner (replaces naive overlap method)
# ─────────────────────────────────────────────────────────────

class StrictPPEAssigner:
    """
    Assigns PPE to workers using torso-center validation.

    Key difference from naive approach:
      - Naive: "vest overlaps with worker box → assign"
      - Strict: "vest covers worker's CENTRAL torso → assign"

    This prevents cross-worker bleeding in crowded scenes.
    """

    VEST_CLASSES = {'safety_vest', 'vest', 'safety vest'}
    HELMET_CLASSES = {'helmet', 'hard-hat', 'head', 'hard_hat'}
    WORKER_CLASSES = {'worker', 'person'}

    def __init__(self, validator: Optional[TorsoCenterValidator] = None):
        self.validator = validator or TorsoCenterValidator()

    def assign(
        self,
        detections: List[Detection],
        debug: bool = False,
    ) -> Tuple[List[WorkerPPE], List[Detection]]:
        """
        Assign PPE to workers with strict torso-center validation.

        Returns:
            (profiles, clean_detections)
        """
        # Separate by type
        workers = []
        vests = []
        helmets = []
        other = []

        for d in detections:
            name = d.class_name.lower()
            if name in self.WORKER_CLASSES:
                workers.append(d)
            elif name in self.VEST_CLASSES:
                vests.append(d)
            elif name in self.HELMET_CLASSES:
                helmets.append(d)
            else:
                other.append(d)

        # Create profiles
        profiles = []
        for wid, w in enumerate(workers):
            profiles.append(WorkerPPE(worker_id=wid, worker_box=w.bbox))

        # ── Assign vests (strict torso-center) ──
        # For each vest, find which worker it GENUINELY belongs to
        # by checking torso-center alignment with ALL workers
        # and picking the one with the highest quality score.
        vest_assignments: Dict[int, Tuple[Detection, float]] = {}  # worker_id → (vest, score)

        for vest in vests:
            best_worker = None
            best_score = 0.0

            if debug:
                logger.info(
                    f"\nVest at {tuple(map(int, vest.bbox))} "
                    f"conf={vest.confidence:.2f}:"
                )

            for wp in profiles:
                belongs, score = self.validator.vest_belongs_to_worker(
                    vest, wp.worker_box, debug=debug
                )

                if debug:
                    logger.info(
                        f"  → Worker {wp.worker_id}: "
                        f"{'BELONGS' if belongs else 'REJECTED'} "
                        f"(score={score:.3f})"
                    )

                if belongs and score > best_score:
                    best_score = score
                    best_worker = wp.worker_id

            if best_worker is not None:
                # Check if this worker already has a better vest
                if best_worker in vest_assignments:
                    existing_vest, existing_score = vest_assignments[best_worker]
                    if best_score > existing_score:
                        vest_assignments[best_worker] = (vest, best_score)
                        if debug:
                            logger.info(
                                f"  Replaced W{best_worker}'s vest "
                                f"(old={existing_score:.3f} → new={best_score:.3f})"
                            )
                else:
                    vest_assignments[best_worker] = (vest, best_score)

                if debug:
                    logger.info(f"  ✓ Assigned to Worker {best_worker}")
            else:
                if debug:
                    logger.info(f"  ✗ Unassigned (failed all worker checks)")

        # Apply vest assignments
        for wid, (vest, score) in vest_assignments.items():
            profiles[wid].vest = vest
            vest.worker_id = wid

        # ── Assign helmets (strict head-region check) ──
        helmet_assignments: Dict[int, Tuple[Detection, float]] = {}

        for helmet in helmets:
            best_worker = None
            best_score = 0.0

            for wp in profiles:
                belongs, score = self.validator.helmet_belongs_to_worker(
                    helmet, wp.worker_box
                )
                if belongs and score > best_score:
                    best_score = score
                    best_worker = wp.worker_id

            if best_worker is not None:
                if best_worker in helmet_assignments:
                    existing_helmet, existing_score = helmet_assignments[best_worker]
                    if best_score > existing_score:
                        helmet_assignments[best_worker] = (helmet, best_score)
                else:
                    helmet_assignments[best_worker] = (helmet, best_score)

        for wid, (helmet, score) in helmet_assignments.items():
            profiles[wid].helmet = helmet
            helmet.worker_id = wid

        # ── Build clean detection list ──
        clean = list(workers) + other
        for wp in profiles:
            if wp.vest:
                clean.append(wp.vest)
            if wp.helmet:
                clean.append(wp.helmet)

        # ── Summary ──
        assigned_vests = sum(1 for wp in profiles if wp.has_vest)
        assigned_helmets = sum(1 for wp in profiles if wp.has_helmet)
        dropped_vests = len(vests) - assigned_vests
        dropped_helmets = len(helmets) - assigned_helmets

        logger.info(
            f"PPE Assignment: {len(workers)} workers | "
            f"Vests: {len(vests)} detected → {assigned_vests} assigned "
            f"({dropped_vests} rejected as bleed) | "
            f"Helmets: {len(helmets)} detected → {assigned_helmets} assigned "
            f"({dropped_helmets} rejected)"
        )

        return profiles, clean


# ─────────────────────────────────────────────────────────────
# Complete 4-Stage Pipeline (adds bleed fix to existing 3-stage)
# ─────────────────────────────────────────────────────────────

class BuildSightPPEPipeline:
    """
    Complete 4-stage PPE detection post-processing:

      Stage 1: Cluster-Merge NMS    → merge duplicate fragments
      Stage 2: Coverage Validation   → reject tiny/oversized fragments
      Stage 3: Torso-Center Assign   → strict worker-PPE ownership
      Stage 4: Compliance Scoring    → per-worker PPE report

    This replaces both the old 3-stage pipeline AND the naive
    spatial association. Torso-center alignment is the key fix
    for cross-worker PPE bleeding in crowded scenes.
    """

    VEST_CLASSES = {'safety_vest', 'vest', 'safety vest'}
    HELMET_CLASSES = {'helmet', 'hard-hat', 'head', 'hard_hat'}
    WORKER_CLASSES = {'worker', 'person'}

    def __init__(
        self,
        # Stage 1: Cluster-Merge
        merge_iou: float = 0.10,
        containment_thresh: float = 0.50,
        proximity_ratio: float = 0.3,
        # Stage 2: Coverage
        min_vest_width_ratio: float = 0.30,
        min_vest_height_ratio: float = 0.10,
        # Stage 3: Torso-Center
        h_center_tolerance: float = 0.30,
        torso_strip_inset: float = 0.20,
        min_strip_overlap: float = 0.25,
    ):
        self.merge_iou = merge_iou
        self.containment_thresh = containment_thresh
        self.proximity_ratio = proximity_ratio
        self.min_vest_w = min_vest_width_ratio
        self.min_vest_h = min_vest_height_ratio

        self.assigner = StrictPPEAssigner(
            TorsoCenterValidator(
                h_center_tolerance=h_center_tolerance,
                torso_strip_inset=torso_strip_inset,
                min_strip_overlap_ratio=min_strip_overlap,
            )
        )

    def process(
        self,
        detections: List[Detection],
        debug: bool = False,
    ) -> Tuple[List[WorkerPPE], List[Detection]]:
        """Run all 4 stages."""
        n_raw = len(detections)

        # ── Stage 1: Cluster-Merge NMS ──
        merged = self._cluster_merge(detections)
        if debug:
            logger.info(f"Stage 1: {n_raw} → {len(merged)} (merged fragments)")

        # ── Stage 2: Coverage Validation ──
        # Pre-filter obvious bad detections before assignment
        validated = self._coverage_filter(merged)
        if debug:
            logger.info(f"Stage 2: {len(merged)} → {len(validated)} (coverage filter)")

        # ── Stage 3: Torso-Center Assignment ──
        profiles, clean = self.assigner.assign(validated, debug=debug)

        # ── Stage 4: Compliance Report ──
        if debug:
            for wp in profiles:
                status = "COMPLIANT" if wp.compliance >= 1.0 else "NON-COMPLIANT"
                logger.info(
                    f"Worker {wp.worker_id}: {status} | "
                    f"vest={wp.has_vest} helmet={wp.has_helmet}"
                )

        return profiles, clean

    # ── Stage 1: Cluster-Merge (from previous pipeline) ──

    def _cluster_merge(self, detections: List[Detection]) -> List[Detection]:
        """Merge overlapping same-class detections."""
        by_class: Dict[str, List[Detection]] = {}
        for d in detections:
            by_class.setdefault(d.class_name.lower(), []).append(d)

        result = []
        for cls_name, dets in by_class.items():
            if cls_name in self.WORKER_CLASSES:
                result.extend(dets)
            else:
                result.extend(self._merge_class_group(dets))
        return result

    def _merge_class_group(self, dets: List[Detection]) -> List[Detection]:
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
            for j in range(i + 1, n):
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
                # Union bounding box
                x1 = min(d.bbox[0] for d in group)
                y1 = min(d.bbox[1] for d in group)
                x2 = max(d.bbox[2] for d in group)
                y2 = max(d.bbox[3] for d in group)
                max_conf = max(d.confidence for d in group)
                merged_conf = min(0.99, max_conf + 0.05 * (len(group) - 1))
                result.append(Detection(
                    bbox=(x1, y1, x2, y2),
                    class_id=group[0].class_id,
                    class_name=group[0].class_name,
                    confidence=merged_conf,
                    merged_from=len(group),
                ))
        return result

    def _should_merge(self, a, b) -> bool:
        # IoU check
        iou = self._iou(a, b)
        if iou > self.merge_iou:
            return True
        # Containment check
        if self._containment(a, b) > self.containment_thresh:
            return True
        # Vertical proximity (for vest fragments)
        if self._vert_proximity(a, b):
            return True
        return False

    def _vert_proximity(self, a, b) -> bool:
        h_overlap = max(0, min(a[2], b[2]) - max(a[0], b[0]))
        min_w = min(a[2] - a[0], b[2] - b[0])
        if min_w <= 0 or h_overlap / min_w < 0.5:
            return False
        top, bot = (a, b) if a[1] < b[1] else (b, a)
        gap = max(0, bot[1] - top[3])
        max_h = max(a[3] - a[1], b[3] - b[1])
        return max_h > 0 and gap / max_h < self.proximity_ratio

    @staticmethod
    def _iou(a, b):
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        aa = (a[2] - a[0]) * (a[3] - a[1])
        ab = (b[2] - b[0]) * (b[3] - b[1])
        return inter / (aa + ab - inter) if (aa + ab - inter) > 0 else 0

    @staticmethod
    def _containment(a, b):
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        smaller = min((a[2]-a[0])*(a[3]-a[1]), (b[2]-b[0])*(b[3]-b[1]))
        return inter / smaller if smaller > 0 else 0

    # ── Stage 2: Coverage Filter ──

    def _coverage_filter(self, detections: List[Detection]) -> List[Detection]:
        """Pre-filter obviously wrong PPE detections by size."""
        workers = [d for d in detections if d.class_name.lower() in self.WORKER_CLASSES]
        if not workers:
            return detections

        # Average worker dimensions for reference
        avg_ww = np.mean([d.bbox[2] - d.bbox[0] for d in workers])
        avg_wh = np.mean([d.bbox[3] - d.bbox[1] for d in workers])

        result = []
        for d in detections:
            name = d.class_name.lower()
            if name in self.VEST_CLASSES:
                vw = d.bbox[2] - d.bbox[0]
                vh = d.bbox[3] - d.bbox[1]
                if vw < avg_ww * self.min_vest_w or vh < avg_wh * self.min_vest_h:
                    logger.debug(f"Coverage filter: dropped vest {vw:.0f}x{vh:.0f}")
                    continue
            result.append(d)
        return result


# ─────────────────────────────────────────────────────────────
# Visualization with Torso Strip Debug Overlay
# ─────────────────────────────────────────────────────────────

def draw_debug_overlay(
    frame: np.ndarray,
    profiles: List[WorkerPPE],
    show_torso_strip: bool = True,
) -> np.ndarray:
    """Draw results with optional torso strip visualization."""
    vis = frame.copy()

    COLORS = [
        (50, 205, 50), (255, 140, 0), (30, 144, 255),
        (255, 0, 255), (0, 255, 255), (255, 255, 0),
    ]

    for wp in profiles:
        color = COLORS[wp.worker_id % len(COLORS)]
        x1, y1, x2, y2 = map(int, wp.worker_box)
        ww = x2 - x1
        wh = y2 - y1

        # Worker box
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        # Torso strip overlay (the region that matters for vest assignment)
        if show_torso_strip:
            sx1 = int(x1 + ww * 0.20)
            sx2 = int(x2 - ww * 0.20)
            sy1 = int(y1 + wh * 0.15)
            sy2 = int(y1 + wh * 0.75)
            overlay = vis.copy()
            cv2.rectangle(overlay, (sx1, sy1), (sx2, sy2), color, -1)
            cv2.addWeighted(overlay, 0.15, vis, 0.85, 0, vis)
            cv2.rectangle(vis, (sx1, sy1), (sx2, sy2), color, 1)

        # Worker label
        comp = int(wp.compliance * 100)
        label = f"W{wp.worker_id} [{comp}%]"
        _put_label(vis, label, (x1, y1 - 8), color)

        # Vest
        if wp.vest:
            vx1, vy1, vx2, vy2 = map(int, wp.vest.bbox)
            cv2.rectangle(vis, (vx1, vy1), (vx2, vy2), (0, 220, 0), 2)
            vlbl = f"vest {wp.vest.confidence:.2f}"
            if wp.vest.merged_from > 1:
                vlbl += f" (m:{wp.vest.merged_from})"
            _put_label(vis, vlbl, (vx1, vy1 - 5), (0, 220, 0), 0.45)

        # Helmet
        if wp.helmet:
            hx1, hy1, hx2, hy2 = map(int, wp.helmet.bbox)
            cv2.rectangle(vis, (hx1, hy1), (hx2, hy2), (0, 200, 255), 2)
            _put_label(vis, f"helmet {wp.helmet.confidence:.2f}",
                      (hx1, hy1 - 5), (0, 200, 255), 0.45)

        # Status badges
        bx = x2 + 5
        _badge(vis, "H", bx, y1, (0, 200, 255) if wp.has_helmet else (0, 0, 220))
        _badge(vis, "V", bx, y1 + 25, (0, 220, 0) if wp.has_vest else (0, 0, 220))

    return vis


def _put_label(img, text, org, color, scale=0.55, thick=2):
    f = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, f, scale, thick)
    x, y = int(org[0]), int(org[1])
    cv2.rectangle(img, (x, y-th-6), (x+tw+4, y+2), color, -1)
    cv2.putText(img, text, (x+2, y-2), f, scale, (255,255,255), thick)


def _badge(img, text, x, y, color):
    cv2.rectangle(img, (x, y), (x+25, y+22), color, -1)
    cv2.putText(img, text, (x+5, y+17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)


# ─────────────────────────────────────────────────────────────
# Runner with YOLOv8
# ─────────────────────────────────────────────────────────────

def run(
    image_path: str,
    model_path: str = "best.pt",
    output_path: str = "output_bleed_fixed.jpg",
    conf: float = 0.25,
    iou: float = 0.45,
    debug: bool = True,
    show: bool = False,
):
    """Run the complete 4-stage pipeline."""
    from ultralytics import YOLO

    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"Cannot read: {image_path}")

    model = YOLO(model_path)
    pipeline = BuildSightPPEPipeline()

    # Detect
    results = model.predict(frame, conf=conf, iou=iou, verbose=False)
    raw_dets = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            cls_id = int(box.cls[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            raw_dets.append(Detection(
                bbox=(x1, y1, x2, y2),
                class_id=cls_id,
                class_name=model.names.get(cls_id, "unknown"),
                confidence=float(box.conf[0]),
            ))

    print(f"\nRaw: {len(raw_dets)} detections")

    # Process
    profiles, clean = pipeline.process(raw_dets, debug=debug)

    # Report
    print(f"\n{'='*55}")
    print(f"  BUILDSIGHT PPE REPORT (4-Stage Pipeline)")
    print(f"{'='*55}")
    for wp in profiles:
        s = "COMPLIANT" if wp.compliance >= 1.0 else "NON-COMPLIANT"
        print(f"\n  Worker {wp.worker_id} — {s} ({wp.compliance*100:.0f}%)")
        print(f"    Vest:   {'YES' if wp.has_vest else 'MISSING'}")
        print(f"    Helmet: {'YES' if wp.has_helmet else 'MISSING'}")
    print(f"\n{'='*55}")

    # Visualize
    vis = draw_debug_overlay(frame, profiles, show_torso_strip=True)
    cv2.imwrite(output_path, vis)
    print(f"Saved: {output_path}")

    if show:
        cv2.imshow("BuildSight 4-Stage", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return profiles


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--model", default="best.pt")
    p.add_argument("--output", default="output_bleed_fixed.jpg")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--show", action="store_true")
    a = p.parse_args()
    run(a.source, a.model, a.output, a.conf, a.iou, a.debug, a.show)
