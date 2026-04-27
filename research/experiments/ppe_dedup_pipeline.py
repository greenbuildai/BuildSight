"""
BuildSight — PPE Deduplication & Vest Fix Pipeline
====================================================
Fixes two critical vest detection failures:
  1. DUPLICATE VESTS: One vest detected as 2-4 separate boxes
  2. PARTIAL VESTS:  Only a fragment of the vest is detected

Root Cause Analysis:
  - Standard NMS fails because vest fragments have IoU < threshold
    (top-half and bottom-half of same vest barely overlap)
  - Box-expand (20%up/15%down) pushes fragments further apart
  - No worker-anchoring: detections float freely, no "1 vest per worker" rule

Solution: Three-stage post-processing pipeline:
  Stage 1: Cluster-Merge NMS  — merge overlapping same-class boxes instead of suppress
  Stage 2: Worker-Anchored PPE — enforce 1 helmet + 1 vest per worker
  Stage 3: Coverage Validation — reject partial vests that cover < 40% of expected area

Author: BuildSight / Green Build AI
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from ultralytics import YOLO
import logging

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("BuildSight.VestFix")


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
    merged_from: int = 1  # how many raw detections were merged into this


@dataclass
class WorkerPPE:
    worker_id: int
    worker_box: Tuple[float, float, float, float]
    helmet: Optional[Detection] = None
    vest: Optional[Detection] = None
    all_ppe: List[Detection] = field(default_factory=list)

    @property
    def has_helmet(self) -> bool:
        return self.helmet is not None

    @property
    def has_vest(self) -> bool:
        return self.vest is not None

    @property
    def compliance(self) -> float:
        score = 0
        if self.has_helmet:
            score += 1
        if self.has_vest:
            score += 1
        return score / 2.0


# ─────────────────────────────────────────────────────────────
# Stage 1: Cluster-Merge NMS
# ─────────────────────────────────────────────────────────────

class ClusterMergeNMS:
    """
    Instead of suppressing overlapping same-class boxes, MERGE them.

    Standard NMS problem:
      - 3 vest fragments: [top-half], [bottom-half], [middle]
      - IoU between top-half and bottom-half < 0.15 → both survive
      - Result: 3 vest boxes on one worker

    Cluster-Merge solution:
      - Group all same-class boxes whose pairwise IoU OR containment
        exceeds a threshold
      - Merge each cluster into one box (confidence-weighted union)
      - Result: 1 vest box per cluster, covering the full vest area
    """

    def __init__(
        self,
        merge_iou_threshold: float = 0.10,
        containment_threshold: float = 0.50,
        proximity_threshold_ratio: float = 0.3,
    ):
        self.merge_iou = merge_iou_threshold
        self.containment = containment_threshold
        self.proximity_ratio = proximity_threshold_ratio

    def process(self, detections: List[Detection]) -> List[Detection]:
        """Merge overlapping same-class detections."""
        if not detections:
            return []

        # Group by class
        by_class: Dict[str, List[Detection]] = {}
        for d in detections:
            by_class.setdefault(d.class_name, []).append(d)

        merged_all = []
        for cls_name, dets in by_class.items():
            if cls_name.lower() in ('worker', 'person'):
                # Don't merge worker boxes — they're different people
                merged_all.extend(dets)
            else:
                merged = self._merge_class(dets)
                merged_all.extend(merged)

        return merged_all

    def _merge_class(self, dets: List[Detection]) -> List[Detection]:
        """Cluster and merge detections of the same class."""
        n = len(dets)
        if n <= 1:
            return dets

        # Build adjacency: two detections are "same object" if
        # IoU > threshold OR one contains the other OR they're very close
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

        # Collect clusters
        clusters: Dict[int, List[int]] = {}
        for i in range(n):
            root = find(i)
            clusters.setdefault(root, []).append(i)

        # Merge each cluster
        result = []
        for indices in clusters.values():
            cluster_dets = [dets[i] for i in indices]
            merged = self._merge_cluster(cluster_dets)
            result.append(merged)

        return result

    def _should_merge(self, box_a, box_b) -> bool:
        """Check if two boxes likely represent the same physical object."""
        iou = self._iou(box_a, box_b)
        if iou > self.merge_iou:
            return True

        # Containment: is one box mostly inside the other?
        cont = self._containment_ratio(box_a, box_b)
        if cont > self.containment:
            return True

        # Vertical proximity: vest fragments are often stacked vertically
        # on the same worker (top-half and bottom-half of vest)
        if self._vertical_proximity(box_a, box_b):
            return True

        return False

    def _vertical_proximity(self, box_a, box_b) -> bool:
        """
        Check if two boxes are vertically stacked with significant
        horizontal overlap — classic vest fragment pattern.

        Pattern: [top-vest] and [bottom-vest] on same worker
        - Horizontal overlap > 60% of narrower box
        - Vertical gap < 30% of taller box height
        """
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        # Horizontal overlap
        h_overlap = max(0, min(ax2, bx2) - max(ax1, bx1))
        min_width = min(ax2 - ax1, bx2 - bx1)
        if min_width <= 0:
            return False
        h_overlap_ratio = h_overlap / min_width

        if h_overlap_ratio < 0.5:
            return False

        # Vertical gap (distance between bottom of top box and top of bottom box)
        top_box = box_a if ay1 < by1 else box_b
        bot_box = box_b if ay1 < by1 else box_a
        v_gap = max(0, bot_box[1] - top_box[3])

        max_height = max(ay2 - ay1, by2 - by1)
        if max_height <= 0:
            return False
        v_gap_ratio = v_gap / max_height

        return v_gap_ratio < self.proximity_ratio

    def _merge_cluster(self, dets: List[Detection]) -> Detection:
        """
        Merge a cluster into one detection using confidence-weighted union.

        Box: Union of all boxes (enclosing box)
        Confidence: Weighted average, boosted slightly for multi-detection clusters
        """
        if len(dets) == 1:
            return dets[0]

        # Union bounding box
        x1 = min(d.bbox[0] for d in dets)
        y1 = min(d.bbox[1] for d in dets)
        x2 = max(d.bbox[2] for d in dets)
        y2 = max(d.bbox[3] for d in dets)

        # Confidence: weighted average (higher conf detections dominate)
        total_conf = sum(d.confidence for d in dets)
        if total_conf > 0:
            # Weighted average, slightly boosted for multi-evidence
            avg_conf = total_conf / len(dets)
            max_conf = max(d.confidence for d in dets)
            # Boost: multiple detections = more evidence the object is real
            merged_conf = min(0.99, max_conf + 0.05 * (len(dets) - 1))
        else:
            merged_conf = 0.0

        return Detection(
            bbox=(x1, y1, x2, y2),
            class_id=dets[0].class_id,
            class_name=dets[0].class_name,
            confidence=merged_conf,
            merged_from=len(dets),
        )

    @staticmethod
    def _iou(a, b) -> float:
        ix1 = max(a[0], b[0])
        iy1 = max(a[1], b[1])
        ix2 = min(a[2], b[2])
        iy2 = min(a[3], b[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        aa = (a[2] - a[0]) * (a[3] - a[1])
        ab = (b[2] - b[0]) * (b[3] - b[1])
        union = aa + ab - inter
        return inter / union if union > 0 else 0

    @staticmethod
    def _containment_ratio(a, b) -> float:
        """What fraction of the smaller box is inside the larger?"""
        ix1 = max(a[0], b[0])
        iy1 = max(a[1], b[1])
        ix2 = min(a[2], b[2])
        iy2 = min(a[3], b[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        smaller = min(area_a, area_b)
        return inter / smaller if smaller > 0 else 0


# ─────────────────────────────────────────────────────────────
# Stage 2: Worker-Anchored PPE Deduplication
# ─────────────────────────────────────────────────────────────

class WorkerAnchoredPPE:
    """
    Enforce physical constraints:
      - Each worker can wear at most 1 helmet and 1 vest
      - PPE must be spatially inside or near its worker
      - If multiple vests remain after merge, keep the best one

    This catches any duplicates that Cluster-Merge missed.
    """

    VEST_CLASSES = {'safety_vest', 'vest', 'safety vest', 'safety_vest'}
    HELMET_CLASSES = {'helmet', 'hard-hat', 'head', 'hard_hat'}
    WORKER_CLASSES = {'worker', 'person'}

    def __init__(
        self,
        vest_expected_ratio: Tuple[float, float] = (0.25, 0.70),
        helmet_expected_ratio: Tuple[float, float] = (0.05, 0.35),
    ):
        # Expected vertical position of PPE relative to worker box
        # (fraction from top of worker box)
        self.vest_y_range = vest_expected_ratio
        self.helmet_y_range = helmet_expected_ratio

    def process(
        self, detections: List[Detection]
    ) -> Tuple[List[WorkerPPE], List[Detection]]:
        """
        Returns:
          - List of WorkerPPE profiles (one per worker)
          - Cleaned detection list (duplicates removed)
        """
        # Separate workers and PPE
        workers = []
        ppe_items = []
        other = []

        for d in detections:
            name = d.class_name.lower()
            if name in self.WORKER_CLASSES:
                workers.append(d)
            elif name in self.VEST_CLASSES or name in self.HELMET_CLASSES:
                ppe_items.append(d)
            else:
                other.append(d)

        # Create worker profiles
        profiles = []
        for wid, w in enumerate(workers):
            profiles.append(WorkerPPE(worker_id=wid, worker_box=w.bbox))

        # Assign PPE to workers
        assigned_ppe = set()

        for ppe in ppe_items:
            best_worker = self._assign_to_worker(ppe, profiles)
            if best_worker is None:
                continue

            wp = profiles[best_worker]
            name = ppe.class_name.lower()

            if name in self.VEST_CLASSES:
                if wp.vest is None:
                    wp.vest = ppe
                    ppe.worker_id = best_worker
                    assigned_ppe.add(id(ppe))
                else:
                    # Already has a vest → keep better one
                    existing_score = self._vest_quality_score(wp.vest, wp.worker_box)
                    new_score = self._vest_quality_score(ppe, wp.worker_box)
                    if new_score > existing_score:
                        wp.vest = ppe
                        ppe.worker_id = best_worker
                        assigned_ppe.add(id(ppe))
                        logger.debug(
                            f"W{best_worker}: Replaced vest "
                            f"(old={existing_score:.2f}, new={new_score:.2f})"
                        )

            elif name in self.HELMET_CLASSES:
                if wp.helmet is None:
                    wp.helmet = ppe
                    ppe.worker_id = best_worker
                    assigned_ppe.add(id(ppe))
                else:
                    # Keep higher confidence helmet
                    if ppe.confidence > wp.helmet.confidence:
                        wp.helmet = ppe
                        ppe.worker_id = best_worker
                        assigned_ppe.add(id(ppe))

            wp.all_ppe = [x for x in [wp.helmet, wp.vest] if x is not None]

        # Build clean detection list
        clean_dets = list(workers) + other
        for wp in profiles:
            if wp.helmet:
                clean_dets.append(wp.helmet)
            if wp.vest:
                clean_dets.append(wp.vest)

        return profiles, clean_dets

    def _assign_to_worker(
        self, ppe: Detection, profiles: List[WorkerPPE]
    ) -> Optional[int]:
        """Find which worker this PPE belongs to."""
        px = (ppe.bbox[0] + ppe.bbox[2]) / 2
        py = (ppe.bbox[1] + ppe.bbox[3]) / 2

        best_idx = None
        best_score = -1

        for i, wp in enumerate(profiles):
            wx1, wy1, wx2, wy2 = wp.worker_box
            ww = wx2 - wx1
            wh = wy2 - wy1

            # Check if PPE centroid is within expanded worker region
            # (expand by 20% to catch edge PPE)
            margin_x = ww * 0.2
            margin_y = wh * 0.2
            if not (wx1 - margin_x <= px <= wx2 + margin_x and
                    wy1 - margin_y <= py <= wy2 + margin_y):
                continue

            # Score: combination of overlap + expected position
            overlap = self._overlap_ratio(ppe.bbox, wp.worker_box)
            position_score = self._position_score(ppe, wp.worker_box)
            score = 0.5 * overlap + 0.5 * position_score

            if score > best_score:
                best_score = score
                best_idx = i

        return best_idx

    def _vest_quality_score(self, vest: Detection, worker_box) -> float:
        """
        Score a vest detection by how well it matches expected vest properties.
        Higher = better vest detection.

        Factors:
          - Coverage: vest should cover significant portion of worker torso
          - Position: vest should be in the middle 25-70% of worker height
          - Aspect ratio: vest is wider than tall (landscape-ish)
          - Confidence: model confidence
        """
        wx1, wy1, wx2, wy2 = worker_box
        ww = wx2 - wx1
        wh = wy2 - wy1

        vx1, vy1, vx2, vy2 = vest.bbox
        vw = vx2 - vx1
        vh = vy2 - vy1

        if ww <= 0 or wh <= 0 or vw <= 0 or vh <= 0:
            return 0.0

        # Coverage: what fraction of worker width does the vest span?
        width_coverage = min(vw / ww, 1.0)

        # Vertical coverage of torso region
        torso_top = wy1 + wh * self.vest_y_range[0]
        torso_bot = wy1 + wh * self.vest_y_range[1]
        torso_h = torso_bot - torso_top

        vest_in_torso_top = max(vy1, torso_top)
        vest_in_torso_bot = min(vy2, torso_bot)
        torso_coverage = max(0, vest_in_torso_bot - vest_in_torso_top) / torso_h if torso_h > 0 else 0

        # Position: is the vest vertically centered in the torso?
        vest_center_y = (vy1 + vy2) / 2
        torso_center_y = (torso_top + torso_bot) / 2
        position_error = abs(vest_center_y - torso_center_y) / wh
        position_score = max(0, 1.0 - position_error * 3)

        # Composite score
        score = (
            0.30 * width_coverage +
            0.30 * torso_coverage +
            0.20 * position_score +
            0.20 * vest.confidence
        )

        return score

    def _position_score(self, ppe: Detection, worker_box) -> float:
        """How well does the PPE position match expected anatomy?"""
        wx1, wy1, wx2, wy2 = worker_box
        wh = wy2 - wy1
        if wh <= 0:
            return 0.0

        py = (ppe.bbox[1] + ppe.bbox[3]) / 2
        rel_y = (py - wy1) / wh  # 0 = top of worker, 1 = bottom

        name = ppe.class_name.lower()
        if name in self.VEST_CLASSES:
            # Vest should be in 25-70% range
            if self.vest_y_range[0] <= rel_y <= self.vest_y_range[1]:
                return 1.0
            else:
                dist = min(abs(rel_y - self.vest_y_range[0]),
                          abs(rel_y - self.vest_y_range[1]))
                return max(0, 1.0 - dist * 3)
        elif name in self.HELMET_CLASSES:
            # Helmet should be in 0-35% range
            if self.helmet_y_range[0] <= rel_y <= self.helmet_y_range[1]:
                return 1.0
            else:
                dist = min(abs(rel_y - self.helmet_y_range[0]),
                          abs(rel_y - self.helmet_y_range[1]))
                return max(0, 1.0 - dist * 3)
        return 0.5

    @staticmethod
    def _overlap_ratio(box_a, box_b) -> float:
        """Fraction of box_a that overlaps with box_b."""
        ix1 = max(box_a[0], box_b[0])
        iy1 = max(box_a[1], box_b[1])
        ix2 = min(box_a[2], box_b[2])
        iy2 = min(box_a[3], box_b[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        return inter / area_a if area_a > 0 else 0


# ─────────────────────────────────────────────────────────────
# Stage 3: Coverage Validation
# ─────────────────────────────────────────────────────────────

class CoverageValidator:
    """
    Reject vest detections that are too small (partial fragments)
    or too large (false positive covering whole body).

    Expected vest size relative to worker:
      Width:  50-100% of worker width
      Height: 20-55% of worker height
    """

    def __init__(
        self,
        min_width_ratio: float = 0.35,
        max_width_ratio: float = 1.15,
        min_height_ratio: float = 0.12,
        max_height_ratio: float = 0.60,
        min_area_ratio: float = 0.06,
    ):
        self.min_w = min_width_ratio
        self.max_w = max_width_ratio
        self.min_h = min_height_ratio
        self.max_h = max_height_ratio
        self.min_area = min_area_ratio

    def validate_vest(self, vest: Detection, worker_box: Tuple) -> bool:
        """Returns True if vest passes coverage validation."""
        wx1, wy1, wx2, wy2 = worker_box
        ww = wx2 - wx1
        wh = wy2 - wy1
        w_area = ww * wh

        vx1, vy1, vx2, vy2 = vest.bbox
        vw = vx2 - vx1
        vh = vy2 - vy1
        v_area = vw * vh

        if ww <= 0 or wh <= 0:
            return True  # Can't validate without worker

        width_ratio = vw / ww
        height_ratio = vh / wh
        area_ratio = v_area / w_area if w_area > 0 else 0

        # Check bounds
        if width_ratio < self.min_w:
            logger.debug(f"Vest rejected: too narrow ({width_ratio:.2f} < {self.min_w})")
            return False
        if width_ratio > self.max_w:
            logger.debug(f"Vest rejected: too wide ({width_ratio:.2f} > {self.max_w})")
            return False
        if height_ratio < self.min_h:
            logger.debug(f"Vest rejected: too short ({height_ratio:.2f} < {self.min_h})")
            return False
        if height_ratio > self.max_h:
            logger.debug(f"Vest rejected: too tall ({height_ratio:.2f} > {self.max_h})")
            return False
        if area_ratio < self.min_area:
            logger.debug(f"Vest rejected: too small ({area_ratio:.2f} < {self.min_area})")
            return False

        return True


# ─────────────────────────────────────────────────────────────
# Main Pipeline: Combines all 3 stages
# ─────────────────────────────────────────────────────────────

class PPEDeduplicationPipeline:
    """
    Complete pipeline:
      Raw detections → Stage 1 (Merge) → Stage 2 (Anchor) → Stage 3 (Validate)
                     → Clean, deduplicated PPE per worker

    Usage:
        pipeline = PPEDeduplicationPipeline()
        model = YOLO("best.pt")

        results = model.predict(frame, conf=0.25, iou=0.45)
        raw_dets = parse_yolo_results(results)

        profiles, clean_dets = pipeline.process(raw_dets)

        for wp in profiles:
            print(f"Worker {wp.worker_id}: vest={wp.has_vest}, helmet={wp.has_helmet}")
    """

    def __init__(
        self,
        merge_iou: float = 0.10,
        containment_thresh: float = 0.50,
        proximity_ratio: float = 0.3,
        min_vest_width_ratio: float = 0.35,
        min_vest_height_ratio: float = 0.12,
        min_vest_area_ratio: float = 0.06,
    ):
        self.merger = ClusterMergeNMS(
            merge_iou_threshold=merge_iou,
            containment_threshold=containment_thresh,
            proximity_threshold_ratio=proximity_ratio,
        )
        self.anchorer = WorkerAnchoredPPE()
        self.validator = CoverageValidator(
            min_width_ratio=min_vest_width_ratio,
            min_height_ratio=min_vest_height_ratio,
            min_area_ratio=min_vest_area_ratio,
        )

    def process(
        self, detections: List[Detection]
    ) -> Tuple[List[WorkerPPE], List[Detection]]:
        """
        Full pipeline: merge → anchor → validate.

        Returns (worker_profiles, clean_detections)
        """
        n_raw = len(detections)

        # ── Stage 1: Cluster-Merge NMS ──
        merged = self.merger.process(detections)
        n_merged = len(merged)
        n_eliminated_s1 = n_raw - n_merged

        # ── Stage 2: Worker-Anchored PPE ──
        profiles, clean_dets = self.anchorer.process(merged)
        n_after_anchor = len(clean_dets)

        # ── Stage 3: Coverage Validation ──
        rejected_vests = 0
        for wp in profiles:
            if wp.vest and not self.validator.validate_vest(wp.vest, wp.worker_box):
                rejected_vests += 1
                wp.vest = None
                wp.all_ppe = [x for x in [wp.helmet, wp.vest] if x is not None]

        # Rebuild clean_dets after validation
        final_dets = []
        for d in clean_dets:
            name = d.class_name.lower()
            if name in WorkerAnchoredPPE.VEST_CLASSES:
                # Only keep if it's still assigned to a worker
                if any(wp.vest is d for wp in profiles):
                    final_dets.append(d)
            else:
                final_dets.append(d)

        logger.info(
            f"Pipeline: {n_raw} raw → {n_merged} merged (S1: -{n_eliminated_s1}) "
            f"→ {n_after_anchor} anchored → {len(final_dets)} final "
            f"(S3: -{rejected_vests} partial vests)"
        )

        return profiles, final_dets


# ─────────────────────────────────────────────────────────────
# Integration with YOLOv8
# ─────────────────────────────────────────────────────────────

def parse_yolo_results(results, class_names: dict) -> List[Detection]:
    """Convert YOLO results to Detection objects."""
    dets = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            cls_id = int(box.cls[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            dets.append(Detection(
                bbox=(x1, y1, x2, y2),
                class_id=cls_id,
                class_name=class_names.get(cls_id, "unknown"),
                confidence=float(box.conf[0]),
            ))
    return dets


def draw_results(
    frame: np.ndarray,
    profiles: List[WorkerPPE],
    clean_dets: List[Detection],
    show_merged_count: bool = True,
) -> np.ndarray:
    """Visualize with per-worker color coding and compliance badges."""
    vis = frame.copy()

    COLORS = [
        (0, 255, 0), (255, 128, 0), (0, 128, 255),
        (255, 0, 255), (255, 255, 0), (0, 255, 255),
    ]
    VEST_COLOR = (0, 220, 0)
    HELMET_COLOR = (0, 200, 255)
    MISSING_COLOR = (0, 0, 220)

    for wp in profiles:
        color = COLORS[wp.worker_id % len(COLORS)]
        x1, y1, x2, y2 = map(int, wp.worker_box)

        # Worker box
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        # Worker label
        comp_pct = int(wp.compliance * 100)
        label = f"W{wp.worker_id} [{comp_pct}%]"
        _draw_label(vis, label, (x1, y1 - 8), color)

        # Vest box
        if wp.vest:
            vx1, vy1, vx2, vy2 = map(int, wp.vest.bbox)
            cv2.rectangle(vis, (vx1, vy1), (vx2, vy2), VEST_COLOR, 2)
            vlabel = f"vest {wp.vest.confidence:.2f}"
            if show_merged_count and wp.vest.merged_from > 1:
                vlabel += f" (merged:{wp.vest.merged_from})"
            _draw_label(vis, vlabel, (vx1, vy1 - 5), VEST_COLOR, font_scale=0.45)

        # Helmet box
        if wp.helmet:
            hx1, hy1, hx2, hy2 = map(int, wp.helmet.bbox)
            cv2.rectangle(vis, (hx1, hy1), (hx2, hy2), HELMET_COLOR, 2)
            hlabel = f"helmet {wp.helmet.confidence:.2f}"
            if show_merged_count and wp.helmet.merged_from > 1:
                hlabel += f" (merged:{wp.helmet.merged_from})"
            _draw_label(vis, hlabel, (hx1, hy1 - 5), HELMET_COLOR, font_scale=0.45)

        # PPE status badges
        badge_x = x2 + 5
        _draw_badge(vis, "H", badge_x, y1, HELMET_COLOR if wp.has_helmet else MISSING_COLOR)
        _draw_badge(vis, "V", badge_x, y1 + 25, VEST_COLOR if wp.has_vest else MISSING_COLOR)

    return vis


def _draw_label(img, text, org, color, font_scale=0.55, thickness=2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = int(org[0]), int(org[1])
    cv2.rectangle(img, (x, y - th - 6), (x + tw + 4, y + 2), color, -1)
    cv2.putText(img, text, (x + 2, y - 2), font, font_scale, (255, 255, 255), thickness)


def _draw_badge(img, text, x, y, color):
    cv2.rectangle(img, (x, y), (x + 25, y + 22), color, -1)
    cv2.putText(img, text, (x + 5, y + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)


# ─────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────

def run_detection(
    image_path: str,
    model_path: str = "best.pt",
    output_path: str = "output_vest_fixed.jpg",
    conf: float = 0.25,
    iou: float = 0.45,
    show: bool = False,
):
    """Run the full pipeline on an image."""
    # Load
    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"Cannot read: {image_path}")

    model = YOLO(model_path)
    pipeline = PPEDeduplicationPipeline()

    # Detect
    results = model.predict(frame, conf=conf, iou=iou, verbose=False)
    raw_dets = parse_yolo_results(results, model.names)

    print(f"\nRaw detections: {len(raw_dets)}")
    for d in raw_dets:
        print(f"  {d.class_name}: {d.confidence:.2f} at {tuple(map(int, d.bbox))}")

    # Process through pipeline
    profiles, clean_dets = pipeline.process(raw_dets)

    # Report
    print(f"\n{'='*50}")
    print(f"  VEST FIX RESULTS")
    print(f"{'='*50}")
    for wp in profiles:
        status = "COMPLIANT" if wp.compliance >= 1.0 else "NON-COMPLIANT"
        print(f"\n  Worker {wp.worker_id} — {status} ({wp.compliance*100:.0f}%)")
        if wp.vest:
            print(f"    Vest:   conf={wp.vest.confidence:.2f}, merged_from={wp.vest.merged_from}")
        else:
            print(f"    Vest:   MISSING")
        if wp.helmet:
            print(f"    Helmet: conf={wp.helmet.confidence:.2f}, merged_from={wp.helmet.merged_from}")
        else:
            print(f"    Helmet: MISSING")
    print(f"\n{'='*50}")

    # Visualize
    vis = draw_results(frame, profiles, clean_dets)
    cv2.imwrite(output_path, vis)
    print(f"\nSaved: {output_path}")

    if show:
        cv2.imshow("BuildSight Vest Fix", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return profiles, clean_dets


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="BuildSight PPE Dedup Pipeline")
    parser.add_argument("--source", required=True, help="Image path")
    parser.add_argument("--model", default="best.pt", help="YOLOv8 weights")
    parser.add_argument("--output", default="output_vest_fixed.jpg")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    run_detection(
        args.source, args.model, args.output,
        conf=args.conf, iou=args.iou, show=args.show,
    )
