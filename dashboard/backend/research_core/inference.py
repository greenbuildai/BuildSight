import cv2
import numpy as np
import logging
import torch
import os
from datetime import datetime
from typing import List, Dict, Any
from ultralytics import YOLO
from . import config
from .schemas import DetectionResult, DetectedObject, ComplianceStatus

logger = logging.getLogger(__name__)

class InferenceService:
    def __init__(self, model_path: str = config.MODEL_PATH):
        # Hardware Acceleration Check
        if torch.cuda.is_available():
            preferred_device = 'cuda'
        else:
            preferred_device = 'cpu'

        logger.info(f"Loading YOLO model from {model_path} to {preferred_device.upper()}...")

        self.model = YOLO(model_path)
        try:
            self.model.to(preferred_device)
            self.device = preferred_device
            logger.info(f"Model successfully loaded on {preferred_device.upper()}")
        except RuntimeError as exc:
            # Fall back to CPU for invalid/unsupported device strings.
            logger.warning("Device init failed (%s). Falling back to CPU.", exc)
            preferred_device = "cpu"
            self.model.to(preferred_device)
            self.device = preferred_device
        
        self.class_names = config.CLASS_NAMES
        # Reverse mapping for internal checks
        self.class_ids = {v: k for k, v in config.CLASS_NAMES.items()}

    def reset(self) -> None:
        tracker = getattr(self.model, "tracker", None)
        if tracker:
            tracker.reset()
        predictor = getattr(self.model, "predictor", None)
        if predictor and getattr(predictor, "trackers", None):
            for trk in predictor.trackers:
                trk.reset()
    
    def detect(self, frame: np.ndarray, frame_id: int) -> DetectionResult:
        # Run inference with ByteTrack
        # persist=True tells the model to keep track of objects between frames
        results = self.model.track(frame, persist=True, verbose=False, tracker="bytetrack.yaml")

        # Diagnostic logging
        first_boxes = results[0].boxes if results else None
        total_objects = len(first_boxes) if first_boxes is not None else 0
        logger.info(f"[YOLO] Frame {frame_id}: Detected {total_objects} total objects")

        persons = []
        helmets = []
        vests = []
        
        # Parse results
        class_counts = {}
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls[0])
                cls_name = self.class_names.get(cls_id, "Unknown")
                bbox = box.xyxy[0].tolist() # [x1, y1, x2, y2]

                # Get Track ID if available (it might be None for first few frames or unstable detections)
                track_id = int(box.id[0]) if box.id is not None else 0

                # High thresholds for PPE and Person classes to prevent false positive buckets and bags
                if cls_id == 10 and box.conf[0] < 0.45: # Helmet
                    continue
                elif cls_id == 16 and box.conf[0] < 0.50: # Vest
                    continue
                elif cls_id == 0: # Person
                    if box.conf[0] < 0.72:
                        continue
                    
                    # Jovi's Geometric Guardrails
                    bx_w = bbox[2] - bbox[0]
                    bx_h = bbox[3] - bbox[1]
                    aspect_ratio = bx_w / max(bx_h, 1e-6)
                    
                    # 1. Human Aspect Ratio Check (Workers are tall: H > W)
                    if aspect_ratio > 1.0:
                        continue # Discard horizontal objects detected as workers
                        
                    # 2. Size Check (Discard unrealistically huge blocks)
                    h, w = frame.shape[:2]
                    if (bx_w * bx_h) > (h * w * 0.15):
                        continue

                elif box.conf[0] < 0.3:
                    continue

                # Count classes for diagnostic logging
                class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

                if cls_name == "Person":
                    persons.append({"bbox": bbox, "id": track_id})
                elif cls_name == "Helmet":
                    helmets.append(bbox)
                elif cls_name == "Safety-vest":
                    vests.append(bbox)

        # Log class distribution
        logger.info(f"[YOLO] Frame {frame_id}: Class distribution: {class_counts}")
        logger.info(f"[PPE] Frame {frame_id}: Persons={len(persons)}, Helmets={len(helmets)}, Vests={len(vests)}")

        detected_objects = []

        # Association Logic (heuristic: containment/overlap)
        for p in persons:
            p_bbox = p["bbox"]
            p_id = p["id"]

            has_helmet = self._check_overlap(p_bbox, helmets, "head")
            has_vest = self._check_overlap(p_bbox, vests, "torso")

            # Log PPE detection result for this person
            logger.info(f"[PPE] Frame {frame_id}, Person {p_id}: helmet={has_helmet}, vest={has_vest}")
            
            helmet_state = "PRESENT" if has_helmet else "ABSENT"
            vest_state = "PRESENT" if has_vest else "ABSENT"
            overall_state = "PRESENT" if (has_helmet and has_vest) else "ABSENT"
            
            detected_objects.append(
                DetectedObject(
                    track_id=p_id, # REAL persistent ID from ByteTrack
                    class_name="Person",
                    bbox=p_bbox,
                    compliance=ComplianceStatus(
                        helmet=has_helmet,
                        vest=has_vest,
                        helmet_state=helmet_state,
                        vest_state=vest_state,
                        overall_state=overall_state
                    ),
                    occluded=False
                )
            )
            
        return DetectionResult(
            frame_id=frame_id,
            timestamp=datetime.now(),
            objects=detected_objects
        )

    def _check_overlap(self, person_box: List[float], equipment_boxes: List[List[float]], region: str) -> bool:
        """
        Check if any equipment box overlaps significantly with the person's specific region.
        """
        px1, py1, px2, py2 = person_box
        p_width = px2 - px1
        p_height = py2 - py1

        # Define regions
        if region == "head":
            # Check top 1/3 of person box
            region_box = [px1, py1, px2, py1 + p_height * 0.35]
        elif region == "torso":
            # Check middle/upper body (exclude legs)
            region_box = [px1, py1 + p_height * 0.1, px2, py1 + p_height * 0.7]
        else:
            region_box = person_box

        max_iou = 0.0
        for e_box in equipment_boxes:
            iou = self._iou(region_box, e_box)
            max_iou = max(max_iou, iou)
            if iou > 0.05:  # More lenient threshold for better recall
                logger.debug(f"[Overlap] Found {region} PPE with IoU {iou:.3f}")
                return True

        if max_iou > 0:
            logger.debug(f"[Overlap] No {region} PPE match, max IoU {max_iou:.3f} < 0.05")

        return False

    def _iou(self, boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # compute the area of intersection rectangle
        interArea = max(0, xB - xA) * max(0, yB - yA)

        # compute the area of both the prediction and ground-truth rectangles
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

# Global instance
inference_service = InferenceService()
