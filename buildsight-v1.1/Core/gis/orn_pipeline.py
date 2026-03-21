from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from orn import ORN, ORNConfig


PPE_PRESENT = 0
PPE_ABSENT = 1
PPE_UNCERTAIN = 2


@dataclass
class TrackState:
    hidden: torch.Tensor | None = None
    last_seen_frame: int = 0
    missing_counts: Dict[str, int] = field(default_factory=lambda: {"helmet": 0, "vest": 0})
    last_alert_frame: int = 0


def build_ppe_heatmap(crop_shape: Tuple[int, int], rel_boxes: Dict[str, list]) -> np.ndarray:
    h, w = crop_shape
    heatmap = np.zeros((2, h, w), dtype=np.float32)
    for idx, key in enumerate(["helmet", "vest"]):
        for (x1, y1, x2, y2) in rel_boxes.get(key, []):
            x1 = max(0, min(w - 1, x1))
            x2 = max(0, min(w - 1, x2))
            y1 = max(0, min(h - 1, y1))
            y2 = max(0, min(h - 1, y2))
            heatmap[idx, y1:y2, x1:x2] = 1.0
    return heatmap


def normalize_crop(crop: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    resized = cv2.resize(crop, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
    return resized.astype(np.float32) / 255.0


def resolve_ppe_state(
    orn_probs: np.ndarray,
    det_present: bool,
    occlusion_score: float,
    present_thresh: float = 0.55,
    absent_thresh: float = 0.55,
) -> int:
    # Prefer ORN probability when occlusion is high, otherwise fuse with detector.
    if occlusion_score > 0.6:
        if orn_probs[PPE_PRESENT] >= present_thresh:
            return PPE_PRESENT
        if orn_probs[PPE_ABSENT] >= absent_thresh:
            return PPE_ABSENT
        return PPE_UNCERTAIN

    if det_present:
        return PPE_PRESENT
    if orn_probs[PPE_ABSENT] >= absent_thresh:
        return PPE_ABSENT
    return PPE_UNCERTAIN


def main() -> None:
    # =========================
    # CONFIG
    # =========================
    YOLO_MODEL_PATH = "best.pt"
    VIDEO_PATH = "rd1.mp4"
    ORN_WEIGHTS = "orn_weights.pt"

    CONF_THRES = 0.25
    IOU_THRES_VEST = 0.15
    PPE_CONFIRM_FRAMES = 4
    MISSING_PERSIST_FRAMES = 15
    ALERT_COOLDOWN_FRAMES = 150  # ~5s at 30 fps

    DANGER_ZONE = np.array(
        [(276, 229), (426, 229), (428, 210), (571, 205),
         (640, 430), (420, 447), (417, 474), (184, 468)],
        dtype=np.int32
    )

    # =========================
    # LOAD MODELS
    # =========================
    detector = YOLO(YOLO_MODEL_PATH)

    orn_config = ORNConfig()
    orn_model = ORN(orn_config)
    if ORN_WEIGHTS:
        orn_model.load_state_dict(torch.load(ORN_WEIGHTS, map_location="cpu"))
    orn_model.eval()
    # For deployment: trace or export ORN to ONNX/TensorRT and replace eager execution here.

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError("Unable to open video.")

    track_states: Dict[int, TrackState] = {}
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        results = detector.track(frame, persist=True, conf=CONF_THRES)

        persons, helmets, vests = [], [], []

        for r in results:
            if r.boxes.id is None:
                continue
            for box, cls, tid in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.id):
                label = detector.names[int(cls)]
                box = box.cpu().numpy().astype(int)
                tid = int(tid)
                if label == "person":
                    persons.append((box, tid))
                elif label == "helmet":
                    helmets.append(box)
                elif label == "safety-vest":
                    vests.append(box)

        cv2.polylines(frame, [DANGER_ZONE], True, (0, 0, 255), 3)

        for person_box, pid in persons:
            x1, y1, x2, y2 = person_box
            foot_point = (int((x1 + x2) / 2), int(y2))
            inside_zone = cv2.pointPolygonTest(DANGER_ZONE, foot_point, False) >= 0
            if not inside_zone:
                continue

            crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
            if crop.size == 0:
                continue

            rel_boxes = {"helmet": [], "vest": []}
            for h in helmets:
                cx = int((h[0] + h[2]) / 2)
                cy = int((h[1] + h[3]) / 2)
                if x1 <= cx <= x2 and y1 <= cy <= y2:
                    rel_boxes["helmet"].append([h[0] - x1, h[1] - y1, h[2] - x1, h[3] - y1])
            for v in vests:
                cx = int((v[0] + v[2]) / 2)
                cy = int((v[1] + v[3]) / 2)
                if x1 <= cx <= x2 and y1 <= cy <= y2:
                    rel_boxes["vest"].append([v[0] - x1, v[1] - y1, v[2] - x1, v[3] - y1])

            crop_norm = normalize_crop(crop, orn_config.input_size)
            heatmap = build_ppe_heatmap(orn_config.input_size, rel_boxes)
            inp = np.concatenate([crop_norm.transpose(2, 0, 1), heatmap], axis=0)
            inp_t = torch.from_numpy(inp).unsqueeze(0)

            state = track_states.get(pid, TrackState())
            with torch.no_grad():
                ppe_logits, visibility_logits, occ_logits, next_hidden = orn_model(inp_t, state.hidden)
            state.hidden = next_hidden
            state.last_seen_frame = frame_idx

            ppe_probs = torch.softmax(ppe_logits, dim=-1).squeeze(0).cpu().numpy()
            occ_prob = torch.softmax(occ_logits, dim=1)[:, 1].mean().item()

            helmet_state = resolve_ppe_state(
                ppe_probs[0],
                det_present=len(rel_boxes["helmet"]) > 0,
                occlusion_score=occ_prob,
            )
            vest_state = resolve_ppe_state(
                ppe_probs[1],
                det_present=len(rel_boxes["vest"]) > 0,
                occlusion_score=occ_prob,
            )

            for key, state_id in [("helmet", helmet_state), ("vest", vest_state)]:
                if state_id == PPE_ABSENT:
                    state.missing_counts[key] += 1
                elif state_id == PPE_PRESENT:
                    state.missing_counts[key] = 0

            missing = []
            if state.missing_counts["helmet"] >= PPE_CONFIRM_FRAMES:
                missing.append("Helmet")
            if state.missing_counts["vest"] >= PPE_CONFIRM_FRAMES:
                missing.append("Vest")

            violation = len(missing) > 0 and min(state.missing_counts.values()) >= MISSING_PERSIST_FRAMES

            color = (0, 255, 0)
            label = f"ID {pid}: SAFE"
            if violation and frame_idx - state.last_alert_frame > ALERT_COOLDOWN_FRAMES:
                color = (0, 0, 255)
                label = f"ID {pid}: MISSING {' & '.join(missing)}"
                print(f"ALERT ID {pid} missing {missing}")
                state.last_alert_frame = frame_idx

            track_states[pid] = state

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, foot_point, 5, (255, 0, 0), -1)
            cv2.putText(frame, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        cv2.imshow("PPE ORN Monitoring", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
