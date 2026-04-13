#!/usr/bin/env python3
"""
ensemble_video.py
=================
BuildSight — Multi-Model Ensemble Video Inference

Processes an uploaded video through the full ensemble pipeline:
  1. YOLOv11 inference per frame
  2. Early-exit: skip YOLOv26 when v11 is uniformly high-confidence (≥0.82)
  3. If not early-exit: YOLOv26 inference + per-class WBF fusion
  4. Adaptive 8-rule post-processing (per site condition)
  5. Write annotated output video

Usage:
  python ensemble_video.py --input site_video.mp4
  python ensemble_video.py --input site_video.mp4 --condition S2_dusty --output result.mp4
  python ensemble_video.py --input site_video.mp4 --condition S4_crowded

Arguments:
  --input      Path to input video (mp4, avi, mov, mkv)
  --output     Output video path (default: <input>_ensemble.mp4)
  --condition  Site condition override, or auto for dynamic routing
  --device     GPU device index (default: 0)

Author: Toni (Claude Sonnet 4.6), updated for site-aware runtime 2026-04-07
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
from site_aware_ensemble import (
    TemporalPPEFilter,
    detect_condition,
    preprocess_frame,
    profile_for_condition,
    wbf_fuse_condition,
)

# ── model paths ────────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent

MODEL_PATHS = {
    "yolo11": "/nfsshare/joseva/buildsight/dataset/Backup_Dataset(annotated)/YOLOv11/runs/detect/runs/train/buildsight/weights/best.pt",
    "yolo26": "/nfsshare/joseva/buildsight/dataset/Backup_Dataset(annotated)/YOLOv26/runs/detect/runs/train/buildsight/weights/best.pt",
}

# Local weights fallback (for running on dev machine)
LOCAL_MODEL_PATHS = {
    "yolo11": str(_HERE.parent / "weights" / "yolov11_buildsight_best.pt"),
    "yolo26": str(_HERE.parent / "weights" / "yolov26_buildsight_best.pt"),
}

# ── WBF parameters ─────────────────────────────────────────────────────────────
MODEL_WEIGHTS   = [0.55, 0.45]   # YOLOv11, YOLOv26

# Early-exit: skip YOLOv26 when v11 detections are ALL above this confidence
EARLY_EXIT_CONF     = 0.65
EARLY_EXIT_MIN_DETS = 1

CLS_NAMES  = {0: "helmet", 1: "safety_vest", 2: "worker"}
CLS_COLORS = {0: (0, 255, 0), 1: (0, 165, 255), 2: (255, 100, 0)}
CONDITIONS = ["S1_normal", "S2_dusty", "S3_low_light", "S4_crowded"]


# ── adaptive post-processing ───────────────────────────────────────────────────

def _load_adaptive():
    sys.path.insert(0, str(_HERE))
    sys.path.insert(0, "/nfsshare/joseva")
    try:
        from adaptive_postprocess import apply_all_rules
        return apply_all_rules
    except ImportError:
        print("[WARN] adaptive_postprocess.py not found — skipping adaptive rules")
        return lambda boxes, *a, **k: (boxes, {"raw": len(boxes), "final": len(boxes)})


# ── frame inference ────────────────────────────────────────────────────────────

def infer_frame(frame, model_v11, model_v26, condition, device, apply_all_rules, temporal_filter, frame_idx):
    img_h, img_w = frame.shape[:2]
    profile = profile_for_condition(condition)
    pre_conf = profile["pre_conf"]

    r11 = model_v11.predict(frame, device=device, verbose=False,
                            conf=pre_conf, iou=0.35)[0]
    boxes11, scores11, labels11 = [], [], []
    for box in (r11.boxes or []):
        xy = box.xyxy[0].cpu().numpy().tolist()
        boxes11.append(xy)
        scores11.append(float(box.conf[0]))
        labels11.append(int(box.cls[0]))

    # Early-exit check
    early_exit = (
        len(scores11) >= EARLY_EXIT_MIN_DETS
        and all(s >= EARLY_EXIT_CONF for s in scores11)
    )

    if early_exit:
        fused = [{"box": b, "score": s, "cls": l}
                 for b, s, l in zip(boxes11, scores11, labels11)]
        skipped_v26 = True
    else:
        r26 = model_v26.predict(frame, device=device, verbose=False,
                                conf=pre_conf, iou=0.35)[0]
        boxes26, scores26, labels26 = [], [], []
        for box in (r26.boxes or []):
            xy = box.xyxy[0].cpu().numpy().tolist()
            boxes26.append(xy)
            scores26.append(float(box.conf[0]))
            labels26.append(int(box.cls[0]))
        all_preds = [(boxes11, scores11, labels11), (boxes26, scores26, labels26)]
        fused = wbf_fuse_condition(all_preds, img_w, img_h, condition, MODEL_WEIGHTS)
        skipped_v26 = False

    final_boxes, stats = apply_all_rules(fused, condition, img_w, img_h, image=frame)
    final_boxes = temporal_filter.update(final_boxes, frame_idx)
    stats["raw"] = len(fused)
    stats["final"] = len(final_boxes)
    return final_boxes, stats, skipped_v26


# ── draw overlay ───────────────────────────────────────────────────────────────

def draw_frame(frame, boxes, condition, stats, fps, frame_idx, skipped_v26):
    for box in boxes:
        x1, y1, x2, y2 = [int(v) for v in box["box"]]
        color = CLS_COLORS[box["cls"]]
        label = f"{CLS_NAMES[box['cls']]} {box['score']:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
        cv2.rectangle(frame, (x1, y1 - th - 5), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)

    # HUD banner
    mode = "V11-ONLY" if skipped_v26 else "WBF-V11+V26"
    raw = stats.get("raw", "?")
    final = stats.get("final", len(boxes))
    banner = (
        f"[BUILDSIGHT ENSEMBLE] {condition} | {mode} | "
        f"raw:{raw}->final:{final} | {fps:.1f} FPS | frame:{frame_idx}"
    )
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(frame, banner, (6, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 200, 255), 1)

    # Legend
    legend_y = frame.shape[0] - 10
    for i, (cls_id, name) in enumerate(CLS_NAMES.items()):
        color = CLS_COLORS[cls_id]
        x = 10 + i * 160
        cv2.rectangle(frame, (x, legend_y - 12), (x + 14, legend_y), color, -1)
        cv2.putText(frame, name, (x + 18, legend_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)

    return frame


# ── main ───────────────────────────────────────────────────────────────────────

def run(args):
    from ultralytics import YOLO

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input video not found: {input_path}")
        sys.exit(1)

    output_path = Path(args.output) if args.output else \
        input_path.parent / (input_path.stem + "_ensemble.mp4")

    fixed_condition = args.condition
    if fixed_condition != "auto" and fixed_condition not in CONDITIONS:
        print(f"ERROR: condition must be one of {CONDITIONS}")
        sys.exit(1)

    print(f"Input    : {input_path}")
    print(f"Output   : {output_path}")
    print(f"Condition: {fixed_condition}")
    print(f"Device   : {args.device}")
    print()

    # Load adaptive post-processing
    apply_all_rules = _load_adaptive()

    # Load models — try SASTRA paths first, fall back to local
    def load_model(key):
        p = MODEL_PATHS[key]
        if not Path(p).exists():
            p = LOCAL_MODEL_PATHS[key]
        if not Path(p).exists():
            print(f"ERROR: weights not found for {key}. Checked:\n  {MODEL_PATHS[key]}\n  {LOCAL_MODEL_PATHS[key]}")
            sys.exit(1)
        print(f"Loading {key}: {p}")
        return YOLO(p)

    model_v11 = load_model("yolo11")
    model_v26 = load_model("yolo26")
    print("Models loaded.\n")

    # Open video
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {input_path}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps      = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video: {width}x{height} @ {src_fps:.1f} FPS, {total_frames} frames")

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        src_fps,
        (width, height),
    )

    frame_idx    = 0
    skipped_v26_count = 0
    total_dets   = 0
    fps_window   = []
    t_start      = time.perf_counter()
    temporal_filter = TemporalPPEFilter()
    condition_counts = {key: 0 for key in CONDITIONS}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()
        rough = model_v11.predict(frame, device=args.device, verbose=False, conf=0.20, iou=0.40)[0]
        rough_workers = [
            box.xyxy[0].cpu().numpy().tolist()
            for box in (rough.boxes or [])
            if int(box.cls[0]) == 2 and float(box.conf[0]) >= 0.20
        ]
        decision = detect_condition(frame, rough_workers)
        condition = fixed_condition if fixed_condition != "auto" else decision.key
        condition_counts[condition] += 1
        prepared_frame = preprocess_frame(frame, condition)
        final_boxes, stats, skipped_v26 = infer_frame(
            prepared_frame, model_v11, model_v26, condition,
            args.device, apply_all_rules, temporal_filter, frame_idx
        )
        t1 = time.perf_counter()

        frame_time = t1 - t0
        fps_window.append(frame_time)
        if len(fps_window) > 30:
            fps_window.pop(0)
        live_fps = 1.0 / (sum(fps_window) / len(fps_window))

        annotated = draw_frame(
            frame.copy(), final_boxes, condition,
            stats, live_fps, frame_idx, skipped_v26
        )
        writer.write(annotated)

        frame_idx += 1
        total_dets += len(final_boxes)
        if skipped_v26:
            skipped_v26_count += 1

        if frame_idx % 50 == 0 or frame_idx == 1:
            mode = "EARLY-EXIT" if skipped_v26 else "FULL-WBF"
            print(
                f"  [{frame_idx}/{total_frames}] {live_fps:.1f} FPS | "
                f"{len(final_boxes)} dets | {mode} | "
                f"{condition} b:{decision.brightness:.0f} c:{decision.contrast:.0f} "
                f"workers:{decision.rough_worker_count}"
            )

    cap.release()
    writer.release()

    elapsed = time.perf_counter() - t_start
    mean_fps = frame_idx / elapsed
    skip_rate = skipped_v26_count / max(frame_idx, 1) * 100

    print(f"\n{'='*60}")
    print(f"DONE")
    print(f"  Frames processed : {frame_idx}")
    print(f"  Mean FPS         : {mean_fps:.1f}")
    print(f"  Early-exit rate  : {skip_rate:.1f}% ({skipped_v26_count} frames — YOLOv26 skipped)")
    print(f"  Total detections : {total_dets}")
    print(f"  Condition usage  : " + ", ".join(f"{k}={v}" for k, v in condition_counts.items() if v))
    print(f"  Output           : {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BuildSight Ensemble Video Inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input",     required=True,        help="Input video path")
    parser.add_argument("--output",    default=None,         help="Output video path (default: <input>_ensemble.mp4)")
    parser.add_argument("--condition", default="auto",
                        choices=CONDITIONS + ["auto"],       help="Site condition for adaptive post-processing")
    parser.add_argument("--device",    type=int,   default=0,    help="GPU device index")
    args = parser.parse_args()
    run(args)
