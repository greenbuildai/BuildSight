#!/usr/bin/env python3
"""
Evaluate baseline vs Gemini-audited PPE detections on a small labeled slice.

Default behavior samples a balanced subset from the local validation dataset:
2 images each from normal, dusty, low-light, and crowded conditions.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT / "dashboard" / "backend"
ANNOTATIONS = ROOT / "Dataset" / "Final_Annotated_Dataset" / "annotations" / "instances_val.json"
VAL_IMAGES = ROOT / "Dataset" / "Final_Annotated_Dataset" / "images" / "val"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import server  # noqa: E402


def iou(box_a, box_b) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0.0:
        return 0.0
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / max(area_a + area_b - inter, 1e-6)


def detect_condition(image_meta: dict) -> str:
    if image_meta.get("crowded_scene"):
        return "S4_crowded"

    value = f"{image_meta.get('environment', '')} {image_meta.get('scene_condition', '')}".lower()
    if "dust" in value:
        return "S2_dusty"
    if "low" in value or "night" in value:
        return "S3_low_light"
    return "S1_normal"


def load_dataset():
    coco = json.loads(ANNOTATIONS.read_text(encoding="utf-8"))
    images = {img["id"]: img for img in coco["images"]}
    gt_by_name = defaultdict(list)
    for ann in coco["annotations"]:
        image_meta = images.get(ann["image_id"])
        if image_meta is None:
            continue
        x, y, w, h = ann["bbox"]
        gt_by_name[image_meta["file_name"]].append(
            {"cls": int(ann["category_id"]), "box": [x, y, x + w, y + h]}
        )
    return list(images.values()), gt_by_name


def pick_sample(images: list[dict], gt_by_name: dict, limit_per_condition: int) -> list[dict]:
    grouped = defaultdict(list)
    for image_meta in images:
        gts = gt_by_name.get(image_meta["file_name"], [])
        has_ppe = any(gt["cls"] in (0, 1) for gt in gts)
        if not has_ppe:
            continue
        grouped[detect_condition(image_meta)].append(image_meta)

    picked = []
    for condition in ("S1_normal", "S2_dusty", "S3_low_light", "S4_crowded"):
        candidates = sorted(grouped.get(condition, []), key=lambda item: item["file_name"])
        picked.extend(candidates[:limit_per_condition])
    return picked


def score_predictions(preds: list[dict], gts: list[dict]) -> dict:
    metrics = {
        "helmet": {"tp": 0, "fp": 0, "fn": 0},
        "safety_vest": {"tp": 0, "fp": 0, "fn": 0},
    }
    class_map = {
        server._class_id_for_name("helmet"): ("helmet", 0),
        server._class_id_for_name("safety_vest"): ("safety_vest", 1),
    }

    for pred_cls_id, (metric_name, gt_cls_id) in class_map.items():
        pred_items = [p for p in preds if p["cls"] == pred_cls_id]
        gt_items = [g for g in gts if g["cls"] == gt_cls_id]
        matched_gt = set()

        for pred in sorted(pred_items, key=lambda item: -item["score"]):
            best_idx = -1
            best_iou = 0.0
            for idx, gt in enumerate(gt_items):
                if idx in matched_gt:
                    continue
                ov = iou(pred["box"], gt["box"])
                if ov > best_iou:
                    best_iou = ov
                    best_idx = idx
            if best_idx >= 0 and best_iou >= 0.5:
                metrics[metric_name]["tp"] += 1
                matched_gt.add(best_idx)
            else:
                metrics[metric_name]["fp"] += 1

        metrics[metric_name]["fn"] += max(0, len(gt_items) - len(matched_gt))

    return metrics


def summarise(metrics: dict) -> dict:
    out = {"per_class": {}, "totals": {"tp": 0, "fp": 0, "fn": 0}}
    for cls_name, values in metrics.items():
        tp = values["tp"]
        fp = values["fp"]
        fn = values["fn"]
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-6)
        out["per_class"][cls_name] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }
        out["totals"]["tp"] += tp
        out["totals"]["fp"] += fp
        out["totals"]["fn"] += fn

    total_tp = out["totals"]["tp"]
    total_fp = out["totals"]["fp"]
    total_fn = out["totals"]["fn"]
    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    out["precision"] = round(precision, 4)
    out["recall"] = round(recall, 4)
    out["f1"] = round(2 * precision * recall / max(precision + recall, 1e-6), 4)
    return out


def merge_metric_totals(target: dict, delta: dict) -> None:
    for cls_name, values in delta.items():
        for key in ("tp", "fp", "fn"):
            target[cls_name][key] += values[key]


def run(limit_per_condition: int, output: Path) -> None:
    images, gt_by_name = load_dataset()
    sample = pick_sample(images, gt_by_name, limit_per_condition)
    if not sample:
        raise RuntimeError("No PPE-positive validation images found for the requested slice.")

    baseline_totals = {
        "helmet": {"tp": 0, "fp": 0, "fn": 0},
        "safety_vest": {"tp": 0, "fp": 0, "fn": 0},
    }
    audited_totals = {
        "helmet": {"tp": 0, "fp": 0, "fn": 0},
        "safety_vest": {"tp": 0, "fp": 0, "fn": 0},
    }
    per_image = []

    for index, image_meta in enumerate(sample, start=1):
        image_path = VAL_IMAGES / image_meta["file_name"]
        img = cv2.imread(str(image_path))
        if img is None:
            continue

        condition = detect_condition(image_meta)
        gts = gt_by_name.get(image_meta["file_name"], [])

        t0 = time.perf_counter()
        baseline_dets, _, baseline_audit, _, _ = server.run_inference(
            img.copy(), condition=condition, gemini_audit=False, use_tiling=True
        )
        baseline_ms = round((time.perf_counter() - t0) * 1000)

        t1 = time.perf_counter()
        audited_dets, _, audited_meta, _, _ = server.run_inference(
            img.copy(), condition=condition, gemini_audit=True, use_tiling=True
        )
        audited_ms = round((time.perf_counter() - t1) * 1000)

        baseline_metrics = score_predictions(baseline_dets, gts)
        audited_metrics = score_predictions(audited_dets, gts)
        merge_metric_totals(baseline_totals, baseline_metrics)
        merge_metric_totals(audited_totals, audited_metrics)

        per_image.append(
            {
                "index": index,
                "file_name": image_meta["file_name"],
                "condition": condition,
                "ground_truth_ppe": sum(1 for gt in gts if gt["cls"] in (0, 1)),
                "baseline_total_dets": len(baseline_dets),
                "audited_total_dets": len(audited_dets),
                "baseline_ms": baseline_ms,
                "audited_ms": audited_ms,
                "gemini_audit": audited_meta,
                "baseline_metrics": baseline_metrics,
                "audited_metrics": audited_metrics,
            }
        )

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "mode_name": server.mode_name,
        "gemini_auditor_enabled": server.gemini_auditor_enabled,
        "gemini_auditor_model": server.gemini_auditor_model_name,
        "sample_size": len(per_image),
        "limit_per_condition": limit_per_condition,
        "baseline": summarise(baseline_totals),
        "audited": summarise(audited_totals),
        "per_image": per_image,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Sample size: {report['sample_size']}")
    print(f"Baseline PPE precision={report['baseline']['precision']} recall={report['baseline']['recall']} f1={report['baseline']['f1']}")
    print(f"Audited  PPE precision={report['audited']['precision']} recall={report['audited']['recall']} f1={report['audited']['f1']}")
    print(f"Saved report -> {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-per-condition", type=int, default=2)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "runtime" / "gemini_auditor_eval_slice.json",
    )
    args = parser.parse_args()
    run(args.limit_per_condition, args.output)


if __name__ == "__main__":
    main()
