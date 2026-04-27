#!/usr/bin/env python3
"""
generate_condition_matrix.py
=============================
Reads all per-model per-condition JSON results from val_condition_eval.py
and generates a comprehensive comparative matrix.

Run after all 16 eval jobs complete (4 models x 4 conditions):
  python scripts/generate_condition_matrix.py

Output:
  - Console: formatted comparative table
  - condition_eval_matrix.json  — full structured results
  - condition_eval_matrix.md    — markdown table for docs/comparative_study.md
"""

import json
from pathlib import Path

RESULTS_DIR = Path("/nfsshare/joseva/condition_eval_results")
OUT_JSON    = RESULTS_DIR / "condition_eval_matrix.json"
OUT_MD      = RESULTS_DIR / "condition_eval_matrix.md"

MODELS     = ["yolo11", "yolo26", "yolact", "samurai"]
CONDITIONS = ["S1_normal", "S2_dusty", "S3_low_light", "S4_crowded"]
MODEL_LABELS = {
    "yolo11":  "YOLOv11",
    "yolo26":  "YOLOv26",
    "yolact":  "YOLACT++",
    "samurai": "SAMURAI_GT (ref)",
}
COND_LABELS = {
    "S1_normal":    "S1 Normal",
    "S2_dusty":     "S2 Dusty",
    "S3_low_light": "S3 Low Light",
    "S4_crowded":   "S4 Crowded",
}
METRICS = ["mAP50", "precision", "recall", "F1", "total_FP", "total_FN", "FPS"]

# ─── Load results ─────────────────────────────────────────────────────────────

matrix = {}
missing = []

for model in MODELS:
    matrix[model] = {}
    for cond in CONDITIONS:
        path = RESULTS_DIR / f"{model}_{cond}.json"
        if path.exists():
            with open(path) as f:
                matrix[model][cond] = json.load(f)
        else:
            matrix[model][cond] = None
            missing.append(f"{model}_{cond}")

if missing:
    print(f"\nWARNING — Missing result files ({len(missing)}):")
    for m in missing:
        print(f"  {m}.json")
    print()

# ─── Console table ────────────────────────────────────────────────────────────

def fmt(val, metric):
    if val is None:
        return "—"
    if metric in ("total_FP", "total_FN", "n_imgs"):
        return str(int(val)) if val != "N/A" else "N/A"
    if metric == "FPS":
        return str(val)
    return f"{float(val):.4f}"

def print_metric_table(metric):
    header = f"\n{'─'*70}\n {metric.upper()}\n{'─'*70}"
    print(header)
    col_w = 16
    row = f"{'Model':<16}" + "".join(f"{COND_LABELS[c]:<{col_w}}" for c in CONDITIONS) + "Average"
    print(row)
    print("─" * (16 + col_w * 4 + 10))
    for model in MODELS:
        vals = []
        for cond in CONDITIONS:
            r = matrix[model][cond]
            v = r.get(metric) if r else None
            vals.append(v)
        avg_vals = [v for v in vals if v is not None and v != "N/A"]
        try:
            avg = f"{sum(float(v) for v in avg_vals)/len(avg_vals):.4f}" if avg_vals else "—"
        except Exception:
            avg = "—"
        row = f"{MODEL_LABELS[model]:<16}" + "".join(f"{fmt(v, metric):<{col_w}}" for v in vals) + avg
        print(row)

for metric in METRICS:
    print_metric_table(metric)

# ─── Per-class breakdown ──────────────────────────────────────────────────────

print(f"\n{'─'*70}")
print(" PER-CLASS AP50 (mAP50 by class across conditions)")
print(f"{'─'*70}")
for cls_name in ["helmet", "safety_vest", "worker"]:
    print(f"\n  Class: {cls_name.upper()}")
    col_w = 16
    row = f"  {'Model':<14}" + "".join(f"{COND_LABELS[c]:<{col_w}}" for c in CONDITIONS)
    print(row)
    print("  " + "─" * (14 + col_w * 4))
    for model in MODELS:
        vals = []
        for cond in CONDITIONS:
            r = matrix[model][cond]
            if r and "per_class" in r and cls_name in r["per_class"]:
                vals.append(r["per_class"][cls_name].get("AP50"))
            else:
                vals.append(None)
        row = f"  {MODEL_LABELS[model]:<14}" + "".join(f"{fmt(v, 'mAP50'):<{col_w}}" for v in vals)
        print(row)

# ─── Best model per condition ─────────────────────────────────────────────────

print(f"\n{'─'*70}")
print(" BEST MODEL PER CONDITION (by mAP50, excluding SAMURAI_GT reference)")
print(f"{'─'*70}")
eval_models = [m for m in MODELS if m != "samurai"]
for cond in CONDITIONS:
    best_model, best_map = None, -1.0
    for model in eval_models:
        r = matrix[model][cond]
        if r and r.get("mAP50") is not None:
            if float(r["mAP50"]) > best_map:
                best_map  = float(r["mAP50"])
                best_model = model
    winner = MODEL_LABELS.get(best_model, "?") if best_model else "?"
    print(f"  {COND_LABELS[cond]:<16}: {winner} (mAP50={best_map:.4f})")

# ─── Most robust model overall ───────────────────────────────────────────────

print(f"\n{'─'*70}")
print(" OVERALL ROBUSTNESS RANKING (mean mAP50 across all conditions)")
print(f"{'─'*70}")
rankings = []
for model in eval_models:
    vals = []
    for cond in CONDITIONS:
        r = matrix[model][cond]
        if r and r.get("mAP50") is not None:
            vals.append(float(r["mAP50"]))
    if vals:
        rankings.append((model, sum(vals)/len(vals), min(vals), max(vals)))
rankings.sort(key=lambda x: -x[1])
for rank, (model, mean_map, min_map, max_map) in enumerate(rankings, 1):
    print(f"  #{rank} {MODEL_LABELS[model]:<14} mean={mean_map:.4f}  min={min_map:.4f}  max={max_map:.4f}")

# ─── Save JSON ────────────────────────────────────────────────────────────────

with open(OUT_JSON, "w") as f:
    json.dump(matrix, f, indent=2)
print(f"\nFull matrix saved: {OUT_JSON}")

# ─── Save Markdown ────────────────────────────────────────────────────────────

def write_markdown():
    lines = []
    lines.append("## Condition-Based Validation Matrix\n")
    lines.append("*Generated by `generate_condition_matrix.py` — BuildSight Phase 2*\n")

    for metric in ["mAP50", "precision", "recall", "F1"]:
        lines.append(f"\n### {metric}\n")
        header = "| Model | " + " | ".join(COND_LABELS[c] for c in CONDITIONS) + " | **Mean** |"
        sep    = "|-------|" + "|".join(["-------"] * len(CONDITIONS)) + "|---------|"
        lines.append(header)
        lines.append(sep)
        for model in MODELS:
            vals = []
            for cond in CONDITIONS:
                r = matrix[model][cond]
                v = r.get(metric) if r else None
                vals.append(fmt(v, metric))
            num_vals = [float(v) for v in vals if v != "—"]
            mean = f"**{sum(num_vals)/len(num_vals):.4f}**" if num_vals else "—"
            row = f"| {MODEL_LABELS[model]} | " + " | ".join(vals) + f" | {mean} |"
            lines.append(row)

    lines.append("\n### Per-Class AP50\n")
    for cls_name in ["helmet", "safety_vest", "worker"]:
        lines.append(f"\n#### {cls_name.replace('_',' ').title()}\n")
        header = "| Model | " + " | ".join(COND_LABELS[c] for c in CONDITIONS) + " |"
        sep    = "|-------|" + "|".join(["-------"] * len(CONDITIONS)) + "|"
        lines.append(header)
        lines.append(sep)
        for model in MODELS:
            vals = []
            for cond in CONDITIONS:
                r = matrix[model][cond]
                if r and "per_class" in r and cls_name in r["per_class"]:
                    v = r["per_class"][cls_name].get("AP50")
                    vals.append(fmt(v, "mAP50"))
                else:
                    vals.append("—")
            row = f"| {MODEL_LABELS[model]} | " + " | ".join(vals) + " |"
            lines.append(row)

    lines.append("\n### False Positive / False Negative Summary\n")
    header = "| Model | Condition | FP | FN | FPS |"
    sep    = "|-------|-----------|----|----|-----|"
    lines.append(header)
    lines.append(sep)
    for model in MODELS:
        for cond in CONDITIONS:
            r = matrix[model][cond]
            fp  = str(r.get("total_FP", "—")) if r else "—"
            fn  = str(r.get("total_FN", "—")) if r else "—"
            fps = str(r.get("FPS", "—")) if r else "—"
            lines.append(f"| {MODEL_LABELS[model]} | {COND_LABELS[cond]} | {fp} | {fn} | {fps} |")

    with open(OUT_MD, "w") as f:
        f.write("\n".join(lines))
    print(f"Markdown matrix saved: {OUT_MD}")

write_markdown()
print("\nDone. Copy condition_eval_matrix.md into docs/comparative_study.md Section 5.7.\n")
