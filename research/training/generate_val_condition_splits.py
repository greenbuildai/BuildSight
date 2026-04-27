#!/usr/bin/env python3
"""
Generate val_condition_splits.json from the COCO val manifest using the same
filename keyword rules used by the condition organization scripts.
"""

import json
import os
from collections import Counter
from pathlib import Path

COCO_JSON = Path(
    "/nfsshare/joseva/buildsight/dataset/Backup_Dataset(annotated)/YOLACT_plusplus/annotations/instances_val.json"
)
OUT_JSON = Path("/nfsshare/joseva/condition_eval_results/val_condition_splits.json")

CONDITION_RULES = [
    (["dusty"], "S2_dusty"),
    (["low_light", "lowlight", "low-light", "night", "dark"], "S3_low_light"),
    (["crowded", "crowd"], "S4_crowded"),
    (["normal"], "S1_normal"),
]


def classify(filename):
    lower_name = filename.lower()
    for keywords, condition in CONDITION_RULES:
        if any(keyword in lower_name for keyword in keywords):
            return condition
    return "unclassified"


def main():
    coco = json.loads(COCO_JSON.read_text())
    splits = {"S1_normal": [], "S2_dusty": [], "S3_low_light": [], "S4_crowded": []}
    counts = Counter()

    for image in coco["images"]:
        file_name = os.path.basename(image["file_name"])
        condition = classify(file_name)
        counts[condition] += 1
        if condition in splits:
            splits[condition].append(file_name)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(splits, indent=2))

    print(json.dumps(dict(counts), indent=2))
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
