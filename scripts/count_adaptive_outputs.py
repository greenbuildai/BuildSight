#!/usr/bin/env python3
"""
Count rendered adaptive post-processing output images under val_annotated_adaptive.
"""

import json
from pathlib import Path

ROOT = Path("/nfsshare/joseva/val_annotated_adaptive")
CONDITIONS = ["S1_normal", "S2_dusty", "S3_low_light", "S4_crowded"]
MODELS = ["YOLOv11", "YOLOv26"]


def main():
    counts = {}
    total = 0
    for condition in CONDITIONS:
        for model in MODELS:
            directory = ROOT / condition / model
            count = sum(
                1
                for path in directory.iterdir()
                if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
            )
            counts[f"{condition}/{model}"] = count
            total += count

    print(json.dumps({"total": total, "per_dir": counts}, indent=2))


if __name__ == "__main__":
    main()
