#!/usr/bin/env python3
"""
Rebuild val_condition_splits.json from the existing condition-organized
annotated folders. This preserves the exact 691-image subset used earlier.
"""

import json
from pathlib import Path

BASE_DIR = Path("/nfsshare/joseva/val_annotated_by_condition")
OUT_JSON = Path("/nfsshare/joseva/condition_eval_results/val_condition_splits.json")
CONDITIONS = ["S1_normal", "S2_dusty", "S3_low_light", "S4_crowded"]


def main():
    data = {}
    for condition in CONDITIONS:
        source_dir = BASE_DIR / condition / "YOLOv11"
        data[condition] = sorted(
            path.name
            for path in source_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(data, indent=2))
    print(json.dumps({key: len(value) for key, value in data.items()}, indent=2))
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
