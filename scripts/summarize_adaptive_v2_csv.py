#!/usr/bin/env python3
"""
Summarize adaptive_postprocess_v2_summary.csv totals and per-condition reductions.
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

CSV_PATH = Path("/nfsshare/joseva/logs/adaptive_postprocess_v2_summary.csv")


def main():
    rows = list(csv.DictReader(CSV_PATH.open()))
    total_raw = sum(int(row["raw"]) for row in rows)
    total_final = sum(int(row["final"]) for row in rows)
    by_condition = defaultdict(lambda: {"raw": 0, "final": 0})

    for row in rows:
        by_condition[row["condition"]]["raw"] += int(row["raw"])
        by_condition[row["condition"]]["final"] += int(row["final"])

    result = {
        "total_raw": total_raw,
        "total_final": total_final,
        "reduction_pct": round(100 * (total_raw - total_final) / max(total_raw, 1), 1),
        "by_condition": {
            condition: {
                "raw": values["raw"],
                "final": values["final"],
                "reduction_pct": round(100 * (values["raw"] - values["final"]) / max(values["raw"], 1), 1),
            }
            for condition, values in sorted(by_condition.items())
        },
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
