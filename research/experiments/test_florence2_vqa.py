"""
Florence-2 VQA Test Drive -- BuildSight
========================================
Tests Florence-2 visual question answering on real construction site
images across all 4 scene conditions. Run from project root.

Usage:
    python test_florence2_vqa.py
"""

import sys
import time
from pathlib import Path

# Resolve project root (2 levels up from research/experiments/)
PROJECT_ROOT = Path(__file__).parent.parent.parent
BACKEND = PROJECT_ROOT / "dashboard" / "backend"
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(PROJECT_ROOT))

import geoai_vlm_util as vlm

# Real construction site frames — paths resolved from project root
_ROOT = Path(__file__).parent.parent.parent
TEST_IMAGES = {
    "normal":    _ROOT / "data/annotated_results/ensemble/normal.jpg",
    "dusty":     _ROOT / "data/annotated_results/ensemble/dusty.jpg",
    "low_light": _ROOT / "data/annotated_results/ensemble/low_light.jpg",
    "crowded":   _ROOT / "data/annotated_results/ensemble/crowded.jpg",
}

# Florence-2 uses MORE_DETAILED_CAPTION — not VQA.
# The caption IS the visual answer; Turner AI interprets it per question.
# We verify caption quality here instead of per-question VQA.
CAPTION_CHECKS = [
    # (keyword that should appear in caption for this scene, description)
    ("person",   "detects a person"),
    ("vest",     "mentions PPE/vest"),
    ("building", "recognises site structure"),
]

SEP = "-" * 68
PLACEHOLDER = "Site activity observed. Processing vision telemetry..."


def load_image_bytes(path: Path) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def run_vqa(image_path: Path, question: str) -> dict:
    jpeg = load_image_bytes(image_path)
    t0 = time.time()
    result = vlm.describe_frame_sync(
        jpeg_bytes=jpeg,
        question=question,
        force_refresh=True,
    )
    result["elapsed_s"] = round(time.time() - t0, 2)
    return result


def main():
    print(SEP)
    print("  BuildSight -- Florence-2 VQA Test Drive")
    print(SEP)

    # Step 1: Load the model
    print("\n[1] Loading Florence-2 model ...")
    t_load = time.time()
    ok = vlm._try_load_vlm()
    load_time = round(time.time() - t_load, 1)

    if not ok:
        print("\n  FAIL -- VLM did not load. Check transformers/torch install.")
        retry_in = vlm._VLM_RETRY_AFTER - time.time()
        print(f"  Retry backoff: {retry_in:.0f}s remaining")
        sys.exit(1)

    info = vlm.get_model_info()
    print(f"\n  Model    : {info['model_id']}")
    print(f"  Device   : {info['device']}")
    print(f"  Dtype    : {info['dtype']}")
    print(f"  Load time: {load_time}s")

    # Step 2: Caption quality across all 4 scene conditions
    print(f"\n[2] Running MORE_DETAILED_CAPTION on {len(TEST_IMAGES)} scenes ...\n")

    total_pass = 0
    total_fail = 0
    results_log = []

    for scene, img_path in TEST_IMAGES.items():
        if not img_path.exists():
            print(f"  SKIP [{scene}] -- image not found: {img_path}")
            continue

        print(f"\n{'='*68}")
        print(f"  SCENE: {scene.upper()}")
        print(f"{'='*68}")

        result  = run_vqa(img_path, "")          # question unused — caption mode
        caption = result.get("description", "")
        source  = result.get("source", "?")
        elapsed = result.get("elapsed_s", 0)

        is_vlm  = source == "florence2"
        is_good = is_vlm and caption and caption != PLACEHOLDER and len(caption.strip()) > 10

        print(f"\n  CAPTION : {caption}")
        print(f"  SOURCE  : {source}  TIME: {elapsed}s")

        # Check expected keywords per scene
        caption_lower = caption.lower()
        checks_passed = 0
        for keyword, label in CAPTION_CHECKS:
            hit = keyword.lower() in caption_lower
            mark = "OK" if hit else "--"
            print(f"  [{mark}] {label} ('{keyword}')")
            if hit:
                checks_passed += 1

        scene_pass = is_good and checks_passed >= 1
        if scene_pass:
            print(f"  [PASS] Scene grounded correctly ({checks_passed}/{len(CAPTION_CHECKS)} checks)")
            total_pass += 1
        else:
            print(f"  [FAIL] Caption insufficient ({checks_passed}/{len(CAPTION_CHECKS)} checks, source={source})")
            total_fail += 1

        results_log.append({
            "scene": scene, "caption": caption, "source": source,
            "elapsed_s": elapsed, "checks": checks_passed, "pass": scene_pass,
        })

    # Step 3: Summary
    total    = total_pass + total_fail
    avg_time = sum(r["elapsed_s"] for r in results_log) / max(len(results_log), 1)

    print(f"\n{SEP}")
    print(f"  RESULTS  : {total_pass}/{total} scenes grounded by Florence-2 caption")
    print(f"  Model    : {vlm._VLM_MODEL_ID_LOADED}")
    print(f"  Device   : {vlm._VLM_DEVICE}")
    print(f"  Avg time : {avg_time:.2f}s per scene")

    if total_pass == total:
        print("\n  [OK] Florence-2 caption grounding works on all 4 scene conditions.")
        print("       Visual context will be injected into Turner AI prompts.")
        print("       Safe to proceed with Turner AI integration.")
    elif total_pass > total // 2:
        print(f"\n  [~~] Partial pass ({total_pass}/{total}). Captions functional.")
        print("       Turner integration can proceed — rule-based fills gaps.")
    else:
        print(f"\n  [!!] Caption quality too low ({total_fail}/{total} failed).")
        print("       Check model loading or switch to microsoft/Florence-2-base.")

    print(SEP)


if __name__ == "__main__":
    main()
