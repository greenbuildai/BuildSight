# Surgical Fix Test (V3) — Image 1772974213746
import os
import sys
import torch
import cv2
import json
from pathlib import Path

# Add scripts directory to path
sys.path.append(r"e:\Company\Green Build AI\Prototypes\BuildSight\scripts")

from annotate_indian_dataset import (
    annotate_image, load_models, 
    CLASS_ID
)
from pipeline_config import DATA_DIR, OUTPUT_DIR

image_path = Path(r"e:\Company\Green Build AI\Prototypes\BuildSight\Dataset\Indian Dataset\Normal_Site_Condition\1772974213746.jpg")
output_root = Path(r"e:\Company\Green Build AI\Prototypes\BuildSight\Dataset\Final_Annotated_Dataset")

if not image_path.exists():
    print(f"Error: image not found at {image_path}")
    exit(1)

print("--- Initializing Models (DINO + SAM) ---")
# load_models returns (dino_model, sam_predictor, device)
dino_model, sam_predictor, device = load_models(skip_sam=False)

print(f"\n--- Firing Surgical Fix Test on {image_path.name} ---")
# Required state for annotate_image
next_track_id = {}
track_id_counter = [1]
annotation_id_start = 1
image_id = 9999
sequence_id = "fix_test"
frame_id = 1

image_record, annotations, ann_id_end = annotate_image(
    str(image_path), dino_model, sam_predictor, device,
    condition="normal", sequence_id=sequence_id, frame_id=frame_id, 
    image_id=image_id, annotation_id_start=annotation_id_start,
    next_track_id=next_track_id, track_id_counter=track_id_counter,
    no_auto_detect=True
)

print("\n--- Results ---")
v_count = sum(1 for a in annotations if a["category_id"] == CLASS_ID["safety_vest"])
h_count = sum(1 for a in annotations if a["category_id"] == CLASS_ID["helmet"])
w_count = sum(1 for a in annotations if a["category_id"] == CLASS_ID["worker"])

print(f"Workers: {w_count}")
print(f"Helmets: {h_count}")
print(f"Vests:   {v_count}")

if v_count > 0:
    print("✅ SUCCESS: Safety vest detected with new 0.15 threshold + orange synonyms!")
else:
    print("❌ FAILED: Vest still missing. This suggests it might be getting filtered by a geometric guard.")
