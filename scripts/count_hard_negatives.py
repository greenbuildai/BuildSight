#!/usr/bin/env python3
"""
Count validation images with zero annotations and persist the candidate list.
"""

import json
from pathlib import Path

COCO_JSON = Path(
    "/nfsshare/joseva/buildsight/dataset/Backup_Dataset(annotated)/YOLACT_plusplus/annotations/instances_val.json"
)
OUT_JSON = Path("/nfsshare/joseva/buildsight/dataset/hard_negatives/hard_negative_candidates.json")


def main():
    coco = json.loads(COCO_JSON.read_text())
    imgs_with_ann = {ann["image_id"] for ann in coco["annotations"]}
    all_imgs = {img["id"]: img["file_name"] for img in coco["images"]}
    no_ann = [all_imgs[image_id] for image_id in all_imgs if image_id not in imgs_with_ann]

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({"count": len(no_ann), "files": no_ann}, indent=2))

    print(f"count={len(no_ann)}")
    for name in no_ann[:20]:
        print(name)
    print(f"wrote={OUT_JSON}")


if __name__ == "__main__":
    main()
