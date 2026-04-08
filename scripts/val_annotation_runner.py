#!/usr/bin/env python3
import argparse
import glob
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch


def yolo_annotate(model, img_paths, out_dir, device, class_names):
    colors = {
        "helmet": (0, 255, 0),
        "safety_vest": (255, 165, 0),
        "worker": (0, 120, 255),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, img_path in enumerate(img_paths, 1):
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        result = model(str(img_path), device=device, verbose=False, conf=0.25)[0]
        out = frame.copy()
        for box in result.boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
            ci = int(box.cls[0])
            sc = float(box.conf[0])
            cn = class_names[ci] if ci < len(class_names) else str(ci)
            color = colors.get(cn, (200, 200, 200))
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.putText(out, f"{cn} {sc:.2f}", (x1, max(y1 - 5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.imwrite(str(out_dir / img_path.name), out)
        if i % 100 == 0:
            print(f"{out_dir.name}: {i}/{len(img_paths)}", flush=True)


def samurai_gt_annotate(ann_path, img_dir, out_dir, common_set=None):
    colors = {
        "helmet": (0, 255, 0),
        "safety_vest": (255, 165, 0),
        "worker": (0, 120, 255),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(ann_path) as f:
        coco = json.load(f)
    cat_map = {c["id"]: c["name"] for c in coco["categories"]}
    ann_by_img = {}
    for a in coco["annotations"]:
        ann_by_img.setdefault(a["image_id"], []).append(a)
    valid_images = [img for img in coco["images"] if (not common_set or img["file_name"] in common_set)]
    total = len(valid_images)
    print(f"SAMURAI_GT: Processing {total} images after filtering.", flush=True)

    for i, img_meta in enumerate(valid_images, 1):
        img_path = img_dir / img_meta["file_name"]
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        for a in ann_by_img.get(img_meta["id"], []):
            x, y, w, h = [int(v) for v in a["bbox"]]
            cn = cat_map.get(a["category_id"], "unknown")
            color = colors.get(cn, (200, 200, 200))
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, cn, (x, max(y - 5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.putText(frame, "SAMURAI: Ground Truth Reference", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 220), 2)
        cv2.imwrite(str(out_dir / img_meta["file_name"]), frame)
        if i % 100 == 0 or i == total:
            print(f"SAMURAI_GT: {i}/{total}", flush=True)


def load_common_set():
    possible_paths = [
        Path("/nfsshare/joseva/common_888.txt"),
        Path("common_888.txt"),
        Path("/home/joseva/common_888.txt"),
        Path("../common_888.txt")
    ]
    for p in possible_paths:
        if p.exists():
            with open(p) as f:
                cset = {line.strip() for line in f if line.strip()}
            print(f"LOAD_SUCCESS: Loaded {len(cset)} common images from {p}", flush=True)
            return cset
    return set()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["yolo", "yolact", "samurai"], required=True)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    common_set = load_common_set()
    
    if not common_set:
        print("CRITICAL_ERROR: common_888.txt NOT FOUND or EMPTY! Filter cannot be applied.", flush=True)
        if args.mode in ["samurai", "yolact"]:
            print("Aborting: Validation runs MUST have the 888 filter applied.", flush=True)
            import sys
            sys.exit(1)

    if args.mode == "yolo":
        from ultralytics import YOLO

        models = {
            "YOLOv11": "/nfsshare/joseva/buildsight/dataset/Backup_Dataset(annotated)/YOLOv11/runs/detect/runs/train/buildsight/weights/best.pt",
            "YOLOv26": "/nfsshare/joseva/buildsight/dataset/Backup_Dataset(annotated)/YOLOv26/runs/detect/runs/train/buildsight/weights/best.pt",
        }
        img_paths = sorted(Path("/nfsshare/joseva/buildsight/dataset/Backup_Dataset(annotated)/YOLOv11/images/val").glob("*.jpg"))
        
        # Filter by common_set
        if common_set:
            img_paths = [p for p in img_paths if p.name in common_set]
            print(f"Filtered YOLO images to {len(img_paths)} using common_set", flush=True)
            
        for name, weights in models.items():
            print(f"Starting {name}", flush=True)
            model = YOLO(weights)
            yolo_annotate(model, img_paths, Path(f"/nfsshare/joseva/val_annotated/{name}"), args.device, ["helmet", "safety_vest", "worker"])
            print(f"{name} done", flush=True)

    elif args.mode == "yolact":
        YOLACT_DIR = "/nfsshare/joseva/yolact"
        os.chdir(YOLACT_DIR)
        import sys

        sys.path.insert(0, YOLACT_DIR)
        sys.path.insert(0, os.path.join(YOLACT_DIR, "external", "DCNv2"))
        from yolact import Yolact
        from data import cfg, set_cfg
        from layers.output_utils import postprocess
        from utils.augmentations import FastBaseTransform

        set_cfg("buildsight_config")
        cfg.mask_proto_debug = False
        net = Yolact()
        net.load_weights("/nfsshare/joseva/yolact/weights/buildsight_207_80000.pth")
        net.eval()
        net.detect.use_fast_nms = True
        net.detect.use_cross_class_nms = False
        net = net.cuda(args.device)

        out_dir = Path("/nfsshare/joseva/val_annotated/YOLACT_plusplus")
        out_dir.mkdir(parents=True, exist_ok=True)
        img_paths = sorted(Path("/nfsshare/joseva/buildsight/dataset/Backup_Dataset(annotated)/YOLOv11/images/val").glob("*.jpg"))
        
        # Filter by common_set
        if common_set:
            img_paths = [p for p in img_paths if p.name in common_set]
            print(f"Filtered YOLACT++ images to {len(img_paths)} using common_set", flush=True)

        colors = {"helmet": (0, 255, 0), "safety_vest": (255, 165, 0), "worker": (0, 120, 255)}
        class_names = ["helmet", "safety_vest", "worker"]
        for i, img_path in enumerate(img_paths, 1):
            out_path = out_dir / img_path.name
            if out_path.exists():
                continue
                
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            h, w = frame.shape[:2]
            with torch.no_grad():
                ft = torch.from_numpy(frame).cuda(args.device).float()
                b = FastBaseTransform()(ft.unsqueeze(0))
                p = net(b)
                cls_t, scr_t, box_t, _ = postprocess(p, w, h, score_threshold=0.20, visualize_lincomb=False, crop_masks=True)
            out = frame.copy()
            if len(cls_t):
                scores = scr_t[0] if isinstance(scr_t, (list, tuple)) else scr_t
                cls_np = cls_t.detach().cpu().numpy()
                score_np = scores.detach().cpu().numpy()
                box_np = box_t.detach().cpu().numpy()
                for cls_id, score, box in zip(cls_np, score_np, box_np):
                    x1, y1, x2, y2 = [int(v) for v in box.tolist()]
                    cn = class_names[int(cls_id)]
                    color = colors.get(cn, (200, 200, 200))
                    cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(out, f"{cn} {score:.2f}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.imwrite(str(out_path), out)
            if i % 100 == 0:
                print(f"YOLACT++: {i}/{len(img_paths)}", flush=True)
        print("YOLACT++ done", flush=True)

    elif args.mode == "samurai":
        samurai_gt_annotate(
            Path("/nfsshare/joseva/buildsight/dataset/Backup_Dataset(annotated)/SAMURAI/annotations/instances_val.json"),
            Path("/nfsshare/joseva/buildsight/dataset/Backup_Dataset(annotated)/SAMURAI/images/val"),
            Path("/nfsshare/joseva/val_annotated/SAMURAI_GT"),
            common_set=common_set
        )
        print("SAMURAI_GT done", flush=True)


if __name__ == "__main__":
    main()
