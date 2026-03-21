# Handback to Gemini

**Task ID**: HANDOFF-002
**Status**: failure
**Last Updated**: 2026-03-21T15:30:00+05:30

## Output Summary
The annotation pipeline started successfully and processed **296 of 1373** Normal_Site_Condition images (~22%) before crashing with a **Segmentation Fault (exit code 139)** at the 35-minute mark. No final COCO JSON or updated manifest was written — in-memory state was lost on crash.

## Files Modified
- `Dataset/Final_Annotated_Dataset/images/` and `labels/` — partially updated (296 new images copied/labels written before crash)

## Commands Run
```
python scripts/annotate_indian_dataset.py --conditions Normal_Site_Condition --skip-sam
```

## Crash Diagnostics

### Symptom
```
296/1373 [35:00<5:52:22, 19.63s/img] ... Segmentation fault (exit code 139)
```

### Root Cause — Progressive Memory Pressure
There is **no `torch.cuda.empty_cache()` or `gc.collect()` call** anywhere inside the per-image loop in `annotate_indian_dataset.py`. After processing ~296 images over 35 minutes:
- GPU VRAM fragments progressively (GroundingDINO keeps intermediate CUDA tensors alive between calls)
- In-memory COCO structures (`coco_by_split`) grow unbounded as every image's annotations accumulate in RAM
- The native CUDA/GroundingDINO C extension eventually segfaults under memory pressure

### Telltale Pattern — Processing Time Inflation
| Image range | Speed |
|---|---|
| 240–260 | ~4–6 s/img (normal) |
| 269–280 | ~10–13 s/img (GC pressure) |
| 283–290 | ~15–19 s/img (thrashing) |
| 293–296 | ~19–24 s/img → **crash** |

The monotonic slowdown followed by segfault is the classic signature of un-released CUDA allocations.

### Hardware Verified
- GPU: NVIDIA GeForce RTX 4050 Laptop GPU — 6.0 GB VRAM ✓
- `--skip-sam` applied (SAM bypassed) ✓
- Crash is NOT a CUDA OOM error — it is a native segfault from memory fragmentation

## Required Fix
Add `torch.cuda.empty_cache()` (and optionally `gc.collect()`) inside the per-image loop in `annotate_indian_dataset.py`, ideally every N images (e.g., every 10). Example location — after `write_yolo_files(...)` on line ~1193:

```python
# Every 10 images, release cached GPU memory
if image_id_counter % 10 == 0:
    torch.cuda.empty_cache()
```

This fix is minimal, non-breaking, and targets the exact failure mode. Gemini should apply this fix and re-issue HANDOFF-003 to re-run the pipeline.

## Open Issues / Blockers
- The 296 partially-written label files are valid (they were written per-image before the crash). A future run with a skip-if-label-exists guard would avoid re-processing them.
- The COCO JSON (`new_crowded_*.json`) was **not written** — the crash happened mid-run before the post-loop save.

## Suggested Next Action
1. **Gemini applies the fix**: add `torch.cuda.empty_cache()` every 10 images in the per-image loop of `scripts/annotate_indian_dataset.py`.
2. **Optional**: add a per-image skip guard (`if label_path.exists(): continue`) to resume from image 297 rather than re-processing the first 296.
3. **Issue HANDOFF-003** for Claude to re-run: `python scripts/annotate_indian_dataset.py --conditions Normal_Site_Condition --skip-sam`
