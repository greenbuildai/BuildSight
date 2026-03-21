# Handoff to Claude Code

**Task ID**: HANDOFF-003
**Status**: ready_for_claude
**Assigned To**: claude

## Objective
Re-run the automated annotation pipeline for Normal_Site_Condition to resume an interrupted job.

## Context
In HANDOFF-002, the pipeline crashed at image 296 due to a progressive GPU memory fragmentation issue (Segmentation Fault / exit code 139).

Gemini has applied the following patches to `scripts/annotate_indian_dataset.py`:
1. **Memory Fix**: Added `torch.cuda.empty_cache()` and `gc.collect()` after `write_yolo_files` (every 10 images) to prevent the CUDA cache from destroying the native C driver over long loops.
2. **Resume Guard**: Added a skip check that looks for the `.txt` label file in `OUTPUT_DIR / "labels" / split`. If it exists, the script simply `continue`s to the next image without invoking DINO.

Because the previous 296 label files are physically present on disk, the script will rapidly skip the first 296 images and resume inference at image 297, eventually finishing all 1373 images safely.

## Implementation Plan
1. Validate that the workspace is clean.
2. Execute the pipeline:
   ```bash
   python scripts/annotate_indian_dataset.py --conditions Normal_Site_Condition --skip-sam
   ```
3. Wait for the `ANNOTATION COMPLETE` message. You should see the progress bar rapidly skip the first 296 images and then slow down to standard inference speed (~5s/img).
4. Monitor system resources. If another crash occurs, note the image ID and exit code in your handback.
5. If successful, complete the execution, update `TASK_STATE.json` to "pending_gemini", and write a detailed handback report in `GEMINI_HANDBACK.md` specifying exactly how many images were skipped vs processed.

## Required Output (in GEMINI_HANDBACK.md)
- **Status**: success / failure
- **Summary**: Did the memory fix prevent the crash? How many images were processed in this run vs skipped? 
- **Files Modified**: e.g., `Dataset/Final_Annotated_Dataset/labels/*`
- **Commands Run**: List commands
- **Open Issues**: Identify any outstanding bugs or COCO/Metadata gaps resulting from the fragmented run.
