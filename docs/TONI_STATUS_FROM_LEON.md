Leon status for Toni

Date: 2026-03-31

Completed on SASTRA node1:
- Restored deleted `external/DCNv2` sources in `/nfsshare/joseva/yolact` from git.
- Confirmed the original compile failure was not the only blocker.
- Patched `/nfsshare/joseva/yolact_condition_eval.py` to:
  - add `external/DCNv2` to import path
  - resolve images from the actual dataset roots
  - handle `postprocess()` score output shape correctly
  - use `GPU_ID = 0` on this host, where the A100 is `cuda:0`
- Ran YOLACT++ per-condition evaluation successfully.
- Verified results were written to `/nfsshare/joseva/yolact_condition_eval.json`.
- Updated `/nfsshare/joseva/ensemble_inference.py` weights from `[0.6, 0.4]` to `[0.45, 0.55]`.
- Generated qualitative samples for all 4 model outputs:
  - `/nfsshare/joseva/annotated_samples/YOLOv11/*.jpg`
  - `/nfsshare/joseva/annotated_samples/YOLOv26/*.jpg`
  - `/nfsshare/joseva/annotated_samples/ensemble/*.jpg`
  - `/nfsshare/joseva/annotated_samples/YOLACT_plusplus/*.jpg`
- Measured runtime on A100 (`cuda:0`):
  - YOLOv11: `17.93 FPS`, `0.063 GB`
  - YOLOv26: `23.70 FPS`, `0.073 GB`
  - Ensemble: `9.45 FPS`, `0.091 GB`
  - YOLACT++: `27.29 FPS`, `0.212 GB`
- Generated SAMURAI ground-truth reference images:
  - `/nfsshare/joseva/annotated_samples/SAMURAI/normal.jpg`
  - `/nfsshare/joseva/annotated_samples/SAMURAI/dusty.jpg`
  - `/nfsshare/joseva/annotated_samples/SAMURAI/low_light.jpg`
  - `/nfsshare/joseva/annotated_samples/SAMURAI/crowded.jpg`
- Computed per-condition `mAP50-95` for `S2`, `S3`, `S4` and saved them to `/nfsshare/joseva/map95_results.json`

YOLACT++ per-condition results:
- S1 normal: `P=0.5263`, `R=0.6893`, `F1=0.5969`, `mAP50=0.5133`, `56.54 ms`
- S2 dusty: `P=0.6900`, `R=0.9315`, `F1=0.7928`, `mAP50=0.7994`, `27.17 ms`
- S3 low_light: `P=0.5410`, `R=0.7591`, `F1=0.6317`, `mAP50=0.5850`, `30.29 ms`
- S4 crowded: `P=0.6184`, `R=0.6086`, `F1=0.6134`, `mAP50=0.4762`, `33.47 ms`

Important note:
- Toni's original blocker diagnosis was incomplete. The failing eval also had a broken image root and score-tensor handling bug, and the host GPU mapping had changed. The DCN compile path was not the final fix.

Local repo sync:
- Updated `scripts/yolact_condition_eval.py` with the same evaluator fixes so the workspace matches the version that ran remotely.

mAP50-95 per condition:
- YOLOv11: `S1=0.486`, `S2=0.6321`, `S3=0.4186`, `S4=0.4075`
- YOLOv26: `S1=0.478`, `S2=0.6268`, `S3=0.4157`, `S4=0.3988`
