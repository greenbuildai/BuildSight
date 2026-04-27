# PPE Detection & Helmet Validation Backup Notes - 2026-04-12

## Current Helmet Confidence Thresholds
| Condition | Adaptive Postprocess (Conf) | Server WBF Post-Gate |
|-----------|-----------------------------|-----------------------|
| S1_normal | 0.32                        | 0.30                  |
| S2_dusty  | 0.18                        | 0.25                  |
| S3_low_lt | 0.18                        | 0.22                  |
| S4_crowd  | 0.18                        | 0.18                  |

*Note: S4 thresholds were recently lowered (2026-04-11) to improve recall on distant/elevated workers, likely contributing to increased false positives.*

## Current PPE-to-Worker Association Rules
- **Proximity**: PPE centroid must be within the worker bounding box (centroid-in-box).
- **Expansion**: Worker boxes are expanded by 20% in some checks (`has_worker_overlap`).
- **IoU**: Minimum IoU of 0.08 is accepted as association in `adaptive_postprocess.py`.
- **Temporal**: `TemporalPPEFilter` allows synthetic PPE generation if a worker has a "streak" of 2+ hits with PPE.

## Current Known Helmet False Positives
- **Bare Heads**: Hair/Skin being treated as helmet regions.
- **Wrists/Palms/Hands**: Small round skin-toned regions near the head or body.
- **Detached Helmets**: Detections appearing alone or associated with incorrect blobs/occlusions.

## Reason for this Backup
- Investigating and fixing false positives where bare heads, wrists, and palms are incorrectly classified as helmets.
- Implementing stricter geometric constraints (helmet must be in the top region of a worker).
- Implementing skin-tone and texture rejection for helmet boxes.
- Strengthening worker-to-helmet association requirements.
