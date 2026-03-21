# Preparing BuildSight Datasets (Protocol v1.0)

This skill executes the structured, scenario-aware annotation and validation pipeline for BuildSight. It ensures data is ready for YOLOv8n (Detection), YOLACT++ (Instance Segmentation), and SAMURAI (Video Segmentation).

## When to use this skill
- When preparing datasets for construction safety monitoring.
- Before training the Vision Model or Video Segmentation models.
- To validate that S4 (Crowded) frames meet the strict safety-critical criteria (>= 5 workers).

## 1. Class Taxonomy (MANDATORY)

| Class ID | Class Name | Rules |
|----------|------------|-------|
| 0 | Person | Every visible worker. No merged boxes. |
| 1 | Helmet | Annotated separately if visible. |
| 2 | Vest | Annotated separately if visible. |

- **Threshold**: Annotate if partially visible (>40%) or truncated but identifiable.
- **Overlap**: Never share bounding boxes for overlapping workers.

## 2. Scenario Criteria & Weights

| Scenario | Weight | Key Filtering Rule |
| :--- | :--- | :--- |
| **S1 (Normal)** | 20% | 1-3 workers, clear visibility, 100-180 pixel intensity. |
| **S2 (Dusty)** | 25% | Haze score > 0.4. Requires Dehazing. |
| **S3 (Low-Light)** | 25% | < 80 pixel intensity. Requires Gamma Correction. |
| **S4 (Crowded)** | 30% | **MANDATORY: ≥ 5 workers.** Overlap present. |

## 3. Mandatory Workflow

### Phase 1: Filtering & Pre-Processing
1.  **Filter**: Move frames to `S1_normal`, `S2_dusty`, `S3_lowlight`, `S4_crowded` based on criteria.
2.  **S4 Audit**: Reject any S4 frame with < 5 workers.
3.  **Enhance**: Run `scripts/scenario_enhancer.py` for S2/S3.

### Phase 2: Hybrid Annotation
1.  **Draft**: Run `scripts/auto_annotate.py` (YOLOv8n) using Class IDs 0, 1, 2.
2.  **Refine**: Manual correction in Label Studio or CVAT.
3.  **Segment**: Create polygon masks for YOLACT++ (COCO JSON).

### Phase 3: Validation & Metadata
1.  **Validate**: Run `scripts/validate_protocol.py` to ensure zero empty files and S4 compliance.
2.  **Metadata**: Generate `metadata.json` for each scenario containing worker count and intensity scores.

## 4. Quality Control Checklist (QC)
- [ ] No empty annotation files.
- [ ] All visible persons (Class 0) annotated.
- [ ] S4 verified ≥5 workers per frame.
- [ ] Correct Class IDs: 0, 1, 2.
- [ ] Inter-annotator IoU ≥ 85%.

## Resources
- `scripts/auto_annotate.py`: YOLOv8n pre-labeler (Person, Helmet, Vest).
- `scripts/validate_protocol.py`: S4 compliance and class distribution report.
- `templates/label_studio_config.xml`: Updated with 0/1/2 Class IDs.
- `scripts/scenario_enhancer.py`: Brightness and Haze processing.

## Dataset Structure
```
dataset/
 ├── S1_normal/
 │    ├── images/
 │    ├── labels_yolo/
 │    ├── masks_coco/
 ├── S2_dusty/ ...
 ├── S3_lowlight/ ...
 ├── S4_crowded/ ...
```
