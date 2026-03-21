---
description: BuildSight Project Implementation Methodology Workflow
---

# BuildSight Project Implementation Methodology

This workflow outlines the step-by-step implementation plan for the BuildSight project, based on the **BuildSight - Methodology.pdf** documentation. It covers the dual-track execution of AI Vision Model Development (Phase A) and GIS-Based Spatial Safety Mapping (Phase B).

## Phase 0: Environment Setup & Prerequisites

### 1. Hardware & Software Configuration
Ensure the development environment meets the following specifications:
- **GPU**: NVIDIA RTX 4050 (6GB VRAM, CUDA 12.1 compatible) or better.
- **CPU**: Intel Core i7-12650H or equivalent (10+ cores).
- **RAM**: 32GB DDR5.
- **Storage**: 512GB+ NVMe SSD.
- **OS**: Ubuntu 22.04 LTS (recommended for GPU drivers).

### 2. Software Stack Installation
Install the necessary software and libraries:
- **Python**: 3.10.12
- **PyTorch**: 2.1.0 with CUDA 12.1
- **Ultralytics YOLOv8**: 8.0.196
- **CUDA Toolkit**: 12.1.0
- **cuDNN**: 8.9.2
- **GIS Tools**:
    - QGIS 3.34 LTR (Desktop)
    - PostgreSQL 15.4 + PostGIS 3.4 (Database)
    - Python Libraries: `geopandas==0.14.0`, `shapely==2.0.2`, `pyproj==3.6.1`, `folium==0.15.0`, `opencv-python==4.8`

---

## Phase A: AI Vision Model Development

### 1. Data Collection
- **Primary Source (CCTV)**:
    - **Setup**: Install 4 fixed CCTV cameras at corners of a G+1 residential site (~2000m²).
    - **Specs**: 1080p (FHD), 25 FPS, H.264, 4Mbps bitrate, 90-110° FOV, mounted at 3-4m height.
    - **Protocol**: RTSP streaming.
    - **Duration**: Record for 30 consecutive days (720 hours total). Active recording based on motion detection (~8-10h/day).
    - **Expected Volume**: ~300 hours footage, ~2.1 TB storage.
- **Supplementary Source (Public Datasets)**:
    - Acquire relevant datasets: **Safety-Helmet-Wearing-Dataset** (Normal), **GDUT-HWD** (PPE), **SFCHD** (Low-light), **ExtCon** (Dust/Fog), **Roboflow Construction Hazards** (Dust), **CrowdHuman** (Dense crowds), **WiderPerson** (Extreme crowds).
    - Total supplementary images: ~5,000 selected for diversity.

### 2. Preprocessing Pipeline
Implement the following preprocessing steps in Python (OpenCV):
1.  **Frame Extraction**:
    - Extract frames at **2 FPS** from native 25 FPS video (every 12.5th frame).
    - Output as JPEG (95% quality).
2.  **Duplicate Removal**:
    - Calculate perceptual hash (pHash) for each frame (8x8 grayscale).
    - Remove if Hamming distance < 5 bits compared to previous frame.
3.  **Resolution Standardization**:
    - Resize all frames to **1280x720 (720p)** with letterboxing to maintain aspect ratio.
4.  **Brightness Normalization**:
    - Check average pixel intensity. If < 70 (0-255 scale), apply **Gamma Correction** ($$\gamma=0.7$$) followed by CLAHE.
5.  **Haze Detection & Enhancement**:
    - Calculate Dark Channel Prior. If Haze Score > 0.4, apply Dehazing (Dark Channel + Guided Filter, $\omega=0.95$, $t_0=0.1$).
6.  **Noise Reduction**:
    - Apply Gaussian Blur (kernel 3x3, $\sigma=0.5$).

### 3. Data Annotation
- **Tools**: Use Label Studio.
- **Classes**: `Worker-Helmet`, `Worker-NoHelmet`, `SafetyVest`, `NoSafetyVest`, `Heavy Equipment`.
- **Workflow**:
    1.  **Pre-Annotation**: Run YOLOv8n (COCO weights) to generate initial bounding boxes.
    2.  **Manual Refinement**: Correct boxes and assign classes.
    3.  **Quality Control**: Secondary review of 10% frames. Target IoU > 0.85.
- **Total Dataset**: Target ~65,000 annotated frames (60k CCTV + 5k Public).

### 4. Dataset Splitting
Partition the dataset:
- **Training Set (70%)**: 45,500 frames (Days 1-21 of CCTV).
- **Validation Set (15%)**: 9,750 frames (Days 22-25 of CCTV).
- **Test Set (15%)**: 9,750 frames (Days 26-30 of CCTV).
    - *Note*: Ensure Test Set spans 4 scenarios: **Normal (20%)**, **Dusty (25%)**, **Low-Light (25%)**, **Crowded (30%)**.

### 5. Model Training (YOLOv8n)
Configure training with these hyperparameters:
- **Input Size**: 640x640
- **Batch Size**: 16 (for 6GB VRAM)
- **Epochs**: 100 (Early stopping patience: 20)
- **Optimizer**: AdamW (LR: 0.01 -> 0.0001 cosine decay)
- **Momentum**: 0.937
- **Weight Decay**: 0.0005
- **Augmentation**: Mosaic (100%), Mixup (50%), Random Rotate (±10°, 50%), HSV Shift, Cutout (30%).

### 6. Model Evaluation
Evaluate using metrics:
- **mAP@0.5** (Target $\ge$ 0.75)
- **Recall** (Target $\ge$ 0.85 - Critical for Safety)
- **Precision** (Target $\ge$ 0.80)
- **Inference Speed** (Target $\ge$ 10 FPS on RTX 4050)
- **Selection Formula**: $Score = 0.20(S1) + 0.25(S2) + 0.25(S3) + 0.30(S4)$. Prioritize high recall in S4 (Crowded).

---

## Phase B: GIS-Based Spatial Safety Mapping

### 1. Coordinate System Definition
- **Local System**: Cartesian (Meters). Origin (0,0) at SW corner of site. Z=0 at ground.
- **Global System**: WGS 84 / UTM Zone 44N (EPSG:32644).
- **Transformation**: `UTM_X = UTM_Origin_X + Local_X`.

### 2. Camera Calibration & Transformation
- **Intrinsic Calibration**:
    - Use 9x6 checkerboard pattern. Capture 20 images.
    - Extract `fx, fy, cx, cy` and distortion coefficients using `cv2.calibrateCamera`.
- **Extrinsic Calibration**:
    - Identify 6+ Ground Control Points (GCPs) with known 3D coords.
    - Solve PnP (`cv2.solvePnP`) to get Rotation (R) and Translation (t).
- **Pixel-to-World Algo**:
    1.  Extract bottom-center of bounding box $(u, v)$.
    2.  Undistort points.
    3.  Back-project to 3D ray ($R^{-1} \times [x_n, y_n, 1]$).
    4.  Intersect ray with floor plane ($Z=FloorHeight$).
    5.  Assign floor level based on Y-coordinate lookup or geometric calculation.

### 3. Spatial Database (PostGIS)
- **Schema**: `worker_detections` table.
    - Columns: `detection_id`, `timestamp`, `camera_id`, `frame_number`, `world_x`, `world_y`, `floor_level`, `class_label`, `confidence`, `geom` (Geometry Point).
- **Indexing**: GiST index on `geom` for spatial queries.

### 4. Heatmap Generation
- **Grid Discretization**: Overlay 2x2m grid on site map.
- **Worker Density**:
    - Count workers per cell over 5-min sliding window.
    - Apply **Kernel Density Estimation (KDE)** (Gaussian kernel, $\sigma=5m$) for smooth visualization.
- **Risk Heatmap**:
    - Compute `Risk_Score = 0.35*Density + 0.30*PPE + 0.20*Hazard + 0.15*Movement`.
    - **Hazard Proximity**: Inverse distance weighting to open edges/equipment.

---

## Phase C: Integration & Alert System

### 1. Alert Logic Configuration
Configure the alert engine with these thresholds:
- **Density Exceedance**: > 0.5 workers/m² for > 2 min (Warning).
- **PPE Non-Compliance**: > 30% workers without PPE (Warning).
- **Restricted Zone Entry**: Any worker in no-access zone (Critical).
- **Hazard Proximity**: Worker within 2m of hazard without PPE (Critical).
- **High Risk Score**: Risk > 0.8 for > 3 min (Critical).

### 2. Alert Delivery
- **Critical**: SMS (Twilio) + Audio Alarm (Site Office).
- **Warning**: Mobile Push (FCM).
- **Dashboard**: Real-time WebSocket feed.

### 3. Final Validation
- Verify **99% System Uptime**.
- Verify **< 500ms End-to-End Latency**.
- Validate Detection Accuracy > 90% in controlled tests.
