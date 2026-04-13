---
name: mastering-geoai
description: Expert GeoAI assistant for BuildSight. Handles spatial projections, homography, real-time heatmaps, QGIS integration, and geoai-py dependency management (PyTorch, spatial libs). Use when implementing GeoAI, mapping pixels to coordinates, or adjusting spatial logic.
---

# Mastering GeoAI for BuildSight

## When to use this skill
- Implementing or modifying the GeoAI Tactical HUD, heatmaps, or risk density features.
- Handling pixel-to-GPS (homography) transformations or spatial scaling in BuildSight.
- Working with QGIS outputs, GIS layers, or site coordinate systems.
- Verifying or troubleshooting `geoai` package installations and deep learning dependencies.

## BuildSight GeoAI Implementation Standards

As an expert GeoAI specialist for BuildSight, your goal is to bridge the gap between computer vision detections (pixel space) and real-world industrial intelligence (geospatial space).

### Core Responsibilities
1. **Homography & Projections:** Ensure accurate pixel-to-GPS mapping. Avoid relying on simple linear fallback unless explicitly instructed. Emphasize proper matrix calibration and coordinate transformations using `OpenCV` and `GeoPandas`.
2. **Dynamic Heatmaps (KDE):** BuildSight uses real-time 2D risk heatmaps (Kernel Density Estimation). Heatmaps must decay over time (e.g., 1.5s trails) instead of showing persistent stale patches across the UI.
3. **Zone Pressure Tracking:** Correlate worker locations to predefined risk zones (e.g., "Crane_Swing_Radius", "Excavation_Edge"). Automatically alert on breaches using geospatial intersection geometry.
4. **Environment Health:** Always ensure the host has the correct `geoai-py` dependencies, spatial processing libraries, and `PyTorch` for CUDA-accelerated inference.

## Workflow 1: Validating the GeoAI Environment
Before writing complex spatial pipeline logic, you must ensure the environment supports GPU integration and spatial mapping computations.

### Step 1: Base Check
```bash
python3 -c "import geoai; print(f'geoai v{geoai.__version__}')"
```
*If this fails, skip to step 3.*

### Step 2: Comprehensive Capabilities Check
If verifying an environment for deep analytics, check spatial and deep learning extras (e.g., CUDA) for full capability:
```bash
python3 -c "
import sys
deps = ['geoai', 'geopandas', 'rasterio', 'rioxarray', 'shapely', 'leafmap', 'numpy', 'pandas', 'matplotlib', 'torch', 'torchvision', 'transformers']
for dep in deps:
    try:
        mod = __import__(dep)
        print(f'{dep}: {getattr(mod, \"__version__\", \"installed\")}')
    except ImportError:
        print(f'{dep}: NOT INSTALLED')

try:
    import torch
    if torch.cuda.is_available():
        print(f'CUDA: {torch.version.cuda} (device: {torch.cuda.get_device_name(0)})')
    else:
        print('CUDA: not available (CPU only)')
except ImportError:
    pass
"
```

### Step 3: Installation Guidelines
If dependencies are missing, provide these exactly to the user (do not execute without permission):
- **Core GeoAI:** `pip install geoai-py`
- **GPU Vision (Crucial for YOLO/Samurai):** `pip install torch torchvision`
- **All Spatial Extras:** `pip install "geoai-py[extra]"`

## Workflow 2: Implementing Spatial Translations
When transforming YOLO bounding boxes to QGIS/Leaflet map coordinates:
1. Always calculate the **bottom-center** of the bounding box `[x_center, y_max]` as the grounding point (representing the worker's feet).
2. Apply the perspective transformation matrix $H$ using `cv2.perspectiveTransform` rather than generic affine scaling.
3. Broadcast the $lat, lng$ updates over the WebSocket `IntelligenceEngine` so the frontend React components receive true geographic points.

## Workflow 3: Frontend GeoAI Visualization
When expanding frontend components like `GeoAIMap.tsx` or `GeoAIPage.tsx`:
- **Leaflet/QGIS Integration:** Use standard `[lat, lng]` notation to interface cleanly with external GIS layers.
- **State Optimization:** Ensure the React UI state defers to the global `SettingsContext` for toggling intensive WebGL/Canvas map features (like KDE Heatmap Overlays) to preserve dashboard performance. Avoid deep prop-drilling for spatial toggles.
