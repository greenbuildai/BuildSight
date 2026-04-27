# BuildSight: Accelerated Execution Timetable & Workflow
**Target Deadline:** 3rd Week of April 2026
**Current Date:** March 11, 2026

## Project Goal
Develop an AI-based Construction Safety Monitoring System integrating YOLO (Detection), YOLACT++ (Instance Segmentation), SAMURAI (Video Tracking), and GIS spatial analysis, customized for Indian site conditions.

---

## 📅 Weekly Execution Timetable

### Week 1: March 11 – March 17
*Focus: Data Finalization & Pre-computation*
*   **Agent (Jovi):** Finalize dataset categorization (S1-S4).
*   **Agent (Jovi):** Run `auto_annotate.py` to draft annotations for YOLO, YOLOACT++, and SAMURAI.
*   **Infrastructure (Jovi):** [x] WhatsApp/Telegram AI Bridge 🟢 **ONLINE**
*   **User:** Validate edge cases in S3 (Low Light) and S4 (Crowded) conditions via Label Studio.
*   **User:** Initialize GIS mapping (QGIS/ArcGIS). Load the base Indian construction site maps and map the physical camera locations.
*   **Agent (Jovi):** Prepare the environment for GPU training.

### Week 2: March 18 – March 24
*Focus: Model Training & GIS Framework (75% Checkpoint)*
*   **Agent (Jovi):** Train YOLO models (Detection) on the datasets.
*   **Agent (Jovi):** Train/Fine-tune YOLACT++ (Segmentation) instances simultaneously on cloud/secondary GPU.
*   **User:** Complete 75% GIS workflow. Establish the spatial grid, geo-fencing (Safe/Hazard zones), and define the Camera-to-Real-World perspective transform (Homography Matrix).
*   **Agent (Jovi):** Ensure SAMURAI logic is scripted for temporal tracking.

### Week 3: March 25 – March 31
*Focus: Comparative Study & Occlusion Handling*
*   **Agent (Jovi):** Generate the comprehensive analytical report using the metrics (mAP50, mAP50-95, Precision, Recall) across all 4 conditions to identify the best model structure.
*   **Agent (Jovi):** Implement Advanced Occlusion Handling (e.g., DeepSORT/ByteTrack) on the winning model to ensure workers are tracked even when hidden behind machinery/pillars.
*   **User:** Review the comparative analysis and formalize the results for the 2nd Review Presentation (2nd week of April).

### Week 4: April 1 – April 7
*Focus: Preparation for 2nd Review*
*   **User:** Draft and refine the project report, methodology slides, and findings for the 2nd Review.
*   **Agent (Jovi):** Generate visualizations, dataset distribution graphs, and comparative metric charts to support the presentation.
*   **Joint:** Ensure the comparative study perfectly answers the core research question regarding Indian site conditions.

### Week 5: April 8 – April 14 (2nd Review Week)
*Focus: 2nd Review & System Integration*
*   **User:** Successfully deliver the 2nd Review Presentation.
*   **Agent (Jovi):** Begin integrating the trained Vision Models with the GIS spatial layer (The final 25% of GIS).
*   **Joint:** Stream the live AI detection coordinates onto the QGIS/ArcGIS map to test real-time spatial safety monitoring.

### Week 6: April 15 – April 21 (Final Deadline)
*Focus: Final Polish & Delivery*
*   **Agent (Jovi):** Package the system into an accessible, scalable dashboard (Flask/Streamlit/React) that can be deployed for any site in India.
*   **User:** Finalize the major project report and thesis document.
*   **Joint:** End-to-end system testing and generation of final demo videos.

---

## ⚙️ Core Technical Workflow

1.  **Data Preparation Pipeline:** Raw Images -> Categorization Script -> Enhancers (Dehazing/Gamma) -> Auto-Annotation -> Manual Refinement -> YOLO/COCO JSON Export.
2.  **AI Vision Pipeline:** Camera Feed -> Frame Extraction -> YOLO/YOLACT++ Inference -> SAMURAI Tracking -> Occlusion Resolution (ByteTrack).
3.  **Spatial Integration (GIS):** AI Bounding Box (Pixels) -> Homography Transform -> GPS Coordinates (Lat/Lng) -> GIS Zone Check (Safe vs. Hazardous).
4.  **Actionable Output:** Alert Generation -> Dashboard Visualization -> Safety Logs inside Supabase/DB.
