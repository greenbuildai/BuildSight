# BuildSight Application

BuildSight is a real-time PPE compliance and hazard monitoring system for construction sites.

## Architecture
- **Backend**: Python (FastAPI), YOLOv8, OpenCV
- **Frontend**: React (Vite), TailwindCSS
- **Communication**: MJPEG (Video), WebSockets (Alerts)

## Setup & Running

### Prerequisites
- Python 3.9+
- Node.js 16+

### 1. Backend Setup
1.  Navigate to Core directory: `cd d:/Jovi/Projects/BuildSight/Core`
2.  Install dependencies:
    ```bash
    pip install -r backend/requirements.txt
    ```
3.  Start the server:
    ```bash
    python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
    ```

### 2. Frontend Setup
1.  Navigate to frontend directory: `cd frontend`
2.  Install dependencies:
    ```bash
    npm install
    ```
3.  Start the development server:
    ```bash
    npm run dev
    ```

### 3. Usage
- Open the frontend URL (typically `http://localhost:5173`).
- The dashboard will show the live camera feed.
- "Red Zone" violations and PPE non-compliance will trigger real-time alerts.

## Configuration
- Modify `backend/config.py` to change Video Source path or Model path.
- Modify `backend/services/gis.py` to update Red Zone coordinates.
