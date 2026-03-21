# BuildSight AI - Quick Startup Guide

## Prerequisites
- Python 3.8+
- Node.js 16+
- CUDA-capable GPU (optional, will fallback to CPU)

## Starting the Application

### Terminal 1: Backend Server
```bash
cd d:\Jovi\Projects\BuildSight\Core
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

**Expected Output:**
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Loading YOLO model...
INFO:     Started reloader process...
INFO:     Application startup complete.
```

### Terminal 2: Frontend Dev Server
```bash
cd d:\Jovi\Projects\BuildSight\Core\frontend
npm run dev
```

**Expected Output:**
```
VITE v5.x.x  ready in xxx ms
➜  Local:   http://localhost:5173/
➜  Network: use --host to expose
```

## Verifying the Application is Running

1. Open browser to http://localhost:5173
2. Check "System Status" indicator shows "Operational" (green)
3. Backend logs should show: `INFO:     Application startup complete`

## Troubleshooting

### Backend Not Running
**Symptom:** System Status shows "Backend Offline" (red)

**Solution:**
1. Open terminal and run backend command above
2. Wait for "Application startup complete" message
3. Refresh browser

### Port Already in Use
**Symptom:** Error: "Address already in use"

**Solution:**
```bash
# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Linux/Mac
lsof -ti:8000 | xargs kill -9
```

### Video Upload Stuck at 0%
**Symptom:** Upload progress doesn't advance

**Diagnosis:**
1. Check browser console (F12) for errors
2. Check "System Status" indicator
3. Verify backend terminal shows no errors

**Common Causes:**
- Backend server not running → Start backend
- File too large (>2GB) → Use smaller file
- Invalid file type → Use video file (.mp4, .avi, .mov, etc.)
- Network connectivity issues → Check firewall/antivirus

**Solution:** Restart backend server and try again

### Browser Console Errors
**How to check:**
1. Press F12 in browser
2. Click "Console" tab
3. Look for red error messages

**Common errors:**
- `Failed to fetch` → Backend not running
- `NetworkError` → Connection blocked by firewall
- `Invalid file type` → Wrong file format

### Backend Logs
The backend terminal will show detailed logs:
- `[INFO]` - Normal operations
- `[ERROR]` - Something went wrong
- Look for upload-related messages when uploading fails

## Development Tips

### Hot Reload
- Frontend: Changes to .jsx files reload automatically
- Backend: Changes to .py files reload automatically (with --reload flag)

### Stopping the Application
1. Press `Ctrl+C` in each terminal
2. Wait for graceful shutdown

### Clearing Upload Cache
If uploads get stuck:
```bash
cd d:\Jovi\Projects\BuildSight\Core\inputs\uploads
# Delete all files in this directory
```

## File Structure

```
BuildSight/Core/
├── backend/
│   ├── main.py          # FastAPI backend server
│   ├── services/        # ORN, inference, streaming
│   └── config.py        # Configuration
├── frontend/
│   └── src/
│       ├── components/  # React components
│       └── lib/         # Utilities (API, health checks)
├── inputs/
│   └── uploads/         # Uploaded video files
└── gis/                 # GIS & ORN models
```

## Next Steps

Once both servers are running:
1. Upload a video file via the UI
2. Watch the PPE detection analyze the video
3. View alerts in the Live Alerts panel
4. Check stats in the Stats Panel

## Support

If issues persist:
1. Check browser console (F12)
2. Check backend terminal logs
3. Verify all dependencies installed
4. Ensure ports 5173 and 8000 are available
