import os

# Removed unstable global OpenCV hardware acceleration environment constraints
# as they can block the process on standard builds.


import cv2

import threading
import time
import asyncio
import shutil
import uuid
import queue
import numpy as np
from pathlib import Path
import mimetypes
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import Optional
from . import config
from .services.inference import inference_service
from .services.gis import gis_service
from .services.stream import stream_service
from .services.input_source import InputSourceManager
from .services.orn import orn_service, PPE_PRESENT, PPE_ABSENT, PPE_UNCERTAIN
from .schemas import AlertEvent

import logging

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="BuildSight API", version="1.0.0")

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/uploads", StaticFiles(directory=config.UPLOADS_DIR), name="uploads")

# Global Video State
class VideoState:
    def __init__(self):
        self.raw_frame = None
        self.processed_frame = None
        self.raw_frame_id = 0
        self.processed_frame_id = 0
        self.lock = threading.Lock()
        self.running = False

video_state = VideoState()
input_source = InputSourceManager(config.VIDEO_SOURCE)
frame_queue: "queue.Queue[tuple[int, np.ndarray]]" = queue.Queue(maxsize=1)
upload_sessions = {}
upload_lock = threading.Lock()

def reset_video_buffers() -> None:
    with video_state.lock:
        video_state.raw_frame = None
        video_state.processed_frame = None
        video_state.raw_frame_id = 0
        video_state.processed_frame_id = 0
    while True:
        try:
            frame_queue.get_nowait()
        except queue.Empty:
            break

def get_video_capture(source: Optional[str], mode: str):
    """Helper to initialize video capture based on mode"""
    if not source:
        return None

    logger.info(f"Switching to {mode.upper()} source: {source}")
    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    
    # Try to enable hardware acceleration, but don't fail if unavailable
    try:
        hw_accel = getattr(cv2, "CAP_PROP_HW_ACCELERATION", None)
        hw_device = getattr(cv2, "CAP_PROP_HW_DEVICE", None)
        accel_any = getattr(cv2, "VIDEO_ACCELERATION_ANY", None)
        if hw_accel is not None and accel_any is not None:
            cap.set(hw_accel, accel_any)
        if hw_device is not None:
            cap.set(hw_device, 0)
    except Exception as e:
        logger.warning(f"Hardware acceleration setup failed (non-critical): {e}")
    
    if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    if not cap.isOpened():
        logger.error(f"Failed to open source: {source}")
        cap.release()
        return None
    return cap

def capture_video():
    """Capture thread: reads frames and keeps the latest available for rendering."""
    cap = None
    frame_id = 0
    frame_interval = 0.0

    while video_state.running:
        # Check for mode switch request
        if input_source.consume_change() or cap is None:
            state = input_source.get_state()
            if cap is not None:
                cap.release()
                cap = None

            if not state.source:
                reset_video_buffers()
                time.sleep(0.2)
                continue

            cap = get_video_capture(state.source, state.mode)
            if cap is None:
                time.sleep(0.5)
                continue

            inference_service.reset()
            orn_service.reset()
            frame_id = 0 # Reset frame counter on source switch
            frame_id = 0 # Reset frame counter on source switch
            fps = cap.get(cv2.CAP_PROP_FPS)
            if not fps or fps <= 0 or fps > 120:
                fps = 30.0 # Default fallback to prevent spin loop
            frame_interval = 1.0 / fps
            reset_video_buffers()
            
        loop_start = time.perf_counter()
        ret, frame = cap.read()
        
        # Handle Read Failure / End of Video
        if not ret:
            state = input_source.get_state()
            if state.mode == "live":
                logger.warning("Live stream interrupted. Attempting reconnect...")
                cap.release()
                time.sleep(1) # Backoff
                cap = get_video_capture(state.source, state.mode)
                continue
            else:
                # Loop file video
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

        frame_id += 1
        with video_state.lock:
            video_state.raw_frame = frame
            video_state.raw_frame_id = frame_id

        try:
            if frame_queue.full():
                frame_queue.get_nowait()
            frame_queue.put_nowait((frame_id, frame.copy()))
        except queue.Full:
            pass

        if frame_interval > 0:
            elapsed = time.perf_counter() - loop_start
            time.sleep(max(0, frame_interval - elapsed))

    if cap is not None:
        cap.release()

def process_frames():
    """Inference thread: processes frames independently to avoid blocking playback."""
    while video_state.running:
        try:
            frame_id, frame = frame_queue.get(timeout=0.2)
        except queue.Empty:
            continue

        if gis_service.red_zone is None:
            h, w = frame.shape[:2]
            config.RED_ZONE_COORDS = [[w//2, h//2], [w, h//2], [w, h], [w//2, h]]
            gis_service.red_zone = np.array(config.RED_ZONE_COORDS, dtype=np.int32)

        result = inference_service.detect(frame, frame_id)
        violations = orn_service.apply(result.objects)

        zone_map = {}
        for obj in result.objects:
            x1, y1, x2, y2 = map(int, obj.bbox)
            if obj.compliance.overall_state == PPE_ABSENT:
                color = (0, 0, 255)
            elif obj.compliance.overall_state == PPE_UNCERTAIN:
                color = (0, 215, 255)
            else:
                color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            helmet_state = obj.compliance.helmet_state or PPE_UNCERTAIN
            vest_state = obj.compliance.vest_state or PPE_UNCERTAIN
            label = f"ID {obj.track_id} H:{helmet_state} V:{vest_state}"
            if obj.occluded:
                label += " OCCLUDED"
            cv2.putText(frame, label, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            center_point = (int((x1+x2)/2), int(y2))
            zone_name = gis_service.get_zone_name(center_point)
            if zone_name:
                zone_map[obj.track_id] = zone_name
                cv2.putText(frame, zone_name.upper(), (center_point[0], center_point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
            else:
                zone_map[obj.track_id] = "General"

        for violation in violations:
            zone_name = zone_map.get(violation.worker_id, "General")
            severity = "high" if zone_name == "Red Zone" else "medium"
            alert = AlertEvent(
                severity=severity,
                message=f"Worker {violation.worker_id} missing {violation.violation}",
                zone_id=zone_name,
                worker_id=violation.worker_id,
                violation=violation.violation,
            )
            asyncio.run_coroutine_threadsafe(stream_service.broadcast_alert(alert), loop)

        if gis_service.red_zone is not None:
            cv2.polylines(frame, [gis_service.red_zone.reshape((-1, 1, 2))], True, (0, 0, 255), 2)

        state = input_source.get_state()
        mode_text = "LIVE CCTV" if state.mode == "live" else "VIDEO ANALYSIS"
        cv2.putText(frame, mode_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        with video_state.lock:
            video_state.processed_frame = frame
            video_state.processed_frame_id = frame_id
            
        # Release GIL to allow other threads (like upload) to run
        time.sleep(0.01)

@app.on_event("startup")
def startup_event():
    global loop
    loop = asyncio.get_event_loop()
    video_state.running = True
    capture_thread = threading.Thread(target=capture_video, daemon=True)
    inference_thread = threading.Thread(target=process_frames, daemon=True)
    capture_thread.start()
    inference_thread.start()

@app.on_event("shutdown")
def shutdown_event():
    video_state.running = False

@app.get("/")
async def root():
    return {"message": "BuildSight Backend Running"}

@app.get("/health")
async def health_check():
    """Backend health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "video_loaded": input_source.has_video(),
        "uploads_dir_exists": config.UPLOADS_DIR.exists(),
        "capture_running": video_state.running
    }

@app.post("/set_input_mode")
async def set_input_mode(mode: str):
    """
    Switch input mode between 'video' and 'live'
    """
    if mode not in ["video", "live"]:
        return {"error": "Invalid mode. Use 'video' or 'live'"}
    try:
        input_source.set_mode(mode)
        logger.info(f"Input mode switch requested: {mode}")
    except ValueError as exc:
        return {"error": str(exc)}
    return {"status": "success", "mode": mode}

@app.get("/config")
async def get_config():
    state = input_source.get_state()
    video_filename = None
    video_url = None
    if state.source:
        source_path = Path(state.source)
        if source_path.is_file() and config.UPLOADS_DIR in source_path.parents:
            video_filename = source_path.name
            video_url = f"/uploads/{source_path.name}"
    return {
        "host": config.HOST,
        "port": config.PORT,
        "video_source": state.source,
        "active_mode": state.mode,
        "video_loaded": input_source.has_video(),
        "video_filename": video_filename,
        "video_url": video_url
    }

@app.post("/upload_video")
async def upload_video(request: Request, file: UploadFile = File(...)):
    logger.info(
        "Upload start: client=%s content-length=%s filename=%s",
        request.client.host if request.client else "unknown",
        request.headers.get("content-length"),
        file.filename,
    )
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")
    if file.content_type and not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Invalid file type")

    suffix = ""
    if "." in file.filename:
        suffix = "." + file.filename.split(".")[-1]
    unique_name = f"upload_{int(time.time())}_{uuid.uuid4().hex}{suffix}"
    dest_path = config.UPLOADS_DIR / unique_name

    logger.info(f"Starting upload for {unique_name}")
    total_bytes = 0
    try:
        with dest_path.open("wb") as out_file:
            while True:
                content = await file.read(1024 * 1024) # 1MB chunks
                if not content:
                    break
                total_bytes += len(content)
                out_file.write(content)
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail="File upload failed")
    finally:
        await file.close()
    logger.info(f"Upload completed: {unique_name} bytes={total_bytes}")

    input_source.set_video_source(str(dest_path))
    reset_video_buffers()

    return {"status": "success", "filename": unique_name, "video_url": f"/uploads/{unique_name}"}

@app.post("/upload_init")
async def upload_init(request: Request):
    data = await request.json()
    filename = data.get("filename") or ""
    content_type = data.get("content_type") or data.get("contentType") or ""
    total_size = int(data.get("size") or 0)

    # Validate file size
    MAX_FILE_SIZE = 2 * 1024 * 1024 * 1024  # 2GB
    if total_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {total_size / 1024 / 1024 / 1024:.2f}GB. Maximum is 2GB."
        )

    if total_size == 0:
        raise HTTPException(
            status_code=400,
            detail="File is empty"
        )

    # Validate content type
    valid_types = ['video/mp4', 'video/avi', 'video/mov', 'video/mkv', 'video/webm', 'video/x-matroska']
    if content_type and not any(content_type.startswith(vt) for vt in valid_types):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid content type: {content_type}. Must be a video file."
        )

    # Validate uploads directory exists
    if not config.UPLOADS_DIR.exists():
        logger.error(f"Uploads directory does not exist: {config.UPLOADS_DIR}")
        raise HTTPException(
            status_code=500,
            detail="Server misconfigured: uploads directory not found"
        )

    safe_name = Path(filename).name if filename else ""
    suffix = Path(safe_name).suffix if safe_name else ""
    if not suffix:
        guessed_ext = mimetypes.guess_extension(content_type.split(";")[0].strip())
        suffix = guessed_ext or ".mp4"

    unique_name = f"upload_{int(time.time())}_{uuid.uuid4().hex}{suffix}"
    final_path = config.UPLOADS_DIR / unique_name
    temp_path = config.UPLOADS_DIR / f".{unique_name}.part"

    with upload_lock:
        upload_sessions[unique_name] = {
            "temp_path": temp_path,
            "final_path": final_path,
            "size": total_size,
            "received": 0,
        }

    temp_path.write_bytes(b"")
    logger.info(
        "Chunked upload init: name=%s size=%s type=%s",
        unique_name,
        total_size,
        content_type,
    )
    return {"upload_id": unique_name}

@app.put("/upload_chunk/{upload_id}")
async def upload_chunk(upload_id: str, request: Request, offset: int = 0):
    with upload_lock:
        session = upload_sessions.get(upload_id)
    if not session:
        raise HTTPException(status_code=404, detail="Upload session not found")

    temp_path = session["temp_path"]
    bytes_written = 0
    try:
        with temp_path.open("r+b") as out_file:
            out_file.seek(offset)
            async for chunk in request.stream():
                if not chunk:
                    continue
                out_file.write(chunk)
                bytes_written += len(chunk)
    except FileNotFoundError:
        with temp_path.open("wb") as out_file:
            out_file.seek(offset)
            async for chunk in request.stream():
                if not chunk:
                    continue
                out_file.write(chunk)
                bytes_written += len(chunk)
    except Exception as exc:
        logger.error("Chunk upload failed: upload_id=%s offset=%s error=%s",
                     upload_id, offset, exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Chunk upload failed at offset {offset}: {str(exc)}"
        )

    with upload_lock:
        session = upload_sessions.get(upload_id)
        if session:
            session["received"] = max(session["received"], offset + bytes_written)

    return {"status": "ok", "written": bytes_written}

@app.post("/upload_complete/{upload_id}")
async def upload_complete(upload_id: str):
    with upload_lock:
        session = upload_sessions.get(upload_id)
    if not session:
        raise HTTPException(status_code=404, detail="Upload session not found")

    temp_path = session["temp_path"]
    final_path = session["final_path"]
    expected_size = session["size"]

    if expected_size:
        actual_size = temp_path.stat().st_size if temp_path.exists() else 0
        if actual_size != expected_size:
            logger.error("Upload incomplete: upload_id=%s expected=%s actual=%s",
                         upload_id, expected_size, actual_size)
            raise HTTPException(
                status_code=400,
                detail=f"Upload incomplete: received {actual_size / 1024 / 1024:.2f}MB of {expected_size / 1024 / 1024:.2f}MB"
            )

    temp_path.replace(final_path)

    with upload_lock:
        upload_sessions.pop(upload_id, None)

    input_source.set_video_source(str(final_path))
    reset_video_buffers()
    logger.info("Chunked upload complete: %s", final_path)
    return {"status": "success", "filename": final_path.name, "video_url": f"/uploads/{final_path.name}"}

@app.put("/upload_video_raw")
async def upload_video_raw(request: Request):
    content_type = request.headers.get("content-type", "")
    filename = request.headers.get("x-filename") or request.headers.get("x-file-name") or ""
    safe_name = Path(filename).name if filename else ""
    suffix = Path(safe_name).suffix if safe_name else ""
    if not suffix:
        guessed_ext = mimetypes.guess_extension(content_type.split(";")[0].strip())
        suffix = guessed_ext or ".mp4"
    unique_name = f"upload_{int(time.time())}_{uuid.uuid4().hex}{suffix}"
    dest_path = config.UPLOADS_DIR / unique_name

    logger.info(
        "Raw upload start: client=%s content-length=%s type=%s filename=%s",
        request.client.host if request.client else "unknown",
        request.headers.get("content-length"),
        content_type,
        safe_name or unique_name,
    )
    total_bytes = 0
    try:
        with dest_path.open("wb") as out_file:
            async for chunk in request.stream():
                if not chunk:
                    continue
                total_bytes += len(chunk)
                out_file.write(chunk)
    except Exception as exc:
        logger.error("Raw upload failed: %s", exc)
        raise HTTPException(status_code=500, detail="File upload failed")

    logger.info("Raw upload completed: %s bytes=%s", unique_name, total_bytes)
    input_source.set_video_source(str(dest_path))
    reset_video_buffers()
    return {"status": "success", "filename": unique_name, "video_url": f"/uploads/{unique_name}"}

def generate_frames():
    while True:
        with video_state.lock:
            if (
                video_state.processed_frame is not None
                and video_state.processed_frame_id == video_state.raw_frame_id
            ):
                frame = video_state.processed_frame
            else:
                frame = video_state.raw_frame
        if frame is None:
            time.sleep(0.05)
            continue
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.04)

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.websocket("/ws/alerts")
async def stream_websocket(websocket: WebSocket):
    await stream_service.connect(websocket)
    try:
        while True:
            # Keep alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        stream_service.disconnect(websocket)
