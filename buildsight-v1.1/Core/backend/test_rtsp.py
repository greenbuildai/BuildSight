import cv2
import time
import os

# usage: set env var RTSP_URL or edit below
rtsp_url = os.environ.get("RTSP_URL", "rtsp://8015164110:joseva%238765@192.168.43.100:554/stream1")

print(f"Testing connection to: {rtsp_url}")
print("Attempting to open video capture...")

cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("❌ FAILED: Could not open video connection.")
    print("Possible causes:")
    print("1. The IP address is unreachable (are you on the same Wi-Fi?).")
    print("2. The username/password is incorrect.")
    print("3. The RTSP port (554) is blocked.")
    print("4. The URL path (/stream1) is incorrect for this camera model.")
else:
    print("✅ SUCCESS: Connected to camera.")
    ret, frame = cap.read()
    if ret:
        print(f"Successfully read a frame (Resolution: {frame.shape[1]}x{frame.shape[0]})")
    else:
        print("⚠️ CONNECTED, but failed to read frame. (Codec issue?)")

cap.release()
