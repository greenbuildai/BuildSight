from ultralytics import YOLO
import cv2

model = YOLO("E:/Company/Green Build AI/Prototypes/BuildSight/yolov8-ppe.pt")
print("Classes:", model.names)

results = model.predict("E:/Company/Green Build AI/Prototypes/BuildSight/live_yolo_preview_newest.jpg", save=True, project="scratch", name="yolov8_test")
print("Length of results:", len(results[0].boxes))

for i, box in enumerate(results[0].boxes):
    print(f"Box {i}: class {model.names[int(box.cls[0])]}, conf: {box.conf[0]:.2f}")
