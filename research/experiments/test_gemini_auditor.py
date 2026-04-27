import cv2
import numpy as np
from gemini_auditor import GeminiAuditor
from pipeline_config import GEMINI_API_KEY, GEMINI_MODEL

def test_auditor():
    # Create fake image
    img = np.zeros((500, 500, 3), dtype=np.uint8)
    
    # Fake boxes [x1, y1, x2, y2]
    # W0 is a valid-looking worker box
    boxes = [
        [50, 50, 150, 400],    # worker
        [200, 50, 300, 150],   # helmet
        [400, 400, 450, 450],  # worker (fake tree patch)
    ]
    labels = ["worker", "helmet", "worker"]
    confidences = [0.8, 0.9, 0.4]
    
    # Draw simple shapes to test the AI
    # Actual human shape in W0
    cv2.rectangle(img, (60, 60), (140, 390), (255, 200, 150), -1) 
    # Helmet in H0
    cv2.circle(img, (250, 100), 40, (0, 0, 255), -1)
    # Tree patch in W1 (green box)
    cv2.rectangle(img, (400, 400), (450, 450), (0, 255, 0), -1)

    print("Initializing GeminiAuditor...")
    auditor = GeminiAuditor(api_key=GEMINI_API_KEY, model_name=GEMINI_MODEL)
    
    print("Testing audit_image()...")
    approvals = auditor.audit_image(img, boxes, labels, confidences)
    
    print("Original Labels:", labels)
    print("Approval Array:", approvals)
    
    filtered_boxes, filtered_confs, filtered_labels = auditor.filter_detections(img, boxes, confidences, labels)
    
    print("Filtered Labels:", filtered_labels)
    assert len(filtered_boxes) == sum(approvals), "Filtering logic failed"
    print("Success! Gemini Auditor output looks correct.")

if __name__ == "__main__":
    test_auditor()
