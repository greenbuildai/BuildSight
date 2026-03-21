import cv2
import numpy as np
import os
import argparse

def apply_gamma_correction(image, gamma=0.7):
    """Enhances visibility in low-light (S3)."""
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def apply_clahe(image):
    """Applies Contrast Limited Adaptive Histogram Equalization."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def enhance_scenario(source_dir, output_dir, scenario):
    os.makedirs(output_dir, exist_ok=True)
    images = [f for f in os.listdir(source_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    print(f"🛠️ Enhancing {len(images)} images for Scenario: {scenario}")
    
    for img_name in images:
        path = os.path.join(source_dir, img_name)
        img = cv2.imread(path)
        
        if scenario == "S3":
            # Low-Light Enhancement
            img = apply_gamma_correction(img, gamma=0.7)
            img = apply_clahe(img)
        elif scenario == "S2":
            # Simple Dehazing Approximation (CLAHE)
            # Full Dark Channel Prior is complex; CLAHE is a robust production baseline
            img = apply_clahe(img)
            
        cv2.imwrite(os.path.join(output_dir, img_name), img)

    print(f"✅ Enhancement complete. Prepared images saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BuildSight Scenario Enhancement")
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scenario", required=True, choices=["S2", "S3"])
    args = parser.parse_args()
    enhance_scenario(args.source, args.output, args.scenario)
