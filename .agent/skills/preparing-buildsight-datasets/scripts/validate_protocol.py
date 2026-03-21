import os
import json
import argparse

def validate_dataset(scenario_dir, scenario_type):
    """
    Validates labels against the BuildSight Protocol.
    Specifically checks S4 for >= 5 workers.
    """
    labels_dir = os.path.join(scenario_dir, "labels")
    if not os.path.exists(labels_dir):
        # Check YOLO labels folder name in protocol
        labels_dir = os.path.join(scenario_dir, "labels_yolo")
    
    if not os.path.exists(labels_dir):
        print(f"❌ Labels directory not found in {scenario_dir}")
        return
    
    report = {
        "scenario": scenario_type,
        "total_frames": 0,
        "class_counts": {0: 0, 1: 0, 2: 0},
        "s4_violations": 0,
        "empty_files": 0,
        "frames": []
    }
    
    label_files = [f for f in os.listdir(labels_dir) if f.endswith(".txt")]
    report["total_frames"] = len(label_files)
    
    for lf in label_files:
        path = os.path.join(labels_dir, lf)
        worker_count = 0
        
        if os.path.getsize(path) == 0:
            report["empty_files"] += 1
            continue
            
        with open(path, "r") as f:
            lines = f.readlines()
            for line in lines:
                parts = line.split()
                if not parts: continue
                cls_id = int(parts[0])
                if cls_id in report["class_counts"]:
                    report["class_counts"][cls_id] += 1
                if cls_id == 0:
                    worker_count += 1
        
        # S4 Validation: MANDATORY ≥ 5 workers
        is_s4_valid = True
        if scenario_type == "S4" and worker_count < 5:
            report["s4_violations"] += 1
            is_s4_valid = False
            
        report["frames"].append({
            "file": lf,
            "worker_count": worker_count,
            "valid": is_s4_valid
        })

    # Save Metadata JSON
    meta_path = os.path.join(scenario_dir, f"metadata_{scenario_type}.json")
    with open(meta_path, "w") as f:
        json.dump(report, f, indent=2)
        
    print(f"📊 Validation Report for {scenario_type}:")
    print(f"   - Total Frames: {report['total_frames']}")
    print(f"   - Total Workers (Class 0): {report['class_counts'][0]}")
    if scenario_type == "S4":
        print(f"   - S4 Violations (<5 workers): {report['s4_violations']}")
    print(f"   - Empty Files: {report['empty_files']}")
    print(f"✅ Metadata saved to: {meta_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BuildSight Protocol Validator")
    parser.add_argument("--dir", required=True, help="Scenario directory (e.g. Dataset/S4_crowded)")
    parser.add_argument("--type", required=True, choices=["S1", "S2", "S3", "S4"], help="Scenario type")
    args = parser.parse_args()
    validate_dataset(args.dir, args.type)
