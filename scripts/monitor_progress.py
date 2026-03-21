import os
import time
import sys

lbl_dir = r"e:\Company\Green Build AI\Prototypes\BuildSight\Dataset\Final_Annotated_Dataset\labels"
total = 1373

print(f"\n[{time.strftime('%H:%M:%S')}] SYNCHRONIZED CLAUDE PROGRESS MONITOR")
print("=" * 60)

count = 0
last_count = -1

while count < total:
    if os.path.exists(lbl_dir):
        # Count label files across all splits (train/val/test) for the full 1373 target
        count = 0
        for root, dirs, files in os.walk(lbl_dir):
            count += len([f for f in files if f.endswith(".txt")])
    
    if count != last_count:
        pct = (count / total) * 100
        bars = int(pct / 2)
        bar_str = "█" * bars + "-" * (50 - bars)
        
        # Overwrite the current line with \r
        sys.stdout.write(f"\rProgress: [{bar_str}] {count}/{total} ({pct:.1f}%) ")
        sys.stdout.flush()
        last_count = count
        
    time.sleep(1)

print("\n\nANNOTATION COMPLETE! 1373/1373 labels found on disk.")
print("=" * 60)
time.sleep(60) # Keep window open for 1 minute before closing
