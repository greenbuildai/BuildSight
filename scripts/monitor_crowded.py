"""
Live monitor for Crowded_Condition annotation progress.
Windows cp1252 safe — no unicode block characters.
"""
import os, time, sys, glob
from pathlib import Path

lbl_dir  = Path(r"e:\Company\Green Build AI\Prototypes\BuildSight\Dataset\Final_Annotated_Dataset\labels")
src_dir  = Path(r"e:\Company\Green Build AI\Prototypes\BuildSight\Dataset\Indian Dataset\Crowded_Condition")
log_file = Path(r"e:\Company\Green Build AI\Prototypes\BuildSight\logs\crowded_full_annotation.log")

# All crowded image stems
crowded_stems = set(f.stem.lower() for f in src_dir.glob("*.jpg"))
crowded_stems |= set(f.stem.lower() for f in src_dir.glob("*.JPG"))
crowded_stems |= set(f.stem.lower() for f in src_dir.glob("*.png"))
total = len(crowded_stems)

print(f"\n[{time.strftime('%H:%M:%S')}] TONI - CROWDED CONDITION LIVE MONITOR")
print(f"Total images: {total}")
print("=" * 60)

last_count = -1
start_time = None
initial_count = None

def get_count():
    c = 0
    for split in ["train", "val", "test"]:
        for f in (lbl_dir / split).glob("*.txt"):
            if f.stem.lower() in crowded_stems:
                c += 1
    return c

while True:
    count = get_count()

    if initial_count is None:
        initial_count = count
        start_time = time.time()

    if count != last_count:
        pct = (count / total) * 100
        bar_len = 40
        filled = int(bar_len * count / total)
        bar = "#" * filled + "-" * (bar_len - filled)

        elapsed = time.time() - start_time
        new_done = count - initial_count
        if new_done > 0 and elapsed > 0:
            speed = elapsed / new_done
            remaining = total - count
            eta_s = int(remaining * speed)
            eta_str = f"{eta_s//60}m {eta_s%60}s"
            spd_str = f"{speed:.2f}s/img  ({1/speed:.1f} img/s)"
        else:
            eta_str = "--"
            spd_str = "RESUME (cached)"

        print(f"[{time.strftime('%H:%M:%S')}] [{bar}] {count}/{total} ({pct:.1f}%) | {spd_str} | ETA: {eta_str}")
        last_count = count

    if count >= total:
        print("\n" + "=" * 60)
        print("CROWDED ANNOTATION COMPLETE!")
        print("=" * 60)
        break

    # Check if annotation process finished (log has COMPLETE marker)
    if log_file.exists():
        try:
            text = log_file.read_text(encoding="utf-8", errors="ignore")
            if "ANNOTATION COMPLETE" in text:
                final = get_count()
                print(f"\n[{time.strftime('%H:%M:%S')}] DONE - {final}/{total} labels on disk")
                print("=" * 60)
                break
        except:
            pass

    time.sleep(3)
