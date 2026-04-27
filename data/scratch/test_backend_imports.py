#!/usr/bin/env python3
"""
Smoke test for BuildSight Backend Imports
=========================================
Checks if all critical backend modules can be imported without NameError or ImportError.
"""

import sys
import os
from pathlib import Path

# Fix path to include project root and backend
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
BACKEND_DIR = PROJECT_ROOT / "dashboard" / "backend"
sys.path.insert(0, str(BACKEND_DIR))

def test_imports():
    print("[START] Starting BuildSight Backend Smoke Test...")
    errors = []

    # 1. Test Intelligence Engine
    print("Testing buildsight_intelligence...", end=" ")
    try:
        import buildsight_intelligence
        from buildsight_intelligence import SpatialWorker, SpatialEvent, WorkerTrack
        print("[OK]")
    except Exception as e:
        print(f"[FAILED]: {e}")
        errors.append(f"buildsight_intelligence: {e}")

    # 2. Test GeoAI Utils
    print("Testing geoai_vlm_util...", end=" ")
    try:
        import geoai_vlm_util
        print("[OK]")
    except Exception as e:
        print(f"[FAILED]: {e}")
        errors.append(f"geoai_vlm_util: {e}")

    print("Testing geoai_sam_util...", end=" ")
    try:
        import geoai_sam_util
        print("[OK]")
    except Exception as e:
        print(f"[FAILED]: {e}")
        errors.append(f"geoai_sam_util: {e}")

    # 3. Test GeoAI Router
    print("Testing geoai.router...", end=" ")
    try:
        # Move to backend dir context for relative imports
        os.chdir(BACKEND_DIR)
        from geoai.router import router as geoai_router
        print("[OK]")
    except Exception as e:
        print(f"[FAILED]: {e}")
        errors.append(f"geoai.router: {e}")

    # 4. Test Main Server
    print("Testing server.py...", end=" ")
    try:
        import server
        print("[OK]")
    except Exception as e:
        print(f"[FAILED]: {e}")
        errors.append(f"server.py: {e}")

    # 5. Test WebSocket Server
    print("Testing geoai_ws_server.py...", end=" ")
    try:
        import geoai_ws_server
        print("[OK]")
    except Exception as e:
        print(f"[FAILED]: {e}")
        errors.append(f"geoai_ws_server.py: {e}")

    print("\n" + "="*40)
    if not errors:
        print("[SUCCESS] All imports verified! Backend is stable.")
        sys.exit(0)
    else:
        print(f"[ERROR] Found {len(errors)} import errors:")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)

if __name__ == "__main__":
    test_imports()
