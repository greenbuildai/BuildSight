#!/usr/bin/env python3
"""
BuildSight Backend Orchestrator
=================================
Launches all backend services in parallel:

  1. FastAPI Detection Server  (port 8000)
  2. GeoAI WebSocket Broadcast (port 8765)

Usage:
  python start_backend.py           # Launch both services
  python start_backend.py --api     # Launch only FastAPI
  python start_backend.py --ws      # Launch only WebSocket

Author: BuildSight / Green Build AI
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

BACKEND_DIR = Path(__file__).parent
PROJECT_ROOT = BACKEND_DIR.parent.parent

# Service definitions
SERVICES = {
    "api": {
        "name": "FastAPI Detection Server",
        "port": 8000,
        "cmd": [sys.executable, str(BACKEND_DIR / "server.py")],
        "health_url": "http://localhost:8000/api/health",
    },
    "ws": {
        "name": "GeoAI WebSocket Server",
        "port": 8765,
        "cmd": [sys.executable, str(BACKEND_DIR / "geoai_ws_server.py")],
        "health_url": None,  # WebSocket --- no HTTP health check
    },
}

# ANSI colors
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


def print_banner():
    """Print the startup banner."""
    print(f"""
{CYAN}{BOLD}
  ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  ---             BuildSight Backend Orchestrator               ---
  ---            Green Build AI | IGBC AP | BOCW Act            ---
  ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------{RESET}
""")


def check_port(port: int) -> bool:
    """Check if a port is already in use."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


def launch_service(key: str) -> subprocess.Popen:
    """Launch a backend service and return the process handle."""
    svc = SERVICES[key]
    name = svc["name"]
    port = svc["port"]
    
    if check_port(port):
        print(f"  {YELLOW}--- Port {port} in use --- {name} may already be running{RESET}")
    
    print(f"  {GREEN}--- Launching {name} on port {port}...{RESET}")
    
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    
    proc = subprocess.Popen(
        svc["cmd"],
        cwd=str(BACKEND_DIR),
        env=env,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    
    return proc


def main():
    parser = argparse.ArgumentParser(description="BuildSight Backend Orchestrator")
    parser.add_argument("--api", action="store_true", help="Launch only FastAPI server")
    parser.add_argument("--ws", action="store_true", help="Launch only WebSocket server")
    args = parser.parse_args()
    
    print_banner()
    
    # Determine which services to launch
    if args.api:
        to_launch = ["api"]
    elif args.ws:
        to_launch = ["ws"]
    else:
        to_launch = ["api", "ws"]
    
    procs = {}
    
    # Register signal handler for clean shutdown
    def shutdown(signum=None, frame=None):
        print(f"\n  {YELLOW}Shutting down services...{RESET}")
        for key, proc in procs.items():
            name = SERVICES[key]["name"]
            if proc.poll() is None:
                print(f"  {RED}--- Stopping {name} (PID {proc.pid}){RESET}")
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
        print(f"  {GREEN}--- All services stopped{RESET}")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    
    # Launch services
    for key in to_launch:
        procs[key] = launch_service(key)
        time.sleep(1)  # Stagger launches
    
    print(f"""
  {CYAN}{'---' * 55}
  Services running:
""")
    for key in to_launch:
        svc = SERVICES[key]
        print(f"    {GREEN}---{RESET} {svc['name']:30s} --- port {svc['port']}")
    
    print(f"""
  {CYAN}Press Ctrl+C to stop all services
  {'---' * 55}{RESET}
""")
    
    # Wait for any process to exit
    try:
        while True:
            for key, proc in list(procs.items()):
                ret = proc.poll()
                if ret is not None:
                    name = SERVICES[key]["name"]
                    if ret != 0:
                        print(f"\n  {RED}--- {name} exited with code {ret}{RESET}")
                    else:
                        print(f"\n  {YELLOW}--- {name} exited normally{RESET}")
                    # Don't auto-restart --- let the operator decide
                    del procs[key]
            
            if not procs:
                print(f"\n  {RED}All services have stopped{RESET}")
                break
            
            time.sleep(1)
    except KeyboardInterrupt:
        shutdown()


if __name__ == "__main__":
    main()
