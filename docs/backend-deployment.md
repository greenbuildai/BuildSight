# BuildSight Backend Deployment

This repo now supports two deployment modes using the same FastAPI backend:

1. Lean local backend for development and demos
2. Production-style backend on a SASTRA GPU host

## Current backend footprint

Measured from the current repo:

- `.venv`: about 5.31 GB
- `weights`: about 1.00 GB
- local fallback YOLO weights: about 23.5 MB

That places the current backend at roughly 6.3 GB. A lean inference-only environment should stay in the 5 to 7 GB range including model weights.

## Lean local backend

### Goal

Keep only inference runtime dependencies and required model weights.

### Files

- `deploy/backend/requirements-inference.txt`
- `deploy/backend/requirements-ai.txt`
- `deploy/backend/backend.env.example`
- `deploy/backend/setup-local-backend.ps1`
- `deploy/backend/start-local-backend.ps1`

### Setup

1. Copy `deploy/backend/backend.env.example` to `deploy/backend/backend.env`
2. Adjust paths only if your models or runtime directory live elsewhere
3. Run:

```powershell
cd E:\Company\Green Build AI\Prototypes\BuildSight
powershell -ExecutionPolicy Bypass -File .\deploy\backend\setup-local-backend.ps1
```

Optional Gemini package:

```powershell
powershell -ExecutionPolicy Bypass -File .\deploy\backend\setup-local-backend.ps1 -InstallGemini
```

### Run

```powershell
cd E:\Company\Green Build AI\Prototypes\BuildSight
powershell -ExecutionPolicy Bypass -File .\deploy\backend\start-local-backend.ps1
```

### Why this is leaner

- separate `.venv-inference` from the larger training environment
- `google-generativeai` is optional
- video uploads and annotated outputs are routed into `runtime/`
- returned annotated MP4 files are deleted after download

## SASTRA GPU backend

### Target layout

Use a dedicated server path such as:

```text
/opt/buildsight
```

Expected contents:

- repo checkout under `/opt/buildsight`
- model weights under `/opt/buildsight/models`
- runtime directory under `/opt/buildsight/runtime`
- inference environment under `/opt/buildsight/.venv-inference`

### Files

- `deploy/sastra/backend.env.example`
- `deploy/sastra/setup-sastra-backend.sh`
- `deploy/sastra/start-backend.sh`
- `deploy/sastra/buildsight-backend.service`
- `deploy/sastra/nginx-buildsight-backend.conf`

### Setup

1. Copy the repo to the SASTRA GPU node
2. Copy model weights into `/opt/buildsight/models`
3. Copy `deploy/sastra/backend.env.example` to `deploy/sastra/backend.env`
4. Edit the environment file with real paths and secrets
5. Run:

```bash
cd /opt/buildsight
bash deploy/sastra/setup-sastra-backend.sh /opt/buildsight
```

### Run directly

```bash
cd /opt/buildsight
bash deploy/sastra/start-backend.sh
```

### Run as a service

1. Copy `deploy/sastra/buildsight-backend.service` to `/etc/systemd/system/buildsight-backend.service`
2. Reload systemd
3. Enable and start the service

```bash
sudo systemctl daemon-reload
sudo systemctl enable buildsight-backend
sudo systemctl start buildsight-backend
sudo systemctl status buildsight-backend
```

### Reverse proxy

Use `deploy/sastra/nginx-buildsight-backend.conf` as a base nginx site config.

## Backend configuration

The backend now supports environment-based configuration:

- `BUILDSIGHT_HOST`
- `BUILDSIGHT_PORT`
- `BUILDSIGHT_LOG_LEVEL`
- `BUILDSIGHT_RUNTIME_DIR`
- `BUILDSIGHT_MODEL_DIR`
- `BUILDSIGHT_MODEL_V11`
- `BUILDSIGHT_MODEL_V26`
- `BUILDSIGHT_LOCAL_MODEL`

## Recommendation

- Use the lean local backend for current development and demo work
- Use the SASTRA GPU backend for multi-user or persistent deployment
- Keep one backend codebase and switch environments via env vars only
