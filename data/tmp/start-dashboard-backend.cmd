@echo off
cd /d "E:\Company\Green Build AI\Prototypes\BuildSight"
"E:\Company\Green Build AI\Prototypes\BuildSight\.venv\Scripts\python.exe" "E:\Company\Green Build AI\Prototypes\BuildSight\dashboard\backend\server.py" >> "E:\Company\Green Build AI\Prototypes\BuildSight\logs\dashboard-backend.combined.log" 2>&1
