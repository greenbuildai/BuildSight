$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot | Split-Path -Parent
$sourceCmd = Join-Path $repoRoot "deploy\backend\start-local-backend.cmd"
$startupDir = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs\Startup"
$targetCmd = Join-Path $startupDir "BuildSightBackend.cmd"

New-Item -ItemType Directory -Path $startupDir -Force | Out-Null
Copy-Item -Path $sourceCmd -Destination $targetCmd -Force

Write-Host "Startup launcher installed at $targetCmd"
