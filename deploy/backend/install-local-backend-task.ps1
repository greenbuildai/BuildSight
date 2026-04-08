$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot | Split-Path -Parent
$taskName = "BuildSightBackend"
$launcher = Join-Path $repoRoot "deploy\backend\start-local-backend.ps1"
$logDir = Join-Path $repoRoot "logs"
$powershellExe = Join-Path $env:WINDIR "System32\WindowsPowerShell\v1.0\powershell.exe"

New-Item -ItemType Directory -Path $logDir -Force | Out-Null

$taskCommand = "`"$powershellExe`" -NoProfile -ExecutionPolicy Bypass -File `"$launcher`""

cmd.exe /c "schtasks.exe /Query /TN $taskName" *> $null
if ($LASTEXITCODE -eq 0) {
  schtasks.exe /Delete /TN $taskName /F | Out-Null
}
schtasks.exe /Create /TN $taskName /SC ONLOGON /RL LIMITED /TR $taskCommand /F | Out-Null
schtasks.exe /Run /TN $taskName | Out-Null

Write-Host "Scheduled task '$taskName' created and started."
