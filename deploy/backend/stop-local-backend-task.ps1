$ErrorActionPreference = "Stop"
$taskName = "BuildSightBackend"

schtasks.exe /End /TN $taskName | Out-Null
Write-Host "Scheduled task '$taskName' stopped."
