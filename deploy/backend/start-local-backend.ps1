$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot | Split-Path -Parent
$envFile = Join-Path $repoRoot "deploy\backend\backend.env"
$leanPython = Join-Path $repoRoot ".venv-inference\Scripts\python.exe"
$defaultPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
$python = if (Test-Path $leanPython) { $leanPython } else { $defaultPython }

if (Test-Path $envFile) {
  Get-Content $envFile | ForEach-Object {
    if ($_ -match '^\s*#' -or $_ -match '^\s*$') { return }
    $pair = $_ -split '=', 2
    if ($pair.Length -eq 2) {
      [System.Environment]::SetEnvironmentVariable($pair[0], $pair[1])
    }
  }
}

$env:PYTHONUNBUFFERED = "1"

& $python (Join-Path $repoRoot "dashboard\backend\server.py")
