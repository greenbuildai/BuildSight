param(
  [string]$VenvPath = ".venv-inference",
  [switch]$InstallGemini
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot | Split-Path -Parent
$venvAbs = Join-Path $repoRoot $VenvPath
$python = Join-Path $venvAbs "Scripts\python.exe"

if (!(Test-Path $python)) {
  py -3.10 -m venv $venvAbs
}

& $python -m pip install --upgrade pip wheel
& $python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
& $python -m pip install -r (Join-Path $repoRoot "deploy\backend\requirements-inference.txt")

if ($InstallGemini) {
  & $python -m pip install -r (Join-Path $repoRoot "deploy\backend\requirements-ai.txt")
}

Write-Host ""
Write-Host "Lean local backend environment is ready at $venvAbs"
Write-Host "Estimated inference footprint target: 5-7 GB including models."
