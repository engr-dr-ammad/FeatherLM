# FeatherLM setup for Windows (PowerShell, CPU). Developed by Ammad Aslam with AI-assisted tooling.
#   powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
$py = if ($env:PYTHON) { $env:PYTHON } else { "python" }
& $py -c "import sys; assert sys.version_info >= (3, 11), 'Python 3.11+ required'"
if (-not (Test-Path .venv)) { & $py -m venv .venv }
. .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -r requirements-dev.txt
if (Get-Command npm -ErrorAction SilentlyContinue) {
  Push-Location frontend; npm ci --no-audit --no-fund; npm run build; Pop-Location
} else { Write-Host "npm not found: skipping the Studio build (install Node.js 20+ for the UI)" }
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
python -c "import torch, psutil; print('OK: torch', torch.__version__, '| RAM GB', round(psutil.virtual_memory().total/2**30,1))"
Write-Host "Next:"
Write-Host "  python -m featherlm.data.prepare --download tinyshakespeare --out data/shakespeare --vocab-size 8192"
Write-Host "  python train.py --config configs/nano_cpu.yaml"
Write-Host "  python -m uvicorn api.main:app --port 8000   (then open http://127.0.0.1:8000)"
