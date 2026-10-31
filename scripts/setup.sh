#!/usr/bin/env bash
# FeatherLM setup for Linux and macOS (CPU). Developed by Ammad Aslam with AI-assisted tooling.
#   ./scripts/setup.sh            # create .venv, install deps, build the Studio frontend
#   ./scripts/setup.sh --no-frontend
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
$PY -c 'import sys; assert sys.version_info >= (3, 11), "Python 3.11+ required"'
[ -d .venv ] || $PY -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip
if [ "$(uname -s)" = "Linux" ]; then
  # CPU-only wheel: no CUDA libraries (~200 MB instead of several GB).
  pip install --index-url https://download.pytorch.org/whl/cpu torch
else
  pip install torch            # macOS wheels are CPU/MPS builds
fi
pip install -r requirements-dev.txt
if [ "${1:-}" != "--no-frontend" ]; then
  if command -v npm >/dev/null; then
    (cd frontend && npm ci --no-audit --no-fund && npm run build)
  else
    echo "npm not found: skipping FeatherLM Studio build (API still works; install Node.js 20+ for the UI)"
  fi
fi
[ -f .env ] || cp .env.example .env
python - <<'PY'
import platform, psutil, torch
vm = psutil.virtual_memory()
print(f"OK: Python {platform.python_version()}, torch {torch.__version__}, {psutil.cpu_count()} CPU threads, "
      f"{vm.total/2**30:.1f} GB RAM ({vm.available/2**30:.1f} GB available), CUDA={torch.cuda.is_available()}")
PY
echo "Next: source .venv/bin/activate && ./scripts/train.sh"
