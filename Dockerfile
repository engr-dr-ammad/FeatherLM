# FeatherLM CPU image: API + FeatherLM Studio. Developed by Ammad Aslam with AI-assisted tooling.
# Build:  docker build -t featherlm:cpu .
FROM node:22-slim AS frontend
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    FEATHERLM_ROOT=/app FEATHERLM_MODELS_DIR=/data/models FEATHERLM_RUNS_DIR=/data/runs \
    FEATHERLM_DATA_DIR=/data/datasets FEATHERLM_RESULTS_DIR=/app/results FEATHERLM_DB=/data/state/featherlm.db
WORKDIR /app
# CPU-only PyTorch wheel (no CUDA libraries) keeps the image small.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch \
 && useradd --create-home --uid 10001 feather
COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY featherlm/ featherlm/
COPY api/ api/
COPY configs/ configs/
COPY results/ results/
COPY train.py finetune.py inference.py evaluate.py pyproject.toml README.md LICENSE ./
COPY --from=frontend /app/frontend/dist frontend/dist
RUN mkdir -p /data/models /data/runs /data/datasets /data/state && chown -R feather:feather /data
USER feather
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=4).status==200 else 1)"
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
