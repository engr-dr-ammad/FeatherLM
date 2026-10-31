#!/usr/bin/env bash
# Deploy FeatherLM (CPU inference) on an Ubuntu VM with Docker.
#   ./scripts/deploy.sh <model_dir>      e.g. ./scripts/deploy.sh runs/featherlm-nano/final
# Requires: Docker Engine + compose plugin, a .env with FEATHERLM_API_KEY set.
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL_DIR=${1:?usage: deploy.sh <model_dir>}
NAME=${MODEL_NAME:-$(basename "$(dirname "$MODEL_DIR")")}
[ -f .env ] || { echo "create .env from .env.example and set FEATHERLM_API_KEY"; exit 1; }
grep -q '^FEATHERLM_API_KEY=.\+' .env || { echo "FEATHERLM_API_KEY must be set in .env"; exit 1; }
for f in config.json model.safetensors tokenizer.json; do [ -f "$MODEL_DIR/$f" ] || { echo "missing $MODEL_DIR/$f"; exit 1; }; done
docker build -t featherlm:cpu .
docker volume create featherlm-models >/dev/null
# Copy the checkpoint into the models volume (only safetensors/JSON files).
docker run --rm -v featherlm-models:/data/models -v "$(realpath "$MODEL_DIR")":/src:ro --user 0 featherlm:cpu \
  sh -c "mkdir -p /data/models/$NAME && cp /src/config.json /src/model.safetensors /src/tokenizer.json /data/models/$NAME/ && \
         (cp /src/metadata.json /data/models/$NAME/ 2>/dev/null || true) && chown -R 10001:10001 /data/models"
grep -q '^FEATHERLM_DEFAULT_MODEL=' .env && sed -i "s#^FEATHERLM_DEFAULT_MODEL=.*#FEATHERLM_DEFAULT_MODEL=models/$NAME#" .env \
  || echo "FEATHERLM_DEFAULT_MODEL=models/$NAME" >> .env
docker compose -f docker-compose.prod.yml --env-file .env up -d
echo "Waiting for health check..."
for _ in $(seq 1 30); do curl -fsS http://127.0.0.1:8000/health && echo && exit 0; sleep 2; done
echo "service did not become healthy; see: docker compose -f docker-compose.prod.yml logs"; exit 1
