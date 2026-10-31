# Deploying FeatherLM

*Developed by Ammad Aslam with AI-assisted tooling.*

Three methods, all CPU-only. No GPU server is needed.

## Method A: local installation

| OS | Setup |
|---|---|
| Linux / macOS | `./scripts/setup.sh` |
| Windows (PowerShell) | `powershell -ExecutionPolicy Bypass -File scripts\setup.ps1` |

The script creates `.venv`, installs the CPU-only PyTorch wheel and the requirements,
builds FeatherLM Studio (if Node.js 20+ is installed) and copies `.env.example` to `.env`.

```bash
source .venv/bin/activate                         # Windows: .\.venv\Scripts\Activate.ps1
# 1. prepare a dataset
python -m featherlm.data.prepare --download tinyshakespeare --out data/shakespeare --vocab-size 8192
# 2. train
python train.py --config configs/nano_cpu.yaml
# 3. launch the API (serves the Studio too)
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
# 4. open http://127.0.0.1:8000 (Studio) or http://127.0.0.1:8000/docs (OpenAPI)
```

Frontend development with hot reload: `cd frontend && npm run dev` (proxies API calls to
port 8000), then open http://localhost:5173.

## Method B: Docker

```bash
docker compose up -d --build                 # API + Studio on http://127.0.0.1:8000
docker compose logs -f
docker compose down
```

* The image uses the CPU-only PyTorch wheel, runs as a non-root user and has a health check.
* Volumes: `featherlm-models` (`/data/models`), `featherlm-runs`, `featherlm-datasets`, `featherlm-state` (SQLite).
* Restart policy `unless-stopped`; memory limit from `FEATHERLM_MEM_LIMIT` (default 4g).
* Copy a trained checkpoint into the models volume:

```bash
docker compose cp runs/featherlm-nano/final featherlm:/data/models/featherlm-nano
```

**Production configuration** (`docker-compose.prod.yml`): inference only.

* `FEATHERLM_ENABLE_TRAINING_API=false`: training/admin routes return 404.
* `FEATHERLM_API_KEY` is mandatory (compose refuses to start without it).
* Read-only root filesystem, models volume mounted read-only, log rotation, CPU/memory limits.
* Published on 127.0.0.1 only, for a TLS-terminating reverse proxy.

> **Verification status:** the Docker files were syntax-checked with `docker compose config`,
> but the image has not been built in the development environment (no Docker daemon and no
> registry access there). Build and run it once on your machine before relying on it.

## Method C: Ubuntu cloud VM

Tested target: Ubuntu 22.04/24.04, 2 vCPU, 4 GB RAM is enough for Nano inference.

```bash
# 1. Environment and Docker
sudo apt-get update && sudo apt-get install -y ca-certificates curl git nginx
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER && newgrp docker

# 2. Code and configuration
git clone <your-repo-url> featherlm && cd featherlm
cp .env.example .env
sed -i "s/^FEATHERLM_API_KEY=.*/FEATHERLM_API_KEY=$(openssl rand -hex 24)/" .env
sed -i "s/^FEATHERLM_ENABLE_TRAINING_API=.*/FEATHERLM_ENABLE_TRAINING_API=false/" .env

# 3. Upload a model from your workstation (safetensors + JSON only)
#    (on your machine)  scp -r runs/featherlm-nano/final user@server:~/featherlm-nano
./scripts/deploy.sh ~/featherlm-nano        # builds the image, fills the volume, starts prod compose

# 4. Reverse proxy + HTTPS
sudo cp deploy/nginx.conf /etc/nginx/sites-available/featherlm   # edit server_name first
sudo ln -s /etc/nginx/sites-available/featherlm /etc/nginx/sites-enabled/
sudo apt-get install -y certbot python3-certbot-nginx
sudo certbot --nginx -d featherlm.example.com
sudo nginx -t && sudo systemctl reload nginx

# 5. Firewall: only SSH and HTTPS
sudo ufw allow OpenSSH && sudo ufw allow 'Nginx Full' && sudo ufw enable
```

Calling the API:

```bash
curl https://featherlm.example.com/v1/completions \
  -H "Authorization: Bearer $FEATHERLM_API_KEY" -H "Content-Type: application/json" \
  -d '{"prompt": "ROMEO:", "max_tokens": 60, "temperature": 0.8}'
```

The OpenAI Python client works for completions by pointing `base_url` at
`https://featherlm.example.com/v1`.

**Persistent storage:** models live in the `featherlm-models` Docker volume
(`/var/lib/docker/volumes/`). Snapshot the VM disk or `docker run --rm -v featherlm-models:/m -v $PWD:/b alpine tar czf /b/models.tgz /m`.

**Logs:** `docker compose -f docker-compose.prod.yml logs --since 1h`. Logs rotate at
10 MB × 5 files. nginx logs are in `/var/log/nginx/`.

**Monitoring:** `GET /health` (no auth; use for uptime checks) and `GET /metrics`
(Prometheus text format, needs the API key): request counts, generated tokens, latency
sums, time to first token, loaded models, process RSS. `docker stats` shows container CPU
and memory.

**Without Docker:** `deploy/featherlm.service` runs uvicorn under systemd with a memory
limit and filesystem protections.

## Security checklist

* Set `FEATHERLM_API_KEY` and `FEATHERLM_ADMIN_KEY` on any network-facing server.
* Keep `FEATHERLM_ENABLE_TRAINING_API=false` on public servers. Without an admin key, admin
  routes accept only direct loopback connections and reject any request carrying proxy headers.
* Model loading uses safetensors only; checkpoint import accepts only
  `config.json`, `model.safetensors`, `tokenizer.json`, `metadata.json` (no pickle), validates
  them by loading, and refuses path traversal. Optimizer files (`optimizer.pt`) are only read
  by the trainer when resuming local runs, with `weights_only=True`.
* Request bodies are limited (`FEATHERLM_MAX_REQUEST_BYTES`), generation length is capped
  (`FEATHERLM_MAX_NEW_TOKENS`), at most `FEATHERLM_MAX_LOADED_MODELS` models stay in memory,
  and a busy model returns 503 after `FEATHERLM_BUSY_TIMEOUT_S`.
* Training jobs: one at a time by default, step cap, memory budget cap, data paths confined
  to the data directory, output confined to the runs directory.
* On shutdown the server sends SIGTERM to training jobs so they checkpoint before exiting.
