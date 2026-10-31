"""Health, metrics, settings and benchmark results."""
from __future__ import annotations

import json
import logging
import os
import time

import psutil
import torch
from fastapi import APIRouter, Request
from fastapi.responses import PlainTextResponse

from featherlm import __version__

from ..schemas.openai import SettingsUpdate
from ..services.security import Admin, ApiKey

router = APIRouter(tags=["system"])


@router.get("/health")
async def health(request: Request):
    reg = request.app.state.registry
    return {"status": "ok", "version": __version__, "uptime_s": round(time.time() - request.app.state.started, 1),
            "models_available": len(reg.discover()), "models_loaded": reg.loaded,
            "training_api": request.app.state.settings.enable_training_api}


@router.get("/metrics", response_class=PlainTextResponse, dependencies=[ApiKey])
async def metrics(request: Request):
    return PlainTextResponse(request.app.state.metrics.prometheus(request.app), media_type="text/plain; version=0.0.4")


@router.get("/system", dependencies=[ApiKey])
async def system(request: Request):
    vm = psutil.virtual_memory()
    proc = psutil.Process()
    return {"cpu_count": psutil.cpu_count(), "cpu_percent": psutil.cpu_percent(interval=None),
            "ram_total_mb": round(vm.total / 2**20), "ram_available_mb": round(vm.available / 2**20),
            "process_rss_mb": round(proc.memory_info().rss / 2**20, 1), "torch_threads": torch.get_num_threads(),
            "torch": torch.__version__, "cuda": torch.cuda.is_available()}


def _public_settings(s) -> dict:
    return {"threads": s.threads, "default_model": s.default_model, "max_new_tokens": s.max_new_tokens,
            "training_memory_budget_mb": s.training_memory_budget_mb, "log_level": s.log_level,
            "models_dir": str(s.models_dir), "runs_dir": str(s.runs_dir), "training_api": s.enable_training_api,
            "auth_enabled": bool(s.api_key), "admin_key_set": bool(s.admin_key),
            "max_loaded_models": s.max_loaded_models, "max_training_jobs": s.max_training_jobs}


@router.get("/settings", dependencies=[ApiKey])
async def get_settings(request: Request):
    return _public_settings(request.app.state.settings)


@router.put("/settings", dependencies=[Admin])
async def update_settings(upd: SettingsUpdate, request: Request):
    s = request.app.state.settings
    changes = upd.model_dump(exclude_none=True)
    for k, v in changes.items():
        setattr(s, k, v)
    if "threads" in changes and changes["threads"]:
        torch.set_num_threads(changes["threads"])
    if "log_level" in changes:
        logging.getLogger("featherlm").setLevel(changes["log_level"])
    request.app.state.jobs.kv_set("settings", {**(request.app.state.jobs.kv_get("settings", {}) or {}), **changes})
    return _public_settings(s)


@router.get("/benchmarks", dependencies=[ApiKey])
async def benchmarks(request: Request):
    d = request.app.state.settings.results_dir
    out = {}
    if d.exists():
        for f in sorted(d.glob("*.json")):
            try:
                out[f.stem] = json.loads(f.read_text())
            except (OSError, json.JSONDecodeError):
                continue
    return {"results": out, "pid": os.getpid()}
