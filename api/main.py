"""FeatherLM API server (FastAPI).

    uvicorn api.main:app --host 127.0.0.1 --port 8000

OpenAPI docs: http://127.0.0.1:8000/docs
Developed by Ammad Aslam with AI-assisted tooling.
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from featherlm import __version__

from .routes import inference, models, system, training
from .services.jobs import JobManager
from .services.metrics import Metrics
from .services.models import ModelRegistry
from .services.settings import Settings

log = logging.getLogger("featherlm")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        log.info("FeatherLM API %s starting (models=%s runs=%s training_api=%s auth=%s)", __version__,
                 settings.models_dir, settings.runs_dir, settings.enable_training_api, bool(settings.api_key))
        yield
        log.info("shutting down: stopping training jobs gracefully")
        app.state.jobs.shutdown()

    app = FastAPI(
        title="FeatherLM API",
        version=__version__,
        description="Inference and training API for FeatherLM, a hardware-efficient language model. "
                    "Developed by Ammad Aslam with AI-assisted tooling.",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.registry = ModelRegistry(settings)
    app.state.jobs = JobManager(settings)
    app.state.metrics = Metrics()
    app.state.started = time.time()
    saved = app.state.jobs.kv_get("settings", {}) or {}
    for k, v in saved.items():
        if hasattr(settings, k):
            setattr(settings, k, v)

    @app.middleware("http")
    async def limits_and_metrics(request: Request, call_next):
        cl = request.headers.get("content-length")
        limit = settings.max_request_bytes
        if request.url.path == "/models/import":
            limit = 2 * 1024**3
        if cl is not None:
            try:
                if int(cl) > limit:
                    return JSONResponse({"detail": "request body too large"}, status_code=413)
            except ValueError:
                return JSONResponse({"detail": "invalid content-length"}, status_code=400)
        elif request.method in ("POST", "PUT", "PATCH") and request.headers.get("transfer-encoding") == "chunked":
            return JSONResponse({"detail": "chunked request bodies are not accepted"}, status_code=411)
        try:
            response = await call_next(request)
        except Exception:  # pragma: no cover - last-resort error handler
            log.exception("unhandled error on %s", request.url.path)
            response = JSONResponse({"detail": "internal server error"}, status_code=500)
        route = request.scope.get("route")
        app.state.metrics.http(getattr(route, "path", "other"), response.status_code)
        return response

    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["*"],
                       allow_headers=["*"], allow_credentials=False)

    app.include_router(system.router)
    app.include_router(models.router)
    app.include_router(inference.router)
    app.include_router(training.router)

    dist = Path(__file__).resolve().parents[1] / "frontend" / "dist"
    if settings.serve_frontend and (dist / "index.html").exists():
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

        @app.get("/favicon.svg", include_in_schema=False)
        async def favicon():
            return FileResponse(dist / "favicon.svg", media_type="image/svg+xml")

        @app.get("/", include_in_schema=False)
        async def index():
            return FileResponse(dist / "index.html")

        @app.get("/studio/{rest:path}", include_in_schema=False)
        async def studio(rest: str):
            return FileResponse(dist / "index.html")
    return app


app = create_app()
