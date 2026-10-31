"""Model listing, details and import."""
from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool

from ..services.models import MAX_IMPORT_BYTES, ModelNotFound
from ..services.security import Admin, ApiKey

router = APIRouter(tags=["models"])


@router.get("/v1/models", dependencies=[ApiKey])
async def list_models(request: Request):
    reg = request.app.state.registry
    items = await run_in_threadpool(reg.list)
    return {"object": "list", "data": [
        {"id": m["id"], "object": "model", "created": 0, "owned_by": "featherlm", "featherlm": m} for m in items]}


@router.get("/v1/models/{model_id:path}", dependencies=[ApiKey])
async def get_model(model_id: str, request: Request):
    try:
        m = request.app.state.registry.describe(model_id)
    except ModelNotFound as e:
        raise HTTPException(404, "model not found") from e
    return {"id": m["id"], "object": "model", "created": 0, "owned_by": "featherlm", "featherlm": m}


@router.post("/models/import", dependencies=[Admin])
async def import_model(request: Request, name: str = Form(...), file: UploadFile = File(...)):
    data = await file.read(MAX_IMPORT_BYTES + 1)
    try:
        model_id = await run_in_threadpool(request.app.state.registry.import_zip, name, data)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return {"id": model_id, "status": "imported"}


@router.post("/models/unload/{model_id:path}", dependencies=[Admin])
async def unload_model(model_id: str, request: Request):
    return {"unloaded": request.app.state.registry.unload(model_id)}
