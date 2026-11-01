"""Teach the chatbot: manage example datasets and start chat training runs (admin only)."""
from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, ConfigDict, Field

from ..schemas.openai import ChatMessage
from ..services.jobs import JobError
from ..services.models import ModelNotFound
from ..services.security import Admin
from ..services.teach import MAX_UPLOAD_BYTES, TeachError

router = APIRouter(prefix="/teach", dependencies=[Admin], tags=["teach"])


class DatasetCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(..., min_length=1, max_length=48)


class ExampleIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str | None = Field(None, max_length=2000)
    answer: str | None = Field(None, max_length=2000)
    messages: list[ChatMessage] | None = Field(None, max_length=64)


class TrainRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(..., min_length=1, max_length=48)
    dataset: str = Field(..., min_length=1, max_length=48)
    base_model: str | None = Field(None, max_length=256)
    steps: int = Field(200, ge=10, le=5000)
    mode: str = Field("full", pattern="^(full|lora)$")
    include_starter: bool = True


def _call(fn, *a, **k):
    try:
        return fn(*a, **k)
    except KeyError as e:
        raise HTTPException(404, f"not found: {e.args[0] if e.args else ''}") from e
    except (TeachError, JobError) as e:
        raise HTTPException(400, str(e)) from e


@router.get("/datasets")
async def list_datasets(request: Request):
    return {"datasets": await run_in_threadpool(request.app.state.teach.list)}


@router.post("/datasets", status_code=201)
async def create_dataset(body: DatasetCreate, request: Request):
    return await run_in_threadpool(_call, request.app.state.teach.create, body.name)


@router.delete("/datasets/{name}")
async def delete_dataset(name: str, request: Request):
    await run_in_threadpool(_call, request.app.state.teach.delete, name)
    return {"deleted": name}


@router.get("/datasets/{name}/examples")
async def list_examples(name: str, request: Request, limit: int = 500):
    rows = await run_in_threadpool(_call, request.app.state.teach.examples, name, max(1, min(limit, 5000)))
    return {"examples": rows}


@router.post("/datasets/{name}/examples", status_code=201)
async def add_example(name: str, body: ExampleIn, request: Request):
    if body.messages:
        conv = [m.model_dump() for m in body.messages]
    elif body.question and body.answer:
        conv = [{"role": "user", "content": body.question}, {"role": "assistant", "content": body.answer}]
    else:
        raise HTTPException(422, "give question and answer, or messages")
    teach = request.app.state.teach
    await run_in_threadpool(_call, teach.ensure, name)
    added = await run_in_threadpool(_call, teach.add, name, [conv])
    return added[0]


@router.delete("/datasets/{name}/examples/{ex_id}")
async def delete_example(name: str, ex_id: str, request: Request):
    await run_in_threadpool(_call, request.app.state.teach.remove, name, ex_id)
    return {"deleted": ex_id}


@router.post("/datasets/{name}/upload", status_code=201)
async def upload(name: str, request: Request, file: UploadFile = File(...)):
    from ..services.teach import parse_upload

    data = await file.read(MAX_UPLOAD_BYTES + 1)
    teach = request.app.state.teach
    convs = await run_in_threadpool(_call, parse_upload, file.filename or "", data)
    await run_in_threadpool(_call, teach.ensure, name)
    added = await run_in_threadpool(_call, teach.add, name, convs)
    return {"added": len(added)}


@router.post("/train", status_code=201)
async def train(body: TrainRequest, request: Request):
    reg, teach, jobs = request.app.state.registry, request.app.state.teach, request.app.state.jobs
    base_id = body.base_model or default_chat_base(reg)
    if not base_id:
        raise HTTPException(400, "no base model found; train FeatherLM Nano first")
    try:
        base_dir = reg.resolve(base_id)
    except ModelNotFound as e:
        raise HTTPException(404, f"base model not found: {base_id}") from e
    built = await run_in_threadpool(_call, teach.build_run, body.name, body.dataset, base_dir, body.steps, body.mode,
                                    body.include_starter)
    job = await run_in_threadpool(_call, jobs.create_chat, body.name, built["config"], f"teach:{body.dataset}")
    return {"job": job, "model_id": f"runs/{body.name}/best", "base_model": base_id,
            "train_examples": built["train_examples"], "taught_examples": built["taught_examples"]}


def default_chat_base(reg) -> str | None:
    """Prefer an existing chat model (starter chatbot), else any base model."""
    models = reg.list()
    chat = [m for m in models if m["capabilities"]["chat"] and m["quantization"] == "none"]
    for pref in ("runs/featherlm-nano-chat/best", "models/featherlm-nano-chat"):
        if any(m["id"] == pref for m in chat):
            return pref
    if chat:
        return chat[0]["id"]
    plain = [m for m in models if m["quantization"] == "none" and m["id"].endswith("/best")]
    return plain[0]["id"] if plain else (models[0]["id"] if models else None)


@router.get("/base-models")
async def base_models(request: Request):
    reg = request.app.state.registry
    models = [m for m in await run_in_threadpool(reg.list) if m["quantization"] == "none"]
    return {"default": default_chat_base(reg), "models": [{"id": m["id"], "chat": m["capabilities"]["chat"],
                                                            "parameters": m["parameters"]} for m in models]}
