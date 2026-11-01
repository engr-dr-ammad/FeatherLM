"""OpenAI-style completion and chat endpoints with SSE streaming."""
from __future__ import annotations

import threading
import time
import uuid

from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse

from featherlm.inference.engine import EngineBusy, GenerationStats
from featherlm.inference.sampling import SamplingParams
from featherlm.inference.streaming import sse_event

from ..schemas.openai import ChatCompletionRequest, CompletionRequest
from ..services.models import ModelNotFound
from ..services.security import ApiKey

router = APIRouter(dependencies=[ApiKey], tags=["inference"])


def _params(req, settings) -> SamplingParams:
    max_new = min(req.max_tokens, settings.max_new_tokens)
    return SamplingParams(max_new_tokens=max_new, temperature=req.temperature, top_k=req.top_k, top_p=req.top_p,
                          repetition_penalty=req.repetition_penalty, stop=req.stop or [], seed=req.seed)


def _engine(request: Request, model: str | None):
    try:
        return request.app.state.registry.get(model)
    except ModelNotFound as e:
        raise HTTPException(404, f"model not found: {e.args[0] if e.args else model}") from e


def _usage(st: GenerationStats) -> dict:
    return {"prompt_tokens": st.prompt_tokens, "completion_tokens": st.completion_tokens,
            "total_tokens": st.prompt_tokens + st.completion_tokens}


def _perf(st: GenerationStats) -> dict:
    return {"time_to_first_token_ms": round(st.time_to_first_token_s * 1000, 1),
            "total_time_ms": round(st.total_time_s * 1000, 1), "tokens_per_sec": round(st.tokens_per_sec, 2),
            "rss_mb": round(st.rss_mb, 1)}


def _run(request: Request, engine, ids, params, chat: bool, model_id: str, stream: bool):
    metrics = request.app.state.metrics
    settings = request.app.state.settings
    rid = ("chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex[:24]
    created = int(time.time())
    obj = "chat.completion" if chat else "text_completion"
    stats = GenerationStats()
    cancel = threading.Event()

    if not stream:
        def work():
            return "".join(engine.stream(ids, params, cancel=cancel, stats=stats, lock_timeout=settings.busy_timeout_s))

        return work, stats, rid, created, obj

    def gen():
        t0 = time.perf_counter()
        try:
            first = True
            for piece in engine.stream(ids, params, cancel=cancel, stats=stats, lock_timeout=settings.busy_timeout_s):
                if chat:
                    delta = {"content": piece}
                    if first:
                        delta["role"] = "assistant"
                    choice = {"index": 0, "delta": delta, "finish_reason": None}
                    yield sse_event({"id": rid, "object": "chat.completion.chunk", "created": created,
                                     "model": model_id, "choices": [choice]})
                else:
                    yield sse_event({"id": rid, "object": "text_completion", "created": created, "model": model_id,
                                     "choices": [{"index": 0, "text": piece, "finish_reason": None}]})
                first = False
            final_choice = {"index": 0, "finish_reason": stats.finish_reason}
            final_choice.update({"delta": {}} if chat else {"text": ""})
            yield sse_event({"id": rid, "object": "chat.completion.chunk" if chat else "text_completion",
                             "created": created, "model": model_id, "choices": [final_choice],
                             "usage": _usage(stats), "featherlm": _perf(stats)})
            yield sse_event("[DONE]")
            metrics.record(stats, time.perf_counter() - t0)
        except EngineBusy as e:
            yield sse_event({"error": {"message": str(e), "type": "server_busy"}})
        finally:
            cancel.set()  # client disconnects close this generator; make sure decoding stops

    return gen, stats, rid, created, obj


@router.post("/v1/completions")
async def completions(req: CompletionRequest, request: Request):
    model_id, engine = await run_in_threadpool(_engine, request, req.model)
    ids = engine.encode_prompt(req.prompt)
    params = _params(req, request.app.state.settings)
    fn, stats, rid, created, obj = _run(request, engine, ids, params, False, model_id, req.stream)
    if req.stream:
        return StreamingResponse(fn(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})
    t0 = time.perf_counter()
    try:
        text = await run_in_threadpool(fn)
    except EngineBusy as e:
        raise HTTPException(503, str(e)) from e
    request.app.state.metrics.record(stats, time.perf_counter() - t0)
    return {"id": rid, "object": obj, "created": created, "model": model_id,
            "choices": [{"index": 0, "text": text, "finish_reason": stats.finish_reason, "logprobs": None}],
            "usage": _usage(stats), "featherlm": _perf(stats)}


@router.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, request: Request):
    model_id, engine = await run_in_threadpool(_engine, request, req.model)
    if not engine.supports_chat:
        raise HTTPException(400, f"model '{model_id}' has no chat template (it is a plain language model); "
                                 "use /v1/completions or an instruction-tuned checkpoint")
    ids = engine.encode_chat([m.model_dump() for m in req.messages], reserve=min(req.max_tokens, 128))
    params = _params(req, request.app.state.settings)
    fn, stats, rid, created, obj = _run(request, engine, ids, params, True, model_id, req.stream)
    if req.stream:
        return StreamingResponse(fn(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})
    t0 = time.perf_counter()
    try:
        text = await run_in_threadpool(fn)
    except EngineBusy as e:
        raise HTTPException(503, str(e)) from e
    request.app.state.metrics.record(stats, time.perf_counter() - t0)
    return {"id": rid, "object": obj, "created": created, "model": model_id,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                         "finish_reason": stats.finish_reason}],
            "usage": _usage(stats), "featherlm": _perf(stats)}
