"""Training job control (admin only)."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool

from ..schemas.openai import TrainingJobRequest
from ..services.jobs import JobError
from ..services.security import Admin

router = APIRouter(prefix="/training", dependencies=[Admin], tags=["training"])


def _call(fn, *a):
    try:
        return fn(*a)
    except KeyError as e:
        raise HTTPException(404, "job not found") from e
    except JobError as e:
        raise HTTPException(400, str(e)) from e


@router.get("/configs")
async def configs(request: Request):
    return {"configs": request.app.state.jobs.available_configs()}


@router.get("/jobs")
async def list_jobs(request: Request):
    return {"jobs": await run_in_threadpool(request.app.state.jobs.list)}


@router.post("/jobs", status_code=201)
async def create_job(req: TrainingJobRequest, request: Request):
    return await run_in_threadpool(_call, request.app.state.jobs.create, req.name, req.config, req.overrides)


@router.get("/jobs/{job_id}")
async def get_job(job_id: str, request: Request):
    return await run_in_threadpool(_call, request.app.state.jobs.get, job_id)


@router.get("/jobs/{job_id}/metrics")
async def job_metrics(job_id: str, request: Request):
    return {"metrics": await run_in_threadpool(_call, request.app.state.jobs.metrics, job_id)}


@router.post("/jobs/{job_id}/stop")
async def stop_job(job_id: str, request: Request):
    return await run_in_threadpool(_call, request.app.state.jobs.stop, job_id)


@router.post("/jobs/{job_id}/resume")
async def resume_job(job_id: str, request: Request):
    return await run_in_threadpool(_call, request.app.state.jobs.resume, job_id)
