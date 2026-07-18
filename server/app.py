from fastapi import FastAPI, HTTPException, Query

from .job_manager import get_logs, get_status, list_results, submit_job
from .schemas import DesignJobRequest, DesignJobStatus, DesignJobSubmitResponse


app = FastAPI(title="FlowDesign Job API")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/jobs", response_model=DesignJobSubmitResponse)
def create_job(request: DesignJobRequest):
    try:
        return submit_job(request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/jobs/{job_id}", response_model=DesignJobStatus)
def read_job(job_id: str):
    try:
        return get_status(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.get("/jobs/{job_id}/logs")
def read_job_logs(job_id: str, tail: int = Query(200, ge=0, le=5000)):
    try:
        return get_logs(job_id, tail=tail)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.get("/jobs/{job_id}/results")
def read_job_results(job_id: str):
    try:
        return list_results(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
