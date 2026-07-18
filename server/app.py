from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import ValidationError

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
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/jobs/upload", response_model=DesignJobSubmitResponse)
async def create_uploaded_job(
    request: Request,
    type: str = Query("antibody"),
    region: str = Query("all"),
    heavy: Optional[str] = Query(None),
    light: Optional[str] = Query(None),
    num_samples: int = Query(1, ge=1, le=8),
    tag: Optional[str] = Query(None),
    no_renumber: bool = Query(False),
):
    try:
        filename = request.headers.get("X-Filename", "input.pdb")
        body = await request.body()
        job_request = DesignJobRequest(
            type=type,
            region=region,
            pdb="uploaded.pdb",
            heavy=heavy,
            light=light,
            num_samples=num_samples,
            tag=tag,
            no_renumber=no_renumber,
        )
        return submit_job(job_request, uploaded_file=(filename, body))
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
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
