from typing import Literal, Optional

from pydantic import BaseModel, Field


class DesignJobRequest(BaseModel):
    type: Literal["nanobody", "antibody"] = "antibody"
    region: Literal["h3", "all"] = "all"
    pdb: str
    heavy: Optional[str] = None
    light: Optional[str] = None
    batch_size: int = Field(8, ge=1)
    num_samples: int = Field(32, ge=1)
    energy: bool = False
    energy_start: int = Field(98, ge=0)
    energy_end: int = Field(99, ge=0)
    energy_warmup: int = Field(0, ge=0)
    device: str = "cuda"
    tag: Optional[str] = None
    no_renumber: bool = False


class DesignJobSubmitResponse(BaseModel):
    job_id: str
    status: str
    job_dir: str


class DesignJobStatus(BaseModel):
    job_id: str
    status: str
    pid: Optional[int] = None
    created_at: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    updated_at: Optional[str] = None
    return_code: Optional[int] = None
    request_json: Optional[str] = None
    config_yml: Optional[str] = None
    run_log: Optional[str] = None
    results_dir: Optional[str] = None
