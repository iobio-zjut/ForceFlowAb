from typing import Optional

from pydantic import BaseModel, Field, validator


TYPE_ALIASES = {
    "antibody": "antibody",
    "ab": "antibody",
    "ordinary": "antibody",
    "nanobody": "nanobody",
    "nano": "nanobody",
    "vhh": "nanobody",
}

REGION_ALIASES = {
    "all": "all",
    "allcdr": "all",
    "all_cdr": "all",
    "multiple_cdrs": "all",
    "h1": "h1",
    "h_cdr1": "h1",
    "hcdr1": "h1",
    "cdrh1": "h1",
    "h2": "h2",
    "h_cdr2": "h2",
    "hcdr2": "h2",
    "cdrh2": "h2",
    "h3": "h3",
    "h_cdr3": "h3",
    "hcdr3": "h3",
    "cdrh3": "h3",
    "l1": "l1",
    "l_cdr1": "l1",
    "lcdr1": "l1",
    "cdrl1": "l1",
    "l2": "l2",
    "l_cdr2": "l2",
    "lcdr2": "l2",
    "cdrl2": "l2",
    "l3": "l3",
    "l_cdr3": "l3",
    "lcdr3": "l3",
    "cdrl3": "l3",
}


def _normalize_key(value):
    return str(value).strip().lower().replace("-", "_")


def _split_region_values(value):
    if isinstance(value, (list, tuple)):
        raw_parts = []
        for item in value:
            raw_parts.extend(str(item).replace(";", ",").split(","))
    else:
        raw_parts = str(value).replace(";", ",").split(",")
    return [_normalize_key(part) for part in raw_parts if str(part).strip()]


class DesignJobRequest(BaseModel):
    type: str = "antibody"
    region: str = "all"
    pdb: str
    heavy: Optional[str] = None
    light: Optional[str] = None
    batch_size: int = Field(8, ge=1)
    num_samples: int = Field(1, ge=1, le=8)
    energy: bool = False
    energy_start: int = Field(69, ge=0)
    energy_end: int = Field(79, ge=0)
    energy_warmup: int = Field(0, ge=0)
    device: str = "cuda"
    tag: Optional[str] = None
    no_renumber: bool = False

    @validator("type", pre=True)
    def normalize_type(cls, value):
        key = _normalize_key(value)
        if key not in TYPE_ALIASES:
            raise ValueError("type must be antibody or nanobody")
        return TYPE_ALIASES[key]

    @validator("region", pre=True)
    def normalize_region(cls, value):
        regions = []
        for key in _split_region_values(value):
            if key not in REGION_ALIASES:
                raise ValueError("region must be all or a comma-separated list of H1, H2, H3, L1, L2, L3")
            normalized = REGION_ALIASES[key]
            if normalized not in regions:
                regions.append(normalized)
        if not regions:
            raise ValueError("region must not be empty")
        if "all" in regions and len(regions) > 1:
            raise ValueError("region=all cannot be combined with other CDR regions")
        return ",".join(regions)

    @validator("region")
    def validate_region_for_type(cls, region, values):
        regions = set(region.split(","))
        if values.get("type") == "nanobody" and regions.intersection({"l1", "l2", "l3"}):
            raise ValueError("nanobody design does not support L1/L2/L3")
        return region


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
