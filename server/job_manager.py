import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .schemas import DesignJobRequest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JOBS_ROOT = PROJECT_ROOT / "jobs"
JOBS_ROOT = Path(os.environ.get("FLOWDESIGN_JOBS_ROOT", DEFAULT_JOBS_ROOT)).resolve()
DP_SCRIPT = PROJECT_ROOT / "DP.sh"
SAFE_UPLOAD_SUFFIXES = {".pdb", ".ent"}


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _job_id() -> str:
    return "job_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _job_dir(job_id: str) -> Path:
    return JOBS_ROOT / job_id


def _status_path(job_id: str) -> Path:
    return _job_dir(job_id) / "status.json"


def _write_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w") as handle:
        json.dump(data, handle, indent=2)
    os.replace(tmp_path, path)


def _safe_upload_name(filename: str) -> str:
    name = Path(filename or "input.pdb").name
    if not name:
        name = "input.pdb"
    suffix = Path(name).suffix.lower()
    if suffix not in SAFE_UPLOAD_SUFFIXES:
        name = f"{name}.pdb"
    return name


def submit_job(request: DesignJobRequest, uploaded_file: Optional[Tuple[str, bytes]] = None) -> Dict:
    if not DP_SCRIPT.is_file():
        raise FileNotFoundError(f"DP.sh not found: {DP_SCRIPT}")

    job_id = _job_id()
    job_dir = _job_dir(job_id)
    results_dir = job_dir / "results"
    job_dir.mkdir(parents=True, exist_ok=False)
    results_dir.mkdir(parents=True, exist_ok=True)

    request_json = job_dir / "request.json"
    status_json = job_dir / "status.json"
    run_log = job_dir / "run.log"
    config_yml = job_dir / "config.yml"

    uploaded_filename = None
    if uploaded_file is not None:
        original_name, content = uploaded_file
        if not content:
            raise ValueError("Uploaded PDB file is empty")
        upload_dir = job_dir / "input"
        upload_dir.mkdir(parents=True, exist_ok=True)
        uploaded_filename = _safe_upload_name(original_name)
        pdb_path = upload_dir / uploaded_filename
        with open(pdb_path, "wb") as handle:
            handle.write(content)
        request.pdb = str(pdb_path)
    elif not Path(request.pdb).is_file():
        raise FileNotFoundError(f"Input PDB not found: {request.pdb}")

    request_data = request.dict()
    request_data.update(
        {
            "job_id": job_id,
            "job_dir": str(job_dir),
            "results_dir": str(results_dir),
            "uploaded_filename": uploaded_filename,
            "created_at": _now(),
        }
    )
    _write_json(request_json, request_data)
    _write_json(
        status_json,
        {
            "job_id": job_id,
            "status": "queued",
            "request_json": str(request_json),
            "config_yml": str(config_yml),
            "run_log": str(run_log),
            "results_dir": str(results_dir),
            "created_at": _now(),
            "updated_at": _now(),
        },
    )

    cmd = [
        str(DP_SCRIPT),
        "--type",
        request.type,
        "--region",
        request.region,
        "--pdb",
        request.pdb,
        "--batch-size",
        str(request.batch_size),
        "--num-samples",
        str(request.num_samples),
        "--job-id",
        job_id,
        "--out",
        str(job_dir),
        "--energy",
        "true" if request.energy else "false",
        "--energy-start",
        str(request.energy_start),
        "--energy-end",
        str(request.energy_end),
        "--energy-warmup",
        str(request.energy_warmup),
        "--device",
        request.device,
    ]
    if request.heavy:
        cmd.extend(["--heavy", request.heavy])
    if request.light is not None:
        cmd.extend(["--light", request.light])
    if request.tag:
        cmd.extend(["--tag", request.tag])
    if request.no_renumber:
        cmd.append("--no-renumber")

    subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

    return {"job_id": job_id, "status": "queued", "job_dir": str(job_dir)}


def get_status(job_id: str) -> Dict:
    path = _status_path(job_id)
    if not path.is_file():
        raise FileNotFoundError(f"Job not found: {job_id}")
    with open(path) as handle:
        return json.load(handle)


def get_logs(job_id: str, tail: int = 200) -> Dict:
    job_dir = _job_dir(job_id)
    log_path = job_dir / "run.log"
    if not job_dir.is_dir():
        raise FileNotFoundError(f"Job not found: {job_id}")
    if not log_path.is_file():
        return {"job_id": job_id, "log": ""}

    with open(log_path, errors="replace") as handle:
        lines = handle.readlines()
    if tail and tail > 0:
        lines = lines[-tail:]
    return {"job_id": job_id, "log": "".join(lines)}


def list_results(job_id: str) -> Dict:
    status = get_status(job_id)
    results_dir = Path(status.get("results_dir") or (_job_dir(job_id) / "results"))
    if not results_dir.is_dir():
        return {"job_id": job_id, "results_dir": str(results_dir), "files": []}

    files: List[Dict] = []
    for path in sorted(results_dir.rglob("*")):
        if path.is_file():
            files.append(
                {
                    "path": str(path),
                    "relative_path": str(path.relative_to(results_dir)),
                    "size": path.stat().st_size,
                    "modified_at": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
                }
            )
    return {"job_id": job_id, "results_dir": str(results_dir), "files": files}
