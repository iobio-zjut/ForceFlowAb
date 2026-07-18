#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ENV_DIR="${FLOWDESIGN_ENV_DIR:-/home/data/Public_tools/anaconda3/envs/ForceFlowAb}"
if [[ -x "$ENV_DIR/bin/python" ]]; then
  export PATH="$ENV_DIR/bin:$PATH"
  PYTHON="$ENV_DIR/bin/python"
else
  PYTHON="${PYTHON:-python}"
fi

ab_type="antibody"
region="all"
pdb_path=""
heavy_chain=""
light_chain=""
batch_size="8"
num_samples="32"
job_id=""
jobs_root="./jobs"
job_dir=""
energy="false"
energy_start="98"
energy_end="99"
energy_warmup="0"
device="cuda"
extra_tag=""
no_renumber="false"
dry_run="false"

usage() {
  cat <<'EOF'
Usage:
  ./DP.sh --type nanobody|antibody --region h3|all --pdb /path/input.pdb [options]

Options:
  --heavy ID             Heavy chain ID. Omit or use auto for auto-infer.
  --light ID|none        Light chain ID. Use none for nanobody/no light chain.
  --batch-size N         Batch size passed to design_for_pdb. Default: 8
  --num-samples N        sampling.num_samples in generated config. Default: 32
  --job-id ID            Job ID. Default: job_YYYYmmdd_HHMMSS_PID
  --jobs-root DIR        Root directory for jobs. Default: ./jobs
  --out DIR              Explicit job directory. Default: JOBS_ROOT/JOB_ID
  --energy true|false    Enable energy guidance. Default: false
  --energy-start N       Energy guidance start_step. Default: 98
  --energy-end N         Energy guidance end_step. Default: 99
  --energy-warmup N      Energy guidance warmup_steps. Default: 0
  --device DEVICE        Model device. Default: cuda
  --tag TAG              Optional tag passed to design_for_pdb.
  --no-renumber          Pass --no_renumber to design_for_pdb.
  --dry-run              Create job files and print command, but do not run design.
  -h, --help             Show this help.
EOF
}

normalize_bool() {
  case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in
    true|1|yes|y|on) echo "true" ;;
    false|0|no|n|off) echo "false" ;;
    *) echo "Invalid boolean: $1" >&2; exit 2 ;;
  esac
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --type) ab_type="$2"; shift 2 ;;
    --region) region="$2"; shift 2 ;;
    --pdb|--pdb-path) pdb_path="$2"; shift 2 ;;
    --heavy) heavy_chain="$2"; shift 2 ;;
    --light) light_chain="$2"; shift 2 ;;
    --batch-size|-b) batch_size="$2"; shift 2 ;;
    --num-samples) num_samples="$2"; shift 2 ;;
    --job-id) job_id="$2"; shift 2 ;;
    --jobs-root) jobs_root="$2"; shift 2 ;;
    --out|-o) job_dir="$2"; shift 2 ;;
    --energy) energy="$(normalize_bool "$2")"; shift 2 ;;
    --energy-start) energy_start="$2"; shift 2 ;;
    --energy-end) energy_end="$2"; shift 2 ;;
    --energy-warmup) energy_warmup="$2"; shift 2 ;;
    --device|-d) device="$2"; shift 2 ;;
    --tag|-t) extra_tag="$2"; shift 2 ;;
    --no-renumber) no_renumber="true"; shift ;;
    --dry-run) dry_run="true"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

ab_type="$(printf '%s' "$ab_type" | tr '[:upper:]' '[:lower:]')"
region="$(printf '%s' "$region" | tr '[:upper:]' '[:lower:]')"

case "$ab_type" in
  nanobody|nano|vhh) ab_type="nanobody" ;;
  antibody|ordinary|ab) ab_type="antibody" ;;
  *) echo "--type must be nanobody or antibody" >&2; exit 2 ;;
esac

case "$region" in
  h3|cdrh3) region="h3" ;;
  all|allcdr|all_cdr|multiple_cdrs) region="all" ;;
  *) echo "--region must be h3 or all" >&2; exit 2 ;;
esac

if [[ -z "$pdb_path" ]]; then
  echo "--pdb is required" >&2
  usage >&2
  exit 2
fi
if [[ ! -f "$pdb_path" ]]; then
  echo "Input PDB not found: $pdb_path" >&2
  exit 1
fi

if [[ "$region" == "h3" ]]; then
  base_config="./configs/test/H3.yml"
elif [[ "$ab_type" == "nanobody" ]]; then
  base_config="./configs/test/nanobody.yml"
else
  base_config="./configs/test/multicdrs.yml"
fi
if [[ ! -f "$base_config" ]]; then
  echo "Config not found: $base_config" >&2
  exit 1
fi

if [[ -z "$job_id" ]]; then
  job_id="job_$(date +%Y%m%d_%H%M%S)_$$"
fi
if [[ -z "$job_dir" ]]; then
  job_dir="${jobs_root%/}/$job_id"
fi

mkdir -p "$job_dir/results"
request_json="$job_dir/request.json"
run_config="$job_dir/config.yml"
status_json="$job_dir/status.json"
run_log="$job_dir/run.log"
results_dir="$job_dir/results"

exec > >(tee -a "$run_log") 2>&1

write_status() {
  local status="$1"
  local return_code="${2:-}"
  "$PYTHON" - "$status_json" "$job_id" "$status" "$return_code" <<'PY'
import json
import os
import sys
from datetime import datetime

path, job_id, status, return_code = sys.argv[1:]
now = datetime.now().isoformat(timespec="seconds")
data = {}
if os.path.exists(path):
    with open(path) as f:
        data = json.load(f)
data.setdefault("job_id", job_id)
data["status"] = status
data["updated_at"] = now
if status == "running":
    data.setdefault("started_at", now)
if status in {"succeeded", "failed"}:
    data["finished_at"] = now
    data["return_code"] = int(return_code or 0)
with open(path, "w") as f:
    json.dump(data, f, indent=2)
PY
}

finish_failed() {
  local rc=$?
  write_status "failed" "$rc"
  exit "$rc"
}
trap finish_failed ERR

"$PYTHON" - "$request_json" "$job_id" "$ab_type" "$region" "$pdb_path" "$heavy_chain" "$light_chain" "$batch_size" "$num_samples" "$job_dir" "$energy" "$energy_start" "$energy_end" "$energy_warmup" "$device" "$extra_tag" "$no_renumber" "$dry_run" <<'PY'
import json
import sys
from datetime import datetime

(
    path, job_id, ab_type, region, pdb_path, heavy, light, batch_size,
    num_samples, job_dir, energy, energy_start, energy_end, energy_warmup,
    device, tag, no_renumber, dry_run
) = sys.argv[1:]
data = {
    "job_id": job_id,
    "type": ab_type,
    "region": region,
    "pdb": pdb_path,
    "heavy": heavy or None,
    "light": None if light.lower() in {"", "none", "null", "-"} else light,
    "batch_size": int(batch_size),
    "num_samples": int(num_samples),
    "job_dir": job_dir,
    "results_dir": f"{job_dir.rstrip('/')}/results",
    "energy": energy == "true",
    "energy_start": int(energy_start),
    "energy_end": int(energy_end),
    "energy_warmup": int(energy_warmup),
    "device": device,
    "tag": tag or None,
    "no_renumber": no_renumber == "true",
    "dry_run": dry_run == "true",
    "created_at": datetime.now().isoformat(timespec="seconds"),
}
with open(path, "w") as f:
    json.dump(data, f, indent=2)
PY

"$PYTHON" - "$base_config" "$run_config" "$num_samples" "$energy" "$energy_start" "$energy_end" "$energy_warmup" <<'PY'
import sys
import yaml

base_config, run_config, num_samples, energy, start_step, end_step, warmup_steps = sys.argv[1:]
with open(base_config) as f:
    cfg = yaml.safe_load(f)

cfg.setdefault("sampling", {})
cfg["sampling"]["num_samples"] = int(num_samples)
cfg["sampling"].pop("single", None)
cfg["sampling"].pop("multi", None)

eg = cfg["sampling"].setdefault("energy_guidance", {})
eg["enabled"] = energy == "true"
eg["start_step"] = int(start_step)
eg["end_step"] = int(end_step)
eg["warmup_steps"] = int(warmup_steps)

with open(run_config, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY

"$PYTHON" - "$status_json" "$job_id" "$$" "$request_json" "$run_config" "$run_log" "$results_dir" <<'PY'
import json
import sys
from datetime import datetime

path, job_id, pid, request_json, config_yml, run_log, results_dir = sys.argv[1:]
now = datetime.now().isoformat(timespec="seconds")
data = {
    "job_id": job_id,
    "status": "queued",
    "pid": int(pid),
    "request_json": request_json,
    "config_yml": config_yml,
    "run_log": run_log,
    "results_dir": results_dir,
    "created_at": now,
    "updated_at": now,
}
with open(path, "w") as f:
    json.dump(data, f, indent=2)
PY

cmd=(
  "$PYTHON" -m diffab.tools.runner.design_for_pdb
  "$pdb_path"
  -c "$run_config"
  -o "$results_dir"
  -b "$batch_size"
  -d "$device"
)

if [[ -n "$extra_tag" ]]; then
  cmd+=(--tag "$extra_tag")
fi

case "$(printf '%s' "$heavy_chain" | tr '[:upper:]' '[:lower:]')" in
  ""|auto|none|null|-) ;;
  *) cmd+=(--heavy "$heavy_chain") ;;
esac

case "$(printf '%s' "$light_chain" | tr '[:upper:]' '[:lower:]')" in
  ""|auto) ;;
  none|null|-) cmd+=(--light "") ;;
  *) cmd+=(--light "$light_chain") ;;
esac

if [[ "$no_renumber" == "true" ]]; then
  cmd+=(--no_renumber)
fi

echo "Job ID: $job_id"
echo "Job dir: $job_dir"
echo "Base config: $base_config"
echo "Run config: $run_config"
echo "Results dir: $results_dir"
echo "Energy guidance enabled: $energy"
echo "Command:"
printf ' %q' "${cmd[@]}"
echo

if [[ "$dry_run" == "true" ]]; then
  write_status "dry_run" "0"
  echo "Dry run only. Design command was not executed."
  exit 0
fi

write_status "running"
"${cmd[@]}"
write_status "succeeded" "0"
