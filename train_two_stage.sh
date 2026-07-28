#!/usr/bin/env bash

# ACTIVATE_ENV=0 bash /xsdata/lzhlzh/26_36/FlowAB/FlowDesign/train_two_stage.sh
# Resume examples:
#   RESUME_CKPT=/path/to/checkpoints/checkpoint_best.pt bash train_two_stage.sh
#   RESUME_CKPT=/path/to/checkpoints/checkpoint_best.pt RESUME_STAGE=1 bash train_two_stage.sh
#   RESUME_CKPT=/path/to/checkpoints/checkpoint_best.pt RESUME_STAGE=2 bash train_two_stage.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

# Optional environment activation.
ENV_NAME="${ENV_NAME:-diffab}"
ACTIVATE_ENV="${ACTIVATE_ENV:-1}"
if [[ "${ACTIVATE_ENV}" == "1" ]]; then
    if command -v activate >/dev/null 2>&1; then
        # shellcheck disable=SC1091
        . activate "${ENV_NAME}"
    else
        echo "[warn] skip environment activation: command 'activate' not found" >&2
    fi
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
LOG_ROOT="${LOG_ROOT:-${ROOT_DIR}/logs}"
RUN_TAG="${RUN_TAG:-two_stage_$(date +%Y%m%d_%H%M%S)}"

STAGE1_CONFIG="${STAGE1_CONFIG:-./configs/train/codesign_muti_rectflow_RF.yml}"
STAGE2_CONFIG="${STAGE2_CONFIG:-./configs/train/codesign_muti_rectflow_finetune_RF.yml}"
RESUME_CKPT="${RESUME_CKPT:-${RESUME:-}}"
RESUME_STAGE="${RESUME_STAGE:-auto}"

# Put shared training args here if needed.
COMMON_ARGS=(
    --device cuda
    --num_workers 8
)

# Put stage-specific args here if needed.
STAGE1_ARGS=()
STAGE2_ARGS=()

mkdir -p "${LOG_ROOT}"

stage1_prefix="$(basename "${STAGE1_CONFIG}")"
stage1_prefix="${stage1_prefix%.*}"
stage1_config_name="$(basename "${STAGE1_CONFIG}")"
stage2_config_name="$(basename "${STAGE2_CONFIG}")"

infer_resume_stage() {
    local ckpt_path="$1"
    local log_dir="$2"

    if [[ -f "${log_dir}/${stage2_config_name}" ]]; then
        printf '2\n'
        return 0
    fi
    if [[ -f "${log_dir}/${stage1_config_name}" ]]; then
        printf '1\n'
        return 0
    fi

    echo "[error] cannot infer resume stage from ${ckpt_path}" >&2
    echo "[error] expected ${log_dir}/${stage1_config_name} or ${log_dir}/${stage2_config_name}" >&2
    echo "[error] set RESUME_STAGE=1 or RESUME_STAGE=2 explicitly" >&2
    return 1
}

run_stage1() {
    local resume_ckpt="${1:-}"
    local -a stage1_cmd=(
        "${PYTHON_BIN}" train.py "${STAGE1_CONFIG}"
        --logdir "${LOG_ROOT}"
        --tag "${RUN_TAG}"
        "${COMMON_ARGS[@]}"
    )

    if [[ -n "${resume_ckpt}" ]]; then
        stage1_cmd+=(--resume "${resume_ckpt}")
    fi
    if (( ${#STAGE1_ARGS[@]} > 0 )); then
        stage1_cmd+=("${STAGE1_ARGS[@]}")
    fi

    echo "[info] start stage 1"
    "${stage1_cmd[@]}"
}

run_stage2() {
    local resume_ckpt="$1"
    local -a stage2_cmd=(
        "${PYTHON_BIN}" train_sec.py "${STAGE2_CONFIG}"
        --resume "${resume_ckpt}"
        "${COMMON_ARGS[@]}"
    )

    if (( ${#STAGE2_ARGS[@]} > 0 )); then
        stage2_cmd+=("${STAGE2_ARGS[@]}")
    fi

    echo "[info] start stage 2"
    echo "[info] stage2 resume ckpt    : ${resume_ckpt}"
    "${stage2_cmd[@]}"
}

echo "[info] project dir           : ${ROOT_DIR}"
echo "[info] log root              : ${LOG_ROOT}"
echo "[info] run tag               : ${RUN_TAG}"
echo "[info] stage1 cfg            : ${STAGE1_CONFIG}"
echo "[info] stage2 cfg            : ${STAGE2_CONFIG}"
echo "[info] resume ckpt (input)   : ${RESUME_CKPT:-<none>}"
echo "[info] resume stage mode     : ${RESUME_STAGE}"

resume_stage=""
resume_log_dir=""
stage1_log_dir=""
stage1_ckpt=""

if [[ -n "${RESUME_CKPT}" ]]; then
    if [[ ! -f "${RESUME_CKPT}" ]]; then
        echo "[error] resume checkpoint not found: ${RESUME_CKPT}" >&2
        exit 1
    fi

    resume_log_dir="$(dirname "$(dirname "${RESUME_CKPT}")")"
    if [[ ! -d "${resume_log_dir}" ]]; then
        echo "[error] resume log dir not found: ${resume_log_dir}" >&2
        exit 1
    fi

    case "${RESUME_STAGE}" in
        auto)
            resume_stage="$(infer_resume_stage "${RESUME_CKPT}" "${resume_log_dir}")"
            ;;
        1|2)
            resume_stage="${RESUME_STAGE}"
            ;;
        *)
            echo "[error] invalid RESUME_STAGE: ${RESUME_STAGE} (expected auto, 1, or 2)" >&2
            exit 1
            ;;
    esac

    echo "[info] inferred resume log   : ${resume_log_dir}"
    echo "[info] inferred resume stage : ${resume_stage}"
fi

if [[ "${resume_stage}" == "2" ]]; then
    echo "[info] skip stage 1          : resuming stage 2 directly"
    run_stage2 "${RESUME_CKPT}"
    echo "[info] two-stage training finished"
    exit 0
fi

if [[ "${resume_stage}" == "1" ]]; then
    run_stage1 "${RESUME_CKPT}"
    stage1_log_dir="${resume_log_dir}"
else
    run_stage1
    mapfile -t stage1_dirs < <(find "${LOG_ROOT}" -mindepth 1 -maxdepth 1 -type d -name "${stage1_prefix}_*_${RUN_TAG}" | sort)

    if [[ "${#stage1_dirs[@]}" -ne 1 ]]; then
        echo "[error] expected exactly one stage-1 log dir, found ${#stage1_dirs[@]}" >&2
        printf '%s\n' "${stage1_dirs[@]}" >&2
        exit 1
    fi

    stage1_log_dir="${stage1_dirs[0]}"
fi

stage1_ckpt="${stage1_log_dir}/checkpoints/checkpoint_best.pt"
if [[ ! -f "${stage1_ckpt}" ]]; then
    echo "[error] stage-1 checkpoint not found: ${stage1_ckpt}" >&2
    exit 1
fi

echo "[info] stage1 log            : ${stage1_log_dir}"
echo "[info] stage1 resume ckpt    : ${stage1_ckpt}"

run_stage2 "${stage1_ckpt}"

echo "[info] two-stage training finished"
