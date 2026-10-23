#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export EPONA_EVAL_SEED="${EPONA_EVAL_SEED:-}"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-${ROOT}/datasets/maps}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-${ROOT}/datasets/openscene}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-${ROOT}/exp}"
export NAVSIM_DEVKIT_ROOT="${ROOT}"
export HYDRA_FULL_ERROR=1
python scripts/test/run_navsim_eval.py \
    --exp_name "${EXP_NAME:-simvta_navsim_v1}" \
    --config configs/navsim_v1.py \
    --resume_path "${RESUME_PATH:-${ROOT}/checkpoints/simvta.pkl}" \
    --vae_ckpt "${VAE_CKPT:-${ROOT}/checkpoints/dcae_td_20000.pkl}" \
    --navsim_log_path "${NAVSIM_LOG_PATH:-${OPENSCENE_DATA_ROOT}/navsim_logs/test}" \
    --sensor_blobs_path "${NUPLAN_SENSOR_BLOBS_ROOT:-${OPENSCENE_DATA_ROOT}/sensor_blobs}" \
    --navsim_exp_root "${NAVSIM_EXP_ROOT}" \
    --metric_cache_path "${METRIC_CACHE_PATH:-${NAVSIM_EXP_ROOT}/metric_cache}" \
    --save_video_path "${NAVSIM_EXP_ROOT}/predictions" \
    --start_id "${START_ID:-0}" \
    --end_id "${END_ID:-99999}"
