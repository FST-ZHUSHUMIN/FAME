#!/bin/bash
set -euo pipefail

ROOT_DIR="/home/zsm/code/0_p_faed_1/0-qwen-image-finetune-update"
SLURM_SCRIPT="${ROOT_DIR}/scripts/run_aux_response_region_stats_slurm.sh"

FULL_OUTPUT_ROOT="/mount/data/zsm/test_results/aux_response_region_stats"
FULL_DATASETS="humanparsing_editing_resize,test_slider"

PYTHON_BIN="${PYTHON_BIN:-/home/zsm/.conda/envs/qwen-finetune/bin/python}"
CONFIG_PATH="${CONFIG_PATH:-${ROOT_DIR}/configs/7_qwen_lora_attentionloss_f9b1_mask_retrain_ori.yaml}"
CKPT_DIR="${CKPT_DIR:-/mount/data/zsm/all_dataset/0_fame/qwen-finetune/project/outputs/7_qwen_lora_attentionloss_f9b1_mask_update/v1/checkpoint-39-5770}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mount/data/zsm/test_results/aux_response_region_stats}"
DATASETS="${DATASETS:-humanparsing_editing_resize,test_slider}"
MAX_SAMPLES_PER_LEAF="${MAX_SAMPLES_PER_LEAF:-0}"
MAX_TOTAL_SAMPLES="${MAX_TOTAL_SAMPLES:-0}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-20}"
TRUE_CFG_SCALE="${TRUE_CFG_SCALE:-4.0}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-4.0}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:- }"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cuda:0}"
MASK_THRESHOLD="${MASK_THRESHOLD:-0.5}"
SAVE_VISUALS="${SAVE_VISUALS:-1}"
OVERWRITE="${OVERWRITE:-0}"
JOB_NAME="${JOB_NAME:-aux-response}"

if [ "${OUTPUT_ROOT}" = "${FULL_OUTPUT_ROOT}" ] && [ "${ALLOW_PARTIAL_FULL_ROOT:-0}" != "1" ]; then
  DATASETS="${FULL_DATASETS}"
fi

if [ ! -f "${SLURM_SCRIPT}" ]; then
  echo "[ERROR] Missing slurm worker script: ${SLURM_SCRIPT}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}"

echo "[Aux Response] Worker: ${SLURM_SCRIPT}"
echo "[Aux Response] Config: ${CONFIG_PATH}"
echo "[Aux Response] Checkpoint: ${CKPT_DIR}"
echo "[Aux Response] Output root: ${OUTPUT_ROOT}"
echo "[Aux Response] Datasets: ${DATASETS}"
echo "[Aux Response] Max samples per leaf: ${MAX_SAMPLES_PER_LEAF}"
echo "[Aux Response] Max total samples: ${MAX_TOTAL_SAMPLES}"
echo "[Aux Response] Num inference steps: ${NUM_INFERENCE_STEPS}"
echo "[Aux Response] True CFG scale: ${TRUE_CFG_SCALE}"
echo "[Aux Response] Guidance scale: ${GUIDANCE_SCALE}"
echo "[Aux Response] Seed: ${SEED}"
echo "[Aux Response] Device: ${DEVICE}"
echo "[Aux Response] Mask threshold: ${MASK_THRESHOLD}"
echo "[Aux Response] Save visuals: ${SAVE_VISUALS}"
echo "[Aux Response] Overwrite: ${OVERWRITE}"

sbatch \
  --job-name="${JOB_NAME}" \
  --export=ALL,\
PYTHON_BIN="${PYTHON_BIN}",\
CONFIG_PATH="${CONFIG_PATH}",\
CKPT_DIR="${CKPT_DIR}",\
OUTPUT_ROOT="${OUTPUT_ROOT}",\
DATASETS="${DATASETS}",\
MAX_SAMPLES_PER_LEAF="${MAX_SAMPLES_PER_LEAF}",\
MAX_TOTAL_SAMPLES="${MAX_TOTAL_SAMPLES}",\
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS}",\
TRUE_CFG_SCALE="${TRUE_CFG_SCALE}",\
GUIDANCE_SCALE="${GUIDANCE_SCALE}",\
NEGATIVE_PROMPT="${NEGATIVE_PROMPT}",\
SEED="${SEED}",\
DEVICE="${DEVICE}",\
MASK_THRESHOLD="${MASK_THRESHOLD}",\
SAVE_VISUALS="${SAVE_VISUALS}",\
OVERWRITE="${OVERWRITE}" \
  "${SLURM_SCRIPT}"
