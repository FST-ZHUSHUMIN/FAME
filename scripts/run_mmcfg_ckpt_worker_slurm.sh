#!/bin/bash
#SBATCH -J mmcfg-ckpt
#SBATCH -p main
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 48:00:00
#SBATCH -o /home/zsm/code/0_p_faed_1/0-qwen-image-finetune-update/logs/%x_%j.out
#SBATCH -e /home/zsm/code/0_p_faed_1/0-qwen-image-finetune-update/logs/%x_%j.err

set -euo pipefail

ROOT_DIR="/home/zsm/code/0_p_faed_1/0-qwen-image-finetune-update"
ANALYZE_BATCH_SCRIPT="${ROOT_DIR}/scripts/run_mmcfg_analysis_batch.py"
LOG_DIR="${ROOT_DIR}/logs"

PYTHON_BIN="${PYTHON_BIN:-/home/zsm/.conda/envs/qwen-finetune/bin/python}"
CONFIG_PATH="${CONFIG_PATH:-${ROOT_DIR}/configs/7_qwen_lora_attentionloss_f9b1_mask_retrain_ori.yaml}"
CKPT_DIR="${CKPT_DIR:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"

TASKS="${TASKS:-top_length_short2long}"
MAX_SAMPLES="${MAX_SAMPLES:-20}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-20}"
TRUE_CFG_SCALE="${TRUE_CFG_SCALE:-4.0}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-4.0}"
SAVE_STEPS="${SAVE_STEPS:-0,mid,last}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:- }"
DEVICE="${DEVICE:-cuda:0}"

mkdir -p "${LOG_DIR}"
cd "${ROOT_DIR}"

source /home/zsm/anaconda3/etc/profile.d/conda.sh
conda activate qwen-finetune

export CUDA_VISIBLE_DEVICES=0

echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "CONFIG_PATH=${CONFIG_PATH}"
echo "CKPT_DIR=${CKPT_DIR}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "TASKS=${TASKS}"
echo "MAX_SAMPLES=${MAX_SAMPLES}"
echo "Start time: $(date)"

nvidia-smi || true

if [ -z "${CKPT_DIR}" ]; then
  echo "[ERROR] CKPT_DIR is empty." >&2
  exit 1
fi

if [ -z "${OUTPUT_ROOT}" ]; then
  echo "[ERROR] OUTPUT_ROOT is empty." >&2
  exit 1
fi

if [ ! -f "${CKPT_DIR}/pytorch_lora_weights.safetensors" ]; then
  echo "[ERROR] Missing LoRA weights: ${CKPT_DIR}/pytorch_lora_weights.safetensors" >&2
  exit 1
fi

if [ ! -f "${CKPT_DIR}/mask_prediction_head.pt" ]; then
  echo "[ERROR] Missing mask head: ${CKPT_DIR}/mask_prediction_head.pt" >&2
  exit 1
fi

"${PYTHON_BIN}" "${ANALYZE_BATCH_SCRIPT}" \
  --config "${CONFIG_PATH}" \
  --ckpt "${CKPT_DIR}" \
  --output_root "${OUTPUT_ROOT}" \
  --tasks "${TASKS}" \
  --max_samples "${MAX_SAMPLES}" \
  --num_inference_steps "${NUM_INFERENCE_STEPS}" \
  --true_cfg_scale "${TRUE_CFG_SCALE}" \
  --guidance_scale "${GUIDANCE_SCALE}" \
  --save_steps "${SAVE_STEPS}" \
  --negative_prompt "${NEGATIVE_PROMPT}" \
  --device "${DEVICE}"

echo "End time: $(date)"
