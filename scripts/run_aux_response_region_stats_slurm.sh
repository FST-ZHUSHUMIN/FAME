#!/bin/bash
#SBATCH -J aux-response
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
SCRIPT_PATH="${ROOT_DIR}/scripts/run_aux_response_region_stats.py"
LOG_DIR="${ROOT_DIR}/logs"

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

if [ "${OUTPUT_ROOT}" = "${FULL_OUTPUT_ROOT}" ] && [ "${ALLOW_PARTIAL_FULL_ROOT:-0}" != "1" ]; then
  DATASETS="${FULL_DATASETS}"
fi

mkdir -p "${LOG_DIR}"
mkdir -p "${OUTPUT_ROOT}"
cd "${ROOT_DIR}"

source /home/zsm/anaconda3/etc/profile.d/conda.sh
conda activate qwen-finetune

export CUDA_VISIBLE_DEVICES=0

echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "SCRIPT_PATH=${SCRIPT_PATH}"
echo "CONFIG_PATH=${CONFIG_PATH}"
echo "CKPT_DIR=${CKPT_DIR}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "DATASETS=${DATASETS}"
echo "MAX_SAMPLES_PER_LEAF=${MAX_SAMPLES_PER_LEAF}"
echo "MAX_TOTAL_SAMPLES=${MAX_TOTAL_SAMPLES}"
echo "NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS}"
echo "TRUE_CFG_SCALE=${TRUE_CFG_SCALE}"
echo "GUIDANCE_SCALE=${GUIDANCE_SCALE}"
echo "SEED=${SEED}"
echo "DEVICE=${DEVICE}"
echo "MASK_THRESHOLD=${MASK_THRESHOLD}"
echo "SAVE_VISUALS=${SAVE_VISUALS}"
echo "OVERWRITE=${OVERWRITE}"
echo "Start time: $(date)"

nvidia-smi || true

if [ ! -f "${SCRIPT_PATH}" ]; then
  echo "[ERROR] Missing response script: ${SCRIPT_PATH}" >&2
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

CMD=(
  "${PYTHON_BIN}" -u "${SCRIPT_PATH}"
  --config "${CONFIG_PATH}"
  --ckpt "${CKPT_DIR}"
  --output_root "${OUTPUT_ROOT}"
  --datasets "${DATASETS}"
  --max_samples_per_leaf "${MAX_SAMPLES_PER_LEAF}"
  --max_total_samples "${MAX_TOTAL_SAMPLES}"
  --num_inference_steps "${NUM_INFERENCE_STEPS}"
  --true_cfg_scale "${TRUE_CFG_SCALE}"
  --guidance_scale "${GUIDANCE_SCALE}"
  --negative_prompt "${NEGATIVE_PROMPT}"
  --seed "${SEED}"
  --device "${DEVICE}"
  --mask_threshold "${MASK_THRESHOLD}"
)

if [ "${SAVE_VISUALS}" = "1" ]; then
  CMD+=(--save_visuals)
fi

if [ "${OVERWRITE}" = "1" ]; then
  CMD+=(--overwrite)
fi

printf '[Run] %q ' "${CMD[@]}"
printf '\n'
"${CMD[@]}"

echo "End time: $(date)"
