#!/bin/bash
set -euo pipefail

ROOT_DIR="/home/zsm/code/0_p_faed_1/0-qwen-image-finetune-update"
SLURM_SCRIPT="${ROOT_DIR}/scripts/run_mmcfg_analysis_batch_slurm.sh"
PYTHON_BIN="${PYTHON_BIN:-/home/zsm/.conda/envs/qwen-finetune/bin/python}"

CONFIG_PATH="${CONFIG_PATH:-${ROOT_DIR}/configs/7_qwen_lora_attentionloss_f9b1_mask_retrain_ori.yaml}"
CKPT_ROOT="${CKPT_ROOT:-/mount/data/zsm/all_dataset/0_fame/qwen-finetune/project/outputs/7_qwen_lora_attentionloss_f9b1_mask_update/v1}"
OUTPUT_ROOT_BASE="${OUTPUT_ROOT_BASE:-/mount/data/zsm/test_results/mmcfg_ckpt_sweep}"

TASKS="${TASKS:-top_length_short2long}"
MAX_SAMPLES="${MAX_SAMPLES:-20}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-20}"
TRUE_CFG_SCALE="${TRUE_CFG_SCALE:-4.0}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-4.0}"
CHECKPOINTS_SPEC="${CHECKPOINTS_SPEC:-checkpoint-39-5770 checkpoint-55-8078 checkpoint-67-9809 checkpoint-83-12117}"
read -r -a CHECKPOINTS <<< "${CHECKPOINTS_SPEC}"

if [ ! -f "${SLURM_SCRIPT}" ]; then
  echo "[ERROR] Missing slurm worker script: ${SLURM_SCRIPT}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT_BASE}"

echo "[MM-CFG Sweep] Config: ${CONFIG_PATH}"
echo "[MM-CFG Sweep] Checkpoint root: ${CKPT_ROOT}"
echo "[MM-CFG Sweep] Output root base: ${OUTPUT_ROOT_BASE}"
echo "[MM-CFG Sweep] Tasks: ${TASKS}"
echo "[MM-CFG Sweep] Max samples: ${MAX_SAMPLES}"
echo "[MM-CFG Sweep] Checkpoints:"
for ckpt_name in "${CHECKPOINTS[@]}"; do
  echo "  - ${ckpt_name}"
done

for ckpt_name in "${CHECKPOINTS[@]}"; do
  CKPT_DIR="${CKPT_ROOT}/${ckpt_name}"
  if [ ! -f "${CKPT_DIR}/pytorch_lora_weights.safetensors" ]; then
    echo "[WARN] Skip ${ckpt_name}: missing LoRA weights" >&2
    continue
  fi
  if [ ! -f "${CKPT_DIR}/mask_prediction_head.pt" ]; then
    echo "[WARN] Skip ${ckpt_name}: missing mask head" >&2
    continue
  fi

  OUTPUT_ROOT="${OUTPUT_ROOT_BASE}/${ckpt_name}"
  JOB_NAME="mmcfg-${ckpt_name#checkpoint-}"

  echo "[MM-CFG Sweep] Submitting ${ckpt_name}"
  sbatch \
    --job-name="${JOB_NAME}" \
    --export=ALL,\
PYTHON_BIN="${PYTHON_BIN}",\
CONFIG_PATH="${CONFIG_PATH}",\
CKPT_DIR="${CKPT_DIR}",\
OUTPUT_ROOT="${OUTPUT_ROOT}",\
TASKS="${TASKS}",\
MAX_SAMPLES="${MAX_SAMPLES}",\
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS}",\
TRUE_CFG_SCALE="${TRUE_CFG_SCALE}",\
GUIDANCE_SCALE="${GUIDANCE_SCALE}" \
    "${SLURM_SCRIPT}"
done
