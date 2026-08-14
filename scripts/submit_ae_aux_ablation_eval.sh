#!/bin/bash
set -euo pipefail

ROOT_DIR="/home/zsm/code/0_p_faed_1/0-qwen-image-finetune-update"
SLURM_SCRIPT="${ROOT_DIR}/scripts/run_ae_aux_ablation_eval_slurm.sh"

DEFAULT_OUTPUT_ROOT="/mount/data/zsm/test_results/aux_ae_ablation"
FULL_OUTPUT_ROOT="/mount/data/zsm/test_results/aux_ae_ablation_full"
FULL_DATASETS="humanparsing_editing_resize,test_slider"
FULL_VARIANTS="A,B,C,D,E"

PYTHON_BIN="${PYTHON_BIN:-/home/zsm/.conda/envs/qwen-finetune/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${DEFAULT_OUTPUT_ROOT}}"
DATASETS="${DATASETS:-${FULL_DATASETS}}"
VARIANTS="${VARIANTS:-${FULL_VARIANTS}}"
MAX_SAMPLES_PER_LEAF="${MAX_SAMPLES_PER_LEAF:-0}"
MAX_TOTAL_SAMPLES="${MAX_TOTAL_SAMPLES:-0}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-20}"
TRUE_CFG_SCALE="${TRUE_CFG_SCALE:-4.0}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-4.0}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:- }"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cuda:0}"
MASK_THRESHOLD="${MASK_THRESHOLD:-8}"
SAVE_PANELS="${SAVE_PANELS:-1}"
SAVE_PRED_MASKS="${SAVE_PRED_MASKS:-0}"
OVERWRITE="${OVERWRITE:-0}"
JOB_NAME="${JOB_NAME:-ae-aux-eval}"
ALLOW_PARTIAL_FULL_ROOT="${ALLOW_PARTIAL_FULL_ROOT:-0}"

if [ ! -f "${SLURM_SCRIPT}" ]; then
  echo "[ERROR] Missing slurm worker script: ${SLURM_SCRIPT}" >&2
  exit 1
fi

# Protect the canonical full output root from stale shell exports such as
# VARIANTS=A or DATASETS=humanparsing_editing_resize.
if [ "${OUTPUT_ROOT}" = "${FULL_OUTPUT_ROOT}" ] && [ "${ALLOW_PARTIAL_FULL_ROOT}" != "1" ]; then
  if [ "${DATASETS}" != "${FULL_DATASETS}" ] || [ "${VARIANTS}" != "${FULL_VARIANTS}" ]; then
    echo "[AE Aux Eval] Detected full output root; overriding stale partial config."
  fi
  DATASETS="${FULL_DATASETS}"
  VARIANTS="${FULL_VARIANTS}"
fi

mkdir -p "${OUTPUT_ROOT}"

echo "[AE Aux Eval] Worker: ${SLURM_SCRIPT}"
echo "[AE Aux Eval] Output root: ${OUTPUT_ROOT}"
echo "[AE Aux Eval] Datasets: ${DATASETS}"
echo "[AE Aux Eval] Variants: ${VARIANTS}"
echo "[AE Aux Eval] Max samples per leaf: ${MAX_SAMPLES_PER_LEAF}"
echo "[AE Aux Eval] Max total samples: ${MAX_TOTAL_SAMPLES}"
echo "[AE Aux Eval] Num inference steps: ${NUM_INFERENCE_STEPS}"
echo "[AE Aux Eval] True CFG scale: ${TRUE_CFG_SCALE}"
echo "[AE Aux Eval] Guidance scale: ${GUIDANCE_SCALE}"
echo "[AE Aux Eval] Seed: ${SEED}"
echo "[AE Aux Eval] Device: ${DEVICE}"
echo "[AE Aux Eval] Save panels: ${SAVE_PANELS}"
echo "[AE Aux Eval] Save pred masks: ${SAVE_PRED_MASKS}"
echo "[AE Aux Eval] Overwrite: ${OVERWRITE}"
echo "[AE Aux Eval] Allow partial full root: ${ALLOW_PARTIAL_FULL_ROOT}"

sbatch \
  --job-name="${JOB_NAME}" \
  --export=ALL,\
PYTHON_BIN="${PYTHON_BIN}",\
OUTPUT_ROOT="${OUTPUT_ROOT}",\
DATASETS="${DATASETS}",\
VARIANTS="${VARIANTS}",\
MAX_SAMPLES_PER_LEAF="${MAX_SAMPLES_PER_LEAF}",\
MAX_TOTAL_SAMPLES="${MAX_TOTAL_SAMPLES}",\
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS}",\
TRUE_CFG_SCALE="${TRUE_CFG_SCALE}",\
GUIDANCE_SCALE="${GUIDANCE_SCALE}",\
NEGATIVE_PROMPT="${NEGATIVE_PROMPT}",\
SEED="${SEED}",\
DEVICE="${DEVICE}",\
MASK_THRESHOLD="${MASK_THRESHOLD}",\
SAVE_PANELS="${SAVE_PANELS}",\
SAVE_PRED_MASKS="${SAVE_PRED_MASKS}",\
OVERWRITE="${OVERWRITE}" \
  "${SLURM_SCRIPT}"
