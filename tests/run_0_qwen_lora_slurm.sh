#!/bin/bash
#SBATCH -J qwen-lora-test
#SBATCH -p main
#SBATCH --gres=gpu:1
#SBATCH --gpu-bind=single:5
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 48:00:00
#SBATCH -o /home/zsm/code/0_p_faed_1/0-qwen-image-finetune-update/logs/%x_%j.out
#SBATCH -e /home/zsm/code/0_p_faed_1/0-qwen-image-finetune-update/logs/%x_%j.err

set -eo pipefail

ROOT_DIR="/home/zsm/code/0_p_faed_1/0-qwen-image-finetune-update"
SCRIPT_PATH="${ROOT_DIR}/tests/0_qwen_lora.py"

mkdir -p "${ROOT_DIR}/logs"
cd "${ROOT_DIR}"

source /home/zsm/anaconda3/etc/profile.d/conda.sh
conda activate qwen-finetune

# Use the single GPU assigned by Slurm inside the job.
export CUDA_VISIBLE_DEVICES=0

echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Start time: $(date)"

nvidia-smi || true

python "${SCRIPT_PATH}"

echo "End time: $(date)"
