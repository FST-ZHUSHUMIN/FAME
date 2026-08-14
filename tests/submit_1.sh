#!/bin/bash
#SBATCH --job-name=my_job       # 任务名（自定义）
SBATCH --output=qwen_lora_attentionloss_f9b1.out     # 输出日志文件
SBATCH --error=qwen_lora_attentionloss_f9b1.err      # 错误日志文件
SBATCH --nodes=1               # 使用节点数
#SBATCH --ntasks=4              # 总任务数（通常等于核数）
#SBATCH --cpus-per-task=1       # 每个任务占用CPU核心数
#SBATCH --mem=8G                # 每个节点内存（总内存）
#SBATCH --time=01:30:00         # 任务最大运行时间（时:分:秒）
#SBATCH --partition=normal      # 队列/分区名（根据集群配置填写）
#SBATCH --gres=gpu:1            # 申请GPU（如有需要，1表示1块GPU）

# 激活环境（根据你的需求，如conda环境）
source ~/.bashrc
conda activate qwen-train

# 执行你的任务命令
CUDA_VISIBLE_DEVICES=4 python3 /home/zsm/code/0-qwen-image-finetune-update/tests/7_qwen_lora_attentionloss_f9b1_mask.py
wait
echo "all end!"