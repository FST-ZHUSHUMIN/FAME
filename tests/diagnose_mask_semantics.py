"""
诊断mask语义：对比GT mask和预测mask
"""
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path

# Add src to path  
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from qflux.trainer.qwen_image_edit_trainer import QwenImageEditTrainer
from qflux.data.config import load_config_from_yaml

def diagnose_mask_semantics():
    # 配置路径
    config_path = "/home/zsm/code/0-qwen-image-finetune-ori/configs/7_qwen_lora_attentionloss_f9b1_mask.yaml"
    checkpoint_path = "/mnt/data/zsm/qwen-finetune/project/outputs/7_qwen_lora_attentionloss_f9b1_mask/v2/checkpoint-39-5770"
    
    # 测试数据
    test_image_path = "/mnt/data/zsm/qwen-finetune/project/qwen_train/training_images/image_0000.png"
    gt_mask_path = "/mnt/data/zsm/qwen-finetune/project/qwen_train/control_images/image_0000_mask.png"
    prompt_path = "/mnt/data/zsm/qwen-finetune/project/qwen_train/training_images/image_0000.txt"
    
    print("=" * 80)
    print("Mask语义诊断")
    print("=" * 80)
    
    # 1. 读取GT mask
    print("\n[1] 读取GT Mask")
    gt_mask_pil = Image.open(gt_mask_path).convert('L')  # 转换为灰度图
    gt_mask_np = np.array(gt_mask_pil)
    print(f"GT Mask shape: {gt_mask_np.shape}")
    print(f"GT Mask min: {gt_mask_np.min()}, max: {gt_mask_np.max()}, mean: {gt_mask_np.mean():.2f}")
    
    # 统计GT mask的分布
    edit_pixels = np.sum(gt_mask_np > 128)  # 白色=编辑区域
    keep_pixels = np.sum(gt_mask_np <= 128)  # 黑色=保持区域
    total = gt_mask_np.size
    print(f"GT Mask语义: 白色(>128)=编辑区域 占 {100*edit_pixels/total:.1f}%")
    print(f"            黑色(<=128)=保持区域 占 {100*keep_pixels/total:.1f}%")
    
    # 2. 加载模型并预测
    print("\n[2] 加载模型")
    config = load_config_from_yaml(config_path)
    config.trainer.checkpoint_path = checkpoint_path
    trainer = QwenImageEditTrainer(config)
    trainer.setup_predict()
    
    # 显式加载mask prediction head
    mask_head_path = os.path.join(checkpoint_path, "mask_prediction_head.pt")
    if os.path.exists(mask_head_path):
        device = trainer.dit.device if hasattr(trainer, 'dit') else 'cuda'
        state_dict = torch.load(mask_head_path, map_location=device, weights_only=True)
        trainer.mask_prediction_head.load_state_dict(state_dict)
        print(f"✓ 加载mask prediction head: {mask_head_path}")
    
    # 3. 推理获取预测mask
    print("\n[3] 执行推理")
    with open(prompt_path, 'r') as f:
        prompt = f.read().strip()
    print(f"Prompt: {prompt}")
    
    control_img = Image.open(test_image_path).convert("RGB")
    setup_seed = lambda seed: torch.manual_seed(seed)
    setup_seed(42)
    generator = torch.Generator(device="cuda").manual_seed(42)
    
    with torch.inference_mode():
        batch = trainer.prepare_predict_batch_data(
            image=control_img,
            prompt=prompt,
            negative_prompt=" ",
            num_inference_steps=20,
            true_cfg_scale=4.0,
            generator=generator,
        )
        embeddings = trainer.prepare_embeddings(batch, stage="predict")
        latents = trainer.sampling_from_embeddings(embeddings)
    
    # 获取预测的2D mask
    pred_mask_2d = embeddings.get("predicted_mask_2d")
    if pred_mask_2d is None:
        print("✗ 没有找到predicted_mask_2d")
        return
    
    # 转换为numpy
    pred_mask_np = pred_mask_2d[0, 0].detach().float().cpu().numpy()
    print(f"\nPredicted Mask shape: {pred_mask_np.shape}")
    print(f"Predicted Mask min: {pred_mask_np.min():.4f}, max: {pred_mask_np.max():.4f}, mean: {pred_mask_np.mean():.4f}")
    
    # 4. 对比分析
    print("\n[4] 语义对比分析")
    print("-" * 80)
    
    # 将GT mask下采样到和预测mask相同的分辨率
    from PIL import Image as PILImage
    gt_mask_resized_pil = gt_mask_pil.resize(
        (pred_mask_np.shape[1], pred_mask_np.shape[0]),
        PILImage.Resampling.NEAREST
    )
    gt_mask_resized_np = np.array(gt_mask_resized_pil).astype(np.float32) / 255.0  # 归一化到[0,1]
    
    # 找到GT中的编辑区域和保持区域
    edit_region_mask = gt_mask_resized_np > 0.5  # 白色区域
    keep_region_mask = gt_mask_resized_np <= 0.5  # 黑色区域
    
    # 统计预测mask在这两个区域的均值
    pred_in_edit_region = pred_mask_np[edit_region_mask].mean() if edit_region_mask.any() else 0
    pred_in_keep_region = pred_mask_np[keep_region_mask].mean() if keep_region_mask.any() else 0
    
    print(f"GT mask中白色区域(编辑区域)的预测值均值: {pred_in_edit_region:.4f}")
    print(f"GT mask中黑色区域(保持区域)的预测值均值: {pred_in_keep_region:.4f}")
    print(f"差值 (编辑区域 - 保持区域): {pred_in_edit_region - pred_in_keep_region:.4f}")
    print()
    
    if pred_in_edit_region > pred_in_keep_region:
        print("✓ 语义正确：编辑区域预测值 > 保持区域预测值")
    else:
        print("✗ 语义反转：编辑区域预测值 < 保持区域预测值")
        print("  模型学习到了相反的语义！需要在推理时反转mask：pred_mask = 1 - pred_mask")
    
    # 5. 可视化对比
    print("\n[5] 保存可视化结果")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 原图
    axes[0, 0].imshow(control_img)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')
    
    # GT mask (原始分辨率)
    axes[0, 1].imshow(gt_mask_np, cmap='gray')
    axes[0, 1].set_title(f"GT Mask (白色=编辑区域)\nMean: {gt_mask_np.mean():.1f}")
    axes[0, 1].axis('off')
    
    # GT mask (下采样)
    axes[0, 2].imshow(gt_mask_resized_np, cmap='gray', vmin=0, vmax=1)
    axes[0, 2].set_title(f"GT Mask Resized\nMean: {gt_mask_resized_np.mean():.3f}")
    axes[0, 2].axis('off')
    
    # 预测mask
    axes[1, 0].imshow(pred_mask_np, cmap='jet', vmin=0, vmax=1)
    axes[1, 0].set_title(f"Predicted Mask\nMean: {pred_mask_np.mean():.3f}")
    axes[1, 0].axis('off')
    
    # 预测mask (归一化增强对比度)
    pred_norm = (pred_mask_np - pred_mask_np.min()) / (pred_mask_np.max() - pred_mask_np.min() + 1e-8)
    axes[1, 1].imshow(pred_norm, cmap='jet', vmin=0, vmax=1)
    axes[1, 1].set_title(f"Predicted Mask (归一化)\nRange: [{pred_mask_np.min():.3f}, {pred_mask_np.max():.3f}]")
    axes[1, 1].axis('off')
    
    # 预测mask反转
    pred_inverted = 1.0 - pred_mask_np
    axes[1, 2].imshow(pred_inverted, cmap='jet', vmin=0, vmax=1)
    axes[1, 2].set_title(f"Predicted Mask (反转)\nMean: {pred_inverted.mean():.3f}")
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    output_path = "/home/zsm/code/0-qwen-image-finetune-ori/mask_semantics_diagnosis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ 保存到: {output_path}")
    
    print("\n" + "=" * 80)
    print("诊断完成")
    print("=" * 80)

if __name__ == "__main__":
    diagnose_mask_semantics()
