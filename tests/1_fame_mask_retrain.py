import sys
from pathlib import Path

# Add src/ to Python path BEFORE any qflux imports
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from typing import List, Tuple
from PIL import Image
import os
import numpy as np
from qflux.utils.seed import seed_everything
from qflux.trainer.qwen_image_edit_trainer import QwenImageEditTrainer
from qflux.data.config import load_config_from_yaml
import torch
from diffusers import QwenImageEditPipeline

try:
    RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
    RESAMPLE_NEAREST = Image.Resampling.NEAREST
except AttributeError:
    RESAMPLE_LANCZOS = Image.LANCZOS
    RESAMPLE_NEAREST = Image.NEAREST

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Load configuration
# config = load_config_from_yaml("/home/zsm/code/qwen-image-finetune/configs/my_training_config.yaml")
config= load_config_from_yaml("./configs/1_fame_config.yaml")
config.model.mg_cfg_background_scale = 2.0
config.model.mg_cfg_mask_dilate_radius = 1
config.model.mg_cfg_mask_blur_kernel = 5
config.model.mg_cfg_start_ratio = 0.1

# ============================================================================
# Checkpoint选择：根据训练版本选择合适的checkpoint
# ============================================================================


# 新版mask_fixed：训练时已修正（pos_weight+初始化），不需要反转
checkpoint_dir = "/path/to/checkpoint"  # 替换XXXX为实际checkpoint号
MASK_INVERT = bool(getattr(config.model, "invert_predicted_mask", False))
print("Mask inversion at inference (config.model.invert_predicted_mask):", MASK_INVERT)

config.model.lora.pretrained_weight = f"{checkpoint_dir}/pytorch_lora_weights.safetensors"

# Initialize trainer
trainer = QwenImageEditTrainer(config)
# trainer._use_official_diffusers = True
trainer.setup_predict()

# 显式加载 mask prediction head（与 LoRA checkpoint 同目录）
mask_head_ckpt = Path(checkpoint_dir) / "mask_prediction_head.pt"
if config.model.use_mask_prediction:
    if trainer.mask_prediction_head is None:
        raise RuntimeError("use_mask_prediction=True, 但 mask_prediction_head 未初始化")
    if mask_head_ckpt.exists():
        mask_state = torch.load(mask_head_ckpt, map_location="cpu")
        trainer.mask_prediction_head.load_state_dict(mask_state, strict=True)
        trainer.mask_prediction_head.to(device=trainer.config.predict.devices.dit, dtype=torch.bfloat16)
        trainer.mask_prediction_head.eval()
        print(f"✓ Loaded mask_prediction_head from: {mask_head_ckpt}")
    else:
        print(f"[Warning] mask_prediction_head checkpoint not found: {mask_head_ckpt}")

SEED = 42

img_dir_list = [
                  '/path/to/test datasets/In-the-wild/top_length/short',
                  '/path/to/test datasets/In-the-wild/dress_length/short',
                  '/path/to/test datasets/In-the-wild/pant_length/short',
                  '/path/to/test datasets/In-the-wild/sleeve_length/short',

                  '/path/to/test datasets/In-the-wild/sleeve_shape/regular',
                  '/path/to/test datasets/In-the-wild/sleeve_shape/regular',
                  '/path/to/test datasets/In-the-wild/sleeve_shape/regular',
                  '/path/to/test datasets/In-the-wild/sleeve_shape/regular',
               
                  '/path/to/test datasets/In-the-wild/collar/r',
                  '/path/to/test datasets/In-the-wild/collar/r',
                  '/path/to/test datasets/In-the-wild/collar/r',
                  '/path/to/test datasets/In-the-wild/collar/r',
                  '/path/to/test datasets/In-the-wild/collar/r',
                  '/path/to/test datasets/In-the-wild/collar/r',
                  '/path/to/test datasets/In-the-wild/collar/r',
                  '/path/to/test datasets/In-the-wild/collar/r',
                  '/path/to/test datasets/In-the-wild/collar/r',
                  ]

output_dir_list = [
                   '/path/to/test_results/1_fame_mask_retrain/In-the-wild/top_length/short2long',
                   '/path/to/test_results/1_fame_mask_retrain/In-the-wild/dress_length/short2long',
                   '/path/to/test_results/1_fame_mask_retrain/In-the-wild/pant_length/short2long',
                   '/path/to/test_results/1_fame_mask_retrain/In-the-wild/sleeve_length/short2long',

                   '/path/to/test_results/1_fame_mask_retrain/In-the-wild/sleeve_shape/regular2bell',
                   '/path/to/test_results/1_fame_mask_retrain/In-the-wild/sleeve_shape/regular2cape',
                   '/path/to/test_results/1_fame_mask_retrain/In-the-wild/sleeve_shape/regular2cap',
                   '/path/to/test_results/1_fame_mask_retrain/In-the-wild/sleeve_shape/regular2leg of mutton',          
               
                   '/path/to/test_results/1_fame_mask_retrain/In-the-wild/collar/r2cami-off_shoulder',
                   '/path/to/test_results/1_fame_mask_retrain/In-the-wild/collar/r2choker',
                   '/path/to/test_results/1_fame_mask_retrain/In-the-wild/collar/r2hoodie',
                   '/path/to/test_results/1_fame_mask_retrain/In-the-wild/collar/r2one-shoulder',
                   '/path/to/test_results/1_fame_mask_retrain/In-the-wild/collar/r2peter pan',
                   '/path/to/test_results/1_fame_mask_retrain/In-the-wild/collar/r2shirt',
                   '/path/to/test_results/1_fame_mask_retrain/In-the-wild/collar/r2square',
                   '/path/to/test_results/1_fame_mask_retrain/In-the-wild/collar/r2turtle',
                   '/path/to/test_results/1_fame_mask_retrain/In-the-wild/collar/r2v',
                   ]

instruction_list = [
      "Change top length from short to long",
      "Change dress length to long",
      "Change pants length to long",
      "Change sleeve length to long",

      "Change sleeve shape to bell sleeve",
      "Change sleeve shape to cape sleeve",
      "Change sleeve shape to cap sleeve",
      "Change sleeve shape to leg of mutton sleeve",
  
      "Change neckline to cami-off shoulder neckline",
      "Change neckline to choker neckline",
      "Change neckline to hoodie collar",
      "Change neckline to one-shoulder neckline",
      "Change neckline to peter pan collar",
      "Change neckline to shirt collar",
      "Change neckline to square neckline",
      "Change neckline to turtle collar",
      "Change neckline to v neckline"
]

# Set global random seed for deterministic VAE encoding
print("\n" + "=" * 80)
print("Setting global random seed for deterministic VAE encoding")
print("=" * 80)
seed_everything(42)

for i in range(len(img_dir_list)):
    img_dir = img_dir_list[i]
    output_dir = output_dir_list[i]
    instruction = instruction_list[i]

    os.makedirs(output_dir_list[i], exist_ok=True)
    img_list = os.listdir(img_dir_list[i])
    
    for j in range(len(img_list)):
        image_path = os.path.join(img_dir_list[i], img_list[j])
        print(f"{j} {len(img_list)}Processing image: {image_path} with instruction: {instruction}")
        save_path = os.path.join(output_dir, img_list[j])
        if os.path.exists(save_path):
            print(f"Image {save_path} already exists. Skipping...")
            continue
        
        input_image = Image.open(image_path)
        input_image = input_image.convert("RGB")
        if input_image.width > 512:
            input_image = input_image.crop((256, 0, 768, 1024))
        
        # Get image dimensions (PIL uses (width, height) format)
        img_width, img_height = input_image.size
        
        # Reset seed before each inference for deterministic VAE encoding
        setup_seed(SEED)
        generator = torch.Generator(device="cuda").manual_seed(SEED)
        # 不传 height 和 width 参数，让 Pipeline 根据图像自动计算
        # 或者确保图像尺寸已经符合要求（16 的倍数）

        # 1. 先准备 batch，获得 prompt_embeds, prompt_embeds_mask, control tensor
        # 传入 negative_prompt 和 true_cfg_scale 以启用 mask-guided CFG
        batch = trainer.prepare_predict_batch_data(
            image=input_image,
            prompt=instruction,
            negative_prompt=" ",  # 空 negative prompt 启用 CFG
            num_inference_steps=20,
            true_cfg_scale=4,  # CFG 引导强度（mask区域会用这个值，非mask区域接近1.0）
            generator=generator,
        )
        embeddings = trainer.prepare_embeddings(batch, stage="predict")
        
        # 2. 使用 embeddings 直接采样（sampling_from_embeddings 内部会预测并使用 mask-guided CFG）
        latents = trainer.sampling_from_embeddings(embeddings)
        pred_mask = embeddings.get("predicted_mask_2d", None)
        if pred_mask is not None:
            print(f"✓ Predicted mask shape: {pred_mask.shape}, used in mask-guided CFG")
        else:
            print("[Warning] No predicted mask returned from sampler, fallback to no mask visualization.")

        # 3. 解码生成结果
        target_height = embeddings["height"]
        target_width = embeddings["width"]
        result_image_tensor = trainer.decode_vae_latent(latents, target_height, target_width)
        
        # Convert tensor to PIL
        result_image_np = result_image_tensor.detach().permute(0, 2, 3, 1).float().cpu().numpy()
        result_image_np = (result_image_np * 255).round().astype("uint8")
        result_image = Image.fromarray(result_image_np[0])
        
        if result_image.size != input_image.size:
            result_image = result_image.resize(input_image.size, RESAMPLE_LANCZOS)

        # 可视化预测的 mask
        if pred_mask is not None:
            # pred_mask: [B, H, W] or [B, 1, H, W] tensor
            mask_np = pred_mask[0].detach().float().cpu().numpy()  # [H, W] or [1, H, W]
            if mask_np.ndim == 3:
                mask_np = mask_np[0]  # [H, W]
            
            # ============================================================================
            # Mask语义修复：应用反转（如果MASK_INVERT=True）
            # ============================================================================
            if MASK_INVERT:
                mask_np = 1.0 - mask_np
                print(f"✓ Applied mask inversion (1.0 - mask)")

            raw_min = float(mask_np.min())
            raw_max = float(mask_np.max())
            raw_mean = float(mask_np.mean())
            print(f"[MaskStats] raw min={raw_min:.4f}, max={raw_max:.4f}, mean={raw_mean:.4f}")

            # ============================================================================
            # 🔥 修复：不要normalize！sigmoid输出本身已经在[0,1]
            # normalize会扭曲绝对值（例如[0.1,0.3]被拉伸成[0,1]）
            # ============================================================================
            # 直接使用原始sigmoid值，clamp到[0,1]确保安全
            mask_vis = np.clip(mask_np, 0.0, 1.0)
            
            # 可视化灰度图 [0, 255]
            mask_vis_u8 = (mask_vis * 255).clip(0, 255).astype(np.uint8)
            mask_image = Image.fromarray(mask_vis_u8, mode='L')
            
            # Resize mask to match input image size
            if mask_image.size != input_image.size:
                mask_image = mask_image.resize(input_image.size, RESAMPLE_NEAREST)
            
            # 创建热力图 (可选：将灰度mask转换为彩色热力图)
            mask_heatmap = Image.new('RGB', mask_image.size)
            for y in range(mask_image.height):
                for x in range(mask_image.width):
                    val = mask_image.getpixel((x, y))
                    # 蓝色 -> 绿色 -> 黄色 -> 红色
                    if val < 85:
                        r, g, b = 0, val * 3, 255 - val * 3
                    elif val < 170:
                        r, g, b = (val - 85) * 3, 255, 0
                    else:
                        r, g, b = 255, 255 - (val - 170) * 3, 0
                    mask_heatmap.putpixel((x, y), (int(r), int(g), int(b)))
            
            # 保存组合图像：原图 + mask热力图 + 结果图
            combined_image = Image.new('RGB', (input_image.width * 3, input_image.height))
            combined_image.paste(input_image, (0, 0))
            combined_image.paste(mask_heatmap, (input_image.width, 0))
            combined_image.paste(result_image, (input_image.width * 2, 0))
            print(f"✓ Saved 3-panel image (input|mask|result) to: {save_path}")
        else:
            # 如果没有 mask，保持原来的双图布局
            combined_image = Image.new('RGB', (input_image.width * 2, input_image.height))
            combined_image.paste(input_image, (0, 0))
            combined_image.paste(result_image, (input_image.width, 0))
            print(f"⚠ Saved 2-panel image (no mask) to: {save_path}")
        
        combined_image.save(save_path)
