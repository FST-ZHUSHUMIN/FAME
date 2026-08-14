import cv2
import numpy as np
from PIL import Image
import os

# ---------------------- 1. 核心配置 ----------------------
# 输入路径
image_path = '/mount/data/zsm/test_results/7_maskpre_2-1-5-01_5770/top_length/short2long_test/0703Halter000000.png'
# 输出路径（最终结果）
output_path = '/mount/data/zsm/heatmap_stretched_final.png'
# 中间结果帧目录
frames_dir = '/mount/data/zsm/heatmap_stretch_frames'
os.makedirs(frames_dir, exist_ok=True)
# 固定画布尺寸（宽x高）
CANVAS_W, CANVAS_H = 512, 1024
# 拉伸目标：热力图内容纵向拉长至画布高度
STRETCH_TO_HEIGHT = CANVAS_H
# 中间帧数量
num_frames = 20
# 起始拉伸比例（从原始高度的50%开始拉伸）
start_ratio = 0.5

# ---------------------- 2. 读取并预处理图像 ----------------------
# 读取+裁剪（固定画布尺寸）
pil_image = Image.open(image_path).convert('RGB')
pil_image = pil_image.crop((512, 1024, 512+CANVAS_W, 1024+CANVAS_H))
image_rgb = np.array(pil_image)  # 保持RGB

# 提取热力图核心区域（基于亮度或某个通道，这里用灰度检测但保持RGB）
gray_for_detection = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
non_bg_mask = gray_for_detection > 0.1  # 背景阈值
non_bg_rows = np.where(non_bg_mask.any(axis=1))[0]
if len(non_bg_rows) == 0:
    initial_top, initial_bottom = 100, 300
else:
    initial_top = non_bg_rows[0]
    initial_bottom = non_bg_rows[-1]

# 原始热力图区域（RGB）
original_heatmap_rgb = image_rgb[initial_top:initial_bottom+1, :, :]

# ---------------------- 3. 生成中间结果帧（上边固定，下边逐渐变长） ----------------------
original_height = initial_bottom - initial_top + 1
# 从顶部开始，初始截取高度为原始高度的20%
initial_crop_ratio = 0.2
start_crop_height = int(original_height * initial_crop_ratio)
# 固定上边位置（从initial_top开始，或从0开始？用户说y从0开始）
fixed_top = 0  # 用户指定y从0开始

for frame_idx in range(num_frames):
    # 计算当前帧的截取高度（从初始高度逐步增加到原始高度）
    progress = frame_idx / (num_frames - 1)
    current_crop_height = int(start_crop_height + (original_height - start_crop_height) * progress)
    current_crop_height = min(current_crop_height, original_height)  # 不超过原始高度
    
    # 固定上边，下边逐渐向下延伸
    crop_top = fixed_top
    crop_bottom = min(CANVAS_H - 1, fixed_top + current_crop_height - 1)
    
    # 截取热力图内容（只截取在热力图区域内的部分）
    valid_top = max(crop_top, initial_top)
    valid_bottom = min(crop_bottom, initial_bottom)
    
    # 放回画布（RGB，背景蓝色）
    frame_canvas = np.full((CANVAS_H, CANVAS_W, 3), [0, 0, 255], dtype=np.uint8)  # 蓝色背景 (RGB)
    if valid_top <= valid_bottom:
        frame_canvas[valid_top:valid_bottom+1, :, :] = image_rgb[valid_top:valid_bottom+1, :, :]
    
    # 保存中间帧
    frame_path = os.path.join(frames_dir, f'frame_{frame_idx:04d}.png')
    cv2.imwrite(frame_path, cv2.cvtColor(frame_canvas, cv2.COLOR_RGB2BGR))  # OpenCV用BGR
    
    print(f'生成中间帧 {frame_idx+1}/{num_frames}: 截取高度 {crop_bottom - crop_top + 1}px (上边固定y={fixed_top})')

# ---------------------- 4. 生成最终结果（完整RGB图像） ----------------------
final_canvas = np.full((CANVAS_H, CANVAS_W, 3), [0, 0, 255], dtype=np.uint8)  # 蓝色背景 (RGB)
final_canvas[initial_top:initial_bottom+1, :, :] = image_rgb[initial_top:initial_bottom+1, :, :]

# ---------------------- 5. 保存最终结果 ----------------------
cv2.imwrite(output_path, cv2.cvtColor(final_canvas, cv2.COLOR_RGB2BGR))

print(f'✅ 完整热力图已保存至：{output_path}')
print(f'✅ 中间结果帧已保存至：{frames_dir} (共{num_frames}帧，上边固定y=0，下边逐渐变长)')
print(f'✅ 热力图从顶部截取，高度从 {int(original_height * initial_crop_ratio)}px 增加至 {original_height}px')