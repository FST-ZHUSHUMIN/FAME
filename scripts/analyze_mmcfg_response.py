#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

# Add src/ to Python path BEFORE any qflux imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))

from qflux.data.config import load_config_from_yaml
from qflux.trainer.base_trainer import BaseTrainer
from qflux.trainer.qwen_image_edit_trainer import QwenImageEditTrainer


try:
    RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
    RESAMPLE_BILINEAR = Image.Resampling.BILINEAR
except AttributeError:
    RESAMPLE_LANCZOS = Image.LANCZOS
    RESAMPLE_BILINEAR = Image.BILINEAR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze MM-CFG response maps without GT masks.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to YAML config.",
    )
    parser.add_argument(
        "--ckpt",
        type=Path,
        required=True,
        help="Checkpoint directory containing pytorch_lora_weights.safetensors and mask_prediction_head.pt",
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        help="Directory of input images. If omitted, --input_image must be provided.",
    )
    parser.add_argument(
        "--input_image",
        type=Path,
        help="Single input image path.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Editing instruction.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory to save analysis results.",
    )
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--true_cfg_scale", type=float, default=4.0)
    parser.add_argument("--guidance_scale", type=float, default=4.0)
    parser.add_argument("--mg_cfg_background_scale", type=float, default=None)
    parser.add_argument("--mg_cfg_mask_dilate_radius", type=int, default=None)
    parser.add_argument("--mg_cfg_mask_blur_kernel", type=int, default=None)
    parser.add_argument("--mg_cfg_start_ratio", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=10)
    parser.add_argument(
        "--save_steps",
        type=str,
        default="0,mid,last",
        help="Comma-separated step spec, e.g. 0,5,mid,last",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=" ",
        help="Use a blank string to enable true CFG.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override predict dit device, e.g. cuda:0",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute samples even if metrics.json already exists.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collect_samples(args: argparse.Namespace) -> list[Path]:
    if args.input_image is not None:
        return [args.input_image]
    if args.input_dir is None:
        raise ValueError("Provide either --input_image or --input_dir")
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    files = sorted([p for p in args.input_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])
    if args.max_samples <= 0:
        return files
    return files[: args.max_samples]


def load_trainer_and_weights(args: argparse.Namespace) -> QwenImageEditTrainer:
    config = load_config_from_yaml(str(args.config))
    config.model.use_mask_prediction = True
    config.cache.use_cache = False

    if args.device is not None:
        config.predict.devices.dit = args.device
        config.predict.devices.vae = args.device
        config.predict.devices.text_encoder = args.device

    if args.mg_cfg_background_scale is not None:
        config.model.mg_cfg_background_scale = args.mg_cfg_background_scale
    if args.mg_cfg_mask_dilate_radius is not None:
        config.model.mg_cfg_mask_dilate_radius = args.mg_cfg_mask_dilate_radius
    if args.mg_cfg_mask_blur_kernel is not None:
        config.model.mg_cfg_mask_blur_kernel = args.mg_cfg_mask_blur_kernel
    if args.mg_cfg_start_ratio is not None:
        config.model.mg_cfg_start_ratio = args.mg_cfg_start_ratio

    lora_path = args.ckpt / "pytorch_lora_weights.safetensors"
    mask_head_path = args.ckpt / "mask_prediction_head.pt"
    if not lora_path.exists():
        raise FileNotFoundError(f"LoRA weights not found: {lora_path}")
    if not mask_head_path.exists():
        raise FileNotFoundError(f"Mask head not found: {mask_head_path}")

    config.model.lora.pretrained_weight = str(lora_path)

    trainer = QwenImageEditTrainer(config)
    trainer.setup_predict()

    if trainer.mask_prediction_head is None:
        raise RuntimeError("Mask prediction head was not initialized.")

    mask_state = torch.load(mask_head_path, map_location="cpu", weights_only=True)
    trainer.mask_prediction_head.load_state_dict(mask_state, strict=True)
    trainer.mask_prediction_head.to(device=trainer.dit.device, dtype=trainer.weight_dtype)
    trainer.mask_prediction_head.eval()

    return trainer


def load_input_image(path: Path) -> Image.Image:
    image = Image.open(path).convert("RGB")
    if image.width > 512:
        image = image.crop((256, 0, 768, 1024))
    return image


def compute_response_token(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt((x.float() ** 2).mean(dim=-1) + 1.0e-8)


def token_response_to_image(response_token: torch.Tensor, pred_mask_2d: torch.Tensor, out_hw: tuple[int, int]) -> torch.Tensor:
    bsz, _, hm, wm = pred_mask_2d.shape
    ht = hm // 2
    wt = wm // 2
    expected = ht * wt
    if response_token.shape[1] != expected:
        raise ValueError(f"Token count mismatch: got T={response_token.shape[1]}, expected={expected}")
    resp_2d = response_token.view(bsz, 1, ht, wt)
    return F.interpolate(resp_2d.float(), size=out_hw, mode="bilinear", align_corners=False)


def normalize_for_vis(x: torch.Tensor) -> np.ndarray:
    arr = x.detach().float().cpu()
    min_v = float(arr.min())
    max_v = float(arr.max())
    if max_v - min_v < 1.0e-8:
        return np.zeros(arr.shape, dtype=np.float32)
    return ((arr - min_v) / (max_v - min_v)).numpy().astype(np.float32)


def heatmap_to_rgb(arr: np.ndarray) -> np.ndarray:
    arr = np.clip(arr, 0.0, 1.0)
    r = np.clip(1.5 * arr - 0.2, 0.0, 1.0)
    g = np.clip(1.5 - 2.0 * np.abs(arr - 0.5), 0.0, 1.0)
    b = np.clip(1.2 * (1.0 - arr), 0.0, 1.0)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255).astype(np.uint8)


def overlay_heatmap(base_image: Image.Image, heatmap: torch.Tensor, alpha: float = 0.55) -> Image.Image:
    base_np = np.array(base_image).astype(np.float32)
    heat_arr = normalize_for_vis(heatmap)
    if heat_arr.shape[:2] != (base_np.shape[0], base_np.shape[1]):
        heat_img = Image.fromarray((heat_arr * 255).astype(np.uint8), mode="L")
        heat_img = heat_img.resize(base_image.size, RESAMPLE_BILINEAR)
        heat_arr = np.array(heat_img).astype(np.float32) / 255.0
    heat_rgb = heatmap_to_rgb(heat_arr).astype(np.float32)
    merged = np.clip((1.0 - alpha) * base_np + alpha * heat_rgb, 0, 255).astype(np.uint8)
    return Image.fromarray(merged)


def overlay_mask_outline(
    base_image: Image.Image,
    mask: torch.Tensor,
    threshold: float = 0.5,
    color: tuple[int, int, int] = (255, 64, 64),
    alpha: float = 0.65,
) -> Image.Image:
    base_np = np.array(base_image).astype(np.float32)
    mask_np = normalize_for_vis(mask)
    if mask_np.shape[:2] != (base_np.shape[0], base_np.shape[1]):
        mask_img = Image.fromarray((mask_np * 255).astype(np.uint8), mode="L")
        mask_img = mask_img.resize(base_image.size, RESAMPLE_BILINEAR)
        mask_np = np.array(mask_img).astype(np.float32) / 255.0
    mask_bin = (mask_np >= threshold).astype(np.uint8)
    if mask_bin.any():
        inner = mask_bin[1:-1, 1:-1]
        eroded = np.zeros_like(mask_bin)
        eroded[1:-1, 1:-1] = (
            inner
            & mask_bin[:-2, 1:-1]
            & mask_bin[2:, 1:-1]
            & mask_bin[1:-1, :-2]
            & mask_bin[1:-1, 2:]
        )
        outline = (mask_bin - eroded).clip(0, 1).astype(np.float32)
    else:
        outline = mask_bin.astype(np.float32)
    color_arr = np.zeros_like(base_np)
    color_arr[..., 0] = color[0]
    color_arr[..., 1] = color[1]
    color_arr[..., 2] = color[2]
    merged = np.where(
        outline[..., None] > 0,
        np.clip((1.0 - alpha) * base_np + alpha * color_arr, 0, 255),
        base_np,
    ).astype(np.uint8)
    return Image.fromarray(merged)


def mask_to_image(mask: torch.Tensor, size: tuple[int, int]) -> Image.Image:
    mask_np = normalize_for_vis(mask)
    mask_u8 = (mask_np * 255).astype(np.uint8)
    img = Image.fromarray(mask_u8, mode="L")
    if img.size != size:
        img = img.resize(size, RESAMPLE_BILINEAR)
    return img.convert("RGB")


def tensor_image_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    image_np = image_tensor.detach().permute(0, 2, 3, 1).float().cpu().numpy()
    image_np = (image_np * 255).round().clip(0, 255).astype(np.uint8)
    return Image.fromarray(image_np[0])


def delta_map_to_image(delta: torch.Tensor, size: tuple[int, int]) -> Image.Image:
    arr = delta.detach().float().cpu().numpy()
    max_abs = float(np.max(np.abs(arr)))
    if max_abs < 1.0e-8:
        norm = np.zeros_like(arr, dtype=np.float32)
    else:
        norm = np.clip(arr / max_abs, -1.0, 1.0).astype(np.float32)
    pos = np.clip(norm, 0.0, 1.0)
    neg = np.clip(-norm, 0.0, 1.0)
    rgb = np.stack(
        [
            1.0 - neg,
            1.0 - (pos + neg),
            1.0 - pos,
        ],
        axis=-1,
    )
    img = Image.fromarray((rgb * 255).astype(np.uint8))
    if img.size != size:
        img = img.resize(size, RESAMPLE_BILINEAR)
    return img


def crop_box_from_mask(mask: torch.Tensor, image_size: tuple[int, int], threshold: float = 0.5) -> tuple[int, int, int, int]:
    mask_np = normalize_for_vis(mask)
    width, height = image_size
    if mask_np.shape[:2] != (height, width):
        mask_img = Image.fromarray((mask_np * 255).astype(np.uint8), mode="L")
        mask_img = mask_img.resize(image_size, RESAMPLE_BILINEAR)
        mask_np = np.array(mask_img).astype(np.float32) / 255.0
    ys, xs = np.where(mask_np >= threshold)
    if len(xs) == 0 or len(ys) == 0:
        return (0, 0, width, height)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    box_w = x1 - x0
    box_h = y1 - y0
    pad_x = max(12, int(box_w * 0.22))
    pad_y = max(12, int(box_h * 0.28))
    x0 = max(0, x0 - pad_x)
    y0 = max(0, y0 - pad_y)
    x1 = min(width, x1 + pad_x)
    y1 = min(height, y1 + pad_y)
    min_w = max(72, width // 6)
    min_h = max(72, height // 6)
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2
    half_w = max((x1 - x0) // 2, min_w // 2)
    half_h = max((y1 - y0) // 2, min_h // 2)
    x0 = max(0, cx - half_w)
    x1 = min(width, cx + half_w)
    y0 = max(0, cy - half_h)
    y1 = min(height, cy + half_h)
    return (x0, y0, x1, y1)


def save_roi_zoom_grid(
    sample_dir: Path,
    input_image: Image.Image,
    final_image_cfg: Image.Image,
    final_image_mm: Image.Image,
    delta_image: Image.Image,
    crop_box: tuple[int, int, int, int],
) -> None:
    labels = ["Input ROI", "CFG ROI", "MM-CFG ROI", "Delta ROI"]
    crops = [
        input_image.crop(crop_box),
        final_image_cfg.crop(crop_box),
        final_image_mm.crop(crop_box),
        delta_image.crop(crop_box),
    ]
    tile_w = max(img.width for img in crops)
    tile_h = max(img.height for img in crops)
    grid = Image.new("RGB", (4 * tile_w, tile_h + 24), color=(255, 255, 255))
    draw = ImageDraw.Draw(grid)
    for i, (label, crop) in enumerate(zip(labels, crops)):
        x = i * tile_w
        crop = crop.resize((tile_w, tile_h), RESAMPLE_BILINEAR)
        grid.paste(crop, (x, 24))
        draw.text((x + 6, 4), label, fill=(0, 0, 0))
    grid.save(sample_dir / "roi_zoom.png")


def parse_step_spec(spec: str, num_steps: int) -> list[int]:
    result: list[int] = []
    for item in [x.strip() for x in spec.split(",") if x.strip()]:
        if item == "mid":
            idx = max(0, num_steps // 2)
        elif item == "last":
            idx = max(0, num_steps - 1)
        else:
            idx = int(item)
        idx = min(max(idx, 0), max(0, num_steps - 1))
        if idx not in result:
            result.append(idx)
    if not result and num_steps > 0:
        result = [0, num_steps // 2, num_steps - 1]
    return result


def compute_predmask_region_metrics(
    cumulative_std: torch.Tensor,
    cumulative_mm: torch.Tensor,
    pred_mask_img: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, float]:
    pred_bin = (pred_mask_img >= threshold).float()
    std_sum = cumulative_std.sum().clamp(min=1.0e-8)
    mm_sum = cumulative_mm.sum().clamp(min=1.0e-8)
    inside_std = (cumulative_std * pred_bin).sum() / std_sum
    inside_mm = (cumulative_mm * pred_bin).sum() / mm_sum
    outside_std = (cumulative_std * (1.0 - pred_bin)).sum() / std_sum
    outside_mm = (cumulative_mm * (1.0 - pred_bin)).sum() / mm_sum
    return {
        "inside_pred_ratio_std": float(inside_std.item()),
        "inside_pred_ratio_mm": float(inside_mm.item()),
        "outside_pred_ratio_std": float(outside_std.item()),
        "outside_pred_ratio_mm": float(outside_mm.item()),
        "concentration_gain": float((inside_mm - inside_std).item()),
        "leakage_reduction": float((outside_std - outside_mm).item()),
    }


def save_comparison_grid(
    sample_dir: Path,
    input_image: Image.Image,
    pred_mask_img: Image.Image,
    final_image_cfg: Image.Image,
    final_image_mm: Image.Image,
    step_records: list[dict],
    save_indices: list[int],
    cumulative_std: torch.Tensor,
    cumulative_mm: torch.Tensor,
) -> None:
    step_panels: list[tuple[str, Image.Image]] = []
    for idx in save_indices:
        rec = step_records[idx]
        step_panels.append((f"STD t={rec['timestep']}", overlay_heatmap(input_image, rec["response_std"])))
        step_panels.append((f"MM t={rec['timestep']}", overlay_heatmap(input_image, rec["response_mm"])))

    panels: list[tuple[str, Image.Image]] = [
        ("Input", input_image),
        ("Pred Mask", pred_mask_img),
        ("Final CFG", final_image_cfg),
        ("Final MM-CFG", final_image_mm),
    ]
    panels.extend(step_panels)
    panels.extend(
        [
            ("Cum STD", overlay_heatmap(input_image, cumulative_std)),
            ("Cum MM", overlay_heatmap(input_image, cumulative_mm)),
        ]
    )

    cols = 4
    while len(panels) % cols != 0:
        panels.append(("", Image.new("RGB", input_image.size, color=(255, 255, 255))))

    tile_w, tile_h = input_image.size
    rows = len(panels) // cols
    grid = Image.new("RGB", (cols * tile_w, rows * (tile_h + 24)), color=(255, 255, 255))
    draw = ImageDraw.Draw(grid)
    for i, (title, panel) in enumerate(panels):
        row = i // cols
        col = i % cols
        x = col * tile_w
        y = row * (tile_h + 24)
        panel = panel.resize((tile_w, tile_h), RESAMPLE_BILINEAR)
        grid.paste(panel, (x, y + 24))
        if title:
            draw.text((x + 6, y + 4), title, fill=(0, 0, 0))
    grid.save(sample_dir / "comparison_grid.png")


def run_mmcfg_analysis(
    trainer: QwenImageEditTrainer,
    input_image: Image.Image,
    prompt: str,
    args: argparse.Namespace,
) -> dict:
    set_seed(args.seed)
    generator = torch.Generator(device=trainer.dit.device).manual_seed(args.seed)
    batch = trainer.prepare_predict_batch_data(
        image=input_image,
        prompt=prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.num_inference_steps,
        true_cfg_scale=args.true_cfg_scale,
        guidance_scale=args.guidance_scale,
        generator=generator,
    )
    embeddings = trainer.prepare_embeddings(batch, stage="predict")

    device = trainer.dit.device
    weight_dtype = trainer.weight_dtype

    num_inference_steps = embeddings["num_inference_steps"]
    true_cfg_scale = float(embeddings["true_cfg_scale"])
    control_latents = embeddings["control_latents"].to(device, dtype=weight_dtype)
    prompt_embeds = embeddings["prompt_embeds"].to(device, dtype=weight_dtype)
    prompt_embeds_mask = embeddings["prompt_embeds_mask"].to(device, dtype=torch.int64)
    img_shapes = embeddings["img_shapes"]
    height = int(embeddings["height"])
    width = int(embeddings["width"])
    guidance_scale = float(embeddings["guidance"])
    negative_prompt = embeddings.get("negative_prompt", None)

    if negative_prompt is None or true_cfg_scale <= 1.0:
        raise RuntimeError("Analysis requires true CFG with negative prompt and true_cfg_scale > 1.")

    negative_prompt_embeds = embeddings["negative_prompt_embeds"].to(device, dtype=weight_dtype)
    negative_prompt_embeds_mask = embeddings["negative_prompt_embeds_mask"].to(device, dtype=torch.int64)
    txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist()
    negative_txt_seq_lens = negative_prompt_embeds_mask.sum(dim=1).tolist()

    batch_size = control_latents.shape[0]
    if batch_size != 1:
        raise RuntimeError("This analysis script currently supports only batch_size=1.")

    num_channels_latents = trainer.dit.config.in_channels // 4
    height_latent = 2 * (height // (trainer.vae_scale_factor * 2))
    width_latent = 2 * (width // (trainer.vae_scale_factor * 2))

    if "latents" in embeddings:
        latents_init = embeddings["latents"].to(device, dtype=weight_dtype)
    else:
        shape = (batch_size, 1, num_channels_latents, height_latent, width_latent)
        latents_init = torch.randn(shape, generator=generator, device=device, dtype=weight_dtype)
        latents_init = trainer._pack_latents(latents_init, batch_size, num_channels_latents, height_latent, width_latent)

    timesteps, _ = trainer.prepare_predict_timesteps(
        num_inference_steps,
        latents_init.shape[1],
        scheduler=trainer.sampling_scheduler,
    )
    scheduler_cfg = copy.deepcopy(trainer.sampling_scheduler)
    scheduler_mm = copy.deepcopy(trainer.sampling_scheduler)
    scheduler_cfg.set_begin_index(0)
    scheduler_mm.set_begin_index(0)

    if trainer.dit.config.guidance_embeds:
        guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
    else:
        guidance = None

    step_records: list[dict] = []
    cumulative_std = None
    cumulative_mm = None
    final_pred_mask_2d = None
    final_pred_mask_img = None
    latents_cfg = latents_init.clone()
    latents_mm = latents_init.clone()

    schedule_ts = scheduler_mm.timesteps.detach().cpu().tolist()
    step_index_map = {int(v): j for j, v in enumerate(schedule_ts)}
    schedule_sigmas = scheduler_mm.sigmas.to(device=device, dtype=weight_dtype)

    bg_scale_cfg = float(getattr(trainer.config.model, "mg_cfg_background_scale", 1.0))
    bg_scale_cfg = min(max(bg_scale_cfg, 1.0), true_cfg_scale)

    start_ratio = float(getattr(trainer.config.model, "mg_cfg_start_ratio", 0.0))
    start_ratio = min(max(start_ratio, 0.0), 1.0)
    start_step = int(round(start_ratio * len(timesteps)))

    dilate_radius = int(getattr(trainer.config.model, "mg_cfg_mask_dilate_radius", 0))
    blur_kernel = int(getattr(trainer.config.model, "mg_cfg_mask_blur_kernel", 0))
    dilate_radius = max(dilate_radius, 0)
    blur_kernel = max(blur_kernel, 0)
    if blur_kernel > 0 and blur_kernel % 2 == 0:
        blur_kernel += 1

    with torch.inference_mode():
        for i, t in enumerate(timesteps):
            latents_cfg = latents_cfg.to(device, dtype=weight_dtype)
            latents_mm = latents_mm.to(device, dtype=weight_dtype)
            timestep_cfg = t.expand(latents_cfg.shape[0]).to(latents_cfg.dtype)
            timestep_mm = t.expand(latents_mm.shape[0]).to(latents_mm.dtype)
            latent_model_input_cfg = torch.cat([latents_cfg, control_latents], dim=1)
            latent_model_input_mm = torch.cat([latents_mm, control_latents], dim=1)

            with trainer.dit.cache_context("cond"):
                noise_pred_cfg = trainer.dit(
                    hidden_states=latent_model_input_cfg,
                    timestep=timestep_cfg / 1000,
                    guidance=guidance,
                    encoder_hidden_states_mask=prompt_embeds_mask,
                    encoder_hidden_states=prompt_embeds,
                    img_shapes=img_shapes,
                    txt_seq_lens=txt_seq_lens,
                    attention_kwargs={},
                    return_dict=False,
                )[0]
                noise_pred_cfg = noise_pred_cfg[:, : latents_cfg.size(1)]

            with trainer.dit.cache_context("cond"):
                noise_pred_mm = trainer.dit(
                    hidden_states=latent_model_input_mm,
                    timestep=timestep_mm / 1000,
                    guidance=guidance,
                    encoder_hidden_states_mask=prompt_embeds_mask,
                    encoder_hidden_states=prompt_embeds,
                    img_shapes=img_shapes,
                    txt_seq_lens=txt_seq_lens,
                    attention_kwargs={},
                    return_dict=False,
                )[0]
                noise_pred_mm = noise_pred_mm[:, : latents_mm.size(1)]

            step_index = step_index_map[int(t.item())]
            sigma_t = schedule_sigmas[step_index].flatten().to(dtype=latents_mm.dtype)
            while len(sigma_t.shape) < latents_mm.ndim:
                sigma_t = sigma_t.unsqueeze(-1)

            pre_x0_mm = latents_mm - sigma_t * noise_pred_mm
            edit_mask_input = pre_x0_mm - control_latents
            pred_logits = trainer.predict_edit_mask(edit_mask_input.detach(), img_shapes)
            pred_mask = torch.sigmoid(pred_logits)

            if bool(getattr(trainer.config.model, "invert_predicted_mask", False)):
                pred_mask = 1.0 - pred_mask

            if pred_mask.shape[-2:] != (height_latent, width_latent):
                pred_mask = F.interpolate(pred_mask, size=(height_latent, width_latent), mode="nearest")

            predicted_mask_seq = None
            pred_mask_2d = None
            if i >= start_step:
                pred_mask_2d = pred_mask
                if pred_mask_2d.ndim == 3:
                    pred_mask_2d = pred_mask_2d.unsqueeze(1)
                if pred_mask_2d.shape[1] != 1:
                    pred_mask_2d = pred_mask_2d[:, :1]

                if dilate_radius > 0:
                    kernel = 2 * dilate_radius + 1
                    pred_mask_2d = F.max_pool2d(pred_mask_2d, kernel_size=kernel, stride=1, padding=dilate_radius)

                if blur_kernel > 1:
                    pad = blur_kernel // 2
                    pred_mask_2d = F.avg_pool2d(pred_mask_2d, kernel_size=blur_kernel, stride=1, padding=pad)

                pred_mask_2d = pred_mask_2d.clamp_(0.0, 1.0)
                predicted_mask_seq = trainer._pack_mask_latent(pred_mask_2d).to(device)
                if predicted_mask_seq.shape[1] != latents_mm.shape[1]:
                    predicted_mask_seq = None
                    pred_mask_2d = None

            with trainer.dit.cache_context("uncond"):
                neg_noise_pred_cfg = trainer.dit(
                    hidden_states=latent_model_input_cfg,
                    timestep=timestep_cfg / 1000,
                    guidance=guidance,
                    encoder_hidden_states_mask=negative_prompt_embeds_mask,
                    encoder_hidden_states=negative_prompt_embeds,
                    img_shapes=img_shapes,
                    txt_seq_lens=negative_txt_seq_lens,
                    attention_kwargs={},
                    return_dict=False,
                )[0]
                neg_noise_pred_cfg = neg_noise_pred_cfg[:, : latents_cfg.size(1)]

            with trainer.dit.cache_context("uncond"):
                neg_noise_pred_mm = trainer.dit(
                    hidden_states=latent_model_input_mm,
                    timestep=timestep_mm / 1000,
                    guidance=guidance,
                    encoder_hidden_states_mask=negative_prompt_embeds_mask,
                    encoder_hidden_states=negative_prompt_embeds,
                    img_shapes=img_shapes,
                    txt_seq_lens=negative_txt_seq_lens,
                    attention_kwargs={},
                    return_dict=False,
                )[0]
                neg_noise_pred_mm = neg_noise_pred_mm[:, : latents_mm.size(1)]

            delta_cfg = noise_pred_cfg - neg_noise_pred_cfg
            delta_mm = noise_pred_mm - neg_noise_pred_mm
            response_std_token = compute_response_token(true_cfg_scale * delta_cfg)

            if predicted_mask_seq is not None:
                scale_seq = bg_scale_cfg + (true_cfg_scale - bg_scale_cfg) * predicted_mask_seq
                response_mm_token = compute_response_token(scale_seq.unsqueeze(-1) * delta_mm)
            else:
                scale_seq = torch.full_like(response_std_token, true_cfg_scale)
                response_mm_token = compute_response_token(true_cfg_scale * delta_mm)

            if pred_mask_2d is None:
                pred_mask_2d_vis = torch.zeros((1, 1, height_latent, width_latent), device=device, dtype=torch.float32)
            else:
                pred_mask_2d_vis = pred_mask_2d.float()

            response_std_img = token_response_to_image(response_std_token, pred_mask_2d_vis, (height, width))[0, 0]
            response_mm_img = token_response_to_image(response_mm_token, pred_mask_2d_vis, (height, width))[0, 0]
            pred_mask_img = F.interpolate(
                pred_mask_2d_vis.float(),
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )[0, 0]

            if cumulative_std is None:
                cumulative_std = response_std_img.clone()
                cumulative_mm = response_mm_img.clone()
            else:
                cumulative_std += response_std_img
                cumulative_mm += response_mm_img

            step_records.append(
                {
                    "step_idx": i,
                    "timestep": int(t.item()),
                    "response_std": response_std_img.detach().cpu(),
                    "response_mm": response_mm_img.detach().cpu(),
                    "pred_mask": pred_mask_img.detach().cpu(),
                }
            )

            if predicted_mask_seq is not None:
                comb_pred_mm = neg_noise_pred_mm + scale_seq.unsqueeze(-1) * delta_mm
                final_pred_mask_2d = pred_mask_2d_vis.detach().cpu()
                final_pred_mask_img = pred_mask_img.detach().cpu()
            else:
                comb_pred_mm = neg_noise_pred_mm + true_cfg_scale * delta_mm

            comb_pred_cfg = neg_noise_pred_cfg + true_cfg_scale * delta_cfg

            cond_norm_cfg = torch.norm(noise_pred_cfg, dim=-1, keepdim=True)
            noise_norm_cfg = torch.norm(comb_pred_cfg, dim=-1, keepdim=True).clamp(min=1.0e-8)
            noise_pred_final_cfg = comb_pred_cfg * (cond_norm_cfg / noise_norm_cfg)

            cond_norm_mm = torch.norm(noise_pred_mm, dim=-1, keepdim=True)
            noise_norm_mm = torch.norm(comb_pred_mm, dim=-1, keepdim=True).clamp(min=1.0e-8)
            noise_pred_final_mm = comb_pred_mm * (cond_norm_mm / noise_norm_mm)

            latents_cfg_dtype = latents_cfg.dtype
            latents_mm_dtype = latents_mm.dtype
            latents_cfg = scheduler_cfg.step(noise_pred_final_cfg, t, latents_cfg, return_dict=False)[0]
            latents_mm = scheduler_mm.step(noise_pred_final_mm, t, latents_mm, return_dict=False)[0]
            if latents_cfg.dtype != latents_cfg_dtype:
                latents_cfg = latents_cfg.to(latents_cfg_dtype)
            if latents_mm.dtype != latents_mm_dtype:
                latents_mm = latents_mm.to(latents_mm_dtype)

    final_image_tensor_cfg = trainer.decode_vae_latent(latents_cfg, height, width)
    final_image_cfg = tensor_image_to_pil(final_image_tensor_cfg)
    final_image_tensor_mm = trainer.decode_vae_latent(latents_mm, height, width)
    final_image_mm = tensor_image_to_pil(final_image_tensor_mm)

    if final_pred_mask_2d is None:
        final_pred_mask_2d = torch.zeros((1, 1, height_latent, width_latent), dtype=torch.float32)
        final_pred_mask_img = torch.zeros((height, width), dtype=torch.float32)

    return {
        "final_latents_cfg": latents_cfg.detach().cpu(),
        "final_latents_mm": latents_mm.detach().cpu(),
        "final_image_cfg": final_image_cfg,
        "final_image_mm": final_image_mm,
        "pred_mask_2d": final_pred_mask_2d,
        "pred_mask_img": final_pred_mask_img,
        "step_records": step_records,
        "cumulative_std": cumulative_std.detach().cpu(),
        "cumulative_mm": cumulative_mm.detach().cpu(),
    }


def analyze_single_sample(
    trainer: QwenImageEditTrainer,
    image_path: Path,
    args: argparse.Namespace,
) -> dict[str, float]:
    input_image = load_input_image(image_path)
    sample_name = image_path.stem
    sample_dir = args.output_dir / sample_name
    sample_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = sample_dir / "metrics.json"

    if metrics_path.exists() and not args.overwrite:
        print(f"[MM-CFG Analysis] Skip existing sample: {sample_name}")
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        return metrics

    result = run_mmcfg_analysis(trainer, input_image, args.prompt, args)
    pred_mask_img_pil = mask_to_image(result["pred_mask_img"], input_image.size)
    mask_overlay_pil = overlay_mask_outline(input_image, result["pred_mask_img"])
    delta_map_pil = delta_map_to_image(result["cumulative_mm"] - result["cumulative_std"], input_image.size)
    roi_box = crop_box_from_mask(result["pred_mask_img"], input_image.size)
    input_image.save(sample_dir / "input.png")
    pred_mask_img_pil.save(sample_dir / "pred_mask.png")
    mask_overlay_pil.save(sample_dir / "mask_overlay.png")
    result["final_image_cfg"].save(sample_dir / "final_output_cfg.png")
    result["final_image_mm"].save(sample_dir / "final_output_mmcfg.png")
    delta_map_pil.save(sample_dir / "delta_map.png")

    save_indices = parse_step_spec(args.save_steps, len(result["step_records"]))
    for idx in save_indices:
        rec = result["step_records"][idx]
        overlay_heatmap(input_image, rec["response_std"]).save(sample_dir / f"step_{idx:03d}_std.png")
        overlay_heatmap(input_image, rec["response_mm"]).save(sample_dir / f"step_{idx:03d}_mm.png")

    overlay_heatmap(input_image, result["cumulative_std"]).save(sample_dir / "cumulative_std.png")
    overlay_heatmap(input_image, result["cumulative_mm"]).save(sample_dir / "cumulative_mm.png")
    save_roi_zoom_grid(
        sample_dir=sample_dir,
        input_image=input_image,
        final_image_cfg=result["final_image_cfg"],
        final_image_mm=result["final_image_mm"],
        delta_image=delta_map_pil,
        crop_box=roi_box,
    )

    save_comparison_grid(
        sample_dir=sample_dir,
        input_image=input_image,
        pred_mask_img=pred_mask_img_pil,
        final_image_cfg=result["final_image_cfg"],
        final_image_mm=result["final_image_mm"],
        step_records=result["step_records"],
        save_indices=save_indices,
        cumulative_std=result["cumulative_std"],
        cumulative_mm=result["cumulative_mm"],
    )

    metrics = compute_predmask_region_metrics(
        cumulative_std=result["cumulative_std"],
        cumulative_mm=result["cumulative_mm"],
        pred_mask_img=result["pred_mask_img"],
    )
    metrics["sample"] = sample_name

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    return metrics


def write_summary_csv(output_dir: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(output_dir / "summary_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer = load_trainer_and_weights(args)
    samples = collect_samples(args)

    if not samples:
        raise RuntimeError("No input images found.")

    all_metrics: list[dict[str, float]] = []
    for image_path in samples:
        print(f"[MM-CFG Analysis] Processing: {image_path}")
        metrics = analyze_single_sample(trainer, image_path, args)
        all_metrics.append(metrics)

    write_summary_csv(args.output_dir, all_metrics)
    print(f"[MM-CFG Analysis] Done. Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
