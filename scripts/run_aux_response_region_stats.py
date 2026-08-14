#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))

from qflux.data.config import load_config_from_yaml
from qflux.trainer.qwen_image_edit_trainer import QwenImageEditTrainer

# Reuse the already-debugged MM-CFG analysis implementation.
from analyze_mmcfg_response import run_mmcfg_analysis


try:
    RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
    RESAMPLE_NEAREST = Image.Resampling.NEAREST
except AttributeError:
    RESAMPLE_LANCZOS = Image.LANCZOS
    RESAMPLE_NEAREST = Image.NEAREST


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

DEFAULT_IMAGE_ROOTS = {
    "humanparsing_editing_resize": Path("/mount/data/zsm/all_dataset/0_fame/aux_data_nanogen/humanparsing_editing_resize"),
    "test_slider": Path("/mount/data/zsm/all_dataset/0_fame/aux_data_nanogen/test_slider"),
}

DEFAULT_MASK_ROOTS = {
    "humanparsing_editing_resize": Path("/mount/data/zsm/all_dataset/0_fame/aux_data_nanogen/humanparsing_editing_resize_mask_diff_fnal"),
    "test_slider": Path("/mount/data/zsm/all_dataset/0_fame/aux_data_nanogen/test_slider_mask_diff_fnal"),
}


@dataclass(frozen=True)
class SampleSpec:
    dataset: str
    attribute: str
    transform: str
    rel_path: str
    image_path: Path
    mask_path: Path
    prompt: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute cumulative response-map region statistics on aux_data_nanogen."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/7_qwen_lora_attentionloss_f9b1_mask_retrain_ori.yaml",
        help="Config of the mask-prediction model used for D/E analysis.",
    )
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=Path("/mount/data/zsm/all_dataset/0_fame/qwen-finetune/project/outputs/7_qwen_lora_attentionloss_f9b1_mask_update/v1/checkpoint-39-5770"),
        help="Checkpoint directory containing LoRA weights and mask_prediction_head.pt.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="humanparsing_editing_resize,test_slider",
        help="Comma-separated aux dataset names.",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("/mount/data/zsm/test_results/aux_response_region_stats"),
        help="Directory for per-sample and summary CSV outputs.",
    )
    parser.add_argument("--max_samples_per_leaf", type=int, default=0)
    parser.add_argument("--max_total_samples", type=int, default=0)
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--true_cfg_scale", type=float, default=4.0)
    parser.add_argument("--guidance_scale", type=float, default=4.0)
    parser.add_argument("--negative_prompt", type=str, default=" ")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--mask_threshold", type=float, default=0.5, help="Threshold on [0,1] masks.")
    parser.add_argument(
        "--save_visuals",
        action="store_true",
        help="Save cumulative response overlays and mask overlays per sample.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute sample outputs even if cached JSON exists.",
    )
    return parser.parse_args()


def parse_csv_list(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def normalize_text_token(token: str) -> str:
    return token.replace("_", " ").strip()


def infer_prompt(attribute: str, transform: str) -> str:
    exact_map = {
        ("top_length", "short2long"): "Change top length from short to long",
        ("dress_length", "short2long"): "Change dress length to long",
        ("pant_length", "short2long"): "Change pants length to long",
        ("sleeve_length", "short2long"): "Change sleeve length to long",
        ("sleeve_shape", "regular2bell"): "Change sleeve shape to bell sleeve",
        ("sleeve_shape", "regular2cape"): "Change sleeve shape to cape sleeve",
        ("sleeve_shape", "regular2cap"): "Change sleeve shape to cap sleeve",
        ("sleeve_shape", "regular2leg of mutton"): "Change sleeve shape to leg of mutton sleeve",
        ("collar", "r2cami-off_shoulder"): "Change neckline to cami-off shoulder neckline",
        ("collar", "r2choker"): "Change neckline to choker neckline",
        ("collar", "r2hoodie"): "Change neckline to hoodie collar",
        ("collar", "r2one-shoulder"): "Change neckline to one-shoulder neckline",
        ("collar", "r2peter pan"): "Change neckline to peter pan collar",
        ("collar", "r2shirt"): "Change neckline to shirt collar",
        ("collar", "r2square"): "Change neckline to square neckline",
        ("collar", "r2turtle"): "Change neckline to turtle collar",
        ("collar", "r2v"): "Change neckline to v neckline",
    }
    if (attribute, transform) in exact_map:
        return exact_map[(attribute, transform)]

    if "2" not in transform:
        raise ValueError(f"Cannot infer prompt from transform: {attribute}/{transform}")

    src_token, dst_token = transform.split("2", 1)
    src_text = normalize_text_token(src_token)
    dst_text = normalize_text_token(dst_token)

    if attribute in {"top_length", "dress_length", "pant_length", "sleeve_length"}:
        label_map = {
            "top_length": "top length",
            "dress_length": "dress length",
            "pant_length": "pants length",
            "sleeve_length": "sleeve length",
        }
        return f"Change {label_map[attribute]} from {src_text} to {dst_text}"
    if attribute == "sleeve_shape":
        return f"Change sleeve shape from {src_text} to {dst_text} sleeve"
    if attribute == "collar":
        return f"Change neckline from {src_text} to {dst_text}"
    raise ValueError(f"Unsupported attribute: {attribute}")


def collect_samples(args: argparse.Namespace) -> list[SampleSpec]:
    selected = parse_csv_list(args.datasets)
    samples: list[SampleSpec] = []
    per_leaf_counter: dict[tuple[str, str], int] = {}

    for dataset_name in selected:
        if dataset_name not in DEFAULT_IMAGE_ROOTS or dataset_name not in DEFAULT_MASK_ROOTS:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        image_root = DEFAULT_IMAGE_ROOTS[dataset_name]
        mask_root = DEFAULT_MASK_ROOTS[dataset_name]

        for mask_path in sorted(mask_root.rglob("*")):
            if not mask_path.is_file() or mask_path.suffix.lower() not in IMAGE_EXTS:
                continue
            rel_path = mask_path.relative_to(mask_root)
            if len(rel_path.parts) < 3:
                continue
            image_path = image_root / rel_path
            if not image_path.exists():
                continue
            attribute = rel_path.parts[0]
            transform = rel_path.parts[1]
            leaf_key = (dataset_name, f"{attribute}/{transform}")
            if args.max_samples_per_leaf > 0 and per_leaf_counter.get(leaf_key, 0) >= args.max_samples_per_leaf:
                continue
            samples.append(
                SampleSpec(
                    dataset=dataset_name,
                    attribute=attribute,
                    transform=transform,
                    rel_path=rel_path.as_posix(),
                    image_path=image_path,
                    mask_path=mask_path,
                    prompt=infer_prompt(attribute, transform),
                )
            )
            per_leaf_counter[leaf_key] = per_leaf_counter.get(leaf_key, 0) + 1
            if args.max_total_samples > 0 and len(samples) >= args.max_total_samples:
                return samples
    return samples


def split_twopanel_image(image: Image.Image) -> tuple[Image.Image, Image.Image]:
    width, height = image.size
    mid = width // 2
    return image.crop((0, 0, mid, height)), image.crop((mid, 0, width, height))


def load_source_and_reference_mask(sample: SampleSpec, threshold: float) -> tuple[Image.Image, np.ndarray]:
    twopanel = Image.open(sample.image_path).convert("RGB")
    source_image, _ = split_twopanel_image(twopanel)
    mask_image = Image.open(sample.mask_path).convert("L")
    if mask_image.size != source_image.size:
        mask_image = mask_image.resize(source_image.size, RESAMPLE_NEAREST)
    mask_np = np.asarray(mask_image, dtype=np.float32) / 255.0
    mask_bin = (mask_np >= threshold).astype(np.float32)
    return source_image, mask_bin


def load_trainer(args: argparse.Namespace) -> QwenImageEditTrainer:
    config = load_config_from_yaml(str(args.config))
    config.cache.use_cache = False
    config.model.use_mask_prediction = True
    if args.device is not None:
        config.predict.devices.dit = args.device
        config.predict.devices.vae = args.device
        config.predict.devices.text_encoder = args.device

    lora_path = args.ckpt / "pytorch_lora_weights.safetensors"
    mask_head_path = args.ckpt / "mask_prediction_head.pt"
    if not lora_path.exists():
        raise FileNotFoundError(f"Missing LoRA weights: {lora_path}")
    if not mask_head_path.exists():
        raise FileNotFoundError(f"Missing mask head: {mask_head_path}")

    config.model.lora.pretrained_weight = str(lora_path)
    trainer = QwenImageEditTrainer(config)
    trainer.setup_predict()
    if trainer.mask_prediction_head is None:
        raise RuntimeError("Mask prediction head is not initialized.")

    mask_state = torch.load(mask_head_path, map_location="cpu")
    trainer.mask_prediction_head.load_state_dict(mask_state, strict=True)
    trainer.mask_prediction_head.to(device=trainer.dit.device, dtype=trainer.weight_dtype)
    trainer.mask_prediction_head.eval()
    return trainer


def mask_ratio(response_map: torch.Tensor, mask_np: np.ndarray) -> float:
    response = response_map.detach().float().cpu().numpy()
    if response.shape != mask_np.shape:
        mask_img = Image.fromarray((mask_np * 255).astype(np.uint8), mode="L")
        mask_img = mask_img.resize((response.shape[1], response.shape[0]), RESAMPLE_NEAREST)
        mask_np = np.asarray(mask_img, dtype=np.float32) / 255.0
        mask_np = (mask_np >= 0.5).astype(np.float32)
    total = float(response.sum())
    if total <= 1.0e-8:
        return math.nan
    return float((response * mask_np).sum() / total)


def compute_region_metrics(
    cumulative_std: torch.Tensor,
    cumulative_mm: torch.Tensor,
    pred_mask_img: torch.Tensor,
    ref_mask_np: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    pred_mask_np = pred_mask_img.detach().float().cpu().numpy()
    pred_bin = (pred_mask_np >= threshold).astype(np.float32)
    ref_bin = ref_mask_np.astype(np.float32)

    inside_pred_std = mask_ratio(cumulative_std, pred_bin)
    inside_pred_mm = mask_ratio(cumulative_mm, pred_bin)
    inside_ref_std = mask_ratio(cumulative_std, ref_bin)
    inside_ref_mm = mask_ratio(cumulative_mm, ref_bin)

    metrics = {
        "inside_pred_ratio_std": inside_pred_std,
        "inside_pred_ratio_mm": inside_pred_mm,
        "outside_pred_ratio_std": (1.0 - inside_pred_std) if not math.isnan(inside_pred_std) else math.nan,
        "outside_pred_ratio_mm": (1.0 - inside_pred_mm) if not math.isnan(inside_pred_mm) else math.nan,
        "inside_gain_pred": (inside_pred_mm - inside_pred_std) if not (math.isnan(inside_pred_std) or math.isnan(inside_pred_mm)) else math.nan,
        "outside_reduction_pred": ((1.0 - inside_pred_std) - (1.0 - inside_pred_mm)) if not (math.isnan(inside_pred_std) or math.isnan(inside_pred_mm)) else math.nan,
        "inside_ref_ratio_std": inside_ref_std,
        "inside_ref_ratio_mm": inside_ref_mm,
        "outside_ref_ratio_std": (1.0 - inside_ref_std) if not math.isnan(inside_ref_std) else math.nan,
        "outside_ref_ratio_mm": (1.0 - inside_ref_mm) if not math.isnan(inside_ref_mm) else math.nan,
        "inside_gain_ref": (inside_ref_mm - inside_ref_std) if not (math.isnan(inside_ref_std) or math.isnan(inside_ref_mm)) else math.nan,
        "outside_reduction_ref": ((1.0 - inside_ref_std) - (1.0 - inside_ref_mm)) if not (math.isnan(inside_ref_std) or math.isnan(inside_ref_mm)) else math.nan,
    }
    return metrics


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: dict) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    ensure_parent(path)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_rows(rows: list[dict], group_keys: list[str], metric_keys: list[str]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        group = tuple(row[key] for key in group_keys)
        grouped.setdefault(group, []).append(row)

    summary_rows: list[dict] = []
    for group, group_rows in sorted(grouped.items()):
        out = {key: value for key, value in zip(group_keys, group)}
        out["num_samples"] = len(group_rows)
        for metric in metric_keys:
            values = [float(r[metric]) for r in group_rows if metric in r and not math.isnan(float(r[metric]))]
            out[metric] = float(sum(values) / len(values)) if values else math.nan
        summary_rows.append(out)
    return summary_rows


def maybe_save_visuals(sample_dir: Path, source_image: Image.Image, result: dict, ref_mask_np: np.ndarray) -> None:
    from analyze_mmcfg_response import overlay_heatmap, overlay_mask_outline, mask_to_image, delta_map_to_image

    sample_dir.mkdir(parents=True, exist_ok=True)
    ref_mask_img = Image.fromarray((ref_mask_np * 255).astype(np.uint8), mode="L")
    if ref_mask_img.size != source_image.size:
        ref_mask_img = ref_mask_img.resize(source_image.size, RESAMPLE_NEAREST)

    overlay_heatmap(source_image, result["cumulative_std"]).save(sample_dir / "cumulative_std_ref.png")
    overlay_heatmap(source_image, result["cumulative_mm"]).save(sample_dir / "cumulative_mm_ref.png")
    delta_map_to_image(result["cumulative_mm"] - result["cumulative_std"], source_image.size).save(sample_dir / "delta_map_ref.png")
    overlay_mask_outline(source_image, torch.from_numpy(ref_mask_np), threshold=0.5, color=(255, 64, 64)).save(sample_dir / "reference_mask_overlay.png")
    mask_to_image(torch.from_numpy(ref_mask_np), source_image.size).save(sample_dir / "reference_mask.png")
    if result.get("pred_mask_img") is not None:
        overlay_mask_outline(source_image, result["pred_mask_img"], threshold=0.5, color=(64, 160, 255)).save(sample_dir / "predicted_mask_overlay.png")


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    samples = collect_samples(args)
    if not samples:
        raise RuntimeError("No aux samples were collected.")

    trainer: QwenImageEditTrainer | None = None
    rows: list[dict] = []
    metric_keys = [
        "inside_pred_ratio_std",
        "inside_pred_ratio_mm",
        "outside_pred_ratio_std",
        "outside_pred_ratio_mm",
        "inside_gain_pred",
        "outside_reduction_pred",
        "inside_ref_ratio_std",
        "inside_ref_ratio_mm",
        "outside_ref_ratio_std",
        "outside_ref_ratio_mm",
        "inside_gain_ref",
        "outside_reduction_ref",
    ]

    try:
        trainer = load_trainer(args)
        for index, sample in enumerate(samples, start=1):
            source_image, ref_mask_np = load_source_and_reference_mask(sample, args.mask_threshold)
            sample_dir = args.output_root / sample.dataset / sample.attribute / sample.transform / Path(sample.rel_path).stem
            metrics_json = sample_dir / "response_metrics.json"

            if metrics_json.exists() and not args.overwrite:
                payload = json.loads(metrics_json.read_text(encoding="utf-8"))
                row = payload["metrics"]
            else:
                result = run_mmcfg_analysis(trainer, source_image, sample.prompt, args)
                metrics = compute_region_metrics(
                    cumulative_std=result["cumulative_std"],
                    cumulative_mm=result["cumulative_mm"],
                    pred_mask_img=result["pred_mask_img"],
                    ref_mask_np=ref_mask_np,
                    threshold=args.mask_threshold,
                )
                row = {
                    "dataset": sample.dataset,
                    "attribute": sample.attribute,
                    "transform": sample.transform,
                    "rel_path": sample.rel_path,
                    "image_path": str(sample.image_path),
                    "mask_path": str(sample.mask_path),
                    "prompt": sample.prompt,
                }
                row.update(metrics)
                save_json(metrics_json, {"sample": row, "metrics": row})
                if args.save_visuals:
                    maybe_save_visuals(sample_dir, source_image, result, ref_mask_np)

            rows.append(row)
            print(f"[AuxResponse] {index}/{len(samples)} {sample.dataset}/{sample.rel_path}")
    finally:
        if trainer is not None:
            del trainer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    write_csv(args.output_root / "response_metrics_per_sample.csv", rows)
    write_csv(args.output_root / "response_metrics_summary_by_dataset.csv", summarize_rows(rows, ["dataset"], metric_keys))
    write_csv(args.output_root / "response_metrics_summary_by_attribute.csv", summarize_rows(rows, ["attribute"], metric_keys))
    write_csv(args.output_root / "response_metrics_summary_by_dataset_attribute.csv", summarize_rows(rows, ["dataset", "attribute"], metric_keys))

    run_config = {
        "config": str(args.config),
        "ckpt": str(args.ckpt),
        "datasets": parse_csv_list(args.datasets),
        "num_inference_steps": args.num_inference_steps,
        "true_cfg_scale": args.true_cfg_scale,
        "guidance_scale": args.guidance_scale,
        "negative_prompt": args.negative_prompt,
        "seed": args.seed,
        "device": args.device,
        "mask_threshold": args.mask_threshold,
        "max_samples_per_leaf": args.max_samples_per_leaf,
        "max_total_samples": args.max_total_samples,
    }
    (args.output_root / "run_config.json").write_text(json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[Done] samples={len(rows)}, output_root={args.output_root}")


if __name__ == "__main__":
    main()
