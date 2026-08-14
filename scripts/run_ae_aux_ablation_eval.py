#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Add src/ to Python path BEFORE any qflux imports.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))

from qflux.data.config import load_config_from_yaml
from qflux.trainer.qwen_image_edit_trainer import QwenImageEditTrainer
from qflux.utils.seed import seed_everything


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

DEFAULT_OUTPUT_ROOT = Path("/mount/data/zsm/test_results/aux_ae_ablation")


@dataclass(frozen=True)
class VariantSpec:
    name: str
    config_path: Path
    checkpoint_dir: Path | None
    use_mask_prediction: bool
    mg_cfg_background_scale: float | None = None
    mg_cfg_mask_dilate_radius: int | None = None
    mg_cfg_mask_blur_kernel: int | None = None
    mg_cfg_start_ratio: float | None = None


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
        description="Run A-E ablation generation and mask-aware evaluation on aux_data_nanogen."
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="humanparsing_editing_resize,test_slider",
        help="Comma-separated dataset names. Default uses both aux datasets.",
    )
    parser.add_argument(
        "--variants",
        type=str,
        default="A,B,C,D,E",
        help="Comma-separated variant names to run.",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for generated results and CSV summaries.",
    )
    parser.add_argument(
        "--max_samples_per_leaf",
        type=int,
        default=0,
        help="Limit samples per attribute/transform leaf. 0 means all.",
    )
    parser.add_argument(
        "--max_total_samples",
        type=int,
        default=0,
        help="Global limit across all selected datasets. 0 means all.",
    )
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--true_cfg_scale", type=float, default=4.0)
    parser.add_argument("--guidance_scale", type=float, default=4.0)
    parser.add_argument("--negative_prompt", type=str, default=" ")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None, help="Override predict device, e.g. cuda:0")
    parser.add_argument(
        "--mask_threshold",
        type=int,
        default=8,
        help="Threshold for binarizing reference masks in [0, 255].",
    )
    parser.add_argument(
        "--save_panels",
        action="store_true",
        help="Save 4-panel visualization: source | result | reference target | reference mask.",
    )
    parser.add_argument(
        "--save_pred_masks",
        action="store_true",
        help="Save predicted masks for variants that use MM-CFG.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-generate outputs even if result images already exist.",
    )
    return parser.parse_args()


def build_variant_specs(true_cfg_scale: float) -> dict[str, VariantSpec]:
    cfg0 = PROJECT_ROOT / "configs/0_qwen_lora.yaml"
    cfg4 = PROJECT_ROOT / "configs/4_qwen_lora_attentionloss_f9b1_2.yaml"
    cfg7 = PROJECT_ROOT / "configs/7_qwen_lora_attentionloss_f9b1_mask_retrain_ori.yaml"
    ckpt_b = Path("/mount/data/zsm/all_dataset/0_fame/qwen-finetune/project/outputs/0_qwen_lora/v0/checkpoint-35-5193")
    ckpt_c = Path("/mount/data/zsm/all_dataset/0_fame/qwen-finetune/project/outputs/4_qwen_lora_attentionloss_f9b1_2/v2/checkpoint-39-5770")
    ckpt_de = Path("/mount/data/zsm/all_dataset/0_fame/qwen-finetune/project/outputs/7_qwen_lora_attentionloss_f9b1_mask_update/v1/checkpoint-39-5770")
    return {
        "A": VariantSpec("A", cfg0, None, False),
        "B": VariantSpec("B", cfg0, ckpt_b, False),
        "C": VariantSpec("C", cfg4, ckpt_c, False),
        "D": VariantSpec("D", cfg7, ckpt_de, False),
        "E": VariantSpec("E", cfg7, ckpt_de, True, 2.0, 1, 5, 0.1),
    }


def parse_csv_list(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def resolve_selected_datasets(names: list[str]) -> dict[str, tuple[Path, Path]]:
    resolved: dict[str, tuple[Path, Path]] = {}
    for name in names:
        if name not in DEFAULT_IMAGE_ROOTS or name not in DEFAULT_MASK_ROOTS:
            raise ValueError(f"Unsupported dataset: {name}")
        resolved[name] = (DEFAULT_IMAGE_ROOTS[name], DEFAULT_MASK_ROOTS[name])
    return resolved


def split_twopanel_image(image: Image.Image) -> tuple[Image.Image, Image.Image]:
    width, height = image.size
    if width % 2 != 0:
        raise ValueError(f"Two-panel image width must be even, got {width}")
    mid = width // 2
    left = image.crop((0, 0, mid, height))
    right = image.crop((mid, 0, width, height))
    return left, right


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


def collect_samples(
    dataset_specs: dict[str, tuple[Path, Path]],
    max_samples_per_leaf: int,
    max_total_samples: int,
) -> list[SampleSpec]:
    samples: list[SampleSpec] = []
    for dataset_name, (image_root, mask_root) in dataset_specs.items():
        per_leaf_counter: dict[tuple[str, str], int] = {}
        for mask_path in sorted(mask_root.rglob("*")):
            if not mask_path.is_file() or mask_path.suffix.lower() not in IMAGE_EXTS:
                continue

            rel_path = mask_path.relative_to(mask_root)
            if len(rel_path.parts) < 3:
                continue

            attribute = rel_path.parts[0]
            transform = rel_path.parts[1]
            image_path = image_root / rel_path
            if not image_path.exists():
                print(f"[Skip] Missing paired image for mask: {mask_path}")
                continue

            leaf_key = (dataset_name, f"{attribute}/{transform}")
            if max_samples_per_leaf > 0 and per_leaf_counter.get(leaf_key, 0) >= max_samples_per_leaf:
                continue

            prompt = infer_prompt(attribute, transform)
            samples.append(
                SampleSpec(
                    dataset=dataset_name,
                    attribute=attribute,
                    transform=transform,
                    rel_path=rel_path.as_posix(),
                    image_path=image_path,
                    mask_path=mask_path,
                    prompt=prompt,
                )
            )
            per_leaf_counter[leaf_key] = per_leaf_counter.get(leaf_key, 0) + 1

            if max_total_samples > 0 and len(samples) >= max_total_samples:
                return samples
    return samples


def load_source_target_mask(sample: SampleSpec, mask_threshold: int) -> tuple[Image.Image, Image.Image, np.ndarray]:
    twopanel = Image.open(sample.image_path).convert("RGB")
    source_image, target_image = split_twopanel_image(twopanel)
    mask_image = Image.open(sample.mask_path).convert("L")
    if mask_image.size != source_image.size:
        mask_image = mask_image.resize(source_image.size, RESAMPLE_NEAREST)
    mask_np = np.array(mask_image, dtype=np.uint8)
    mask_bin = (mask_np >= mask_threshold).astype(np.float32)
    return source_image, target_image, mask_bin


def setup_seed(seed: int) -> None:
    seed_everything(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_variant_trainer(spec: VariantSpec, args: argparse.Namespace) -> QwenImageEditTrainer:
    config = load_config_from_yaml(str(spec.config_path))
    config.cache.use_cache = False
    config.model.use_mask_prediction = spec.use_mask_prediction

    if args.device is not None:
        config.predict.devices.dit = args.device
        config.predict.devices.vae = args.device
        config.predict.devices.text_encoder = args.device

    if spec.checkpoint_dir is None:
        config.model.lora.pretrained_weight = None
    else:
        lora_path = spec.checkpoint_dir / "pytorch_lora_weights.safetensors"
        if not lora_path.exists():
            raise FileNotFoundError(f"LoRA weights not found: {lora_path}")
        config.model.lora.pretrained_weight = str(lora_path)

    if spec.use_mask_prediction:
        if spec.mg_cfg_background_scale is not None:
            config.model.mg_cfg_background_scale = spec.mg_cfg_background_scale
        if spec.mg_cfg_mask_dilate_radius is not None:
            config.model.mg_cfg_mask_dilate_radius = spec.mg_cfg_mask_dilate_radius
        if spec.mg_cfg_mask_blur_kernel is not None:
            config.model.mg_cfg_mask_blur_kernel = spec.mg_cfg_mask_blur_kernel
        if spec.mg_cfg_start_ratio is not None:
            config.model.mg_cfg_start_ratio = spec.mg_cfg_start_ratio

    trainer = QwenImageEditTrainer(config)
    trainer.setup_predict()

    if spec.use_mask_prediction:
        if spec.checkpoint_dir is None:
            raise ValueError(f"Variant {spec.name} requires a checkpoint directory for mask head.")
        mask_head_path = spec.checkpoint_dir / "mask_prediction_head.pt"
        if not mask_head_path.exists():
            raise FileNotFoundError(f"Mask prediction head not found: {mask_head_path}")
        if trainer.mask_prediction_head is None:
            raise RuntimeError("Mask prediction head is not initialized in trainer.")
        mask_state = torch.load(mask_head_path, map_location="cpu")
        trainer.mask_prediction_head.load_state_dict(mask_state, strict=True)
        trainer.mask_prediction_head.to(device=trainer.dit.device, dtype=trainer.weight_dtype)
        trainer.mask_prediction_head.eval()
        print(f"[Info] Loaded mask head for variant {spec.name}: {mask_head_path}")

    return trainer


def decode_to_pil(trainer: QwenImageEditTrainer, embeddings: dict, latents: torch.Tensor, source_size: tuple[int, int]) -> Image.Image:
    target_height = embeddings["height"]
    target_width = embeddings["width"]
    image_tensor = trainer.decode_vae_latent(latents, target_height, target_width)
    image_np = image_tensor.detach().permute(0, 2, 3, 1).float().cpu().numpy()
    image_np = (image_np * 255).round().clip(0, 255).astype(np.uint8)
    image = Image.fromarray(image_np[0])
    if image.size != source_size:
        image = image.resize(source_size, RESAMPLE_LANCZOS)
    return image


def predicted_mask_to_image(pred_mask: torch.Tensor | None, size: tuple[int, int]) -> Image.Image | None:
    if pred_mask is None:
        return None
    mask_np = pred_mask[0].detach().float().cpu().numpy()
    if mask_np.ndim == 3:
        mask_np = mask_np[0]
    mask_np = np.clip(mask_np, 0.0, 1.0)
    mask_u8 = (mask_np * 255).round().astype(np.uint8)
    mask_image = Image.fromarray(mask_u8, mode="L")
    if mask_image.size != size:
        mask_image = mask_image.resize(size, RESAMPLE_NEAREST)
    return mask_image


def mask_to_rgb(mask_bin: np.ndarray) -> Image.Image:
    mask_u8 = (mask_bin * 255).astype(np.uint8)
    mask_rgb = np.zeros((mask_u8.shape[0], mask_u8.shape[1], 3), dtype=np.uint8)
    mask_rgb[..., 0] = mask_u8
    return Image.fromarray(mask_rgb)


def build_panel(source: Image.Image, generated: Image.Image, target: Image.Image, mask_bin: np.ndarray) -> Image.Image:
    mask_rgb = mask_to_rgb(mask_bin)
    width, height = source.size
    canvas = Image.new("RGB", (width * 4, height))
    canvas.paste(source, (0, 0))
    canvas.paste(generated, (width, 0))
    canvas.paste(target, (width * 2, 0))
    canvas.paste(mask_rgb, (width * 3, 0))
    return canvas


def masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    denom = float(mask.sum())
    if denom <= 0.0:
        return math.nan
    return float((values * mask).sum() / denom)


def compute_metrics(
    source: Image.Image,
    generated: Image.Image,
    target: Image.Image,
    mask_bin: np.ndarray,
) -> dict[str, float]:
    src = np.asarray(source, dtype=np.float32) / 255.0
    gen = np.asarray(generated, dtype=np.float32) / 255.0
    tgt = np.asarray(target, dtype=np.float32) / 255.0
    fg = mask_bin.astype(np.float32)
    bg = 1.0 - fg

    diff_gen_src = np.abs(gen - src).mean(axis=2)
    diff_gen_tgt = np.abs(gen - tgt).mean(axis=2)

    total_change = float(diff_gen_src.sum())
    inside_change_sum = float((diff_gen_src * fg).sum())

    return {
        "mask_area_ratio": float(fg.mean()),
        "inside_change_l1": masked_mean(diff_gen_src, fg),
        "outside_change_l1": masked_mean(diff_gen_src, bg),
        "inside_target_l1": masked_mean(diff_gen_tgt, fg),
        "outside_target_l1": masked_mean(diff_gen_tgt, bg),
        "outside_preservation_l1": masked_mean(diff_gen_src, bg),
        "change_concentration": inside_change_sum / (total_change + 1.0e-8),
    }


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: list[dict]) -> None:
    ensure_parent(path)
    if not rows:
        with path.open("w", newline="", encoding="utf-8") as f:
            f.write("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_rows(rows: list[dict], group_keys: list[str], metric_keys: list[str]) -> list[dict]:
    grouped: dict[tuple, dict[str, list[float]]] = {}
    for row in rows:
        group = tuple(row[key] for key in group_keys)
        grouped.setdefault(group, {metric: [] for metric in metric_keys})
        for metric in metric_keys:
            value = row.get(metric)
            if value is None or (isinstance(value, float) and math.isnan(value)):
                continue
            grouped[group][metric].append(float(value))

    summary_rows: list[dict] = []
    for group, metric_lists in sorted(grouped.items()):
        row = {key: value for key, value in zip(group_keys, group)}
        for metric, values in metric_lists.items():
            row[metric] = float(sum(values) / len(values)) if values else math.nan
        row["num_samples"] = len(next(iter(metric_lists.values()))) if metric_lists else 0
        summary_rows.append(row)
    return summary_rows


def run_variant(
    spec: VariantSpec,
    trainer: QwenImageEditTrainer,
    samples: list[SampleSpec],
    args: argparse.Namespace,
    output_root: Path,
) -> list[dict]:
    rows: list[dict] = []
    for index, sample in enumerate(samples, start=1):
        source_image, target_image, mask_bin = load_source_target_mask(sample, args.mask_threshold)
        variant_dir = output_root / spec.name / sample.dataset / sample.attribute / sample.transform
        result_path = variant_dir / Path(sample.rel_path).name
        panel_path = variant_dir / f"{Path(sample.rel_path).stem}__panel.png"
        pred_mask_path = variant_dir / f"{Path(sample.rel_path).stem}__pred_mask.png"

        generated_image: Image.Image
        pred_mask_image: Image.Image | None = None

        if result_path.exists() and not args.overwrite:
            generated_image = Image.open(result_path).convert("RGB")
        else:
            setup_seed(args.seed)
            batch = trainer.prepare_predict_batch_data(
                image=source_image,
                prompt=sample.prompt,
                negative_prompt=args.negative_prompt,
                num_inference_steps=args.num_inference_steps,
                true_cfg_scale=args.true_cfg_scale,
                guidance_scale=args.guidance_scale,
            )
            embeddings = trainer.prepare_embeddings(batch, stage="predict")
            latents = trainer.sampling_from_embeddings(embeddings)
            generated_image = decode_to_pil(trainer, embeddings, latents, source_image.size)
            pred_mask_image = predicted_mask_to_image(embeddings.get("predicted_mask_2d"), source_image.size)

            ensure_parent(result_path)
            generated_image.save(result_path)
            if args.save_panels:
                build_panel(source_image, generated_image, target_image, mask_bin).save(panel_path)
            if args.save_pred_masks and pred_mask_image is not None:
                pred_mask_image.save(pred_mask_path)

        metrics = compute_metrics(source_image, generated_image, target_image, mask_bin)
        row = {
            "variant": spec.name,
            "dataset": sample.dataset,
            "attribute": sample.attribute,
            "transform": sample.transform,
            "rel_path": sample.rel_path,
            "image_path": str(sample.image_path),
            "mask_path": str(sample.mask_path),
            "result_path": str(result_path),
            "prompt": sample.prompt,
        }
        row.update(metrics)
        rows.append(row)
        print(f"[{spec.name}] {index}/{len(samples)} {sample.dataset}/{sample.rel_path}")
    return rows


def cleanup_trainer(trainer: QwenImageEditTrainer | None) -> None:
    if trainer is None:
        return
    try:
        del trainer
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    variant_specs = build_variant_specs(args.true_cfg_scale)
    selected_variant_names = parse_csv_list(args.variants)
    selected_dataset_names = parse_csv_list(args.datasets)

    for variant_name in selected_variant_names:
        if variant_name not in variant_specs:
            raise ValueError(f"Unsupported variant: {variant_name}")

    dataset_specs = resolve_selected_datasets(selected_dataset_names)
    samples = collect_samples(dataset_specs, args.max_samples_per_leaf, args.max_total_samples)
    if not samples:
        raise RuntimeError("No samples were collected from the selected datasets.")

    manifest_rows = [asdict(sample) for sample in samples]
    for row in manifest_rows:
        row["image_path"] = str(row["image_path"])
        row["mask_path"] = str(row["mask_path"])
    write_csv(args.output_root / "sample_manifest.csv", manifest_rows)

    run_config = {
        "datasets": selected_dataset_names,
        "variants": selected_variant_names,
        "num_inference_steps": args.num_inference_steps,
        "true_cfg_scale": args.true_cfg_scale,
        "guidance_scale": args.guidance_scale,
        "negative_prompt": args.negative_prompt,
        "seed": args.seed,
        "mask_threshold": args.mask_threshold,
        "max_samples_per_leaf": args.max_samples_per_leaf,
        "max_total_samples": args.max_total_samples,
        "variant_specs": {name: asdict(variant_specs[name]) for name in selected_variant_names},
    }
    for name, spec in run_config["variant_specs"].items():
        for key, value in list(spec.items()):
            if isinstance(value, Path):
                spec[key] = str(value)
    (args.output_root / "run_config.json").write_text(json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8")

    all_rows: list[dict] = []
    metric_keys = [
        "mask_area_ratio",
        "inside_change_l1",
        "outside_change_l1",
        "inside_target_l1",
        "outside_target_l1",
        "outside_preservation_l1",
        "change_concentration",
    ]

    for variant_name in selected_variant_names:
        spec = variant_specs[variant_name]
        trainer: QwenImageEditTrainer | None = None
        try:
            print(f"\n[Load] Variant {variant_name}")
            trainer = load_variant_trainer(spec, args)
            variant_rows = run_variant(spec, trainer, samples, args, args.output_root)
            all_rows.extend(variant_rows)
            write_csv(args.output_root / f"metrics_per_sample_{variant_name}.csv", variant_rows)
        finally:
            cleanup_trainer(trainer)

    write_csv(args.output_root / "metrics_per_sample_all.csv", all_rows)
    write_csv(
        args.output_root / "metrics_summary_by_variant.csv",
        summarize_rows(all_rows, ["variant"], metric_keys),
    )
    write_csv(
        args.output_root / "metrics_summary_by_variant_dataset.csv",
        summarize_rows(all_rows, ["variant", "dataset"], metric_keys),
    )
    write_csv(
        args.output_root / "metrics_summary_by_variant_attribute.csv",
        summarize_rows(all_rows, ["variant", "attribute"], metric_keys),
    )
    print(f"\n[Done] samples={len(samples)}, variants={len(selected_variant_names)}, output_root={args.output_root}")


if __name__ == "__main__":
    main()
