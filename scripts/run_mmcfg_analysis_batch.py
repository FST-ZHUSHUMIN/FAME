#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANALYZE_SCRIPT = PROJECT_ROOT / "scripts" / "analyze_mmcfg_response.py"


DEFAULT_TASKS = [
    {
        "name": "top_length_short2long",
        "input_dir": "/mount/data/zsm/all_dataset/0_fame/humanparsing_editing_resize/top_length/short",
        "prompt": "Change top length from short to long",
        "output_subdir": "top_length/short2long",
    },
    {
        "name": "dress_length_short2long",
        "input_dir": "/mount/data/zsm/all_dataset/0_fame/humanparsing_editing_resize/dress_length/short",
        "prompt": "Change dress length to long",
        "output_subdir": "dress_length/short2long",
    },
    {
        "name": "pant_length_short2long",
        "input_dir": "/mount/data/zsm/all_dataset/0_fame/humanparsing_editing_resize/pant_length/short",
        "prompt": "Change pants length to long",
        "output_subdir": "pant_length/short2long",
    },
    {
        "name": "sleeve_length_short2long",
        "input_dir": "/mount/data/zsm/all_dataset/0_fame/humanparsing_editing_resize/sleeve_length/short",
        "prompt": "Change sleeve length to long",
        "output_subdir": "sleeve_length/short2long",
    },
    {
        "name": "sleeve_shape_regular2bell",
        "input_dir": "/mount/data/zsm/all_dataset/0_fame/humanparsing_editing_resize/sleeve_shape/regular",
        "prompt": "Change sleeve shape to bell sleeve",
        "output_subdir": "sleeve_shape/regular2bell",
    },
    {
        "name": "sleeve_shape_regular2cape",
        "input_dir": "/mount/data/zsm/all_dataset/0_fame/humanparsing_editing_resize/sleeve_shape/regular",
        "prompt": "Change sleeve shape to cape sleeve",
        "output_subdir": "sleeve_shape/regular2cape",
    },
    {
        "name": "sleeve_shape_regular2cap",
        "input_dir": "/mount/data/zsm/all_dataset/0_fame/humanparsing_editing_resize/sleeve_shape/regular",
        "prompt": "Change sleeve shape to cap sleeve",
        "output_subdir": "sleeve_shape/regular2cap",
    },
    {
        "name": "sleeve_shape_regular2leg_of_mutton",
        "input_dir": "/mount/data/zsm/all_dataset/0_fame/humanparsing_editing_resize/sleeve_shape/regular",
        "prompt": "Change sleeve shape to leg of mutton sleeve",
        "output_subdir": "sleeve_shape/regular2leg of mutton",
    },
    {
        "name": "collar_r2cami_off_shoulder",
        "input_dir": "/mount/data/zsm/all_dataset/0_fame/humanparsing_editing_resize/collar/r",
        "prompt": "Change neckline to cami-off shoulder neckline",
        "output_subdir": "collar/r2cami-off_shoulder",
    },
    {
        "name": "collar_r2choker",
        "input_dir": "/mount/data/zsm/all_dataset/0_fame/humanparsing_editing_resize/collar/r",
        "prompt": "Change neckline to choker neckline",
        "output_subdir": "collar/r2choker",
    },
    {
        "name": "collar_r2hoodie",
        "input_dir": "/mount/data/zsm/all_dataset/0_fame/humanparsing_editing_resize/collar/r",
        "prompt": "Change neckline to hoodie collar",
        "output_subdir": "collar/r2hoodie",
    },
    {
        "name": "collar_r2one_shoulder",
        "input_dir": "/mount/data/zsm/all_dataset/0_fame/humanparsing_editing_resize/collar/r",
        "prompt": "Change neckline to one-shoulder neckline",
        "output_subdir": "collar/r2one-shoulder",
    },
    {
        "name": "collar_r2peter_pan",
        "input_dir": "/mount/data/zsm/all_dataset/0_fame/humanparsing_editing_resize/collar/r",
        "prompt": "Change neckline to peter pan collar",
        "output_subdir": "collar/r2peter pan",
    },
    {
        "name": "collar_r2v",
        "input_dir": "/mount/data/zsm/all_dataset/0_fame/humanparsing_editing_resize/collar/r",
        "prompt": "Change neckline to v neckline",
        "output_subdir": "collar/r2v",
    },
    {
        "name": "collar_r2square",
        "input_dir": "/mount/data/zsm/all_dataset/0_fame/humanparsing_editing_resize/collar/r",
        "prompt": "Change neckline to square neckline",
        "output_subdir": "collar/r2square",
    },
    {
        "name": "collar_r2shirt",
        "input_dir": "/mount/data/zsm/all_dataset/0_fame/humanparsing_editing_resize/collar/r",
        "prompt": "Change neckline to shirt collar",
        "output_subdir": "collar/r2shirt",
    },
    {
        "name": "collar_r2turtle",
        "input_dir": "/mount/data/zsm/all_dataset/0_fame/humanparsing_editing_resize/collar/r",
        "prompt": "Change neckline to turtle collar",
        "output_subdir": "collar/r2turtle",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch launcher for MM-CFG response analysis.")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config.")
    parser.add_argument("--ckpt", type=Path, required=True, help="Checkpoint directory.")
    parser.add_argument(
        "--output_root",
        type=Path,
        required=True,
        help="Root directory to save all task outputs.",
    )
    parser.add_argument("--python_bin", type=str, default=sys.executable, help="Python interpreter to run the analysis.")
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--true_cfg_scale", type=float, default=4.0)
    parser.add_argument("--guidance_scale", type=float, default=4.0)
    parser.add_argument("--mg_cfg_background_scale", type=float, default=None)
    parser.add_argument("--mg_cfg_mask_dilate_radius", type=int, default=None)
    parser.add_argument("--mg_cfg_mask_blur_kernel", type=int, default=None)
    parser.add_argument("--mg_cfg_start_ratio", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=20)
    parser.add_argument("--save_steps", type=str, default="0,mid,last")
    parser.add_argument("--negative_prompt", type=str, default=" ")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--tasks",
        type=str,
        default="all",
        help="Comma-separated task names to run. Use 'all' to run all default tasks.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without executing them.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Force re-running samples even when metrics.json already exists.",
    )
    return parser.parse_args()


def select_tasks(task_spec: str) -> list[dict]:
    if task_spec.strip().lower() == "all":
        return DEFAULT_TASKS
    selected = []
    requested = {item.strip() for item in task_spec.split(",") if item.strip()}
    for task in DEFAULT_TASKS:
        if task["name"] in requested:
            selected.append(task)
    missing = requested - {task["name"] for task in selected}
    if missing:
        raise ValueError(f"Unknown task names: {sorted(missing)}")
    return selected


def build_command(args: argparse.Namespace, task: dict) -> list[str]:
    output_dir = args.output_root / task["output_subdir"]
    cmd = [
        args.python_bin,
        str(ANALYZE_SCRIPT),
        "--config",
        str(args.config),
        "--ckpt",
        str(args.ckpt),
        "--input_dir",
        task["input_dir"],
        "--prompt",
        task["prompt"],
        "--output_dir",
        str(output_dir),
        "--num_inference_steps",
        str(args.num_inference_steps),
        "--true_cfg_scale",
        str(args.true_cfg_scale),
        "--guidance_scale",
        str(args.guidance_scale),
        "--seed",
        str(args.seed),
        "--max_samples",
        str(args.max_samples),
        "--save_steps",
        args.save_steps,
        "--negative_prompt",
        args.negative_prompt,
    ]
    if args.mg_cfg_background_scale is not None:
        cmd += ["--mg_cfg_background_scale", str(args.mg_cfg_background_scale)]
    if args.mg_cfg_mask_dilate_radius is not None:
        cmd += ["--mg_cfg_mask_dilate_radius", str(args.mg_cfg_mask_dilate_radius)]
    if args.mg_cfg_mask_blur_kernel is not None:
        cmd += ["--mg_cfg_mask_blur_kernel", str(args.mg_cfg_mask_blur_kernel)]
    if args.mg_cfg_start_ratio is not None:
        cmd += ["--mg_cfg_start_ratio", str(args.mg_cfg_start_ratio)]
    if args.device is not None:
        cmd += ["--device", args.device]
    if args.overwrite:
        cmd += ["--overwrite"]
    return cmd


def main() -> None:
    args = parse_args()
    tasks = select_tasks(args.tasks)
    args.output_root.mkdir(parents=True, exist_ok=True)

    print("[MM-CFG Batch] Selected tasks:")
    for task in tasks:
        print(f"  - {task['name']}: {task['input_dir']} -> {task['output_subdir']}")

    for task in tasks:
        cmd = build_command(args, task)
        printable = " ".join(f'"{item}"' if " " in item else item for item in cmd)
        print(f"\n[MM-CFG Batch] Running task: {task['name']}")
        print(printable)
        if args.dry_run:
            continue
        subprocess.run(cmd, check=True)

    print(f"\n[MM-CFG Batch] Done. Outputs saved under: {args.output_root}")


if __name__ == "__main__":
    main()
