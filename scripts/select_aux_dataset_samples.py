#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Iterable


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DEFAULT_SOURCES = [
    "/home/zsm/code/2_p_gradient_editing/GRAG-Image-Editing/test_slider",
    "/mount/data/zsm/all_dataset/0_fame/humanparsing_editing_resize",
]
DEFAULT_OUTPUT_ROOT = "/mount/data/zsm/all_dataset/0_fame/aux_dataset"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Randomly sample images from leaf attribute directories and copy them "
            "into an auxiliary dataset while preserving the source directory structure."
        )
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=DEFAULT_SOURCES,
        help="Source dataset roots to sample from.",
    )
    parser.add_argument(
        "--output-root",
        default=DEFAULT_OUTPUT_ROOT,
        help="Output root directory for the auxiliary dataset.",
    )
    parser.add_argument(
        "--samples-per-leaf",
        type=int,
        default=10,
        help="Number of images to sample from each leaf directory.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed used for deterministic sampling.",
    )
    parser.add_argument(
        "--copy-mode",
        choices=("copy", "symlink"),
        default="copy",
        help="Whether to copy files or create symlinks in the output directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing sampled files if they already exist.",
    )
    return parser.parse_args()


def iter_leaf_image_dirs(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if not path.is_dir():
            continue
        image_files = [p for p in sorted(path.iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
        if image_files:
            yield path


def deterministic_sample(files: list[Path], k: int, seed: int, namespace: str) -> list[Path]:
    if len(files) <= k:
        return files
    digest = hashlib.md5(f"{seed}:{namespace}".encode("utf-8")).hexdigest()
    local_seed = int(digest[:8], 16)
    rng = random.Random(local_seed)
    return sorted(rng.sample(files, k))


def copy_or_link(src: Path, dst: Path, mode: str, overwrite: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src)


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, str | int]] = []
    summary: dict[str, dict[str, int]] = defaultdict(lambda: {"leaf_dirs": 0, "sampled_images": 0})

    for source_str in args.sources:
        source_root = Path(source_str).resolve()
        if not source_root.exists():
            raise FileNotFoundError(f"Source root does not exist: {source_root}")

        dataset_name = source_root.name
        dataset_output_root = output_root / dataset_name

        for leaf_dir in iter_leaf_image_dirs(source_root):
            rel_leaf = leaf_dir.relative_to(source_root)
            image_files = [p for p in sorted(leaf_dir.iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
            selected = deterministic_sample(
                files=image_files,
                k=args.samples_per_leaf,
                seed=args.seed,
                namespace=f"{dataset_name}/{rel_leaf.as_posix()}",
            )

            summary[dataset_name]["leaf_dirs"] += 1
            summary[dataset_name]["sampled_images"] += len(selected)

            for index, src_path in enumerate(selected, start=1):
                dst_path = dataset_output_root / rel_leaf / src_path.name
                copy_or_link(src=src_path, dst=dst_path, mode=args.copy_mode, overwrite=args.overwrite)
                manifest_rows.append(
                    {
                        "dataset": dataset_name,
                        "leaf_dir": rel_leaf.as_posix(),
                        "sample_index": index,
                        "source_path": str(src_path),
                        "output_path": str(dst_path),
                        "filename": src_path.name,
                    }
                )

    manifest_path = output_root / "selection_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["dataset", "leaf_dir", "sample_index", "source_path", "output_path", "filename"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary_path = output_root / "selection_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "sources": [str(Path(p).resolve()) for p in args.sources],
                "output_root": str(output_root.resolve()),
                "samples_per_leaf": args.samples_per_leaf,
                "seed": args.seed,
                "copy_mode": args.copy_mode,
                "datasets": summary,
                "total_selected": len(manifest_rows),
            },
            f,
            indent=2,
            ensure_ascii=True,
        )

    print(f"Selected {len(manifest_rows)} images into: {output_root}")
    print(f"Manifest: {manifest_path}")
    print(f"Summary:  {summary_path}")


if __name__ == "__main__":
    main()
