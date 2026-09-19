#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch-generate images from attacked prompt folders, compute PSNR, parse LPIPS,
and summarize results for N in {0, 25, 50, 100, 500, 1000}.
"""

import re
import csv
import math
import argparse
import subprocess
from pathlib import Path
import sys
import numpy as np
from PIL import Image


DEFAULT_NS = "0,25,50,100,250,500,1000,2000,4000,16000"


def list_gt_images(frame_path: Path):
    return sorted([p for p in frame_path.glob("*.png") if p.is_file()])


def list_generated_images(results_dir: Path):
    return sorted([
        p for p in results_dir.glob("*.png")
        if p.is_file() and re.fullmatch(r"\d{5}\.png", p.name)
    ])


def load_rgb(path: Path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)


def compute_psnr(gt: np.ndarray, pred: np.ndarray, max_val: float = 255.0):
    if gt.shape != pred.shape:
        raise ValueError(f"Shape mismatch: gt={gt.shape}, pred={pred.shape}")
    mse = np.mean((gt - pred) ** 2)
    if mse <= 1e-12:
        return float("inf")
    return 10.0 * math.log10((max_val ** 2) / mse)


def match_gt_and_pred(frame_path: Path, results_dir: Path):
    gt_files = list_gt_images(frame_path)
    pairs = []
    for gt in gt_files:
        pred = results_dir / gt.name
        if pred.exists():
            pairs.append((gt, pred))
    return pairs


def parse_mean_lpips(log_path: Path):
    if not log_path.exists():
        return None
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    matches = re.findall(r"mean\s+lpips\s*:\s*([0-9]*\.?[0-9]+)", text, flags=re.IGNORECASE)
    if not matches:
        return None
    return float(matches[-1])


def run_generation(frame_path: Path, generation_script: str, rank: int, interval: int, python_exec: str):
    cmd = [
        python_exec,
        generation_script,
        "-frame_path", str(frame_path),
        "-rank", str(rank),
        "-interval", str(interval),
    ]
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def evaluate_one_folder(frame_path: Path, rank: int, interval: int):
    results_dir = frame_path / "results" / f"rank{rank}_interval{interval}"
    log_path = results_dir / "log.txt"

    pairs = match_gt_and_pred(frame_path, results_dir)
    if not pairs:
        return {
            "num_images": 0,
            "mean_psnr": None,
            "std_psnr": None,
            "mean_lpips": parse_mean_lpips(log_path),
        }, []

    psnrs = []
    details = []
    for gt_path, pred_path in pairs:
        gt = load_rgb(gt_path)
        pred = load_rgb(pred_path)
        psnr = compute_psnr(gt, pred)
        psnrs.append(psnr)
        details.append({
            "image_name": gt_path.name,
            "psnr": psnr,
        })

    metrics = {
        "num_images": len(psnrs),
        "mean_psnr": float(np.mean(psnrs)),
        "std_psnr": float(np.std(psnrs)),
        "mean_lpips": parse_mean_lpips(log_path),
    }
    return metrics, details


def find_condition_dirs(attack_root: Path, n_value: int):
    n_dir = attack_root / "prompt_outputs" / f"N{n_value:06d}"
    if not n_dir.exists():
        return []
    return sorted([p for p in n_dir.iterdir() if p.is_dir()])


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--attack_root", type=str, required=True)
    parser.add_argument("--generation_script", type=str, default="generation.py")
    parser.add_argument("--python_exec", type=str, default=sys.executable)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--interval", type=int, default=1)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument(
        "--only_conditions",
        type=str,
        default="",
        help='Comma-separated condition names, e.g. "baseline_attack,ours_diff_state_attack"'
    )
    parser.add_argument("--output_csv", type=str, default="attack_generation_summary.csv")
    parser.add_argument("--detail_csv", type=str, default="attack_generation_detail.csv")
    parser.add_argument("--leak_counts", default=DEFAULT_NS)
    args = parser.parse_args()

    attack_root = Path(args.attack_root).resolve()
    generation_script = str(Path(args.generation_script))

    keep_conditions = None
    if args.only_conditions.strip():
        keep_conditions = {x.strip() for x in args.only_conditions.split(",") if x.strip()}

    summary_rows = []
    detail_rows = []

    for n_value in [int(x) for x in args.leak_counts.split(",")]:
        condition_dirs = find_condition_dirs(attack_root, n_value)
        if not condition_dirs:
            print(f"[SKIP] N={n_value}: folder not found")
            continue

        print(f"\n========== N = {n_value} ==========")
        for frame_path in condition_dirs:
            condition_name = frame_path.name
            if keep_conditions is not None and condition_name not in keep_conditions:
                print(f"[SKIP] condition={condition_name} (filtered)")
                continue

            results_dir = frame_path / "results" / f"rank{args.rank}_interval{args.interval}"
            existing_outputs = list_generated_images(results_dir)

            if args.skip_existing and existing_outputs:
                print(f"[SKIP generation] {frame_path} (found {len(existing_outputs)} generated images)")
            else:
                run_generation(
                    frame_path=frame_path,
                    generation_script=generation_script,
                    rank=args.rank,
                    interval=args.interval,
                    python_exec=args.python_exec,
                )

            metrics, details = evaluate_one_folder(
                frame_path=frame_path,
                rank=args.rank,
                interval=args.interval,
            )

            summary_row = {
                "n_leak": n_value,
                "condition": condition_name,
                "num_images": metrics["num_images"],
                "mean_psnr": metrics["mean_psnr"],
                "std_psnr": metrics["std_psnr"],
                "mean_lpips": metrics["mean_lpips"],
                "frame_path": str(frame_path),
            }
            summary_rows.append(summary_row)

            for item in details:
                detail_rows.append({
                    "n_leak": n_value,
                    "condition": condition_name,
                    "image_name": item["image_name"],
                    "psnr": item["psnr"],
                    "frame_path": str(frame_path),
                })

            print(
                f"[DONE] N={n_value:4d} | {condition_name:24s} | "
                f"images={metrics['num_images']:3d} | "
                f"mean_psnr={metrics['mean_psnr']} | "
                f"mean_lpips={metrics['mean_lpips']}"
            )

    output_csv = Path(args.output_csv).resolve()
    detail_csv = Path(args.detail_csv).resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    detail_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "n_leak", "condition", "num_images",
                "mean_psnr", "std_psnr", "mean_lpips", "frame_path"
            ]
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    with open(detail_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "n_leak", "condition", "image_name", "psnr", "frame_path"
            ]
        )
        writer.writeheader()
        writer.writerows(detail_rows)

    print("\n========== SUMMARY ==========")
    for row in summary_rows:
        print(
            f"N={row['n_leak']:4d} | "
            f"{row['condition']:24s} | "
            f"PSNR={row['mean_psnr']} | "
            f"LPIPS={row['mean_lpips']}"
        )

    print(f"\nSummary CSV: {output_csv}")
    print(f"Detail  CSV: {detail_csv}")


if __name__ == "__main__":
    main()
