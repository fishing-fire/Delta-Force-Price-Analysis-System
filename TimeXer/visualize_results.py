from __future__ import annotations

import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

# Ensure parent directory is in python path to import utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.tools import visual_experiment_results

def _pick_latest_run(results_root: Path) -> Path | None:
    if not results_root.exists():
        return None
    candidates: list[Path] = []
    for p in results_root.iterdir():
        if not p.is_dir():
            continue
        if (p / "pred.npy").exists() and (p / "true.npy").exists():
            candidates.append(p)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)

def main(folder_path: str):
    if not os.path.exists(folder_path):
        print(f"Error: Folder not found: {folder_path}")
        return

    pred_path = os.path.join(folder_path, 'pred.npy')
    true_path = os.path.join(folder_path, 'true.npy')

    if not os.path.exists(pred_path) or not os.path.exists(true_path):
        print(f"Error: pred.npy or true.npy not found in {folder_path}")
        return

    print(f"Loading results from {folder_path}...")
    preds = np.load(pred_path)
    trues = np.load(true_path)

    print(f"Shapes - Preds: {preds.shape}, Trues: {trues.shape}")
    
    print("Generating visualizations...")
    visual_experiment_results(trues, preds, folder_path)
    print(f"Done! Check the 'visualization' subfolder in {folder_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, default="", help="相对 TimeXer 目录的结果目录（包含 pred.npy/true.npy）")
    parser.add_argument("--results_root", type=str, default="results", help="相对 TimeXer 目录的 results 根目录")
    args = parser.parse_args()

    timexer_root = Path(__file__).resolve().parent
    if args.run_dir.strip():
        p = Path(args.run_dir.strip())
        run_dir = (timexer_root / p) if not p.is_absolute() else p
    else:
        run_dir = _pick_latest_run(timexer_root / args.results_root)

    if not run_dir:
        print("Error: 未找到可用结果目录。请通过 --run_dir 指定，或确保 results_root 下存在含 pred.npy/true.npy 的子目录。")
        raise SystemExit(1)

    main(str(run_dir.resolve()))
