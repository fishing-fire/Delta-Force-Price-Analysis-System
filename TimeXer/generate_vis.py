from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
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

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, default="", help="相对 TimeXer 目录的结果目录（包含 pred.npy/true.npy）")
    parser.add_argument("--results_root", type=str, default="results", help="相对 TimeXer 目录的 results 根目录")
    args = parser.parse_args()

    timexer_root = Path(__file__).resolve().parent
    run_dir: Path | None
    if args.run_dir.strip():
        p = Path(args.run_dir.strip())
        run_dir = (timexer_root / p) if not p.is_absolute() else p
    else:
        run_dir = _pick_latest_run(timexer_root / args.results_root)

    if not run_dir:
        print("Error: 未找到可用结果目录。请通过 --run_dir 指定，或确保 results_root 下存在含 pred.npy/true.npy 的子目录。")
        return 1
    run_dir = run_dir.resolve()
    if not run_dir.exists():
        print(f"Error: Path not found: {run_dir}")
        return 1

    pred_path = run_dir / "pred.npy"
    true_path = run_dir / "true.npy"
    if not pred_path.exists() or not true_path.exists():
        print(f"Error: pred.npy 或 true.npy 不存在：{run_dir}")
        return 1

    try:
        preds = np.load(pred_path)
        trues = np.load(true_path)
        print(f"Loading results from: {run_dir.relative_to(timexer_root).as_posix() if run_dir.is_relative_to(timexer_root) else str(run_dir)}")
        print(f"Loaded preds shape: {preds.shape}")
        print(f"Loaded trues shape: {trues.shape}")
        print("Generating visualizations...")
        visual_experiment_results(trues, preds, str(run_dir))
        print("Visualization generated successfully!")
        return 0
    except Exception as e:
        print(f"An error occurred: {e}")
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
