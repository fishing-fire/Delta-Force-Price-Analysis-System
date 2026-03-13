from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backup_root",
        type=str,
        default=str(Path(__file__).resolve().parent / "backup" / "category compare collection_category"),
    )
    args = parser.parse_args()

    backup_root = Path(args.backup_root)
    results_root = backup_root / "Collection_有外生变量" / "results"
    rows = []

    for exp_dir in sorted([p for p in results_root.iterdir() if p.is_dir()]):
        p = exp_dir / "exog_contrib" / "exog_contrib.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        df["exp_name"] = exp_dir.name
        rows.append(df)

    if not rows:
        return 0

    all_df = pd.concat(rows, axis=0, ignore_index=True)
    agg = (
        all_df.groupby("exog", as_index=False)
        .agg(
            exp_count=("exp_name", "nunique"),
            mean_delta_mae=("delta_mae", "mean"),
            median_delta_mae=("delta_mae", "median"),
            mean_delta_rmse=("delta_rmse", "mean"),
            mean_delta_mape=("delta_mape", "mean"),
        )
        .sort_values("mean_delta_mae", ascending=False)
    )

    out_dir = results_root / "_exog_contrib_agg"
    out_dir.mkdir(parents=True, exist_ok=True)
    agg.to_csv(out_dir / "exog_contrib_agg.csv", index=False, encoding="utf-8-sig")

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 4.5), dpi=140)
    ax = fig.add_subplot(111)
    ax.bar(agg["exog"].tolist(), agg["mean_delta_mae"].tolist())
    ax.set_title("Exogenous contribution summary (mean delta MAE)")
    ax.set_ylabel("mean ΔMAE")
    ax.set_xlabel("exogenous variable")
    ax.tick_params(axis="x", labelrotation=30)
    fig.tight_layout()
    fig.savefig(out_dir / "exog_contrib_agg_mean_delta_mae.png")
    plt.close(fig)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

