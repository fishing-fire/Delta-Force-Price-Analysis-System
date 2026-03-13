from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from utils.metrics import metric, metric_per_channel


EXOG_COLS = {
    "is_holiday",
    "in_CS",
    "is_CS",
    "is_need",
    "is_make",
    "is_active",
    "is_public",
}


def _read_meta_from_log(log_path: Path) -> dict | None:
    if not log_path.exists():
        return None
    try:
        with log_path.open("r", encoding="utf-8", errors="ignore") as f:
            for _ in range(800):
                line = f.readline()
                if not line:
                    break
                line = line.lstrip("\ufeff").strip()
                if line.startswith("META:"):
                    payload = line[len("META:") :].strip()
                    return json.loads(payload)
    except Exception:
        return None
    return None


def _read_data_path_from_log(log_path: Path) -> str | None:
    if not log_path.exists():
        return None
    try:
        with log_path.open("r", encoding="utf-8", errors="ignore") as f:
            for _ in range(2000):
                line = f.readline()
                if not line:
                    break
                line = line.lstrip("\ufeff").strip()
                if line.startswith("Data Path:"):
                    value = line[len("Data Path:") :].strip()
                    value = value.strip('"').strip("'")
                    return value if value else None
    except Exception:
        return None
    return None


def _norm_key(s: str) -> str:
    return re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "", s).lower()


def _infer_dataset_csv(exp_name: str, dataset_dir: Path) -> Path | None:
    prefix = exp_name.split("_", 1)[0]
    alias = {
        "Arrow": "箭矢",
        "300BLK": ".300BLK",
    }.get(prefix, prefix)
    key = _norm_key(alias)
    for p in dataset_dir.glob("*.csv"):
        if _norm_key(p.stem) == key:
            return p
    for p in dataset_dir.glob("*.csv"):
        if _norm_key(p.stem).startswith(key):
            return p
    return None


def _looks_standardized(values: np.ndarray) -> bool:
    v = values[np.isfinite(values)]
    if v.size == 0:
        return False
    max_abs = float(np.max(np.abs(v)))
    std = float(np.std(v))
    return max_abs < 20 and std < 10


def _inverse_y_from_dataset_csv(preds: np.ndarray, trues: np.ndarray, data_csv: Path) -> tuple[np.ndarray, np.ndarray]:
    df_raw = pd.read_csv(data_csv, encoding="utf-8-sig")
    cols = list(df_raw.columns)
    if cols and cols[0].strip().lower() == "date":
        cols_data = cols[1:]
    else:
        cols_data = cols

    present_exog = [c for c in cols_data if c in EXOG_COLS]
    target_cols = [c for c in cols_data if c not in present_exog]
    if len(target_cols) == 0:
        return preds, trues

    num_train = int(len(df_raw) * 0.7)
    train_x = df_raw.loc[: max(num_train - 1, 0), cols_data].to_numpy(dtype=float, copy=False)
    means = train_x.mean(axis=0)
    scales = train_x.std(axis=0, ddof=0)
    scales = np.where(scales == 0, 1.0, scales)

    target_indices = [cols_data.index(c) for c in target_cols]
    target_means = means[target_indices]
    target_scales = scales[target_indices]

    preds_raw = preds * target_scales + target_means
    trues_raw = trues * target_scales + target_means
    return preds_raw, trues_raw


def _normalize_bool(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("true", "1", "yes", "y", "on"):
            return True
        if v in ("false", "0", "no", "n", "off"):
            return False
    return value


def _extract_params(meta: dict | None, folder_name: str) -> dict:
    if not meta:
        return {
            "Model Name": "Unknown",
            "Task Name": "Unknown",
            "Model ID": folder_name,
            "Data": "Unknown",
            "Features": None,
            "Seq Len": None,
            "Label Len": None,
            "Pred Len": None,
            "D Model": None,
            "N Heads": None,
            "E Layers": None,
            "D Layers": None,
            "D FF": None,
            "Expand": None,
            "D Conv": None,
            "Factor": None,
            "Embed": None,
            "Distil": None,
            "Description": None,
            "Iteration": None,
        }
    return {
        "Model Name": meta.get("model", "Unknown"),
        "Task Name": meta.get("task_name", "Unknown"),
        "Model ID": folder_name,
        "Data": meta.get("data", "Unknown"),
        "Features": meta.get("features", None),
        "Seq Len": meta.get("seq_len", None),
        "Label Len": meta.get("label_len", None),
        "Pred Len": meta.get("pred_len", None),
        "D Model": meta.get("d_model", None),
        "N Heads": meta.get("n_heads", None),
        "E Layers": meta.get("e_layers", None),
        "D Layers": meta.get("d_layers", None),
        "D FF": meta.get("d_ff", None),
        "Expand": meta.get("expand", None),
        "D Conv": meta.get("d_conv", None),
        "Factor": meta.get("factor", None),
        "Embed": meta.get("embed", None),
        "Distil": _normalize_bool(meta.get("distil", None)),
        "Description": meta.get("des", None),
        "Iteration": meta.get("itr_index", None),
    }


def _recompute_one(exp_dir: Path, dataset_dir: Path) -> tuple[dict | None, str | None]:
    pred_path = exp_dir / "pred.npy"
    true_path = exp_dir / "true.npy"
    if not pred_path.exists() or not true_path.exists():
        return None, "missing pred.npy/true.npy"

    preds = np.load(pred_path)
    trues = np.load(true_path)

    if preds.shape != trues.shape:
        return None, f"shape mismatch pred={preds.shape} true={trues.shape}"

    if _looks_standardized(trues):
        data_path = _read_data_path_from_log(exp_dir / "log.txt")
        data_csv = None
        if data_path:
            p = Path(data_path)
            data_csv = p if p.is_absolute() else (dataset_dir / data_path)
        if data_csv is None or not data_csv.exists():
            data_csv = _infer_dataset_csv(exp_dir.name, dataset_dir)
        if data_csv is not None and data_csv.exists():
            preds, trues = _inverse_y_from_dataset_csv(preds, trues, data_csv)

    mae, mse, rmse, mape, mspe = metric(preds, trues)
    df = pd.DataFrame(
        {
            "MAE": [float(mae)],
            "MSE": [float(mse)],
            "RMSE": [float(rmse)],
            "MAPE": [float(mape)],
            "MSPE": [float(mspe)],
        }
    )
    df.to_csv(exp_dir / "metrics.csv", index=False, encoding="utf-8-sig")

    mae_d, mse_d, rmse_d, mape_d, mspe_d = metric_per_channel(preds, trues)
    num_channels = int(preds.shape[-1])
    df_d = pd.DataFrame(
        {
            "Channel": [f"Channel_{i}" for i in range(num_channels)],
            "MAE": mae_d.astype(float),
            "MSE": mse_d.astype(float),
            "RMSE": rmse_d.astype(float),
            "MAPE": mape_d.astype(float),
            "MSPE": mspe_d.astype(float),
        }
    )
    df_d.to_csv(exp_dir / "metrics_detail.csv", index=False, encoding="utf-8-sig")

    np.save(exp_dir / "metrics.npy", np.array([mae, mse, rmse, mape, mspe], dtype=float))

    meta = _read_meta_from_log(exp_dir / "log.txt")
    row = _extract_params(meta, folder_name=exp_dir.name)
    row.update(
        {
            "MAE": float(mae),
            "MSE": float(mse),
            "RMSE": float(rmse),
            "MAPE": float(mape),
            "MSPE": float(mspe),
        }
    )
    return row, None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backup_root",
        type=str,
        default=str(Path(__file__).resolve().parent / "backup" / "category compare collection_category"),
    )
    args = parser.parse_args()
    backup_root = Path(args.backup_root)

    groups = [
        backup_root / "NoCollection_无外生变量" / "results",
        backup_root / "Collection_有外生变量" / "results",
    ]

    rows: list[dict] = []
    errors: list[str] = []

    timexer_root = Path(__file__).resolve().parent
    ds_category = timexer_root / "dataset" / "bullet" / "category"
    ds_collection_category = timexer_root / "dataset" / "bullet" / "collection_category"

    for group_dir in groups:
        if not group_dir.exists():
            continue
        dataset_dir = ds_collection_category if "Collection_有外生变量" in str(group_dir) else ds_category
        for exp_dir in sorted([p for p in group_dir.iterdir() if p.is_dir()]):
            row, err = _recompute_one(exp_dir, dataset_dir=dataset_dir)
            if err:
                errors.append(f"{exp_dir}: {err}")
                continue
            if row:
                rows.append(row)

    if rows:
        df_all = pd.DataFrame(rows)
        cols = [
            "Model Name",
            "Task Name",
            "Model ID",
            "MAE",
            "MSE",
            "RMSE",
            "MAPE",
            "MSPE",
            "Data",
            "Features",
            "Seq Len",
            "Label Len",
            "Pred Len",
            "D Model",
            "N Heads",
            "E Layers",
            "D Layers",
            "D FF",
            "Expand",
            "D Conv",
            "Factor",
            "Embed",
            "Distil",
            "Description",
            "Iteration",
        ]
        cols = [c for c in cols if c in df_all.columns]
        df_all = df_all[cols]
        out_path = backup_root / "summary_metrics.csv"
        df_all.to_csv(out_path, index=False, encoding="utf-8-sig")

    if errors:
        err_path = backup_root / "recompute_metrics_errors.txt"
        err_path.write_text("\n".join(errors), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

