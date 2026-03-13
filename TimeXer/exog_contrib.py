from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data_provider.data_factory import data_provider
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from utils.metrics import metric


EXOG_COLS = [
    "is_holiday",
    "in_CS",
    "is_CS",
    "is_need",
    "is_make",
    "is_active",
    "is_public",
]


def _read_meta(log_path: Path) -> dict | None:
    if not log_path.exists():
        return None
    try:
        with log_path.open("r", encoding="utf-8", errors="ignore") as f:
            for _ in range(1200):
                line = f.readline()
                if not line:
                    break
                line = line.lstrip("\ufeff").strip()
                if line.startswith("META:"):
                    return json.loads(line[len("META:") :].strip())
    except Exception:
        return None
    return None


def _find_arg_value(log_text: str, key: str) -> str | None:
    m = re.search(rf"{re.escape(key)}\s*:\s*([^\r\n]+)", log_text)
    if not m:
        return None
    return m.group(1).strip()


def _read_log_text(log_path: Path) -> str:
    if not log_path.exists():
        return ""
    return log_path.read_text(encoding="utf-8", errors="ignore")


def _infer_exog_indices(data_csv: Path) -> list[tuple[str, int]]:
    header = data_csv.read_text(encoding="utf-8-sig", errors="ignore").splitlines()[0]
    cols = [c.strip() for c in header.split(",") if c.strip()]
    if cols and cols[0].lower() == "date":
        cols = cols[1:]
    pairs = []
    for i, c in enumerate(cols):
        if c in EXOG_COLS:
            pairs.append((c, i))
    return pairs


def _align_outputs_and_y(outputs: torch.Tensor, batch_y: torch.Tensor, features: str) -> tuple[torch.Tensor, torch.Tensor]:
    target_dim = batch_y.shape[-1]
    out_dim = outputs.shape[-1]
    if out_dim < target_dim:
        raise RuntimeError(f"Model output channel mismatch: outputs={out_dim} < y={target_dim} (features={features})")
    if out_dim == target_dim:
        return outputs, batch_y
    if features == "MS":
        return outputs[:, :, -target_dim:], batch_y
    return outputs[:, :, :target_dim], batch_y


@dataclass(frozen=True)
class RunResult:
    mae: float
    mse: float
    rmse: float
    mape: float
    mspe: float


def _to_result(preds: np.ndarray, trues: np.ndarray) -> RunResult:
    mae, mse, rmse, mape, mspe = metric(preds, trues)
    return RunResult(float(mae), float(mse), float(rmse), float(mape), float(mspe))


def _infer_with_mask(exp: Exp_Long_Term_Forecast, mask_idx: int | None) -> tuple[np.ndarray, np.ndarray]:
    test_data, test_loader = exp._get_data(flag="test")
    preds: list[np.ndarray] = []
    trues: list[np.ndarray] = []

    exp.model.eval()
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
            batch_x = batch_x.float().to(exp.device)
            batch_y = batch_y.float().to(exp.device)
            batch_x_mark = batch_x_mark.float().to(exp.device)
            batch_y_mark = batch_y_mark.float().to(exp.device)

            if mask_idx is not None:
                batch_x[:, :, mask_idx] = 0.0

            dec_inp = torch.zeros_like(batch_y[:, -exp.args.pred_len :, :]).float()
            dec_inp = torch.cat([batch_y[:, : exp.args.label_len, :], dec_inp], dim=1).float().to(exp.device)

            if exp.args.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            else:
                outputs = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            outputs = outputs[:, -exp.args.pred_len :, :]
            batch_y = batch_y[:, -exp.args.pred_len :, :]
            outputs, batch_y = _align_outputs_and_y(outputs, batch_y, exp.args.features)

            outputs_np = outputs.detach().cpu().numpy()
            batch_y_np = batch_y.detach().cpu().numpy()

            if getattr(test_data, "scale", False) and getattr(exp.args, "inverse", False):
                if exp.args.features == "MS":
                    outputs_np = np.tile(outputs_np, [1, 1, batch_y_np.shape[-1]])
                if hasattr(test_data, "inverse_transform_y") and outputs_np.shape[-1] == batch_y_np.shape[-1]:
                    outputs_np = test_data.inverse_transform_y(outputs_np.reshape(-1, outputs_np.shape[-1])).reshape(
                        outputs_np.shape
                    )
                    batch_y_np = test_data.inverse_transform_y(batch_y_np.reshape(-1, batch_y_np.shape[-1])).reshape(
                        batch_y_np.shape
                    )
                else:
                    shape = batch_y_np.shape
                    outputs_np = test_data.inverse_transform(outputs_np.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y_np = test_data.inverse_transform(batch_y_np.reshape(shape[0] * shape[1], -1)).reshape(shape)

            f_dim = -1 if exp.args.features == "MS" else 0
            outputs_np = outputs_np[:, :, f_dim:]
            batch_y_np = batch_y_np[:, :, f_dim:]

            preds.append(outputs_np)
            trues.append(batch_y_np)

    preds_all = np.concatenate(preds, axis=0).reshape(-1, exp.args.pred_len, preds[0].shape[-1])
    trues_all = np.concatenate(trues, axis=0).reshape(-1, exp.args.pred_len, trues[0].shape[-1])
    return preds_all, trues_all


def _build_args_from_meta(meta: dict, exp_name: str, dataset_root: Path, data_path: str) -> argparse.Namespace:
    args = argparse.Namespace()
    args.task_name = meta.get("task_name", "long_term_forecast")
    args.is_training = 0
    args.model_id = exp_name
    args.model = meta.get("model", "TimeXer")
    args.data = meta.get("data", "custom")
    args.root_path = str(dataset_root)
    args.data_path = data_path
    args.features = meta.get("features", "M")
    args.target = meta.get("target", "OT")
    args.freq = meta.get("freq", "h")
    args.checkpoints = ""
    args.seq_len = int(meta.get("seq_len", 96))
    args.label_len = int(meta.get("label_len", 48))
    args.pred_len = int(meta.get("pred_len", 96))
    args.seasonal_patterns = "Monthly"
    args.inverse = True
    args.expand = int(meta.get("expand", 2))
    args.d_conv = int(meta.get("d_conv", 4))
    args.top_k = 5
    args.num_kernels = 6
    args.enc_in = int(meta.get("enc_in", 7))
    args.dec_in = int(meta.get("dec_in", 7))
    args.c_out = int(meta.get("c_out", 7))
    args.d_model = int(meta.get("d_model", 512))
    args.n_heads = int(meta.get("n_heads", 8))
    args.e_layers = int(meta.get("e_layers", 2))
    args.d_layers = int(meta.get("d_layers", 1))
    args.d_ff = int(meta.get("d_ff", 2048))
    args.moving_avg = 25
    args.factor = int(meta.get("factor", 1))
    args.distil = bool(meta.get("distil", True))
    args.dropout = float(meta.get("dropout", 0.1))
    args.embed = meta.get("embed", "timeF")
    args.activation = "gelu"
    args.output_attention = False
    args.channel_independence = 1
    args.decomp_method = "moving_avg"
    args.use_norm = 1
    args.down_sampling_layers = 0
    args.down_sampling_window = 1
    args.down_sampling_method = None
    args.seg_len = 48
    args.num_workers = int(meta.get("num_workers", 8))
    args.itr = 1
    args.train_epochs = 1
    args.batch_size = int(meta.get("batch_size", 64))
    args.patience = 1
    args.learning_rate = float(meta.get("learning_rate", 0.0001))
    args.des = meta.get("des", "")
    args.loss = meta.get("loss", "MSE")
    args.lradj = meta.get("lradj", "plateau")
    args.use_amp = bool(meta.get("use_amp", True))
    args.use_gpu = True
    args.gpu = 0
    args.use_multi_gpu = False
    args.devices = "0"
    args.device_ids = [0]
    args.p_hidden_dims = [128, 128]
    args.p_hidden_layers = 2
    args.use_dtw = False
    args.augmentation_ratio = 0
    args.seed = 2
    args.jitter = False
    args.scaling = False
    args.permutation = False
    args.randompermutation = False
    args.magwarp = False
    args.timewarp = False
    args.windowslice = False
    args.windowwarp = False
    args.rotation = False
    args.spawner = False
    args.dtwwarp = False
    args.shapedtwwarp = False
    args.wdba = False
    args.discdtw = False
    args.discsdtw = False
    args.extra_tag = "no_full_vis"
    args.patch_len = 16
    args.results_path = os.path.abspath(str(dataset_root))
    return args


def _load_meta_fallback(log_text: str, meta: dict | None) -> dict:
    out = {} if not meta else dict(meta)
    dp = _find_arg_value(log_text, "Data Path")
    if dp is not None:
        out["data_path"] = dp.split()[0]
    tgt = _find_arg_value(log_text, "Target")
    if tgt is not None:
        out["target"] = tgt.split()[0]
    freq = _find_arg_value(log_text, "Freq")
    if freq is not None:
        out["freq"] = freq.split()[0]
    dropout = _find_arg_value(log_text, "Dropout")
    if dropout is not None:
        out["dropout"] = float(dropout.split()[0])
    lr = _find_arg_value(log_text, "Learning Rate")
    if lr is not None:
        out["learning_rate"] = float(lr.split()[0])
    nw = _find_arg_value(log_text, "Num Workers")
    if nw is not None:
        out["num_workers"] = int(nw.split()[0])
    bs = _find_arg_value(log_text, "Batch Size")
    if bs is not None:
        out["batch_size"] = int(bs.split()[0])
    ua = _find_arg_value(log_text, "Use Amp")
    if ua is not None:
        out["use_amp"] = bool(int(ua.split()[0]))
    inv = _find_arg_value(log_text, "Inverse")
    if inv is not None:
        out["inverse"] = bool(int(inv.split()[0]))
    return out


def _run_one(backup_root: Path, exp_name: str) -> Path:
    exp_dir = backup_root / "Collection_有外生变量" / "results" / exp_name
    ckpt_dir = backup_root / "Collection_有外生变量" / "checkpoints" / exp_name
    ckpt_path = ckpt_dir / "checkpoint.pth"

    log_path = exp_dir / "log.txt"
    log_text = _read_log_text(log_path)
    meta = _read_meta(log_path)
    meta2 = _load_meta_fallback(log_text, meta)

    data_path = meta2.get("data_path")
    if not data_path:
        raise RuntimeError(f"无法从 log.txt 推断 Data Path: {log_path}")

    dataset_root = Path(__file__).resolve().parent / "dataset" / "bullet" / "collection_category"
    data_csv = dataset_root / data_path
    if not data_csv.exists():
        raise RuntimeError(f"数据文件不存在: {data_csv}")

    exog_pairs = _infer_exog_indices(data_csv)
    if len(exog_pairs) == 0:
        raise RuntimeError(f"未在数据表头中找到外生变量列: {data_csv}")

    args = _build_args_from_meta(meta2, exp_name=exp_name, dataset_root=dataset_root, data_path=data_path)
    args.checkpoints = str(ckpt_dir.parent)

    train_data, _ = data_provider(args, flag="train")
    args.enc_in = int(train_data.data_x.shape[-1])
    args.dec_in = int(train_data.data_x.shape[-1])
    args.c_out = int(train_data.data_y.shape[-1])

    exp = Exp_Long_Term_Forecast(args)
    exp.model.load_state_dict(torch.load(ckpt_path, map_location=exp.device))

    baseline_pred = np.load(exp_dir / "pred.npy")
    baseline_true = np.load(exp_dir / "true.npy")
    baseline = _to_result(baseline_pred, baseline_true)

    rows = []
    for name, idx in exog_pairs:
        preds_m, trues_m = _infer_with_mask(exp, mask_idx=idx)
        r = _to_result(preds_m, trues_m)
        rows.append(
            {
                "exog": name,
                "mask_idx": idx,
                "baseline_mae": baseline.mae,
                "masked_mae": r.mae,
                "delta_mae": r.mae - baseline.mae,
                "baseline_rmse": baseline.rmse,
                "masked_rmse": r.rmse,
                "delta_rmse": r.rmse - baseline.rmse,
                "baseline_mape": baseline.mape,
                "masked_mape": r.mape,
                "delta_mape": r.mape - baseline.mape,
            }
        )

    out_dir = exp_dir / "exog_contrib"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).sort_values("delta_mae", ascending=False)
    df.to_csv(out_dir / "exog_contrib.csv", index=False, encoding="utf-8-sig")

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 4.5), dpi=140)
    ax = fig.add_subplot(111)
    ax.bar(df["exog"].tolist(), df["delta_mae"].tolist())
    ax.set_title(f"{exp_name} exog contribution (delta MAE, mask=0)")
    ax.set_ylabel("delta MAE = MAE(mask) - MAE(full)")
    ax.set_xlabel("exogenous variable")
    ax.tick_params(axis="x", labelrotation=30)
    fig.tight_layout()
    fig.savefig(out_dir / "exog_contrib_delta_mae.png")
    plt.close(fig)

    return out_dir


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backup_root",
        type=str,
        default=str(Path(__file__).resolve().parent / "backup" / "category compare collection_category"),
    )
    parser.add_argument("--exp_name", type=str, default="5.56_Collection_Category_Exp_Large_M_768_A")
    parser.add_argument("--all", action="store_true", default=False)
    args = parser.parse_args()

    backup_root = Path(args.backup_root)
    if args.all:
        results_root = backup_root / "Collection_有外生变量" / "results"
        for p in sorted([p for p in results_root.iterdir() if p.is_dir()]):
            _run_one(backup_root, p.name)
        return 0

    _run_one(backup_root, args.exp_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

