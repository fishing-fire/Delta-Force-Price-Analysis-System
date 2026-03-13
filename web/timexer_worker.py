import json
import os
import re
import sys
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional
import unicodedata

import numpy as np


EXOG_COLS = [
    "is_holiday",
    "in_CS",
    "is_CS",
    "is_need",
    "is_make",
    "is_active",
    "is_public",
]


def _write(obj):
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


if hasattr(sys.stdin, "reconfigure"):
    sys.stdin.reconfigure(encoding="utf-8")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


@dataclass(frozen=True)
class ModelConfig:
    model_group: str
    model_id: str
    category_csv: str
    dropout: float


def _project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _timexer_root():
    return os.path.join(_project_root(), "TimeXer")


def _default_data_dir():
    return os.path.join(_project_root(), "测试新子弹价格数据", "总子弹数据")


def _default_category_exog_dir():
    return os.path.join(_project_root(), "测试新子弹价格数据", "带外生变量的总子弹数据")


MODEL_CONFIGS = {
    "5.56x45mm": ModelConfig(
        model_group="5.56x45mm",
        model_id="V18_7d_556x45mm_Collection_Category_Exp_Long_M_336_A_B_Best_LRDown_M_336_D512_P168",
        category_csv="5.56x45mm.csv",
        dropout=0.3,
    ),
    ".300BLK": ModelConfig(
        model_group=".300BLK",
        model_id="V18_7d_300BLK_Collection_Category_Opt2_192_Plateau4_B_Best_LRDown_M_192_D512_P168",
        category_csv=".300BLK.csv",
        dropout=0.1,
    ),
    "箭矢": ModelConfig(
        model_group="箭矢",
        model_id="V18_7d_Arrow_Collection_Category_Deep6_240_B_Best_LRDown_M_240_D512_P168",
        category_csv="箭矢.csv",
        dropout=0.1,
    ),
    "9x19mm": ModelConfig(
        model_group="9x19mm",
        model_id="V18_7d_9x19mm_Collection_Category_MidLarge_336_D_Best_LongSeq_M_336_D640_P168",
        category_csv="9x19mm.csv",
        dropout=0.1,
    ),
    "9x39mm": ModelConfig(
        model_group="9x39mm",
        model_id="V18_7d_9x39mm_Collection_Category_Opt2_192_Plateau4_B_Best_LRDown_M_192_D512_P168",
        category_csv="9x39mm.csv",
        dropout=0.1,
    ),
    "7.62x39mm": ModelConfig(
        model_group="7.62x39mm",
        model_id="V18_7d_762x39mm_Collection_Category_Opt2_192_Plateau4_B_Best_LRDown_M_192_D512_P168",
        category_csv="7.62x39mm.csv",
        dropout=0.1,
    ),
    "7.62x51mm": ModelConfig(
        model_group="7.62x51mm",
        model_id="V18_7d_762x51mm_Collection_Category_Deep6_240_B_Best_LRDown_M_240_D512_P168",
        category_csv="7.62x51mm.csv",
        dropout=0.1,
    ),
    "7.62x54R": ModelConfig(
        model_group="7.62x54R",
        model_id="V17_7d_762x54R_Collection_Category_Exp_Long_M_336_A_M_336_D512_P168",
        category_csv="7.62x54R.csv",
        dropout=0.1,
    ),
    "5.45x39mm": ModelConfig(
        model_group="5.45x39mm",
        model_id="V17_7d_545x39mm_Collection_Category_MidLarge_640_M_240_D640_P168",
        category_csv="5.45x39mm.csv",
        dropout=0.1,
    ),
    "5.7x28mm": ModelConfig(
        model_group="5.7x28mm",
        model_id="V17_7d_57x28mm_Collection_Category_Deep6_240_M_240_D512_P168",
        category_csv="5.7x28mm.csv",
        dropout=0.1,
    ),
    "5.8x42mm": ModelConfig(
        model_group="5.8x42mm",
        model_id="V18_7d_58x42mm_Collection_Category_Deep6_240_B_Best_LRDown_M_240_D512_P168",
        category_csv="5.8x42mm.csv",
        dropout=0.1,
    ),
    "6.8x51mm": ModelConfig(
        model_group="6.8x51mm",
        model_id="V17_7d_68x51mm_Collection_Category_Deep6_240_M_240_D512_P168",
        category_csv="6.8x51mm.csv",
        dropout=0.1,
    ),
    "4.6x30mm": ModelConfig(
        model_group="4.6x30mm",
        model_id="V18_7d_46x30mm_Collection_Category_Opt2_192_Plateau4_E_Best_Patch32_M_192_D512_P168",
        category_csv="4.6x30mm.csv",
        dropout=0.1,
    ),
    "12.7x55mm": ModelConfig(
        model_group="12.7x55mm",
        model_id="V18_7d_127x55mm_Collection_Category_Deep6_240_C_Best_LRUp_M_240_D512_P168",
        category_csv="12.7x55mm.csv",
        dropout=0.1,
    ),
    "12 Gauge": ModelConfig(
        model_group="12 Gauge",
        model_id="V18_7d_12Gauge_Collection_Category_LongCtx2_480_D_Best_LongSeq_M_480_D512_P168",
        category_csv="12 Gauge.csv",
        dropout=0.1,
    ),
    ".357 Magnum": ModelConfig(
        model_group=".357 Magnum",
        model_id="V18_7d_357Magnum_Collection_Category_LongCtx2_384_E_Best_Patch32_M_384_D512_P168",
        category_csv=".357 Magnum.csv",
        dropout=0.1,
    ),
    "45-70 Govt": ModelConfig(
        model_group="45-70 Govt",
        model_id="V17_7d_4570Govt_Collection_Category_Exp_Long_M_336_A_M_336_D512_P168",
        category_csv="45-70 Govt.csv",
        dropout=0.1,
    ),
    ".45 ACP": ModelConfig(
        model_group=".45 ACP",
        model_id="V18_7d_45ACP_Collection_Category_Exp_Long_M_336_A_B_Best_LRDown_M_336_D512_P168",
        category_csv=".45 ACP.csv",
        dropout=0.1,
    ),
    ".50 AE": ModelConfig(
        model_group=".50 AE",
        model_id="V17_7d_50AE_Collection_Category_Opt2_192_Plateau4_M_192_D512_P168",
        category_csv=".50 AE.csv",
        dropout=0.1,
    ),
}


sys.path.insert(0, _timexer_root())

import torch  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast  # noqa: E402
from utils.timefeatures import time_features  # noqa: E402


def _load_meta(model_id: str):
    log_path = os.path.join(_timexer_root(), "results", model_id, "log.txt")
    with open(log_path, "r", encoding="utf-8") as f:
        first = f.readline().strip()
    if not first.startswith("META:"):
        raise RuntimeError(f"未找到 META 配置: {log_path}")
    return json.loads(first[len("META:") :].strip())


def _build_args_from_meta(meta: dict, enc_in: int, dec_in: int, c_out: int, model_id: str, dropout: float):
    distil_val = meta.get("distil", True)
    if isinstance(distil_val, str):
        distil_val = distil_val.lower() == "true"

    patch_len_val = meta.get("patch_len")
    if patch_len_val is None:
        probe = f"{meta.get('des', '')} {meta.get('model_id', '')} {model_id}"
        m = re.search(r"Patch(\d+)", probe)
        patch_len_val = int(m.group(1)) if m else 16
    else:
        patch_len_val = int(patch_len_val)

    return SimpleNamespace(
        task_name=meta.get("task_name", "long_term_forecast"),
        is_training=0,
        model_id=model_id,
        model=meta.get("model", "TimeXer"),
        data=meta.get("data", "custom"),
        root_path=".",
        data_path=".",
        features=meta.get("features", "M"),
        target="OT",
        freq="h",
        checkpoints=os.path.join(_timexer_root(), "checkpoints"),
        seq_len=int(meta.get("seq_len", 96)),
        label_len=int(meta.get("label_len", 48)),
        pred_len=int(meta.get("pred_len", 72)),
        seasonal_patterns="Monthly",
        inverse=True,
        expand=int(meta.get("expand", 2)),
        d_conv=int(meta.get("d_conv", 4)),
        top_k=5,
        num_kernels=6,
        enc_in=int(enc_in),
        dec_in=int(dec_in),
        c_out=int(c_out),
        d_model=int(meta.get("d_model", 512)),
        n_heads=int(meta.get("n_heads", 8)),
        e_layers=int(meta.get("e_layers", 4)),
        d_layers=int(meta.get("d_layers", 2)),
        d_ff=int(meta.get("d_ff", 2048)),
        moving_avg=25,
        factor=int(meta.get("factor", 3)),
        distil=distil_val,
        dropout=float(dropout),
        embed=meta.get("embed", "timeF"),
        activation="gelu",
        output_attention=False,
        channel_independence=1,
        decomp_method="moving_avg",
        use_norm=1,
        down_sampling_layers=0,
        down_sampling_window=1,
        down_sampling_method=None,
        seg_len=48,
        num_workers=0,
        itr=1,
        train_epochs=1,
        batch_size=1,
        patience=1,
        learning_rate=0.0001,
        des=meta.get("des", "web_infer"),
        loss="MSE",
        lradj="plateau",
        use_amp=False,
        use_gpu=True,
        gpu=0,
        use_multi_gpu=False,
        devices="0",
        device_ids=[0],
        p_hidden_dims=[128, 128],
        p_hidden_layers=2,
        use_dtw=False,
        augmentation_ratio=0,
        seed=2021,
        jitter=False,
        scaling=False,
        permutation=False,
        randompermutation=False,
        magwarp=False,
        timewarp=False,
        windowslice=False,
        windowwarp=False,
        rotation=False,
        spawner=False,
        dtwwarp=False,
        shapedtwwarp=False,
        wdba=False,
        discdtw=False,
        discsdtw=False,
        extra_tag="",
        patch_len=patch_len_val,
    )


def _read_category_columns(category_csv: str):
    category_path = os.path.join(_timexer_root(), "dataset", "bullet", "category", category_csv)
    df_head = pd.read_csv(category_path, nrows=1)
    cols = []
    for c in df_head.columns:
        if c is None:
            continue
        name = str(c).strip().lstrip("\ufeff")
        if name == "date":
            continue
        cols.append(name)
    if not cols:
        raise RuntimeError(f"category 文件列为空: {category_path}")
    return cols


def _read_category_exog_info(category_exog_dir: str, category_csv: str):
    category_path = os.path.join(category_exog_dir, category_csv)
    df_head = pd.read_csv(category_path, nrows=1)
    cols = []
    for c in df_head.columns:
        if c is None:
            continue
        name = str(c).strip().lstrip("\ufeff")
        cols.append(name)
    if not cols or cols[0] != "date":
        raise RuntimeError(f"带外生口径文件表头异常: {category_path}")
    present_exog = [c for c in EXOG_COLS if c in cols]
    target_cols = [c for c in cols[1:] if c not in present_exog]
    if not target_cols:
        raise RuntimeError(f"带外生口径文件未找到 targets: {category_path}")
    input_cols = target_cols + present_exog
    return category_path, target_cols, present_exog, input_cols


def _norm_text(s):
    return unicodedata.normalize("NFKC", str(s)).strip().lstrip("\ufeff")


def _load_bullet_series(data_dir: str, bullet_name: str):
    file_path = os.path.join(data_dir, f"{bullet_name}.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)
    df = pd.read_csv(file_path)
    if "时间戳" not in df.columns or "均价" not in df.columns:
        raise RuntimeError(f"CSV 列不匹配: {file_path}")
    out = df[["时间戳", "均价"]].copy()
    out["时间戳"] = pd.to_numeric(out["时间戳"], errors="coerce").astype("Int64")
    out["均价"] = pd.to_numeric(out["均价"], errors="coerce")
    out = out.dropna(subset=["时间戳", "均价"])
    out = out.rename(columns={"时间戳": "ts", "均价": bullet_name})
    out["ts"] = out["ts"].astype(np.int64)
    out = out.groupby("ts", as_index=False).last()
    return out


def _build_multivariate_frame(data_dir: str, columns: list[str]):
    merged = None
    for col in columns:
        df_col = _load_bullet_series(data_dir, col)
        merged = df_col if merged is None else pd.merge(merged, df_col, on="ts", how="outer")

    merged = merged.sort_values("ts").reset_index(drop=True)
    if merged.empty:
        raise RuntimeError("合并后数据为空")

    start_ts = int(merged["ts"].min())
    end_ts = int(merged["ts"].max())
    step = 60 * 60 * 1000
    full_ts = np.arange(start_ts - (start_ts % step), end_ts + step, step, dtype=np.int64)

    merged = merged.set_index("ts").reindex(full_ts)
    merged.index.name = "ts"
    merged = merged.reset_index()
    for col in columns:
        merged[col] = merged[col].astype(float)
    merged[columns] = merged[columns].ffill().bfill()
    merged["date"] = pd.to_datetime(merged["ts"], unit="ms")
    return merged[["date", "ts"] + columns]


def _time_features(dates, freq: str):
    feats = time_features(pd.to_datetime(np.array(dates, dtype="datetime64[ns]")), freq=freq)
    return feats.transpose(1, 0).astype(np.float32)


class TimeXerRunner:
    def __init__(self):
        self._models = {}

    def _get_or_load(self, model_group: str, enc_in: int, dec_in: int, c_out: int):
        cfg = MODEL_CONFIGS.get(model_group)
        if not cfg:
            raise RuntimeError(f"未知模型组: {model_group}")

        key = (cfg.model_id, int(enc_in), int(dec_in), int(c_out))
        if key in self._models:
            return self._models[key]

        meta = _load_meta(cfg.model_id)
        args = _build_args_from_meta(meta, enc_in, dec_in, c_out, cfg.model_id, cfg.dropout)
        exp = Exp_Long_Term_Forecast(args)
        checkpoint_path = os.path.join(args.checkpoints, args.model_id, "checkpoint.pth")
        exp.model.load_state_dict(torch.load(checkpoint_path, map_location=exp.device))
        exp.model.eval()
        self._models[key] = (exp, args)
        return exp, args

    def forecast(self, model_group: str, bullet: str, data_dir: str, category_exog_dir: Optional[str]):
        cfg = MODEL_CONFIGS.get(model_group)
        if not cfg:
            raise RuntimeError(f"未知模型组: {model_group}")

        bullet_n = _norm_text(bullet)
        if category_exog_dir:
            category_path, target_cols, present_exog, input_cols = _read_category_exog_info(category_exog_dir, cfg.category_csv)
            col_norm_to_idx = {_norm_text(c): i for i, c in enumerate(target_cols)}
            if bullet_n not in col_norm_to_idx:
                raise RuntimeError(f"子弹不在该模型组范围内: {bullet_n}")

            df = pd.read_csv(category_path)
            if "date" not in df.columns:
                raise RuntimeError(f"带外生口径文件缺少 date: {category_path}")
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)
            for c in input_cols:
                if c not in df.columns:
                    raise RuntimeError(f"带外生口径文件缺少列 {c}: {category_path}")
                df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
            df[input_cols] = df[input_cols].ffill().bfill()
            exp, args = self._get_or_load(model_group, len(input_cols), len(target_cols), len(target_cols))
            values_x = df[input_cols].to_numpy(np.float32)
            values_y = df[target_cols].to_numpy(np.float32)
            target_count = len(target_cols)
        else:
            columns = _read_category_columns(cfg.category_csv)
            col_norm_to_idx = {_norm_text(c): i for i, c in enumerate(columns)}
            if bullet_n not in col_norm_to_idx:
                raise RuntimeError(f"子弹不在该模型组范围内: {bullet_n}")

            df = _build_multivariate_frame(data_dir, columns)
            exp, args = self._get_or_load(model_group, len(columns), len(columns), len(columns))
            values_x = df[columns].to_numpy(np.float32)
            values_y = values_x
            target_count = len(columns)

        seq_len = int(args.seq_len)
        label_len = int(args.label_len)
        pred_len = int(args.pred_len)

        n = values_x.shape[0]
        if n < seq_len:
            pad_x = np.repeat(values_x[:1], seq_len - n, axis=0)
            values_x = np.concatenate([pad_x, values_x], axis=0)
            pad_y = np.repeat(values_y[:1], seq_len - n, axis=0)
            values_y = np.concatenate([pad_y, values_y], axis=0)
            df = pd.concat(
                [df.iloc[:1].copy().assign(date=df.iloc[0]["date"]) for _ in range(seq_len - n)] + [df],
                ignore_index=True,
            )
            n = values_x.shape[0]

        train_end = max(int(n * 0.7), 1)
        scaler = StandardScaler()
        scaler.fit(values_x[:train_end])
        scaled_x = scaler.transform(values_x).astype(np.float32)

        x = scaled_x[-seq_len:, :]
        x_times = df["date"].iloc[-seq_len:].to_list()
        x_mark = _time_features(x_times, args.freq)

        if target_count == values_x.shape[1]:
            y_label = scaled_x[-label_len:, :]
        else:
            y_label = scaled_x[-label_len:, :target_count]
        last_time = pd.to_datetime(df["date"].iloc[-1])
        future_times = pd.date_range(last_time + pd.Timedelta(hours=1), periods=pred_len, freq="H").to_pydatetime().tolist()
        y_times = df["date"].iloc[-label_len:].to_list() + future_times
        y_mark = _time_features(y_times, args.freq)

        batch_x = torch.from_numpy(x[None, :, :]).float().to(exp.device)
        batch_x_mark = torch.from_numpy(x_mark[None, :, :]).float().to(exp.device)
        batch_y_mark = torch.from_numpy(y_mark[None, :, :]).float().to(exp.device)
        dec_zeros = torch.zeros((1, pred_len, y_label.shape[1]), dtype=torch.float32, device=exp.device)
        dec_inp = torch.cat([torch.from_numpy(y_label[None, :, :]).float().to(exp.device), dec_zeros], dim=1)

        with torch.no_grad():
            outputs = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            outputs = outputs[:, -pred_len:, :].detach().cpu().numpy().astype(np.float32)

        means = scaler.mean_.astype(np.float32)
        scales = scaler.scale_.astype(np.float32)
        inv = outputs * scales[None, None, :target_count] + means[None, None, :target_count]

        col_idx = col_norm_to_idx[bullet_n]
        pred_values = inv[0, :, int(col_idx)].tolist()
        pred_ts = [int(pd.Timestamp(t).value // 1_000_000) for t in future_times]

        return {
            "modelId": cfg.model_id,
            "predLen": pred_len,
            "points": [{"ts": ts, "price": float(v)} for ts, v in zip(pred_ts, pred_values)],
        }


def main():
    data_dir = os.environ.get("BULLET_DATA_DIR") or _default_data_dir()
    category_exog_dir = os.environ.get("BULLET_CATEGORY_EXOG_DIR")
    if category_exog_dir:
        category_exog_dir = os.path.abspath(category_exog_dir)
    elif os.path.isdir(_default_category_exog_dir()):
        category_exog_dir = os.path.abspath(_default_category_exog_dir())
    runner = TimeXerRunner()
    _write({"type": "ready"})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception:
            continue
        request_id = req.get("requestId")
        msg_type = req.get("type")
        if not request_id:
            continue

        try:
            if msg_type == "ping":
                _write({"ok": True, "requestId": request_id, "type": "pong"})
                continue
            if msg_type != "forecast":
                raise RuntimeError("未知请求类型")

            model_group = req.get("modelGroup")
            bullet = req.get("bullet")
            if not model_group or not bullet:
                raise RuntimeError("缺少 modelGroup 或 bullet")

            result = runner.forecast(model_group, bullet, data_dir, category_exog_dir)
            _write({"ok": True, "requestId": request_id, **result})
        except Exception as e:
            _write({"ok": False, "requestId": request_id, "error": str(e)})


if __name__ == "__main__":
    main()

