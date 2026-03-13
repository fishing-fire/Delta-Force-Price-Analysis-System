from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.paths import rel_posix


EXOG_COLS = {
    "is_holiday",
    "in_CS",
    "is_CS",
    "is_need",
    "is_make",
    "is_active",
    "is_public",
}


def _read_header_cols(csv_path: Path) -> list[str]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        return next(reader)


def _infer_dims(cols: list[str]) -> tuple[int, int, list[str], list[str]]:
    cols_ = cols[:]
    if cols_ and cols_[0].strip().lower() == "date":
        cols_ = cols_[1:]
    present_exog = [c for c in cols_ if c in EXOG_COLS]
    target_cols = [c for c in cols_ if c not in present_exog]
    return len(cols_), len(target_cols), present_exog, target_cols


def _slug(s: str) -> str:
    s = s.strip()
    s = s.replace(" ", "")
    s = s.replace("\t", "")
    s = s.replace(":", "_")
    s = s.replace("/", "_")
    s = s.replace("\\", "_")
    s = re.sub(r"[<>\"|?*]", "_", s)
    return s


def _ascii_slug(s: str) -> str:
    t = _slug(s)
    t = re.sub(r"[^0-9A-Za-z._-]+", "_", t)
    t = t.strip("._-")
    if t:
        return t
    return "ds_" + s.encode("utf-8").hex()[:24]


@dataclass(frozen=True)
class BenchConfig:
    seq_len: int
    label_len: int
    pred_len: int
    d_model: int
    n_heads: int
    e_layers: int
    d_layers: int
    d_ff: int
    dropout: float
    learning_rate: float
    train_epochs: int
    batch_size: int
    patience: int
    num_workers: int
    use_amp: bool
    lradj: str
    loss: str
    skip_existing: bool


def _iter_csv_files(dataset_dir: Path, selected: set[str] | None) -> Iterable[Path]:
    for p in sorted(dataset_dir.glob("*.csv")):
        stem = p.stem
        if selected is None or stem in selected:
            yield p


def _run_one(
    timexer_root: Path,
    run_dir: Path,
    dataset_path: Path,
    dataset_id: str,
    model: str,
    seed: int,
    cfg: BenchConfig,
) -> tuple[str, int]:
    model_id = f"{dataset_id}_{model}_s{seed}"
    metrics_path = run_dir / "results" / model_id / "metrics.csv"
    if cfg.skip_existing and metrics_path.exists():
        return model_id, 0

    root_path_rel = os.path.relpath(dataset_path.parent, start=run_dir).replace("\\", "/")
    cmd = [
        sys.executable,
        str(timexer_root / "run.py"),
        "--task_name",
        "long_term_forecast",
        "--is_training",
        "1",
        "--model_id",
        model_id,
        "--model",
        model,
        "--data",
        "custom",
        "--root_path",
        root_path_rel,
        "--data_path",
        dataset_path.name,
        "--features",
        "M",
        "--freq",
        "h",
        "--checkpoints",
        "./checkpoints/",
        "--seq_len",
        str(cfg.seq_len),
        "--label_len",
        str(cfg.label_len),
        "--pred_len",
        str(cfg.pred_len),
        "--d_model",
        str(cfg.d_model),
        "--n_heads",
        str(cfg.n_heads),
        "--e_layers",
        str(cfg.e_layers),
        "--d_layers",
        str(cfg.d_layers),
        "--d_ff",
        str(cfg.d_ff),
        "--dropout",
        str(cfg.dropout),
        "--learning_rate",
        str(cfg.learning_rate),
        "--train_epochs",
        str(cfg.train_epochs),
        "--batch_size",
        str(cfg.batch_size),
        "--patience",
        str(cfg.patience),
        "--num_workers",
        str(cfg.num_workers),
        "--lradj",
        cfg.lradj,
        "--loss",
        cfg.loss,
        "--seed",
        str(seed),
        "--des",
        "BenchV2",
        "--extra_tag",
        "no_full_vis",
    ]
    if cfg.use_amp:
        cmd.append("--use_amp")

    env = os.environ.copy()
    proc = subprocess.run(cmd, cwd=str(run_dir), env=env)
    return model_id, proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, default="TimeXer,iTransformer,PatchTST,TimesNet,DLinear,TSMixer,TimeMixer")
    parser.add_argument("--seeds", type=str, default="1,2,3")
    parser.add_argument("--datasets", type=str, default="all")
    parser.add_argument("--skip_existing", action="store_true", default=True)
    parser.add_argument("--no_skip_existing", action="store_true", default=False)
    parser.add_argument("--train_epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--use_amp", action="store_true", default=True)
    parser.add_argument("--no_amp", action="store_true", default=False)
    args = parser.parse_args()

    timexer_root = Path(__file__).resolve().parents[1]
    dataset_dir = timexer_root / "dataset" / "bullet" / "collection_category"
    base_dir = timexer_root / "backup" / "collection_category_benchmark_v2"

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    selected_datasets: set[str] | None
    if args.datasets.strip().lower() == "all":
        selected_datasets = None
    else:
        selected_datasets = {d.strip() for d in args.datasets.split(",") if d.strip()}

    skip_existing = args.skip_existing and not args.no_skip_existing
    use_amp = args.use_amp and not args.no_amp

    cfg = BenchConfig(
        seq_len=96,
        label_len=48,
        pred_len=96,
        d_model=512,
        n_heads=8,
        e_layers=2,
        d_layers=1,
        d_ff=2048,
        dropout=0.1,
        learning_rate=float(args.learning_rate),
        train_epochs=int(args.train_epochs),
        batch_size=int(args.batch_size),
        patience=int(args.patience),
        num_workers=8,
        use_amp=use_amp,
        lradj="plateau",
        loss="MSE",
        skip_existing=skip_existing,
    )

    base_dir.mkdir(parents=True, exist_ok=True)
    (base_dir / "runs_failed.jsonl").touch(exist_ok=True)
    (base_dir / "runs_ok.jsonl").touch(exist_ok=True)

    (base_dir / "benchmark_config.json").write_text(
        json.dumps(
            {
                "models": models,
                "seeds": seeds,
                "dataset_dir": rel_posix(dataset_dir, timexer_root),
                "base_dir": rel_posix(base_dir, timexer_root),
                "fixed_args": {
                    "task_name": "long_term_forecast",
                    "data": "custom",
                    "features": "M",
                    "freq": "h",
                },
                "hyperparams": {
                    "seq_len": cfg.seq_len,
                    "label_len": cfg.label_len,
                    "pred_len": cfg.pred_len,
                    "d_model": cfg.d_model,
                    "n_heads": cfg.n_heads,
                    "e_layers": cfg.e_layers,
                    "d_layers": cfg.d_layers,
                    "d_ff": cfg.d_ff,
                    "dropout": cfg.dropout,
                    "learning_rate": cfg.learning_rate,
                    "train_epochs": cfg.train_epochs,
                    "batch_size": cfg.batch_size,
                    "patience": cfg.patience,
                    "num_workers": cfg.num_workers,
                    "use_amp": cfg.use_amp,
                    "lradj": cfg.lradj,
                    "loss": cfg.loss,
                    "skip_existing": cfg.skip_existing,
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    csv_files = list(_iter_csv_files(dataset_dir, selected_datasets))
    if not csv_files:
        raise FileNotFoundError(f"No CSV found in {dataset_dir}")

    for csv_path in csv_files:
        cols = _read_header_cols(csv_path)
        enc_in, c_out, present_exog, target_cols = _infer_dims(cols)
        dataset_key = _slug(csv_path.stem)
        dataset_id = _ascii_slug(csv_path.stem)

        for model in models:
            run_dir = base_dir / model / dataset_key
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "dataset_info.json").write_text(
                json.dumps(
                    {
                            "dataset_csv": rel_posix(csv_path, timexer_root),
                        "dataset_key": dataset_key,
                        "dataset_id": dataset_id,
                        "enc_in": enc_in,
                        "c_out": c_out,
                        "present_exog": present_exog,
                        "target_cols": target_cols,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            for seed in seeds:
                model_id, code = _run_one(
                    timexer_root=timexer_root,
                    run_dir=run_dir,
                    dataset_path=csv_path,
                    dataset_id=dataset_id,
                    model=model,
                    seed=seed,
                    cfg=cfg,
                )

                record = {
                    "dataset": dataset_key,
                    "model": model,
                    "seed": seed,
                    "model_id": model_id,
                    "run_dir": rel_posix(run_dir, timexer_root),
                    "returncode": code,
                }
                if code == 0:
                    with (base_dir / "runs_ok.jsonl").open("a", encoding="utf-8") as f:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                else:
                    record["enc_in"] = enc_in
                    record["c_out"] = c_out
                    with (base_dir / "runs_failed.jsonl").open("a", encoding="utf-8") as f:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

