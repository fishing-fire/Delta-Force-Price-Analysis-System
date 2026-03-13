from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.paths import rel_link, rel_posix


def _mean(xs: list[float]) -> float | None:
    if not xs:
        return None
    return sum(xs) / len(xs)


def _std(xs: list[float]) -> float | None:
    if len(xs) < 2:
        return 0.0 if xs else None
    m = _mean(xs)
    if m is None:
        return None
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return math.sqrt(var)


def _read_metrics_csv(path: Path) -> dict[str, float] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        row = next(reader, None)
        if not row:
            return None
        return {
            "MAE": float(row["MAE"]),
            "MSE": float(row["MSE"]),
            "RMSE": float(row["RMSE"]),
            "MAPE": float(row["MAPE"]),
            "MSPE": float(row["MSPE"]),
        }


@dataclass(frozen=True)
class RunEntry:
    dataset: str
    model: str
    seed: int | None
    model_id: str
    run_dir: Path
    result_dir: Path
    metrics: dict[str, float]


def _iter_runs(base_dir: Path) -> list[RunEntry]:
    runs: list[RunEntry] = []
    for model_dir in sorted([p for p in base_dir.iterdir() if p.is_dir()]):
        for dataset_dir in sorted([p for p in model_dir.iterdir() if p.is_dir()]):
            results_dir = dataset_dir / "results"
            if not results_dir.exists():
                continue
            for model_id_dir in sorted([p for p in results_dir.iterdir() if p.is_dir()]):
                metrics = _read_metrics_csv(model_id_dir / "metrics.csv")
                if not metrics:
                    continue
                model_id = model_id_dir.name
                m = re.search(r"_s(\d+)$", model_id)
                seed = int(m.group(1)) if m else None
                runs.append(
                    RunEntry(
                        dataset=dataset_dir.name,
                        model=model_dir.name,
                        seed=seed,
                        model_id=model_id,
                        run_dir=dataset_dir,
                        result_dir=model_id_dir,
                        metrics=metrics,
                    )
                )
    return runs


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines: list[str] = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=str, default="all")
    args = parser.parse_args()

    timexer_root = Path(__file__).resolve().parents[1]
    base_dir = timexer_root / "backup" / "collection_category_benchmark_v2"
    if not base_dir.exists():
        raise FileNotFoundError(str(base_dir))

    runs = _iter_runs(base_dir)
    if args.seeds.strip().lower() != "all":
        allowed = {int(s.strip()) for s in args.seeds.split(",") if s.strip()}
        runs = [r for r in runs if r.seed in allowed]
    if not runs:
        raise RuntimeError(f"No valid runs found under {base_dir}")

    compare_csv = base_dir / "metrics_compare.csv"
    with compare_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "model",
                "seed",
                "model_id",
                "MAE",
                "MSE",
                "RMSE",
                "MAPE",
                "MSPE",
                "run_dir",
                "result_dir",
                "log_path",
                "metrics_detail_path",
                "test_pdf_path",
                "checkpoint_path",
            ],
        )
        writer.writeheader()
        for r in runs:
            run_dir_rel = rel_posix(r.run_dir, timexer_root)
            result_dir_rel = rel_posix(r.result_dir, timexer_root)
            log_rel = rel_posix(r.result_dir / "log.txt", timexer_root)
            detail_rel = rel_posix(r.result_dir / "metrics_detail.csv", timexer_root)
            pdf_rel = rel_posix(r.run_dir / "test_results" / r.model_id / "0.pdf", timexer_root)
            ckpt_rel = rel_posix(r.run_dir / "checkpoints" / r.model_id / "checkpoint.pth", timexer_root)
            writer.writerow(
                {
                    "dataset": r.dataset,
                    "model": r.model,
                    "seed": r.seed,
                    "model_id": r.model_id,
                    "MAE": r.metrics["MAE"],
                    "MSE": r.metrics["MSE"],
                    "RMSE": r.metrics["RMSE"],
                    "MAPE": r.metrics["MAPE"],
                    "MSPE": r.metrics["MSPE"],
                    "run_dir": run_dir_rel,
                    "result_dir": result_dir_rel,
                    "log_path": log_rel,
                    "metrics_detail_path": detail_rel,
                    "test_pdf_path": pdf_rel,
                    "checkpoint_path": ckpt_rel,
                }
            )

    grouped: dict[tuple[str, str], list[RunEntry]] = {}
    for r in runs:
        grouped.setdefault((r.dataset, r.model), []).append(r)

    dataset_models: dict[str, list[str]] = {}
    for dataset, model in grouped.keys():
        dataset_models.setdefault(dataset, []).append(model)

    summaries: list[dict[str, Any]] = []
    for (dataset, model), entries in grouped.items():
        maes = [e.metrics["MAE"] for e in entries]
        rmses = [e.metrics["RMSE"] for e in entries]
        summaries.append(
            {
                "dataset": dataset,
                "model": model,
                "runs": len(entries),
                "MAE_mean": _mean(maes),
                "MAE_std": _std(maes),
                "RMSE_mean": _mean(rmses),
                "RMSE_std": _std(rmses),
            }
        )

    summary_csv = base_dir / "summary_by_dataset_model.csv"
    with summary_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "model",
                "runs",
                "MAE_mean",
                "MAE_std",
                "RMSE_mean",
                "RMSE_std",
            ],
        )
        writer.writeheader()
        for row in sorted(summaries, key=lambda x: (x["dataset"], x["model"])):
            writer.writerow(row)

    ranks: dict[tuple[str, str], int] = {}
    best_by_dataset: dict[str, dict[str, Any]] = {}
    for dataset in sorted(dataset_models.keys()):
        rows = [r for r in summaries if r["dataset"] == dataset and isinstance(r["MAE_mean"], float)]
        rows_sorted = sorted(rows, key=lambda x: x["MAE_mean"])
        if rows_sorted:
            best_by_dataset[dataset] = rows_sorted[0]
        for i, row in enumerate(rows_sorted, start=1):
            ranks[(dataset, row["model"])] = i

    models = sorted({r.model for r in runs})
    avg_rank: dict[str, float] = {}
    win_count: dict[str, int] = {m: 0 for m in models}
    for dataset, best in best_by_dataset.items():
        win_count[best["model"]] = win_count.get(best["model"], 0) + 1
    for model in models:
        rs = [rank for (ds, m), rank in ranks.items() if m == model]
        avg_rank[model] = float(sum(rs) / len(rs)) if rs else float("inf")

    def fmt_pm(mean: float | None, std: float | None) -> str:
        if mean is None or std is None:
            return "-"
        return f"{mean:.4f}±{std:.4f}"

    def first_ok_run(dataset: str, model: str) -> RunEntry | None:
        xs = [r for r in runs if r.dataset == dataset and r.model == model]
        if not xs:
            return None
        return sorted(xs, key=lambda e: e.seed if e.seed is not None else 0)[0]

    now = datetime.now()
    report_path = base_dir / f"benchmark_report_{now:%Y%m%d_%H%M%S}.md"
    lines: list[str] = []
    lines.append("# collection_category 多模型对比报告（统一参数，M+外生）")
    lines.append("")
    lines.append(f"- 生成时间：{now:%Y-%m-%d %H:%M:%S}")
    lines.append(f"- 结果目录：{rel_posix(base_dir, timexer_root)}")
    lines.append(f"- 运行数：{len(runs)}")
    lines.append("")
    lines.append("## 总览（按 MAE 的平均排名/胜率）")
    lines.append("")
    overall_rows: list[list[str]] = []
    for model in sorted(models, key=lambda m: avg_rank.get(m, float("inf"))):
        overall_rows.append([model, f"{avg_rank[model]:.3f}", str(win_count.get(model, 0))])
    lines.append(_md_table(["模型", "平均排名(越小越好)", "数据集最优次数"], overall_rows))
    lines.append("")
    lines.append("## 各数据集结果（按 MAE_mean 排序）")
    lines.append("")
    for dataset in sorted(dataset_models.keys()):
        rows = [r for r in summaries if r["dataset"] == dataset and isinstance(r["MAE_mean"], float)]
        rows_sorted = sorted(rows, key=lambda x: x["MAE_mean"])
        if not rows_sorted:
            continue
        lines.append(f"### {dataset}")
        lines.append("")
        table_rows: list[list[str]] = []
        for row in rows_sorted:
            table_rows.append(
                [
                    row["model"],
                    fmt_pm(row["MAE_mean"], row["MAE_std"]),
                    fmt_pm(row["RMSE_mean"], row["RMSE_std"]),
                    str(row["runs"]),
                ]
            )
        lines.append(_md_table(["模型", "MAE(mean±std)", "RMSE(mean±std)", "runs"], table_rows))
        lines.append("")
        best = rows_sorted[0]
        lines.append(f"- 最优：{best['model']}（MAE_mean={best['MAE_mean']:.4f}）")
        if len(rows_sorted) > 1:
            second = rows_sorted[1]
            delta = second["MAE_mean"] - best["MAE_mean"]
            rel = delta / second["MAE_mean"] * 100 if second["MAE_mean"] else None
            rel_s = f"{rel:.2f}%" if rel is not None else "-"
            lines.append(f"- 与次优差距：MAE 改善 {delta:+.4f}（相对 {rel_s}）")
        ex = first_ok_run(dataset, best["model"])
        if ex:
            log_p = ex.result_dir / "log.txt"
            detail_p = ex.result_dir / "metrics_detail.csv"
            pdf_p = ex.run_dir / "test_results" / ex.model_id / "0.pdf"
            parts = []
            parts.append(f"[log.txt]({rel_link(log_p, base_dir)})" if log_p.exists() else "-")
            parts.append(f"[metrics_detail.csv]({rel_link(detail_p, base_dir)})" if detail_p.exists() else "-")
            parts.append(f"[test_report.pdf]({rel_link(pdf_p, base_dir)})" if pdf_p.exists() else "-")
            lines.append("- 样例产物：" + " ".join(parts))
        lines.append("")

    lines.append("## 复现实用清单")
    lines.append("")
    lines.append(f"- 明细指标：[{compare_csv.name}]({rel_link(compare_csv, base_dir)})")
    lines.append(f"- 汇总表：[{summary_csv.name}]({rel_link(summary_csv, base_dir)})")
    lines.append(f"- 逐数据集汇总：[{summary_csv.name}]({rel_link(summary_csv, base_dir)})")

    report_text = "\n".join(lines)
    report_path.write_text(report_text, encoding="utf-8")
    (base_dir / "latest_report.md").write_text(report_text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

