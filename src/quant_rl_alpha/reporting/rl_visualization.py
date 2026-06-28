from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_rl_alpha.alpha import AlphaPool, mutual_ic
from quant_rl_alpha.alpha.pool import align_values_to_labels
from quant_rl_alpha.expression import evaluate, parse_rpn
from quant_rl_alpha.expression.rpn import RPNError
from quant_rl_alpha.utils.config import load_config
from quant_rl_alpha.utils.paths import ensure_dir, project_root


@dataclass(frozen=True)
class RLFactorReportResult:
    report_path: Path
    mode: str
    pool_size: int
    target_pool_size: int
    missing_artifacts: tuple[str, ...] = ()


def build_rl_factor_report(config_name: str | Path = "rl") -> RLFactorReportResult:
    rl_config = load_config(config_name)
    alpha_config = load_config("alpha")
    target_pool_size = int(alpha_config["pool_size"])
    report_path = _project_path(rl_config["outputs"]["report"])
    artifact_paths = _artifact_paths(rl_config)
    missing = tuple(name for name, path in artifact_paths.items() if not path.is_file())

    if missing:
        html = _preflight_report(rl_config, alpha_config, artifact_paths, missing)
        result = RLFactorReportResult(report_path, "preflight", 0, target_pool_size, missing)
    else:
        html, pool_size = _final_report(rl_config, alpha_config, artifact_paths)
        result = RLFactorReportResult(report_path, "final", pool_size, target_pool_size)

    ensure_dir(report_path.parent)
    report_path.write_text(html, encoding="utf-8")
    return result


def write_pool_ic_live_report(
    pool: AlphaPool | pd.DataFrame,
    path: str | Path,
    *,
    iteration: int,
    total_iterations: int,
    refresh_seconds: int = 20,
) -> Path:
    report_path = _project_path(path)
    frame = pool.summary_frame() if isinstance(pool, AlphaPool) else pool.copy()
    columns = [column for column in ["name", "formula", "ic"] if column in frame.columns]
    display = frame.loc[:, columns].copy() if columns else pd.DataFrame()
    if "ic" in display.columns:
        display = display.sort_values("ic", key=lambda values: values.abs(), ascending=False)
    sections = [
        _section(
            "Status",
            _kv_table(
                {
                    "iteration": f"{iteration}/{total_iterations}",
                    "pool_size": len(frame),
                    "updated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
            ),
        ),
        _section(
            "Current Factor IC",
            (
                _bar_svg(display["name"], display["ic"], "current factor IC")
                + _frame_table(display)
                if not display.empty and {"name", "ic"}.issubset(display.columns)
                else _empty("No factors in pool yet.")
            ),
        ),
    ]
    html = _html_doc("RL Pool IC Live", sections, refresh_seconds=refresh_seconds)
    ensure_dir(report_path.parent)
    report_path.write_text(html, encoding="utf-8")
    return report_path


def _preflight_report(
    rl_config: dict[str, Any],
    alpha_config: dict[str, Any],
    artifact_paths: dict[str, Path],
    missing: tuple[str, ...],
) -> str:
    data_config = rl_config["data"]
    panel_summary = _daily_panel_summary(data_config["daily_panel"])
    labels = _read_labels(data_config["labels"])
    train_labels = _filter_labels(
        labels, data_config["train_start"], data_config["train_end"]
    )
    validation_labels = _filter_labels(
        labels, data_config["validation_start"], data_config["validation_end"]
    )
    monthly = _monthly_label_summary(
        labels,
        data_config["train_start"],
        data_config["validation_end"],
    )
    config_rows = {
        "device": rl_config["device"],
        "train_iterations": rl_config["train_iterations"],
        "episodes_per_iteration": rl_config["episodes_per_iteration"],
        "target_pool_size": alpha_config["pool_size"],
        "min_valid_dates": alpha_config["min_valid_dates"],
        "min_valid_ratio": alpha_config["min_valid_ratio"],
        "min_stocks_per_date": alpha_config["min_stocks_per_date"],
    }
    sections = [
        _section("Status", _status_block("Pre-flight report", "warn")),
        _section("Missing Artifacts", _kv_table({name: artifact_paths[name] for name in missing})),
        _section("Data Coverage", _kv_table(panel_summary)),
        _section(
            "Label Split",
            _kv_table(
                {
                    "train_labels": len(train_labels),
                    "validation_labels": len(validation_labels),
                    "train_range": _date_range(train_labels),
                    "validation_range": _date_range(validation_labels),
                }
            ),
        ),
        _section("Monthly Label Count", _line_svg(monthly["count"], "labels per month")),
        _section("Monthly Future Return Mean", _line_svg(monthly["mean"], "monthly mean")),
        _section("Monthly Future Return Std", _line_svg(monthly["std"], "monthly std")),
        _section("Config Summary", _kv_table(config_rows)),
    ]
    return _html_doc("RL Alpha Pre-flight Report", sections)


def _final_report(
    rl_config: dict[str, Any],
    alpha_config: dict[str, Any],
    artifact_paths: dict[str, Path],
) -> tuple[str, int]:
    pool = pd.read_parquet(artifact_paths["pool"])
    metrics = pd.read_parquet(artifact_paths["metrics"])
    validation = pd.read_parquet(artifact_paths["validation"])
    target_pool_size = int(alpha_config["pool_size"])
    pool_size = len(pool)
    factor_validation = validation[validation["name"] != "__pool__"].copy()
    factors = pool.merge(
        factor_validation.loc[:, ["name", "validation_ic", "validation_rank_ic"]],
        on="name",
        how="left",
    )
    pool_summary = _pool_summary(metrics, validation)
    sections = [
        _section("Status", _pool_status(pool_size, target_pool_size)),
        _section("Pool Summary", _kv_table(pool_summary)),
        _section("Factor Pool", _frame_table(factors)),
        _section("Factor Weights", _bar_svg(factors["name"], factors["weight"], "weight")),
        _section(
            "Train IC vs Validation IC",
            _scatter_svg(factors["ic"], factors["validation_ic"], factors["name"]),
        ),
        _section("Mutual IC Heatmap", _mutual_ic_heatmap(pool, rl_config)),
        _section("Training Diagnostics", _training_diagnostics(metrics)),
        _section("Validation Metrics", _frame_table(validation)),
    ]
    return _html_doc("RL Alpha Factor Pool Report", sections), pool_size


def _artifact_paths(config: dict[str, Any]) -> dict[str, Path]:
    outputs = config["outputs"]
    return {
        "pool": _project_path(outputs["pool"]),
        "metrics": _project_path(outputs["metrics"]),
        "validation": _project_path(outputs["validation"]),
    }


def _daily_panel_summary(path: str | Path) -> dict[str, object]:
    frame = pd.read_parquet(_project_path(path), columns=["date", "symbol"])
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    return {
        "daily_panel": _project_path(path),
        "rows": len(frame),
        "symbols": frame["symbol"].nunique(),
        "start": frame["date"].min().date(),
        "end": frame["date"].max().date(),
    }


def _read_labels(path: str | Path) -> pd.DataFrame:
    labels = pd.read_parquet(
        _project_path(path),
        columns=["date", "symbol", "label_end_date", "future_20d_return"],
    )
    labels["date"] = pd.to_datetime(labels["date"]).dt.normalize()
    labels["label_end_date"] = pd.to_datetime(labels["label_end_date"]).dt.normalize()
    return labels


def _filter_labels(labels: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    start_date = pd.Timestamp(start).normalize()
    end_date = pd.Timestamp(end).normalize()
    return labels[
        (labels["date"] >= start_date)
        & (labels["date"] <= end_date)
        & (labels["label_end_date"] <= end_date)
    ].copy()


def _monthly_label_summary(labels: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    filtered = _filter_labels(labels, start, end)
    return filtered.groupby("date")["future_20d_return"].agg(["count", "mean", "std"])


def _pool_summary(metrics: pd.DataFrame, validation: pd.DataFrame) -> dict[str, object]:
    latest = metrics.tail(1).to_dict("records")
    pool_row = validation[validation["name"] == "__pool__"].tail(1).to_dict("records")
    train = latest[0] if latest else {}
    valid = pool_row[0] if pool_row else {}
    return {
        "train_pool_ic": train.get("pool_ic", ""),
        "train_pool_loss": train.get("pool_loss", ""),
        "validation_pool_ic": valid.get("validation_ic", ""),
        "validation_pool_rank_ic": valid.get("validation_rank_ic", ""),
        "final_valid_ratio": train.get("valid_ratio", ""),
        "final_pool_size": train.get("pool_size", ""),
    }


def _mutual_ic_heatmap(pool: pd.DataFrame, rl_config: dict[str, Any]) -> str:
    if pool.empty:
        return _empty("No factors in pool.")
    data_config = rl_config["data"]
    labels = _filter_labels(
        _read_labels(data_config["labels"]),
        data_config["train_start"],
        data_config["train_end"],
    )
    daily_panel = pd.read_parquet(
        _project_path(data_config["daily_panel"]),
        columns=["date", "symbol", "open", "close", "high", "low", "volume", "vwap"],
    )
    daily_panel["date"] = pd.to_datetime(daily_panel["date"]).dt.normalize()
    daily_panel = daily_panel[daily_panel["date"] <= pd.Timestamp(data_config["train_end"])]

    names: list[str] = []
    values: list[pd.DataFrame] = []
    skipped: list[str] = []
    for _, row in pool.iterrows():
        tokens = str(row["tokens"]).split()
        if not tokens:
            continue
        name = str(row["name"])
        try:
            expr = parse_rpn(tuple(tokens))
            aligned = align_values_to_labels(evaluate(expr, daily_panel), labels)
        except (RPNError, ValueError):
            skipped.append(name)
            continue
        names.append(name)
        values.append(aligned)
    if not values:
        return _empty("No tokenized factors available for mutual IC.")

    matrix = np.eye(len(values), dtype=float)
    for row in range(len(values)):
        for col in range(row + 1, len(values)):
            value = mutual_ic(values[row], values[col])
            matrix[row, col] = value
            matrix[col, row] = value
    note = (
        _empty(f"Skipped dimension-invalid legacy factors: {', '.join(skipped)}")
        if skipped
        else ""
    )
    return _heatmap_table(names, matrix) + note


def _training_diagnostics(metrics: pd.DataFrame) -> str:
    charts = []
    columns = ["mean_reward", "valid_ratio", "pool_size", "pool_ic", "pool_loss", "invalid_count"]
    for column in columns:
        if column in metrics.columns:
            charts.append(_line_svg(metrics[column], column))
    return "\n".join(charts) if charts else _empty("No training metrics available.")


def _html_doc(
    title: str,
    sections: list[str],
    *,
    refresh_seconds: int | None = None,
) -> str:
    refresh = (
        f'  <meta http-equiv="refresh" content="{int(refresh_seconds)}">\n'
        if refresh_seconds is not None
        else ""
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
{refresh}  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>
    body {{ margin: 24px; font-family: Segoe UI, Arial, sans-serif; color: #172033; }}
    h1 {{ font-size: 28px; margin: 0 0 18px; }}
    h2 {{ font-size: 18px; margin: 28px 0 10px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border: 1px solid #d8dee9; padding: 6px 8px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ background: #f3f6fa; }}
    .status {{ padding: 10px 12px; border-radius: 6px; font-weight: 600; }}
    .ok {{ background: #e7f6ee; color: #17613a; }}
    .warn {{ background: #fff5db; color: #7a4b00; }}
    .bad {{ background: #fde8e8; color: #9b1c1c; }}
    .empty {{ color: #687386; font-style: italic; }}
    .chart {{ margin: 8px 0 16px; max-width: 920px; }}
    .heatmap td {{ width: 28px; height: 24px; padding: 2px; font-size: 10px; }}
  </style>
</head>
<body>
  <h1>{escape(title)}</h1>
  {''.join(sections)}
</body>
</html>
"""


def _section(title: str, body: str) -> str:
    return f"<section><h2>{escape(title)}</h2>{body}</section>"


def _status_block(text: str, level: str) -> str:
    return f'<div class="status {level}">{escape(text)}</div>'


def _pool_status(pool_size: int, target_pool_size: int) -> str:
    if pool_size < min(25, target_pool_size):
        return _status_block(
            f"Pool size warning: {pool_size}/{target_pool_size}; expected 25-30 factors.",
            "bad",
        )
    if pool_size <= target_pool_size:
        return _status_block(f"Pool size OK: {pool_size}/{target_pool_size}.", "ok")
    return _status_block(f"Pool size exceeds target: {pool_size}/{target_pool_size}.", "bad")


def _kv_table(values: dict[str, object]) -> str:
    rows = "".join(
        f"<tr><th>{escape(str(key))}</th><td>{escape(_format_value(value))}</td></tr>"
        for key, value in values.items()
    )
    return f"<table>{rows}</table>"


def _frame_table(frame: pd.DataFrame, max_rows: int = 80) -> str:
    if frame.empty:
        return _empty("No rows available.")
    display = frame.head(max_rows).copy()
    header = "".join(f"<th>{escape(str(column))}</th>" for column in display.columns)
    rows = []
    for _, row in display.iterrows():
        cells = "".join(
            f"<td>{escape(_format_value(row[column]))}</td>" for column in display.columns
        )
        rows.append(f"<tr>{cells}</tr>")
    suffix = _empty(f"Showing first {max_rows} rows.") if len(frame) > max_rows else ""
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(rows)}</tbody></table>{suffix}"


def _line_svg(values: pd.Series, title: str) -> str:
    clean = pd.to_numeric(values, errors="coerce").reset_index(drop=True)
    valid = clean.dropna()
    if len(valid) < 2:
        return _empty(f"Not enough data for {title}.")
    width, height, pad = 720, 220, 28
    x_values = np.linspace(pad, width - pad, len(clean))
    y_min = float(valid.min())
    y_max = float(valid.max())
    if abs(y_max - y_min) < 1e-12:
        y_max = y_min + 1.0
    points = []
    for index, value in enumerate(clean):
        if np.isfinite(value):
            y = height - pad - (float(value) - y_min) / (y_max - y_min) * (height - 2 * pad)
            points.append(f"{x_values[index]:.1f},{y:.1f}")
    return f"""<svg class="chart" viewBox="0 0 {width} {height}" role="img">
  <text x="{pad}" y="18" font-size="13">{escape(title)}</text>
  <line x1="{pad}" y1="{height - pad}" x2="{width - pad}" y2="{height - pad}" stroke="#9aa4b2"/>
  <line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height - pad}" stroke="#9aa4b2"/>
  <polyline fill="none" stroke="#2563eb" stroke-width="2" points="{' '.join(points)}"/>
</svg>"""


def _bar_svg(labels: pd.Series, values: pd.Series, title: str) -> str:
    frame = pd.DataFrame(
        {"label": labels.astype(str), "value": pd.to_numeric(values, errors="coerce")}
    )
    frame = frame.dropna().assign(abs_value=lambda item: item["value"].abs())
    frame = frame.sort_values("abs_value", ascending=False).head(30)
    if frame.empty:
        return _empty(f"No data for {title}.")
    width = 720
    row_height = 22
    height = 32 + row_height * len(frame)
    max_abs = float(frame["abs_value"].max()) or 1.0
    zero = 260
    scale = 420 / max_abs
    rows = []
    for index, row in enumerate(frame.itertuples(index=False), start=0):
        y = 28 + index * row_height
        value = float(row.value)
        x = zero if value >= 0 else zero + value * scale
        bar_width = abs(value) * scale
        color = "#2563eb" if value >= 0 else "#dc2626"
        rows.append(
            f'<text x="4" y="{y + 13}" font-size="11">{escape(row.label[:32])}</text>'
            f'<rect x="{x:.1f}" y="{y}" width="{bar_width:.1f}" height="14" fill="{color}"/>'
            f'<text x="{zero + 426}" y="{y + 12}" font-size="11">{value:.4g}</text>'
        )
    return f"""<svg class="chart" viewBox="0 0 {width} {height}" role="img">
  <text x="4" y="16" font-size="13">{escape(title)}</text>
  <line x1="{zero}" y1="24" x2="{zero}" y2="{height - 4}" stroke="#172033"/>
  {''.join(rows)}
</svg>"""


def _scatter_svg(x_values: pd.Series, y_values: pd.Series, labels: pd.Series) -> str:
    frame = pd.DataFrame(
        {
            "x": pd.to_numeric(x_values, errors="coerce"),
            "y": pd.to_numeric(y_values, errors="coerce"),
            "label": labels.astype(str),
        }
    ).dropna()
    if frame.empty:
        return _empty("No IC pairs available.")
    width, height, pad = 520, 360, 36
    x_min, x_max = _range(frame["x"])
    y_min, y_max = _range(frame["y"])
    circles = []
    for row in frame.itertuples(index=False):
        x = pad + (float(row.x) - x_min) / (x_max - x_min) * (width - 2 * pad)
        y = height - pad - (float(row.y) - y_min) / (y_max - y_min) * (height - 2 * pad)
        circles.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#2563eb">'
            f"<title>{escape(row.label)}: train={row.x:.4g}, validation={row.y:.4g}</title>"
            "</circle>"
        )
    return f"""<svg class="chart" viewBox="0 0 {width} {height}" role="img">
  <line x1="{pad}" y1="{height - pad}" x2="{width - pad}" y2="{height - pad}" stroke="#9aa4b2"/>
  <line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height - pad}" stroke="#9aa4b2"/>
  <text x="{width / 2 - 42}" y="{height - 6}" font-size="12">train IC</text>
  <text x="4" y="20" font-size="12">validation IC</text>
  {''.join(circles)}
</svg>"""


def _heatmap_table(labels: list[str], matrix: np.ndarray) -> str:
    header = "<th></th>" + "".join(f"<th>{escape(label[:10])}</th>" for label in labels)
    rows = []
    for row_index, label in enumerate(labels):
        cells = [f"<th>{escape(label[:20])}</th>"]
        for col_index in range(len(labels)):
            value = float(matrix[row_index, col_index])
            tooltip = f"{escape(labels[row_index])} / {escape(labels[col_index])}: {value:.4f}"
            cells.append(
                f'<td title="{tooltip}" style="background:{_heat_color(value)}">'
                f"{value:.2f}</td>"
            )
        rows.append(f"<tr>{''.join(cells)}</tr>")
    return (
        f'<table class="heatmap"><thead><tr>{header}</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table>'
    )


def _heat_color(value: float) -> str:
    if not np.isfinite(value):
        return "#f3f6fa"
    alpha = min(abs(value), 1.0) * 0.85 + 0.05
    return f"rgba(37, 99, 235, {alpha:.2f})" if value >= 0 else f"rgba(220, 38, 38, {alpha:.2f})"


def _range(values: pd.Series) -> tuple[float, float]:
    lower = float(values.min())
    upper = float(values.max())
    if abs(upper - lower) < 1e-12:
        upper = lower + 1.0
    return lower, upper


def _date_range(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    return f"{frame['date'].min().date()} to {frame['date'].max().date()}"


def _format_value(value: object) -> str:
    if isinstance(value, float | np.floating):
        return "" if not np.isfinite(value) else f"{float(value):.6g}"
    return str(value)


def _empty(text: str) -> str:
    return f'<p class="empty">{escape(text)}</p>'


def _project_path(path: str | Path) -> Path:
    resolved = Path(path)
    return resolved if resolved.is_absolute() else project_root() / resolved
