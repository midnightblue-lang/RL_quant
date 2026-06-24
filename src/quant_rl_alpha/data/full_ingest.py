from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pandas as pd

from quant_rl_alpha.data.akshare_client import AkshareClient, DownloadResult, download_daily_bars
from quant_rl_alpha.data.cache import standard_daily_dir
from quant_rl_alpha.data.pipeline import build_quality_report
from quant_rl_alpha.data.schema import normalize_symbol
from quant_rl_alpha.utils.config import load_config
from quant_rl_alpha.utils.logging import get_logger
from quant_rl_alpha.utils.paths import ensure_dir, project_root

LOGGER = get_logger(__name__)


DEFAULT_FULL_PATHS = {
    "symbol_list": "data/reports/full_a_stock_symbols.csv",
    "manifest": "data/reports/full_download_manifest.csv",
    "failures": "data/reports/full_download_failures.csv",
    "quality_report": "data/reports/full_data_quality.md",
    "summary_report": "data/reports/full_data_ingestion.md",
}
CACHE_TEXT_FIELDS = (
    "first_date", "last_date", "source_in_file", "adjust_in_file",
    "download_started_at", "download_finished_at",
)
EMPTY_RESULT_TEXT_FIELDS = CACHE_TEXT_FIELDS + ("standard_path", "raw_path", "error", "error_type")


@dataclass(frozen=True)
class FullIngestResult:
    symbols: pd.DataFrame
    manifest: pd.DataFrame
    quality: pd.DataFrame
    symbol_list_path: Path
    manifest_path: Path
    failures_path: Path
    quality_report_path: Path
    summary_report_path: Path


def download_full_market_data(client: AkshareClient | None = None) -> FullIngestResult:
    config = load_config("data")
    full_config = config.get("full_market", {})
    client = client or AkshareClient()

    paths = _full_paths(full_config)
    symbols = _load_full_symbols(client, config, full_config)
    _write_csv(symbols, paths["symbol_list"])

    endpoint = str(config["endpoint"])
    end_date = config["end_date"] or pd.Timestamp.today().strftime("%Y-%m-%d")
    names = dict(zip(symbols["symbol"], symbols["name"], strict=True))
    flush_every = int(full_config.get("manifest_flush_every", 20))
    progress_every = int(full_config.get("progress_every", 50))
    run_id = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    metadata = {
        "run_id": run_id,
        "provider": str(config.get("provider", "akshare")),
        "endpoint": endpoint,
        "adjust": str(config["adjust"]),
        "start_date": str(config["start_date"]),
        "end_date": str(end_date),
        "retry_times": int(config["retry_times"]),
        "akshare_version": _package_version("akshare"),
    }

    results: list[DownloadResult] = []
    cache_audits: dict[str, dict[str, object]] = {}
    LOGGER.info("Starting full-market download for %s symbols", len(symbols))
    for index, symbol in enumerate(symbols["symbol"].tolist(), start=1):
        started_at = _now_text()
        result = download_daily_bars(
            client,
            [symbol],
            start_date=str(config["start_date"]),
            end_date=str(end_date),
            adjust=str(config["adjust"]),
            endpoint=endpoint,
            names=names,
            skip_existing=bool(config["skip_existing"]),
            retry_times=int(config["retry_times"]),
            sleep_seconds=float(config["sleep_seconds"]),
        )[0]
        results.append(result)
        cache_audits[normalize_symbol(symbol)] = _inspect_result_cache(result) | {
            "download_started_at": started_at,
            "download_finished_at": _now_text(),
        }

        if flush_every > 0 and (index % flush_every == 0 or index == len(symbols)):
            _write_manifest_files(
                symbols,
                results,
                paths["manifest"],
                paths["failures"],
                metadata=metadata,
                cache_audits=cache_audits,
            )
        if progress_every > 0 and (index % progress_every == 0 or index == len(symbols)):
            LOGGER.info("Full-market download progress: %s/%s", index, len(symbols))

    quality = build_quality_report(
        input_dir=standard_daily_dir(), output_path=paths["quality_report"]
    )
    manifest = _write_manifest_files(
        symbols,
        results,
        paths["manifest"],
        paths["failures"],
        metadata=metadata,
        cache_audits=cache_audits,
        quality=quality,
    )
    write_full_ingest_report(manifest, quality, paths["summary_report"])
    return FullIngestResult(
        symbols=symbols,
        manifest=manifest,
        quality=quality,
        symbol_list_path=paths["symbol_list"],
        manifest_path=paths["manifest"],
        failures_path=paths["failures"],
        quality_report_path=paths["quality_report"],
        summary_report_path=paths["summary_report"],
    )


def build_download_manifest(
    symbols: pd.DataFrame,
    results: list[DownloadResult],
    *,
    metadata: dict[str, object] | None = None,
    cache_audits: dict[str, dict[str, object]] | None = None,
    quality: pd.DataFrame | None = None,
) -> pd.DataFrame:
    metadata = metadata or {}
    cache_audits = cache_audits or {}
    quality_issues = _quality_issue_by_symbol(quality)
    result_by_symbol = {normalize_symbol(result.symbol): result for result in results}
    updated_at = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = []
    for row in symbols.loc[:, ["symbol", "name"]].itertuples(index=False):
        symbol = normalize_symbol(row.symbol)
        result = result_by_symbol.get(symbol)
        audit = cache_audits.get(symbol, {})
        base = _manifest_metadata(metadata) | {"symbol": symbol, "name": row.name}
        if result is None:
            pending = base | {
                "endpoint": str(metadata.get("endpoint", "")),
                "status": "pending",
                "skipped": False,
                "attempts": 0,
                "standard_rows": 0,
                "raw_rows": 0,
                "has_quality_issue": "",
                "updated_at": updated_at,
            }
            pending.update(dict.fromkeys(EMPTY_RESULT_TEXT_FIELDS, ""))
            rows.append(pending)
            continue

        status = "failed" if result.error else "skipped" if result.skipped else "downloaded"
        manifest_row = base | {
            "endpoint": result.endpoint,
            "status": status,
            "skipped": result.skipped,
            "attempts": result.attempts,
            "standard_rows": int(audit.get("standard_rows", result.rows)),
            "raw_rows": int(audit.get("raw_rows", 0)),
            "has_quality_issue": quality_issues.get(symbol, ""),
            "standard_path": _text_or_empty(result.standard_path),
            "raw_path": _text_or_empty(result.raw_path),
            "error": _text_or_empty(result.error),
            "error_type": _text_or_empty(result.error_type),
            "updated_at": updated_at,
        }
        manifest_row.update({field: str(audit.get(field, "")) for field in CACHE_TEXT_FIELDS})
        rows.append(manifest_row)
    return pd.DataFrame(rows)


def write_full_ingest_report(
    manifest: pd.DataFrame, quality: pd.DataFrame, path: str | Path
) -> Path:
    output = Path(path)
    ensure_dir(output.parent)

    status_counts = manifest["status"].value_counts().to_dict() if not manifest.empty else {}
    failed = int(status_counts.get("failed", 0))
    downloaded = int(status_counts.get("downloaded", 0))
    skipped = int(status_counts.get("skipped", 0))
    pending = int(status_counts.get("pending", 0))
    issue_count = int(quality["has_issue"].sum()) if "has_issue" in quality else 0
    total_rows = int(manifest["standard_rows"].sum()) if "standard_rows" in manifest else 0

    lines = [
        "# 全量数据接入报告",
        "",
        "## 下载概览",
        "",
        f"- 股票列表数量：{len(manifest)}",
        f"- 新下载股票数：{downloaded}",
        f"- 跳过已缓存股票数：{skipped}",
        f"- 失败股票数：{failed}",
        f"- 未处理股票数：{pending}",
        f"- 标准化行情总行数：{total_rows}",
        "",
        "## 质量概览",
        "",
        f"- 参与质量检查的缓存股票数：{len(quality)}",
        f"- 存在质量提示的股票数：{issue_count}",
        "",
        "## 清洗口径提醒",
        "",
        "- 原始 AKShare 返回和标准化行情分层保存，不覆盖原始字段。",
        "- 只做字段标准化、日期解析、数值类型转换、成交量单位统一和排序。",
        "- 不前向填充 OHLCV，不伪造停牌交易日，不自动删除大涨跌幅样本。",
        "- 前复权 OHLC 与原始成交额/成交量计算出的 VWAP 可能口径不一致，只记录提示。",
        "- 当前全量股票列表来自 AKShare 当前可获取列表，不等价于无幸存者偏差的历史全市场。",
        "",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def _load_full_symbols(
    client: AkshareClient,
    config: dict[str, object],
    full_config: dict[str, object],
) -> pd.DataFrame:
    source = str(config.get("symbol_list_endpoint", "code_name"))
    symbols = client.list_a_stocks(exclude_bj=bool(config["exclude_bj"]), source=source)
    required = {"symbol", "name"}
    missing = required - set(symbols.columns)
    if missing:
        raise ValueError(f"Full symbol list missing columns: {sorted(missing)}")

    result = symbols.loc[:, ["symbol", "name"]].copy()
    result["symbol"] = result["symbol"].map(normalize_symbol)
    result["name"] = result["name"].fillna("").astype(str)
    result = result.drop_duplicates("symbol").sort_values("symbol").reset_index(drop=True)

    limit = full_config.get("symbol_limit")
    if limit is not None:
        limit_value = int(limit)
        if limit_value <= 0:
            raise ValueError("full_market.symbol_limit must be positive when set")
        result = result.head(limit_value).reset_index(drop=True)
    if result.empty:
        raise ValueError("Full symbol list is empty")
    return result


def _full_paths(full_config: dict[str, object]) -> dict[str, Path]:
    configured = full_config.get("paths", {})
    if configured is not None and not isinstance(configured, dict):
        raise ValueError("full_market.paths must be a mapping")

    paths = DEFAULT_FULL_PATHS | (configured or {})
    return {key: _project_path(value) for key, value in paths.items()}


def _write_manifest_files(
    symbols: pd.DataFrame,
    results: list[DownloadResult],
    manifest_path: Path,
    failures_path: Path,
    *,
    metadata: dict[str, object] | None = None,
    cache_audits: dict[str, dict[str, object]] | None = None,
    quality: pd.DataFrame | None = None,
) -> pd.DataFrame:
    manifest = build_download_manifest(
        symbols,
        results,
        metadata=metadata,
        cache_audits=cache_audits,
        quality=quality,
    )
    _write_csv(manifest, manifest_path)
    failures = manifest[manifest["status"] == "failed"].copy()
    _write_csv(failures, failures_path)
    return manifest


def _write_csv(frame: pd.DataFrame, path: str | Path) -> Path:
    output = Path(path)
    ensure_dir(output.parent)
    frame.to_csv(output, index=False, encoding="utf-8")
    return output


def _project_path(value: object) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return project_root() / path


def _text_or_empty(value: object | None) -> str:
    return "" if value is None else str(value)


def _manifest_metadata(metadata: dict[str, object]) -> dict[str, object]:
    return {
        "run_id": metadata.get("run_id", ""),
        "provider": metadata.get("provider", ""),
        "adjust": metadata.get("adjust", ""),
        "start_date": metadata.get("start_date", ""),
        "end_date": metadata.get("end_date", ""),
        "retry_times": metadata.get("retry_times", ""),
        "akshare_version": metadata.get("akshare_version", ""),
    }


def _inspect_result_cache(result: DownloadResult) -> dict[str, object]:
    audit: dict[str, object] = {
        "raw_rows": 0,
        "standard_rows": result.rows,
        "first_date": "",
        "last_date": "",
        "source_in_file": "",
        "adjust_in_file": "",
    }
    if result.raw_path is not None and result.raw_path.exists():
        audit["raw_rows"] = len(pd.read_parquet(result.raw_path))
    if result.standard_path is None or not result.standard_path.exists():
        return audit

    frame = pd.read_parquet(result.standard_path)
    if frame.empty:
        audit["standard_rows"] = 0
        return audit

    dates = pd.to_datetime(frame["date"])
    audit["standard_rows"] = len(frame)
    audit["first_date"] = dates.min().strftime("%Y-%m-%d")
    audit["last_date"] = dates.max().strftime("%Y-%m-%d")
    audit["source_in_file"] = "|".join(sorted(frame["source"].dropna().astype(str).unique()))
    audit["adjust_in_file"] = "|".join(sorted(frame["adjust"].dropna().astype(str).unique()))
    return audit


def _quality_issue_by_symbol(quality: pd.DataFrame | None) -> dict[str, bool]:
    if quality is None or quality.empty or "has_issue" not in quality:
        return {}
    frame = quality.loc[:, ["symbol", "has_issue"]].copy()
    frame["symbol"] = frame["symbol"].map(normalize_symbol)
    return dict(zip(frame["symbol"], frame["has_issue"].astype(bool), strict=True))


def _package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return ""


def _now_text() -> str:
    return pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
