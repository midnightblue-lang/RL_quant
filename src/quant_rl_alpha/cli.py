from __future__ import annotations

import argparse
from collections.abc import Callable

from quant_rl_alpha.data.full_ingest import download_full_market_data
from quant_rl_alpha.data.pipeline import build_quality_report, download_configured_sample
from quant_rl_alpha.data.stage2_pipeline import build_stage2_outputs
from quant_rl_alpha.model import build_xgboost_dataset, run_xgboost_training
from quant_rl_alpha.reporting import build_rl_factor_report
from quant_rl_alpha.rl.experiment import run_rl_alpha_mining

CommandAction = Callable[[], None]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="quant-rl-alpha")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("download-sample", help="Download configured AKShare sample symbols.")
    subparsers.add_parser("quality-report", help="Build a quality report from cached daily bars.")
    subparsers.add_parser(
        "stage1-sample",
        help="Download sample data and build the quality report.",
    )
    subparsers.add_parser(
        "stage2-build",
        help="Build daily panel, monthly universe, and forward-return labels.",
    )
    subparsers.add_parser(
        "download-full",
        help="Download full current A-share daily bars and build data audit reports.",
    )
    subparsers.add_parser(
        "stage25-full-data",
        help="Download full data, audit it, then rebuild Stage 2 outputs.",
    )
    stage5_train = subparsers.add_parser(
        "stage5-train",
        help="Run configured PPO formula-alpha mining on the training split.",
    )
    stage5_train.add_argument("--config", default="rl", help="RL config name under config/*.yml.")
    stage5_report = subparsers.add_parser(
        "stage5-report",
        help="Build the configured RL factor pool visualization report.",
    )
    stage5_report.add_argument("--config", default="rl", help="RL config name under config/*.yml.")
    stage6_dataset = subparsers.add_parser(
        "stage6-build-dataset",
        help="Build the Stage 6 XGBoost wide feature dataset.",
    )
    stage6_dataset.add_argument(
        "--config",
        default="xgboost",
        help="XGBoost config name under config/*.yml.",
    )
    stage6_train = subparsers.add_parser(
        "stage6-train",
        help="Run Stage 6 monthly walk-forward XGBoost predictions.",
    )
    stage6_train.add_argument(
        "--config",
        default="xgboost",
        help="XGBoost config name under config/*.yml.",
    )
    args = parser.parse_args(argv)

    commands: dict[str, tuple[CommandAction, ...]] = {
        "download-sample": (_print_download_results,),
        "quality-report": (_print_quality_report,),
        "stage1-sample": (_print_download_results, _print_quality_report),
        "stage2-build": (_print_stage2_outputs,),
        "download-full": (_print_full_ingest,),
        "stage25-full-data": (_print_full_ingest, _print_stage2_outputs),
        "stage5-train": (lambda: _print_stage5_training(args.config),),
        "stage5-report": (lambda: _print_stage5_report(args.config),),
        "stage6-build-dataset": (lambda: _print_stage6_dataset(args.config),),
        "stage6-train": (lambda: _print_stage6_training(args.config),),
    }
    actions = commands.get(args.command)
    if actions is not None:
        for action in actions:
            action()
        return 0
    raise ValueError(f"Unsupported command: {args.command}")


def _print_download_results() -> None:
    results = download_configured_sample()
    for result in results:
        status = "skipped" if result.skipped else "downloaded"
        if result.error:
            status = f"failed: {result.error}"
        print(f"{result.symbol} [{result.endpoint}] {status}, rows={result.rows}")


def _print_quality_report() -> None:
    summary = build_quality_report()
    print(f"quality report symbols={len(summary)}")
    if not summary.empty:
        print(summary[["symbol", "rows", "has_issue"]].to_string(index=False))


def _print_stage2_outputs() -> None:
    panel, universe, labels = build_stage2_outputs()
    print(f"daily_panel rows={len(panel)}")
    print(f"monthly_universe rows={len(universe)}")
    print(f"labels rows={len(labels)}")


def _print_full_ingest() -> None:
    result = download_full_market_data()
    status_counts = result.manifest["status"].value_counts().to_dict()
    print(f"full symbols={len(result.symbols)}")
    print(f"downloaded={int(status_counts.get('downloaded', 0))}")
    print(f"skipped={int(status_counts.get('skipped', 0))}")
    print(f"failed={int(status_counts.get('failed', 0))}")
    print(f"quality symbols={len(result.quality)}")
    print(f"manifest={result.manifest_path}")
    print(f"summary={result.summary_report_path}")


def _print_stage5_training(config_name: str = "rl") -> None:
    result = run_rl_alpha_mining(config_name)
    print(f"pool rows={result.pool_rows}")
    print(f"metric rows={result.metric_rows}")
    print(f"validation rows={result.validation_rows}")
    print(f"pool={result.pool_path}")
    print(f"metrics={result.metrics_path}")
    print(f"validation={result.validation_path}")
    print(f"config={result.config_path}")


def _print_stage5_report(config_name: str = "rl") -> None:
    result = build_rl_factor_report(config_name)
    print(f"report mode={result.mode}")
    print(f"pool rows={result.pool_size}/{result.target_pool_size}")
    if result.missing_artifacts:
        print(f"missing={','.join(result.missing_artifacts)}")
    print(f"report={result.report_path}")


def _print_stage6_dataset(config_name: str = "xgboost") -> None:
    result = build_xgboost_dataset(config_name)
    print(f"dataset rows={result.rows}")
    print(f"alpha features={result.alpha_count}")
    print(f"dataset={result.dataset_path}")


def _print_stage6_training(config_name: str = "xgboost") -> None:
    result = run_xgboost_training(config_name)
    print(f"dataset={result.dataset_path}")
    print(f"predictions rows={result.prediction_rows}")
    print(f"metrics rows={result.metric_rows}")
    print(f"feature importance rows={result.feature_importance_rows}")
    print(f"predictions={result.predictions_path}")
    print(f"metrics={result.metrics_path}")
    print(f"feature_importance={result.feature_importance_path}")


if __name__ == "__main__":
    raise SystemExit(main())
