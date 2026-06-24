from __future__ import annotations

import argparse
from collections.abc import Callable

from quant_rl_alpha.data.pipeline import build_quality_report, download_configured_sample
from quant_rl_alpha.data.stage2_pipeline import build_stage2_outputs

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
    args = parser.parse_args(argv)

    commands: dict[str, tuple[CommandAction, ...]] = {
        "download-sample": (_print_download_results,),
        "quality-report": (_print_quality_report,),
        "stage1-sample": (_print_download_results, _print_quality_report),
        "stage2-build": (_print_stage2_outputs,),
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


if __name__ == "__main__":
    raise SystemExit(main())
