from quant_rl_alpha.cli import main


class FakeFullIngestResult:
    def __init__(self):
        import pandas as pd

        self.symbols = pd.DataFrame({"symbol": ["000001"]})
        self.manifest = pd.DataFrame({"status": ["downloaded"]})
        self.quality = pd.DataFrame({"symbol": ["000001"]})
        self.manifest_path = "manifest.csv"
        self.summary_report_path = "summary.md"


class FakeRLTrainingResult:
    def __init__(self):
        self.pool_rows = 30
        self.metric_rows = 1000
        self.validation_rows = 31
        self.pool_path = "pool.parquet"
        self.metrics_path = "metrics.parquet"
        self.validation_path = "validation.parquet"
        self.config_path = "config.yml"


class FakeRLReportResult:
    def __init__(self):
        self.mode = "preflight"
        self.pool_size = 0
        self.target_pool_size = 30
        self.missing_artifacts = ("pool",)
        self.report_path = "rl_factor_report.html"


class FakeStage6DatasetResult:
    def __init__(self):
        self.rows = 100
        self.alpha_count = 30
        self.dataset_path = "xgboost_dataset.parquet"


class FakeStage6TrainingResult:
    def __init__(self):
        self.dataset_path = "xgboost_dataset.parquet"
        self.prediction_rows = 50
        self.metric_rows = 2
        self.feature_importance_rows = 30
        self.predictions_path = "xgboost_predictions.parquet"
        self.metrics_path = "xgboost_metrics.parquet"
        self.feature_importance_path = "xgboost_feature_importance.parquet"


def test_cli_quality_report_command(monkeypatch) -> None:
    calls: list[str] = []

    def fake_quality_report():
        calls.append("quality")
        import pandas as pd

        return pd.DataFrame({"symbol": ["000001"], "rows": [1], "has_issue": [False]})

    monkeypatch.setattr("quant_rl_alpha.cli.build_quality_report", fake_quality_report)

    assert main(["quality-report"]) == 0
    assert calls == ["quality"]


def test_cli_stage2_build_command(monkeypatch) -> None:
    calls: list[str] = []

    def fake_stage2_outputs():
        calls.append("stage2")
        import pandas as pd

        return pd.DataFrame({"x": [1]}), pd.DataFrame({"x": [1]}), pd.DataFrame({"x": [1]})

    monkeypatch.setattr("quant_rl_alpha.cli.build_stage2_outputs", fake_stage2_outputs)

    assert main(["stage2-build"]) == 0
    assert calls == ["stage2"]


def test_cli_download_full_command(monkeypatch) -> None:
    calls: list[str] = []

    def fake_full_ingest():
        calls.append("full")
        return FakeFullIngestResult()

    monkeypatch.setattr("quant_rl_alpha.cli.download_full_market_data", fake_full_ingest)

    assert main(["download-full"]) == 0
    assert calls == ["full"]


def test_cli_stage25_full_data_command(monkeypatch) -> None:
    calls: list[str] = []

    def fake_full_ingest():
        calls.append("full")
        return FakeFullIngestResult()

    def fake_stage2_outputs():
        calls.append("stage2")
        import pandas as pd

        return pd.DataFrame({"x": [1]}), pd.DataFrame({"x": [1]}), pd.DataFrame({"x": [1]})

    monkeypatch.setattr("quant_rl_alpha.cli.download_full_market_data", fake_full_ingest)
    monkeypatch.setattr("quant_rl_alpha.cli.build_stage2_outputs", fake_stage2_outputs)

    assert main(["stage25-full-data"]) == 0
    assert calls == ["full", "stage2"]


def test_cli_stage5_train_command(monkeypatch) -> None:
    calls: list[str] = []

    def fake_training(config_name="rl"):
        calls.append(config_name)
        return FakeRLTrainingResult()

    monkeypatch.setattr("quant_rl_alpha.cli.run_rl_alpha_mining", fake_training)

    assert main(["stage5-train"]) == 0
    assert calls == ["rl"]

    calls.clear()
    assert main(["stage5-train", "--config", "rl_small"]) == 0
    assert calls == ["rl_small"]


def test_cli_stage5_report_command(monkeypatch) -> None:
    calls: list[str] = []

    def fake_report(config_name="rl"):
        calls.append(config_name)
        return FakeRLReportResult()

    monkeypatch.setattr("quant_rl_alpha.cli.build_rl_factor_report", fake_report)

    assert main(["stage5-report"]) == 0
    assert calls == ["rl"]

    calls.clear()
    assert main(["stage5-report", "--config", "rl_small"]) == 0
    assert calls == ["rl_small"]


def test_cli_stage6_build_dataset_command(monkeypatch) -> None:
    calls: list[str] = []

    def fake_dataset(config_name="xgboost"):
        calls.append(config_name)
        return FakeStage6DatasetResult()

    monkeypatch.setattr("quant_rl_alpha.cli.build_xgboost_dataset", fake_dataset)

    assert main(["stage6-build-dataset"]) == 0
    assert calls == ["xgboost"]

    calls.clear()
    assert main(["stage6-build-dataset", "--config", "custom_xgb"]) == 0
    assert calls == ["custom_xgb"]


def test_cli_stage6_train_command(monkeypatch) -> None:
    calls: list[str] = []

    def fake_training(config_name="xgboost"):
        calls.append(config_name)
        return FakeStage6TrainingResult()

    monkeypatch.setattr("quant_rl_alpha.cli.run_xgboost_training", fake_training)

    assert main(["stage6-train"]) == 0
    assert calls == ["xgboost"]

    calls.clear()
    assert main(["stage6-train", "--config", "custom_xgb"]) == 0
    assert calls == ["custom_xgb"]
