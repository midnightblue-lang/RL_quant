from quant_rl_alpha.cli import main


class FakeFullIngestResult:
    def __init__(self):
        import pandas as pd

        self.symbols = pd.DataFrame({"symbol": ["000001"]})
        self.manifest = pd.DataFrame({"status": ["downloaded"]})
        self.quality = pd.DataFrame({"symbol": ["000001"]})
        self.manifest_path = "manifest.csv"
        self.summary_report_path = "summary.md"


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
