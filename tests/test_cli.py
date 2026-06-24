from quant_rl_alpha.cli import main


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
