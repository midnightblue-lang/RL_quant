import random

import numpy as np

from quant_rl_alpha.utils.config import load_config
from quant_rl_alpha.utils.paths import config_dir, data_dir, project_root
from quant_rl_alpha.utils.seed import set_seed


def test_project_root_is_stable() -> None:
    root = project_root()
    assert (root / "PROJECT_PLAN.md").is_file()
    assert config_dir() == root / "config"
    assert data_dir() == root / "data"


def test_config_files_load() -> None:
    assert load_config("data")["provider"] == "akshare"
    assert load_config("universe")["top_n"] == 1500
    assert load_config("expression")["max_tokens"] == 20
    assert load_config("alpha")["pool_size"] == 30
    assert load_config("alpha")["l1_alpha"] == 0.005
    assert load_config("rl")["pool_size"] == 30
    assert load_config("rl")["clip_epsilon"] == 0.2
    assert load_config("rl")["ppo_epochs"] == 4
    assert load_config("rl")["data"]["train_end"] == "2020-12-31"
    assert load_config("rl")["outputs"]["pool"] == "data/features/rl_alpha_pool.parquet"
    assert (
        load_config("rl")["outputs"]["validation"]
        == "data/reports/rl_validation_metrics.parquet"
    )
    assert load_config("rl")["outputs"]["report"] == "data/reports/rl_factor_report.html"
    assert load_config("rl")["outputs"]["pool_ic_live_report"] == (
        "data/reports/rl_pool_ic_live.html"
    )
    assert load_config("xgboost")["target"] == "future_20d_rank"
    assert load_config("backtest")["execution"]["trade_time"] == "next_open"


def test_seed_is_reproducible() -> None:
    set_seed(42)
    first_random = random.random()
    first_numpy = np.random.random()

    set_seed(42)
    assert random.random() == first_random
    assert np.random.random() == first_numpy
