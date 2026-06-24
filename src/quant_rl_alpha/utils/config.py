from pathlib import Path
from typing import Any

import yaml

from quant_rl_alpha.utils.paths import config_dir


def load_config(name: str | Path) -> dict[str, Any]:
    """Load one YAML config file from the project config directory."""
    path = Path(name)
    if not path.suffix:
        path = path.with_suffix(".yml")
    if not path.is_absolute():
        path = config_dir() / path
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    return data
