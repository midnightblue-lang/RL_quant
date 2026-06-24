from pathlib import Path


def project_root() -> Path:
    """Return the repository root by locating PROJECT_PLAN.md."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "PROJECT_PLAN.md").is_file():
            return parent
    raise RuntimeError("Could not locate project root containing PROJECT_PLAN.md")


def config_dir() -> Path:
    return project_root() / "config"


def data_dir() -> Path:
    return project_root() / "data"


def ensure_dir(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved
