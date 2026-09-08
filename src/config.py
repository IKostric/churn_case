"""Configuration loading and path resolution."""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"

logger = logging.getLogger(__name__)


def load_config(path: str | Path | None = None) -> dict:
    """Read config.yaml and attach resolved output paths (created on demand)."""
    path = Path(path) if path is not None else DEFAULT_CONFIG
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    output_dir = (PROJECT_ROOT / cfg["output_dir"]).resolve()
    cfg["paths"] = {
        "project_root": PROJECT_ROOT,
        "data_dir": (PROJECT_ROOT / cfg["data_dir"]).resolve(),
        "output_dir": output_dir,
        "features": output_dir / "features",
        "metrics": output_dir / "metrics",
        "figures": output_dir / "figures",
        "predictions": output_dir / "predictions",
        "models": output_dir / "models",
    }
    for key, directory in cfg["paths"].items():
        if key not in ("project_root", "data_dir"):
            directory.mkdir(parents=True, exist_ok=True)
    return cfg


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
