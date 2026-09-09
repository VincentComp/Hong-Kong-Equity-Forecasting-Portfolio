from __future__ import annotations

from pathlib import Path

import yaml


def load_yaml(path: str | Path) -> dict:
    """Load and return one YAML configuration file."""

    with open(path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if config is None:
        raise ValueError(f"Configuration file is empty: {path}")

    return config


def load_project_config(
    base_config_path: str | Path,
    model_config_path: str | Path,
) -> tuple[dict, dict]:
    """Load the shared project config and one model-specific config."""

    base_config = load_yaml(base_config_path)
    model_config = load_yaml(model_config_path)

    return base_config, model_config