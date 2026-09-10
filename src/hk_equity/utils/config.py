"""Load and validate YAML configuration files for the project.

This module keeps configuration loading in one place. It is used by scripts
that need shared project settings from ``configs/base.yaml`` and model-specific
settings from files under ``configs/models/``.
"""

from __future__ import annotations

from pathlib import Path

import yaml


def load_yaml(path: str | Path) -> dict:#Load the yaml
    """Load one YAML configuration file and return it as a Python dictionary.

    This function opens the YAML file at ``path`` using UTF-8 encoding and
    converts its contents into a Python dictionary with ``yaml.safe_load``.

    It can load both:
    - The shared project configuration, for example ``configs/base.yaml``.
    - A model-specific configuration, for example
      ``configs/models/ma_4w_v1.yaml``.

    Args:
        path: Path to the YAML configuration file. It can be a string or a
            ``pathlib.Path`` object.

    Returns:
        A dictionary containing the settings defined in the YAML file.

    Raises:
        ValueError: If the YAML file exists but contains no configuration data.

    Example:
        >>> base_config = load_yaml("configs/base.yaml")
        >>> base_config["paths"]["raw_data"]
        'data/raw'
    """

    with open(path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if config is None:
        raise ValueError(f"Configuration file is empty: {path}")

    return config


def load_project_config( #load the base and para of the model yaml
    base_config_path: str | Path,
    model_config_path: str | Path,
) -> tuple[dict, dict]:
    """Load the shared project config and one selected model config together.

    The project uses two levels of configuration:
    - ``base.yaml`` stores settings shared by all models, including tickers,
      dates, data source settings, forecast settings, and project paths.
    - A model YAML file stores one model's name, run ID, version, and model
      parameters, such as moving-average lookback weeks or EWMA span.

    This function loads both files through ``load_yaml()`` and returns them
    separately. It does not merge the dictionaries, so shared settings remain
    in ``base_config`` and model-specific settings remain in ``model_config``.

    Args:
        base_config_path: Path to the shared project YAML configuration file,
            normally ``configs/base.yaml``.
        model_config_path: Path to one model-specific YAML configuration file,
            for example ``configs/models/ma_4w_v1.yaml``.

    Returns:
        A tuple containing:
        - ``base_config``: Dictionary loaded from the shared project config.
        - ``model_config``: Dictionary loaded from the selected model config.

    Example:
        >>> base_config, model_config = load_project_config(
        ...     base_config_path="configs/base.yaml",
        ...     model_config_path="configs/models/ma_4w_v1.yaml",
        ... )
        >>> model_config["model"]["run_id"]
        'ma_4w_v1'
    """

    base_config = load_yaml(base_config_path)
    model_config = load_yaml(model_config_path)

    return base_config, model_config