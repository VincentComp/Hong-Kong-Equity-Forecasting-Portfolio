"""Load feature-set YAML files and build WeeklyFeatureSpec objects."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.hk_equity.features.weekly_features import (
    WeeklyFeatureSpec,
)
from src.hk_equity.utils.config import load_yaml


def load_feature_set_config(
    feature_set_name: str,
    features_directory: str = "configs/features",
) -> dict[str, Any]:
    """Load one feature-set YAML configuration.

    Args:
        feature_set_name: Feature-set name without the .yaml extension.
        features_directory: Directory containing feature-set YAML files.

    Returns:
        Loaded feature-set configuration dictionary.
    """

    feature_config_path = (
        Path(features_directory)
        / f"{feature_set_name}.yaml"
    )

    if not feature_config_path.exists():
        raise FileNotFoundError(
            f"Cannot find feature-set config: {feature_config_path}"
        )

    feature_config = load_yaml(
        str(feature_config_path)
    )

    configured_name = feature_config["feature_set"]["name"]

    if configured_name != feature_set_name:
        raise ValueError(
            "Feature-set name does not match its file name. "
            f"File requested: '{feature_set_name}'; "
            f"config name: '{configured_name}'."
        )

    return feature_config


def build_weekly_feature_spec(
    feature_config: dict[str, Any],
) -> WeeklyFeatureSpec:
    """Convert a feature-set YAML configuration into WeeklyFeatureSpec."""

    generation = feature_config["generation"]

    return WeeklyFeatureSpec(
        lag_weeks=tuple(generation["lag_weeks"]),
        rolling_windows=tuple(
            generation["rolling_windows"]
        ),
        include_rolling_std=generation.get(
            "include_rolling_std",
            True,
        ),
        include_downside_deviation=generation.get(
            "include_downside_deviation",
            True,
        ),
        include_market_features=generation.get(
            "include_market_features",
            True,
        ),
        include_excess_return_features=generation.get(
            "include_excess_return_features",
            True,
        ),
        include_ratio_features=generation.get(
            "include_ratio_features",
            False,
        ),
        include_cross_sectional_features=generation.get(
            "include_cross_sectional_features",
            True,
        ),
        cross_sectional_columns=tuple(
            generation.get(
                "cross_sectional_columns",
                (),
            )
        ),
        winsorize_features=generation.get(
            "winsorize_features",
            False,
        ),
        winsorize_lower=float(
            generation.get(
                "winsorize_lower",
                0.05,
            )
        ),
        winsorize_upper=float(
            generation.get(
                "winsorize_upper",
                0.95,
            )
        ),
    )


def get_feature_set_version(
    feature_config: dict[str, Any],
) -> str:
    """Return the configured feature-set version."""

    return str(
        feature_config["feature_set"]["version"]
    )