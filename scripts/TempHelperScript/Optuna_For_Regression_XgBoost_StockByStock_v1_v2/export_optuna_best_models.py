"""Export frozen Optuna best-model YAML configs into model group folders.

This script reads each completed Optuna study's best_model_config.yaml,
changes trial-style model identifiers to *_best, and exports the result to:

    configs/models/regression/groups/
    configs/models/tree_boosting/groups/

Examples:
    python -m scripts.export_optuna_best_models --dry-run

    python -m scripts.export_optuna_best_models \
        --source-directory outputs/tuning \
        --stable-only

    python -m scripts.export_optuna_best_models \
        --source-directory outputs/tuning \
        --stable-only \
        --overwrite
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


DEFAULT_SOURCE_DIRECTORY = Path(
    "outputs/tuning"
)

REGRESSION_OUTPUT_DIRECTORY = Path(
    "configs/models/regression/groups"
)

TREE_BOOSTING_OUTPUT_DIRECTORY = Path(
    "configs/models/tree_boosting/groups"
)

BEST_MODEL_CONFIG_FILENAME = (
    "best_model_config.yaml"
)

BENCHMARK_TICKER = "^HSI"


def parse_arguments() -> argparse.Namespace:
    """Read command-line settings for Optuna best-model export."""

    parser = argparse.ArgumentParser(
        description=(
            "Export Optuna best-model YAML files into model group folders."
        ),
    )

    parser.add_argument(
        "--source-directory",
        default=str(DEFAULT_SOURCE_DIRECTORY),
        help=(
            "Directory containing Optuna study output folders. "
            "Default: outputs/tuning"
        ),
    )

    parser.add_argument(
        "--stable-only",
        action="store_true",
        help=(
            "Export only studies whose folder name contains '_stable'."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "List source and target files without writing YAML files."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Allow overwriting an existing exported *_best.yaml file."
        ),
    )

    return parser.parse_args()


def load_yaml(
    yaml_path: Path,
) -> dict[str, Any]:
    """Load one best-model YAML file."""

    with yaml_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            f"YAML root must be a mapping: {yaml_path}"
        )

    return config


def validate_source_config(
    config: dict[str, Any],
    source_path: Path,
) -> None:
    """Validate required sections before exporting a frozen model."""

    required_sections = {
        "model",
        "parameters",
        "training",
        "features",
    }

    missing_sections = (
        required_sections - set(config.keys())
    )

    if missing_sections:
        raise KeyError(
            "Best-model config is missing sections "
            f"{sorted(missing_sections)}: {source_path}"
        )

    model = config["model"]

    required_model_keys = {
        "name",
        "label",
        "version",
        "run_id",
    }

    missing_model_keys = (
        required_model_keys - set(model.keys())
    )

    if missing_model_keys:
        raise KeyError(
            "model section is missing keys "
            f"{sorted(missing_model_keys)}: {source_path}"
        )

    training = config["training"]

    if "tickers" not in training:
        raise KeyError(
            "training.tickers is required: "
            f"{source_path}"
        )

    training_tickers = training["tickers"]

    if not isinstance(training_tickers, list):
        raise TypeError(
            "training.tickers must be a YAML list: "
            f"{source_path}"
        )

    if BENCHMARK_TICKER in training_tickers:
        raise ValueError(
            "HSI benchmark must never be in training.tickers: "
            f"{source_path}"
        )

    features = config["features"]

    required_feature_keys = {
        "feature_set",
        "feature_map_path",
        "selected_columns",
    }

    missing_feature_keys = (
        required_feature_keys - set(features.keys())
    )

    if missing_feature_keys:
        raise KeyError(
            "features section is missing keys "
            f"{sorted(missing_feature_keys)}: {source_path}"
        )


def get_destination_directory(
    model_name: str,
) -> Path:
    """Return the correct group directory for a model family."""

    normalized_name = model_name.lower()

    if normalized_name == "general_regression":
        return REGRESSION_OUTPUT_DIRECTORY

    if normalized_name == "tree_boosting":
        return TREE_BOOSTING_OUTPUT_DIRECTORY

    raise ValueError(
        "Unsupported model.name for export: "
        f"'{model_name}'. Expected 'general_regression' "
        "or 'tree_boosting'."
    )


def build_best_run_id(
    original_run_id: str,
) -> str:
    """Convert a trial run ID into one stable frozen best-model run ID."""

    trial_marker = "_trial_"

    if trial_marker in original_run_id:
        return original_run_id.split(
            trial_marker,
            maxsplit=1,
        )[0] + "_best"

    if original_run_id.endswith("_best"):
        return original_run_id

    return original_run_id + "_best"


def get_display_family(
    model_name: str,
) -> str:
    """Return a readable family name for model labels."""

    normalized_name = model_name.lower()

    if normalized_name == "general_regression":
        return "Regression"

    if normalized_name == "tree_boosting":
        return "XGBoost"

    raise ValueError(
        f"Unsupported model name: {model_name}"
    )


def get_version_label(
    best_run_id: str,
) -> str:
    """Derive a readable version label from a frozen run ID."""

    marker = "_mae_optuna_"

    if marker not in best_run_id:
        return "optuna_best"

    version_part = best_run_id.split(
        marker,
        maxsplit=1,
    )[1]

    if version_part.endswith("_best"):
        version_part = version_part.removesuffix(
            "_best"
        )

    return version_part


def build_export_config(
    source_config: dict[str, Any],
) -> dict[str, Any]:
    """Create a frozen best-model config with updated model metadata."""

    config = source_config.copy()
    config["model"] = source_config["model"].copy()

    original_run_id = str(
        source_config["model"]["run_id"]
    )

    best_run_id = build_best_run_id(
        original_run_id=original_run_id,
    )

    model_name = str(
        source_config["model"]["name"]
    )

    display_family = get_display_family(
        model_name=model_name,
    )

    version_label = get_version_label(
        best_run_id=best_run_id,
    )

    base_ticker = str(
        source_config["training"]["tickers"][0]
    )

    config["model"]["label"] = (
        f"{base_ticker} {display_family} "
        f"Optuna {version_label} Best"
    )

    config["model"]["version"] = version_label

    config["model"]["run_id"] = best_run_id

    return config


def discover_best_model_files(
    source_directory: Path,
    stable_only: bool,
) -> list[Path]:
    """Find completed Optuna best-model YAML artifacts."""

    if not source_directory.exists():
        raise FileNotFoundError(
            "Cannot find Optuna output directory: "
            f"{source_directory}"
        )

    best_model_paths = sorted(
        source_directory.glob(
            f"*/{BEST_MODEL_CONFIG_FILENAME}"
        )
    )

    if stable_only:
        best_model_paths = [
            path
            for path in best_model_paths
            if "_stable" in path.parent.name
        ]

    if not best_model_paths:
        stable_message = (
            " with '_stable' in the study folder name"
            if stable_only
            else ""
        )

        raise FileNotFoundError(
            "No best_model_config.yaml files found"
            f"{stable_message} in: {source_directory}"
        )

    return best_model_paths


def write_yaml(
    config: dict[str, Any],
    output_path: Path,
) -> None:
    """Write one frozen model YAML config."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(
            config,
            file,
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
        )


def main() -> None:
    """Export all discovered Optuna best-model configs."""

    args = parse_arguments()

    source_directory = Path(
        args.source_directory
    )

    source_paths = discover_best_model_files(
        source_directory=source_directory,
        stable_only=args.stable_only,
    )

    export_records = []

    for source_path in source_paths:
        source_config = load_yaml(
            yaml_path=source_path,
        )

        validate_source_config(
            config=source_config,
            source_path=source_path,
        )

        export_config = build_export_config(
            source_config=source_config,
        )

        model_name = str(
            export_config["model"]["name"]
        )

        destination_directory = (
            get_destination_directory(
                model_name=model_name,
            )
        )

        best_run_id = str(
            export_config["model"]["run_id"]
        )

        destination_path = (
            destination_directory
            / f"{best_run_id}.yaml"
        )

        export_records.append({
            "source_path": source_path,
            "destination_path": destination_path,
            "model_name": model_name,
            "label": export_config["model"]["label"],
            "run_id": best_run_id,
            "config": export_config,
        })

    existing_targets = [
        record["destination_path"]
        for record in export_records
        if record["destination_path"].exists()
    ]

    if existing_targets and not args.overwrite:
        formatted_targets = "\n".join(
            str(path)
            for path in existing_targets
        )

        raise FileExistsError(
            "Refusing to overwrite existing exported best-model files. "
            "Use --overwrite only when you intentionally want to replace "
            "them:\n"
            f"{formatted_targets}"
        )

    print("\n" + "=" * 70)
    print("OPTUNA BEST-MODEL EXPORT")
    print("=" * 70)
    print(f"Source directory: {source_directory}")
    print(f"Stable-only filter: {args.stable_only}")
    print(f"Dry run: {args.dry_run}")
    print(f"Best models found: {len(export_records)}")
    print("=" * 70)

    for position, record in enumerate(
        export_records,
        start=1,
    ):
        print(
            f"\n{position:02d}. "
            f"model={record['model_name']} | "
            f"run_id={record['run_id']}"
        )
        print(f"    label: {record['label']}")
        print(f"    source: {record['source_path']}")
        print(
            f"    destination: "
            f"{record['destination_path']}"
        )

    if args.dry_run:
        print(
            "\nDry run completed. "
            "No model YAML files were exported."
        )
        return

    for record in export_records:
        write_yaml(
            config=record["config"],
            output_path=record["destination_path"],
        )

    regression_count = sum(
        record["model_name"].lower()
        == "general_regression"
        for record in export_records
    )

    xgboost_count = sum(
        record["model_name"].lower()
        == "tree_boosting"
        for record in export_records
    )

    print("\n" + "=" * 70)
    print("OPTUNA BEST-MODEL EXPORT COMPLETED")
    print("=" * 70)
    print(f"Exported total: {len(export_records)}")
    print(f"Regression configs: {regression_count}")
    print(f"XGBoost configs: {xgboost_count}")
    print(
        "Regression destination: "
        f"{REGRESSION_OUTPUT_DIRECTORY}"
    )
    print(
        "XGBoost destination: "
        f"{TREE_BOOSTING_OUTPUT_DIRECTORY}"
    )


if __name__ == "__main__":
    main()