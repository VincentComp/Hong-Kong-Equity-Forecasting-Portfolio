"""Export temporary portfolio-wide Optuna best models into global config folders.

This script reads completed temporary portfolio Optuna artifacts from:

    outputs/tuning/temp_portfolio_*_mae_optuna_v*/

and writes frozen global model YAML files to:

    configs/models/tree_boosting/global/
    configs/models/regression/global/

Examples:
    python -m scripts.export_portfolio_optuna_global_models --dry-run

    python -m scripts.export_portfolio_optuna_global_models \
        --model-family xgboost \
        --version v1

    python -m scripts.export_portfolio_optuna_global_models

    python -m scripts.export_portfolio_optuna_global_models \
        --overwrite
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from src.hk_equity.utils.config import load_yaml


BASE_CONFIG_PATH = "configs/base.yaml"
DEFAULT_SOURCE_DIRECTORY = Path("outputs/tuning")

REGRESSION_OUTPUT_DIRECTORY = Path(
    "configs/models/regression/global"
)

XGBOOST_OUTPUT_DIRECTORY = Path(
    "configs/models/tree_boosting/global"
)

BEST_MODEL_CONFIG_FILENAME = "best_model_config.yaml"
TUNING_METADATA_FILENAME = "tuning_metadata.json"

SUPPORTED_MODEL_FAMILIES = {
    "regression",
    "xgboost",
}

SUPPORTED_VERSIONS = {
    "v1",
    "v2",
}


def parse_arguments() -> argparse.Namespace:
    """Read command-line settings for global best-model export."""

    parser = argparse.ArgumentParser(
        description=(
            "Export temporary portfolio-wide Optuna best models "
            "into global model configuration folders."
        ),
    )

    parser.add_argument(
        "--base-config",
        default=BASE_CONFIG_PATH,
        help="Path to shared project configuration YAML.",
    )

    parser.add_argument(
        "--source-directory",
        default=str(DEFAULT_SOURCE_DIRECTORY),
        help=(
            "Directory containing temporary portfolio Optuna outputs. "
            "Default: outputs/tuning"
        ),
    )

    parser.add_argument(
        "--model-family",
        choices=sorted(SUPPORTED_MODEL_FAMILIES),
        default=None,
        help=(
            "Optional filter. Export only one model family."
        ),
    )

    parser.add_argument(
        "--version",
        choices=sorted(SUPPORTED_VERSIONS),
        default=None,
        help=(
            "Optional filter. Export only one validation version."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Show source and destination files without writing YAML files."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Allow overwriting an existing global best-model YAML file."
        ),
    )

    return parser.parse_args()


def load_model_yaml(
    yaml_path: Path,
) -> dict[str, Any]:
    """Load one Optuna best-model YAML artifact."""

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


def load_metadata(
    metadata_path: Path,
) -> dict[str, Any]:
    """Load one optional tuning metadata JSON artifact."""

    if not metadata_path.exists():
        return {}

    with metadata_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    if not isinstance(metadata, dict):
        raise ValueError(
            f"Metadata JSON root must be a mapping: {metadata_path}"
        )

    return metadata


def get_model_family(
    model_config: dict[str, Any],
) -> str:
    """Map a model YAML model.name to an export model family."""

    model_name = str(
        model_config["model"].get(
            "name",
            "",
        )
    ).lower()

    if model_name == "tree_boosting":
        return "xgboost"

    if model_name == "general_regression":
        return "regression"

    raise ValueError(
        "Unsupported model.name for global export: "
        f"'{model_name}'."
    )


def get_version(
    study_directory: Path,
    metadata: dict[str, Any],
) -> str:
    """Read a v1/v2 validation version from metadata or study name."""

    metadata_version = str(
        metadata.get(
            "tuning_version",
            "",
        )
    ).lower()

    if metadata_version in SUPPORTED_VERSIONS:
        return metadata_version

    study_name = study_directory.name.lower()

    for version in sorted(SUPPORTED_VERSIONS):
        if study_name.endswith(f"_{version}"):
            return version

        if f"_{version}_" in study_name:
            return version

    raise ValueError(
        "Cannot identify v1 or v2 from study directory: "
        f"{study_directory}"
    )


def get_destination_directory(
    model_family: str,
) -> Path:
    """Return the correct global model directory."""

    if model_family == "xgboost":
        return XGBOOST_OUTPUT_DIRECTORY

    if model_family == "regression":
        return REGRESSION_OUTPUT_DIRECTORY

    raise ValueError(
        f"Unsupported model family: {model_family}"
    )


def get_global_run_id(
    model_family: str,
    version: str,
) -> str:
    """Return a stable global portfolio best-model identifier."""

    return (
        f"{model_family}_portfolio_"
        f"mae_optuna_{version}_best"
    )


def get_global_label(
    model_family: str,
    version: str,
) -> str:
    """Return a readable global portfolio model label."""

    if model_family == "xgboost":
        family_label = "XGBoost"
    elif model_family == "regression":
        family_label = "Regression"
    else:
        raise ValueError(
            f"Unsupported model family: {model_family}"
        )

    return (
        f"{family_label} Global Portfolio "
        f"Optuna {version.upper()} Best"
    )


def validate_model_config(
    model_config: dict[str, Any],
    source_path: Path,
    portfolio_tickers: list[str],
    benchmark_ticker: str,
) -> None:
    """Validate that one temporary best model is a full portfolio model."""

    required_sections = {
        "model",
        "parameters",
        "training",
        "features",
    }

    missing_sections = (
        required_sections - set(model_config)
    )

    if missing_sections:
        raise KeyError(
            "Best-model YAML is missing sections "
            f"{sorted(missing_sections)}: {source_path}"
        )

    training_tickers = model_config[
        "training"
    ].get(
        "tickers",
    )

    if not isinstance(training_tickers, list):
        raise TypeError(
            "training.tickers must be a YAML list: "
            f"{source_path}"
        )

    normalized_training_tickers = [
        str(ticker)
        for ticker in training_tickers
    ]

    if benchmark_ticker in normalized_training_tickers:
        raise ValueError(
            "HSI benchmark must not be in training.tickers: "
            f"{source_path}"
        )

    if len(normalized_training_tickers) != len(
        set(normalized_training_tickers)
    ):
        raise ValueError(
            "training.tickers contains duplicates: "
            f"{source_path}"
        )

    if set(normalized_training_tickers) != set(
        portfolio_tickers
    ):
        missing_tickers = sorted(
            set(portfolio_tickers)
            - set(normalized_training_tickers)
        )

        unexpected_tickers = sorted(
            set(normalized_training_tickers)
            - set(portfolio_tickers)
        )

        raise ValueError(
            "Global portfolio model must train on every portfolio "
            "ticker exactly once. "
            f"Missing: {missing_tickers}; "
            f"unexpected: {unexpected_tickers}; "
            f"source: {source_path}"
        )

    selected_columns = model_config[
        "features"
    ].get(
        "selected_columns",
    )

    if not isinstance(selected_columns, list):
        raise TypeError(
            "features.selected_columns must be a YAML list: "
            f"{source_path}"
        )

    if not selected_columns:
        raise ValueError(
            "features.selected_columns cannot be empty: "
            f"{source_path}"
        )


def build_global_model_config(
    source_config: dict[str, Any],
    model_family: str,
    version: str,
    source_study_directory: Path,
) -> dict[str, Any]:
    """Create a frozen global config from one Optuna best-model artifact."""

    config = source_config.copy()
    config["model"] = source_config["model"].copy()

    global_run_id = get_global_run_id(
        model_family=model_family,
        version=version,
    )

    config["model"]["label"] = get_global_label(
        model_family=model_family,
        version=version,
    )

    config["model"]["version"] = version
    config["model"]["run_id"] = global_run_id

    config["metadata"] = {
        "tuning_scope": "portfolio",
        "model_family": model_family,
        "tuning_version": version,
        "source_study_directory": str(
            source_study_directory
        ),
        "source_model_run_id": str(
            source_config["model"].get(
                "run_id",
                "",
            )
        ),
        "selection_rule": (
            "Frozen Optuna best model selected by equal-weight "
            "portfolio MAE across all 10 portfolio stocks."
        ),
        "benchmark_rule": (
            "HSI (^HSI) is benchmark-only and is excluded from "
            "training.tickers and portfolio predictions."
        ),
    }

    return config


def discover_source_paths(
    source_directory: Path,
) -> list[Path]:
    """Find temporary portfolio Optuna best-model YAML artifacts."""

    if not source_directory.exists():
        raise FileNotFoundError(
            "Cannot find source directory: "
            f"{source_directory}"
        )

    source_paths = sorted(
        source_directory.glob(
            "temp_portfolio_*_mae_optuna_v*/"
            f"{BEST_MODEL_CONFIG_FILENAME}"
        )
    )

    if not source_paths:
        raise FileNotFoundError(
            "No temporary portfolio best-model YAML files found in: "
            f"{source_directory}"
        )

    return source_paths


def write_yaml(
    output_path: Path,
    config: dict[str, Any],
) -> None:
    """Write one frozen global model YAML file."""

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
    """Export temporary portfolio Optuna winners as global model configs."""

    args = parse_arguments()

    base_config = load_yaml(
        args.base_config
    )

    portfolio_tickers = list(
        base_config["tickers"].keys()
    )

    benchmark_ticker = str(
        base_config["market"]["ticker"]
    )

    if benchmark_ticker in portfolio_tickers:
        raise ValueError(
            "Benchmark ticker must not be a portfolio ticker: "
            f"{benchmark_ticker}"
        )

    if len(portfolio_tickers) != 10:
        raise ValueError(
            "Expected exactly 10 portfolio stocks, found: "
            f"{len(portfolio_tickers)}"
        )

    source_directory = Path(
        args.source_directory
    )

    source_paths = discover_source_paths(
        source_directory=source_directory,
    )

    export_records = []

    for source_path in source_paths:
        source_study_directory = source_path.parent

        source_config = load_model_yaml(
            yaml_path=source_path,
        )

        metadata = load_metadata(
            metadata_path=(
                source_study_directory
                / TUNING_METADATA_FILENAME
            ),
        )

        model_family = get_model_family(
            model_config=source_config,
        )

        version = get_version(
            study_directory=source_study_directory,
            metadata=metadata,
        )

        if args.model_family is not None:
            if model_family != args.model_family:
                continue

        if args.version is not None:
            if version != args.version:
                continue

        validate_model_config(
            model_config=source_config,
            source_path=source_path,
            portfolio_tickers=portfolio_tickers,
            benchmark_ticker=benchmark_ticker,
        )

        global_config = build_global_model_config(
            source_config=source_config,
            model_family=model_family,
            version=version,
            source_study_directory=source_study_directory,
        )

        global_run_id = global_config["model"][
            "run_id"
        ]

        output_path = (
            get_destination_directory(
                model_family=model_family,
            )
            / f"{global_run_id}.yaml"
        )

        export_records.append({
            "source_path": source_path,
            "output_path": output_path,
            "model_family": model_family,
            "version": version,
            "config": global_config,
        })

    if not export_records:
        raise ValueError(
            "No completed temporary portfolio Optuna studies matched "
            "the selected filters."
        )

    duplicate_outputs = []
    seen_outputs: set[Path] = set()

    for record in export_records:
        output_path = record["output_path"]

        if output_path in seen_outputs:
            duplicate_outputs.append(output_path)

        seen_outputs.add(output_path)

    if duplicate_outputs:
        raise ValueError(
            "Multiple source studies map to the same global output: "
            f"{sorted(str(path) for path in duplicate_outputs)}"
        )

    existing_outputs = [
        record["output_path"]
        for record in export_records
        if record["output_path"].exists()
    ]

    if existing_outputs and not args.overwrite:
        formatted_paths = "\n".join(
            str(path)
            for path in existing_outputs
        )

        raise FileExistsError(
            "Refusing to overwrite existing global model configs. "
            "Use --overwrite only when replacement is intentional:\n"
            f"{formatted_paths}"
        )

    print("\n" + "=" * 70)
    print("PORTFOLIO OPTUNA GLOBAL MODEL EXPORT")
    print("=" * 70)
    print(f"Source directory: {source_directory}")
    print(f"Portfolio stocks: {len(portfolio_tickers)}")
    print(f"Benchmark excluded: {benchmark_ticker}")
    print(f"Dry run: {args.dry_run}")
    print(f"Models selected for export: {len(export_records)}")
    print("=" * 70)

    for position, record in enumerate(
        export_records,
        start=1,
    ):
        model = record["config"]["model"]

        print(
            f"\n{position:02d}. "
            f"family={record['model_family']} | "
            f"version={record['version']}"
        )
        print(f"    label: {model['label']}")
        print(f"    run_id: {model['run_id']}")
        print(f"    source: {record['source_path']}")
        print(f"    destination: {record['output_path']}")

    if args.dry_run:
        print(
            "\nDry run completed. "
            "No global model YAML files were written."
        )
        return

    for record in export_records:
        write_yaml(
            output_path=record["output_path"],
            config=record["config"],
        )

    print("\n" + "=" * 70)
    print("PORTFOLIO OPTUNA GLOBAL MODEL EXPORT COMPLETED")
    print("=" * 70)

    for record in export_records:
        print(f"- {record['output_path']}")


if __name__ == "__main__":
    main()