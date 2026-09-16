"""Build portfolio-router YAML configs from frozen Optuna best models.

Expected exported model-config naming:

    {ticker_id}_regression_mae_optuna_{version}_stable_best.yaml
    {ticker_id}_xgboost_mae_optuna_{version}_stable_best.yaml

Examples:
    python -m scripts.build_optuna_portfolios --dry-run

    python -m scripts.build_optuna_portfolios

    python -m scripts.build_optuna_portfolios \
        --model-family regression \
        --version v1

    python -m scripts.build_optuna_portfolios \
        --overwrite
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from src.hk_equity.utils.config import load_yaml


# =============================================================================
# Shared project settings
# =============================================================================

BASE_CONFIG_PATH = "configs/base.yaml"

REGRESSION_GROUP_DIRECTORY = Path(
    "configs/models/regression/groups"
)

XGBOOST_GROUP_DIRECTORY = Path(
    "configs/models/tree_boosting/groups"
)

PORTFOLIO_OUTPUT_DIRECTORY = Path(
    "configs/models/portfolios"
)

SUPPORTED_MODEL_FAMILIES = [
    "regression",
    "xgboost",
]

SUPPORTED_VERSIONS = [
    "v1",
    "v2",
]


# =============================================================================
# Command-line arguments
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    """Read command-line options for portfolio-router generation."""

    parser = argparse.ArgumentParser(
        description=(
            "Build portfolio-router YAML configs from Optuna "
            "best-model YAML files."
        ),
    )

    parser.add_argument(
        "--model-family",
        choices=SUPPORTED_MODEL_FAMILIES,
        default=None,
        help=(
            "Optional filter. Build only regression or xgboost "
            "portfolio routers."
        ),
    )

    parser.add_argument(
        "--version",
        choices=SUPPORTED_VERSIONS,
        default=None,
        help=(
            "Optional filter. Build only v1 or v2 portfolio routers."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Show discovered model files and planned portfolio YAML "
            "files without writing anything."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Allow overwriting an existing portfolio-router YAML."
        ),
    )

    return parser.parse_args()


# =============================================================================
# Model config discovery and validation
# =============================================================================

def get_group_directory(
    model_family: str,
) -> Path:
    """Return the group-model directory for the selected family."""

    if model_family == "regression":
        return REGRESSION_GROUP_DIRECTORY

    if model_family == "xgboost":
        return XGBOOST_GROUP_DIRECTORY

    raise ValueError(
        f"Unsupported model family: {model_family}"
    )


def get_expected_model_name(
    model_family: str,
) -> str:
    """Return the expected registered model.name for one family."""

    if model_family == "regression":
        return "general_regression"

    if model_family == "xgboost":
        return "tree_boosting"

    raise ValueError(
        f"Unsupported model family: {model_family}"
    )


def get_best_model_filename(
    ticker: str,
    model_family: str,
    version: str,
) -> str:
    """Return the expected exported best-model filename."""

    ticker_id = ticker.replace(
        ".HK",
        "",
    ).lower()

    return (
        f"{ticker_id}_{model_family}_mae_optuna_"
        f"{version}_stable_best.yaml"
    )


def load_model_config(
    model_path: Path,
) -> dict[str, Any]:
    """Load one frozen best-model YAML config."""

    with model_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        model_config = yaml.safe_load(file)

    if not isinstance(model_config, dict):
        raise ValueError(
            f"YAML root must be a mapping: {model_path}"
        )

    return model_config


def validate_best_model_config(
    model_config: dict[str, Any],
    model_path: Path,
    ticker: str,
    model_family: str,
) -> None:
    """Validate that one exported YAML is usable in a portfolio router."""

    required_sections = {
        "model",
        "parameters",
        "training",
        "features",
    }

    missing_sections = (
        required_sections - set(model_config.keys())
    )

    if missing_sections:
        raise KeyError(
            "Model config is missing sections "
            f"{sorted(missing_sections)}: {model_path}"
        )

    expected_model_name = get_expected_model_name(
        model_family=model_family,
    )

    actual_model_name = str(
        model_config["model"].get(
            "name",
            "",
        )
    ).lower()

    if actual_model_name != expected_model_name:
        raise ValueError(
            "Unexpected model.name in best model config. "
            f"Expected '{expected_model_name}', "
            f"found '{actual_model_name}': {model_path}"
        )

    run_id = str(
        model_config["model"].get(
            "run_id",
            "",
        )
    )

    if not run_id.endswith("_best"):
        raise ValueError(
            "Expected a frozen *_best model config, "
            f"found run_id='{run_id}': {model_path}"
        )

    training_tickers = model_config[
        "training"
    ].get(
        "tickers",
        [],
    )

    if not training_tickers:
        raise ValueError(
            f"training.tickers is empty: {model_path}"
        )

    if str(training_tickers[0]) != ticker:
        raise ValueError(
            "The first training ticker must be the base ticker. "
            f"Expected '{ticker}', found "
            f"'{training_tickers[0]}': {model_path}"
        )

    if "^HSI" in training_tickers:
        raise ValueError(
            "HSI (^HSI) must never appear in training.tickers: "
            f"{model_path}"
        )


def discover_best_models(
    portfolio_tickers: list[str],
    model_family: str,
    version: str,
) -> dict[str, Path]:
    """Find exactly one frozen best model YAML for every portfolio ticker."""

    group_directory = get_group_directory(
        model_family=model_family,
    )

    if not group_directory.exists():
        raise FileNotFoundError(
            "Cannot find model group directory: "
            f"{group_directory}"
        )

    model_paths = {}

    for ticker in portfolio_tickers:
        filename = get_best_model_filename(
            ticker=ticker,
            model_family=model_family,
            version=version,
        )

        model_path = group_directory / filename

        if not model_path.exists():
            raise FileNotFoundError(
                "Missing exported best-model YAML for "
                f"{ticker}, family={model_family}, "
                f"version={version}:\n{model_path}"
            )

        model_config = load_model_config(
            model_path=model_path,
        )

        validate_best_model_config(
            model_config=model_config,
            model_path=model_path,
            ticker=ticker,
            model_family=model_family,
        )

        model_paths[ticker] = model_path

    return model_paths


# =============================================================================
# Portfolio-router config construction
# =============================================================================

def get_portfolio_run_id(
    model_family: str,
    version: str,
) -> str:
    """Return one stable portfolio-router run ID."""

    return (
        f"{model_family}_optuna_"
        f"{version}_stable_portfolio"
    )


def get_portfolio_label(
    model_family: str,
    version: str,
) -> str:
    """Return a readable portfolio-router output label."""

    if model_family == "regression":
        family_label = "Regression"

    elif model_family == "xgboost":
        family_label = "XGBoost"

    else:
        raise ValueError(
            f"Unsupported model family: {model_family}"
        )

    return (
        f"{family_label} Optuna "
        f"{version.upper()} Stable Portfolio"
    )


def get_profile_name(
    ticker: str,
    model_family: str,
    version: str,
) -> str:
    """Create a readable and unique router profile key."""

    ticker_id = ticker.replace(
        ".HK",
        "_HK",
    )

    return (
        f"{model_family}_{ticker_id}_{version}"
    )


def build_portfolio_router_config(
    portfolio_tickers: list[str],
    best_model_paths: dict[str, Path],
    model_family: str,
    version: str,
) -> dict[str, Any]:
    """Build one complete portfolio-router YAML config."""

    run_id = get_portfolio_run_id(
        model_family=model_family,
        version=version,
    )

    profiles = {}
    assignments = {}

    for ticker in portfolio_tickers:
        profile_name = get_profile_name(
            ticker=ticker,
            model_family=model_family,
            version=version,
        )

        profiles[profile_name] = {
            "config_path": str(
                best_model_paths[ticker]
            ),
        }

        assignments[ticker] = profile_name

    return {
        "model": {
            "name": "portfolio_router",
            "label": get_portfolio_label(
                model_family=model_family,
                version=version,
            ),
            "version": f"{version}_stable",
            "run_id": run_id,
        },
        "profiles": profiles,
        "assignments": assignments,
        "metadata": {
            "model_family": model_family,
            "tuning_version": version,
            "selection_rule": (
                "Each portfolio ticker uses its own frozen "
                "Optuna *_best model config."
            ),
            "benchmark_rule": (
                "HSI (^HSI) is benchmark-only and is excluded "
                "from assignments."
            ),
        },
    }


def write_portfolio_config(
    portfolio_config: dict[str, Any],
    output_path: Path,
) -> None:
    """Write one generated portfolio-router YAML config."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(
            portfolio_config,
            file,
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
        )


# =============================================================================
# Main execution
# =============================================================================

def main() -> None:
    """Build all requested regression and XGBoost portfolio routers."""

    args = parse_arguments()

    base_config = load_yaml(
        BASE_CONFIG_PATH
    )

    portfolio = base_config["tickers"]
    portfolio_tickers = list(portfolio.keys())

    benchmark_ticker = base_config["market"]["ticker"]

    if benchmark_ticker in portfolio_tickers:
        raise ValueError(
            "HSI benchmark must not be in portfolio tickers: "
            f"{benchmark_ticker}"
        )

    if len(portfolio_tickers) != 10:
        raise ValueError(
            "Expected exactly 10 portfolio stocks, found: "
            f"{len(portfolio_tickers)}"
        )

    selected_families = (
        [args.model_family]
        if args.model_family is not None
        else SUPPORTED_MODEL_FAMILIES
    )

    selected_versions = (
        [args.version]
        if args.version is not None
        else SUPPORTED_VERSIONS
    )

    portfolio_records = []

    for model_family in selected_families:
        for version in selected_versions:
            best_model_paths = discover_best_models(
                portfolio_tickers=portfolio_tickers,
                model_family=model_family,
                version=version,
            )

            portfolio_config = build_portfolio_router_config(
                portfolio_tickers=portfolio_tickers,
                best_model_paths=best_model_paths,
                model_family=model_family,
                version=version,
            )

            run_id = portfolio_config["model"]["run_id"]

            output_path = (
                PORTFOLIO_OUTPUT_DIRECTORY
                / f"{run_id}.yaml"
            )

            portfolio_records.append({
                "model_family": model_family,
                "version": version,
                "best_model_paths": best_model_paths,
                "portfolio_config": portfolio_config,
                "output_path": output_path,
            })

    existing_paths = [
        record["output_path"]
        for record in portfolio_records
        if record["output_path"].exists()
    ]

    if existing_paths and not args.overwrite:
        formatted_paths = "\n".join(
            str(path)
            for path in existing_paths
        )

        raise FileExistsError(
            "Refusing to overwrite existing portfolio YAML files. "
            "Use --overwrite only when replacement is intentional:\n"
            f"{formatted_paths}"
        )

    print("\n" + "=" * 70)
    print("OPTUNA PORTFOLIO ROUTER GENERATOR")
    print("=" * 70)
    print(f"Portfolio stocks: {len(portfolio_tickers)}")
    print(f"Benchmark excluded: {benchmark_ticker}")
    print(f"Dry run: {args.dry_run}")
    print("=" * 70)

    for record in portfolio_records:
        portfolio_config = record[
            "portfolio_config"
        ]

        print(
            "\nPortfolio router:"
            f"\n- Family: {record['model_family']}"
            f"\n- Version: {record['version']}"
            f"\n- Label: {portfolio_config['model']['label']}"
            f"\n- Run ID: {portfolio_config['model']['run_id']}"
            f"\n- Output: {record['output_path']}"
        )

        print("- Assigned frozen models:")

        for ticker in portfolio_tickers:
            print(
                f"  {ticker} -> "
                f"{record['best_model_paths'][ticker]}"
            )

    if args.dry_run:
        print(
            "\nDry run completed. "
            "No portfolio-router YAML files were written."
        )
        return

    for record in portfolio_records:
        write_portfolio_config(
            portfolio_config=record[
                "portfolio_config"
            ],
            output_path=record["output_path"],
        )

    print("\n" + "=" * 70)
    print("OPTUNA PORTFOLIO ROUTER GENERATION COMPLETED")
    print("=" * 70)
    print(
        f"Portfolio YAML files written: "
        f"{len(portfolio_records)}"
    )

    for record in portfolio_records:
        print(f"- {record['output_path']}")


if __name__ == "__main__":
    main()