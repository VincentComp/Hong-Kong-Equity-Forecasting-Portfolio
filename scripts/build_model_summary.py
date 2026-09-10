"""Build one model-performance summary table from existing backtest results.

Examples:
    python -m scripts.build_model_summary

    python -m scripts.build_model_summary \
        --start 2025-01-01 \
        --end 2025-12-31 \
        --output-name model_summary_validation

    python -m scripts.build_model_summary \
        --start 2025-01-01 \
        --end 2025-12-31 \
        --output-name model_summary_validation \
        --run-ids ma_4w_v1 ewma_span4_v1

This script:
1. Scans outputs/backtests/ for all model result folders.
2. Loads overall_metrics.csv for each model in the specified period.
3. Merges all results into one summary table (CSV + Excel).

It does not run any backtests. It only aggregates existing results.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import os

import pandas as pd

from src.hk_equity.utils.config import load_yaml


#Pass the input command
def parse_arguments() -> argparse.Namespace:
    """Read command-line settings for building a model summary."""

    parser = argparse.ArgumentParser(
        description=(
            "Build one model-performance summary table from existing "
            "backtest results."
        )
    )

    parser.add_argument(
        "--base-config",
        default="configs/base.yaml",
        help="Path to shared project configuration YAML.",
    )

    parser.add_argument(
        "--start",
        default=None,
        help=(
            "Evaluation start date in YYYY-MM-DD format. "
            "Default: test_start in base.yaml."
        ),
    )

    parser.add_argument(
        "--end",
        default=None,
        help=(
            "Evaluation end date in YYYY-MM-DD format. "
            "Default: test_end in base.yaml."
        ),
    )

    parser.add_argument(
        "--output-name",
        default=None,
        help=(
            "Optional summary output name without file extension. "
            "Default: model_summary_<start>_to_<end>."
        ),
    )

    parser.add_argument(
        "--run-ids",
        nargs="*",
        default=None,
        help=(
            "Optional model run_ids to filter which models to include. "
            "Matches against folder names in outputs/backtests/. "
            "If omitted, includes all models with results in the period."
        ),
    )

    return parser.parse_args()


#Scan outputs/backtests/ -> to select valid backtest csv
def discover_model_run_ids(
    backtests_directory: str,
    evaluation_label: str,
    selected_run_ids: list[str] | None = None,
) -> list[str]:
    """Discover model run_ids that have results for the specified period."""

    if not Path(backtests_directory).exists(): #if directory not exist -> end
        raise FileNotFoundError(
            f"Backtests directory not found: {backtests_directory}"
        )

    all_run_ids = [#get all the model run id
        item for item in os.listdir(backtests_directory)
        if (Path(backtests_directory) / item).is_dir()
    ]


    available_run_ids = []
    for run_id in all_run_ids:
        period_dir = (
            Path(backtests_directory)
            / run_id
            / evaluation_label
        )

        if period_dir.exists():
            available_run_ids.append(run_id)

    #if have specific model, then build summary for specified model only
    if selected_run_ids is None:
        return sorted(available_run_ids)

    #if have no specific model, then build summary for all existing model csv
    filtered = [
        rid for rid in available_run_ids
        if rid in selected_run_ids
    ]

    if not filtered: #if nothing remain -> return error
        raise ValueError(
            f"No model results found for run_ids: {selected_run_ids} "
            f"in period: {evaluation_label}"
        )

    return sorted(filtered)


def load_model_metrics(#read the valid result metrics
    backtests_directory: str,
    model_run_id: str,
    evaluation_label: str,
) -> pd.DataFrame:
    """Load overall_metrics.csv for one model and period."""

    metrics_path = (
        Path(backtests_directory)
        / model_run_id
        / evaluation_label
        / "overall_metrics.csv"
    )

    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Cannot find: {metrics_path}"
        )

    metrics = pd.read_csv(
        metrics_path,
        dtype={
            "model": "string",
            "model_key": "string",
            "model_run_id": "string",
            "model_version": "string",
            "model_parameters": "string",
            "evaluation_start": "string",
            "evaluation_end": "string",
        },
    )

    return metrics


def main() -> None:
    """Aggregate all model results into one summary table."""

    args = parse_arguments()

    base_config = load_yaml(
        args.base_config
    )

    evaluation_start = (
        args.start
        if args.start is not None
        else base_config["periods"]["test_start"]
    )

    evaluation_end = (
        args.end
        if args.end is not None
        else base_config["periods"]["test_end"]
    )

    evaluation_label = (
        f"{evaluation_start}_to_{evaluation_end}"
    )

    backtests_directory = (
        base_config["paths"]["backtests"]
    )

    reports_tables_dir = (
        base_config["paths"]["reports_tables"]
    )

    Path(reports_tables_dir).mkdir(
        parents=True,
        exist_ok=True,
    )

    model_run_ids = discover_model_run_ids(
        backtests_directory=backtests_directory,
        evaluation_label=evaluation_label,
        selected_run_ids=args.run_ids,
    )

    print(
        "\nBuilding model summary"
        f"\nEvaluation period: {evaluation_label}"
        f"\nModels found: {len(model_run_ids)}"
        f"\nModels: {', '.join(model_run_ids)}"
    )

    all_metrics = []

    for model_run_id in model_run_ids:
        metrics = load_model_metrics(
            backtests_directory=backtests_directory,
            model_run_id=model_run_id,
            evaluation_label=evaluation_label,
        )

        all_metrics.append(metrics)

    if not all_metrics:
        raise ValueError(
            f"No model results found for period: {evaluation_label}"
        )

    summary_table = pd.concat(
        all_metrics,
        ignore_index=True,
    )

    if args.output_name is None:
        output_name = f"model_summary_{evaluation_label}"
    else:
        output_name = args.output_name

    csv_path = (
        Path(reports_tables_dir)
        / f"{output_name}.csv"
    )

    excel_path = (
        Path(reports_tables_dir)
        / f"{output_name}.xlsx"
    )

    summary_table.to_csv(
        csv_path,
        index=False,
    )

    summary_table.to_excel(
        excel_path,
        index=False,
    )

    print("\nModel summary built successfully.\n")
    print(summary_table.to_string(index=False))
    print(f"\nCSV: {csv_path}")
    print(f"Excel: {excel_path}")


if __name__ == "__main__":
    main()