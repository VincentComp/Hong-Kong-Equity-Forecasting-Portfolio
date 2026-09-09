"""Build one model-comparison summary table from existing backtest outputs.

This script is read-only:
    - It does NOT download new data.
    - It does NOT run or retrain any model.
    - It only aggregates previously saved backtest result CSV files.

Examples:
    python scripts/build_model_summary.py

    python scripts/build_model_summary.py \
        --start 2025-01-01 \
        --end 2025-12-31

    python scripts/build_model_summary.py \
        --start 2026-01-01 \
        --end 2026-08-31

Expected backtest structure:
    outputs/backtests/
    ├── ma_4w_v1/
    │   └── 2026-01-01_to_2026-08-31/
    │       ├── overall_metrics.csv
    │       ├── rank_ic_by_week.csv
    │       └── top_3_by_week.csv
    │
    └── ewma_span4_v1/
        └── 2026-01-01_to_2026-08-31/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.hk_equity.utils.config import load_yaml


def parse_arguments() -> argparse.Namespace:
    """Read settings for the model-summary aggregation."""

    parser = argparse.ArgumentParser(
        description=(
            "Build a model comparison table from existing backtest CSV files."
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
            "Optional custom output file name without extension. "
            "Example: model_summary_validation"
        ),
    )

    return parser.parse_args()


def safe_mean(
    data: pd.Series,
) -> float:
    """Return mean after removing missing values; return NaN if no values exist."""

    valid_data = data.dropna()

    if valid_data.empty:
        return np.nan

    return valid_data.mean()


def load_optional_csv(
    path: Path,
) -> pd.DataFrame:
    """Load optional CSV safely.

    Missing or empty files are treated as empty DataFrames. This is expected
    for models such as the zero-return baseline, which cannot produce a
    meaningful stock ranking.
    """

    if not path.exists():
        return pd.DataFrame()

    # Zero-return model may create an empty ranking file.
    if path.stat().st_size == 0:
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def summarise_one_model_run(
    run_directory: Path,
) -> dict:
    """Read metrics and strategy diagnostics for one completed model backtest."""

    overall_metrics_path = (
        run_directory
        / "overall_metrics.csv"
    )

    if not overall_metrics_path.exists():
        raise FileNotFoundError(
            f"Missing required file: {overall_metrics_path}"
        )

    overall_metrics = pd.read_csv(
        overall_metrics_path
    )

    if overall_metrics.empty:
        raise ValueError(
            f"overall_metrics.csv is empty: {overall_metrics_path}"
        )

    # Each backtest run should have exactly one row of overall metrics.
    overall_row = overall_metrics.iloc[0].to_dict()

    rank_ic = load_optional_csv(
        run_directory / "rank_ic_by_week.csv"
    )

    top_3 = load_optional_csv(
        run_directory / "top_3_by_week.csv"
    )

    average_rank_ic = np.nan
    positive_rank_ic_week_ratio = np.nan
    number_of_rank_ic_weeks = 0

    if not rank_ic.empty and "rank_ic" in rank_ic.columns:
        valid_rank_ic = rank_ic["rank_ic"].dropna()

        if not valid_rank_ic.empty:
            average_rank_ic = valid_rank_ic.mean()

            positive_rank_ic_week_ratio = (
                valid_rank_ic > 0
            ).mean()

            number_of_rank_ic_weeks = len(
                valid_rank_ic
            )

    average_top_3_actual_return = np.nan
    average_equal_weight_return = np.nan
    average_top_3_minus_equal_weight = np.nan
    number_of_top_3_weeks = 0

    if not top_3.empty:
        required_columns = [
            "top_n_average_actual_return",
            "equal_weight_average_actual_return",
            "top_n_minus_equal_weight",
        ]

        if all(
            column in top_3.columns
            for column in required_columns
        ):
            average_top_3_actual_return = safe_mean(
                top_3["top_n_average_actual_return"]
            )

            average_equal_weight_return = safe_mean(
                top_3["equal_weight_average_actual_return"]
            )

            average_top_3_minus_equal_weight = safe_mean(
                top_3["top_n_minus_equal_weight"]
            )

            number_of_top_3_weeks = len(
                top_3.dropna(
                    subset=[
                        "top_n_minus_equal_weight",
                    ]
                )
            )

    # Add the metrics calculated from Rank IC and Top-3 selection.
    summary_row = {
        **overall_row,
        "average_rank_ic": average_rank_ic,
        "positive_rank_ic_week_ratio": (
            positive_rank_ic_week_ratio
        ),
        "number_of_rank_ic_weeks": number_of_rank_ic_weeks,
        "average_top_3_actual_return": (
            average_top_3_actual_return
        ),
        "average_equal_weight_return": (
            average_equal_weight_return
        ),
        "average_top_3_minus_equal_weight": (
            average_top_3_minus_equal_weight
        ),
        "number_of_top_3_weeks": number_of_top_3_weeks,
        "backtest_run_directory": str(run_directory),
    }

    # Zero forecast is only a benchmark, because it has no direction or ranking.
    if summary_row.get("model_key") == "zero":
        summary_row["role"] = "Baseline"
    else:
        summary_row["role"] = "Candidate"

    return summary_row


def build_summary(
    backtests_directory: Path,
    evaluation_label: str,
) -> pd.DataFrame:
    """Discover all completed model runs for one evaluation period."""

    overall_metric_files = list(
        backtests_directory.glob(
            f"*/{evaluation_label}/overall_metrics.csv"
        )
    )

    if not overall_metric_files:
        raise FileNotFoundError(
            "No completed backtest results were found for:\n"
            f"{backtests_directory}/*/{evaluation_label}/overall_metrics.csv\n\n"
            "Run one or more models first. For example:\n"
            "python scripts/run_backtest.py "
            "--model-config configs/models/moving_average.yaml"
        )

    summary_rows = []

    for overall_metric_path in overall_metric_files:
        run_directory = overall_metric_path.parent

        try:
            summary_rows.append(
                summarise_one_model_run(
                    run_directory=run_directory,
                )
            )

        except Exception as error:
            print(
                f"Warning: skipped {run_directory} because: {error}"
            )

    if not summary_rows:
        raise ValueError(
            "No valid backtest result folders could be summarised."
        )

    summary = pd.DataFrame(summary_rows)

    # Ensure expected columns exist even if a model cannot provide ranking metrics.
    expected_columns = [
        "model",
        "model_key",
        "model_run_id",
        "model_version",
        "model_parameters",
        "evaluation_start",
        "evaluation_end",
        "observations",
        "MAE",
        "RMSE",
        "directional_accuracy",
        "mean_predicted_return",
        "mean_actual_return",
        "forecast_bias",
        "average_rank_ic",
        "positive_rank_ic_week_ratio",
        "number_of_rank_ic_weeks",
        "average_top_3_actual_return",
        "average_equal_weight_return",
        "average_top_3_minus_equal_weight",
        "number_of_top_3_weeks",
        "role",
        "backtest_run_directory",
    ]

    for column in expected_columns:
        if column not in summary.columns:
            summary[column] = np.nan

    summary = summary[
        expected_columns
    ].sort_values(
        by=["MAE", "RMSE"],
        ascending=[True, True],
        na_position="last",
    ).reset_index(drop=True)

    return summary


def print_summary(
    summary: pd.DataFrame,
) -> None:
    """Print the key metrics in a readable terminal table."""

    display_columns = [
        "role",
        "model_run_id",
        "model",
        "MAE",
        "RMSE",
        "directional_accuracy",
        "average_rank_ic",
        "average_top_3_minus_equal_weight",
    ]

    printable_summary = summary[
        display_columns
    ].copy()

    percentage_columns = [
        "MAE",
        "RMSE",
        "directional_accuracy",
        "average_top_3_minus_equal_weight",
    ]

    for column in percentage_columns:
        printable_summary[column] = (
            printable_summary[column]
            .map(
                lambda value: (
                    f"{value:.2%}"
                    if pd.notna(value)
                    else "N/A"
                )
            )
        )

    printable_summary["average_rank_ic"] = (
        printable_summary["average_rank_ic"]
        .map(
            lambda value: (
                f"{value:.4f}"
                if pd.notna(value)
                else "N/A"
            )
        )
    )

    print(
        "\nModel Performance Summary:\n"
    )

    print(
        printable_summary.to_string(
            index=False,
        )
    )


def main() -> None:
    """Build and save one summary table for a selected evaluation period."""

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

    backtests_directory = Path(
        base_config["paths"]["backtests"]
    )

    summary = build_summary(
        backtests_directory=backtests_directory,
        evaluation_label=evaluation_label,
    )

    reports_tables_directory = Path(
        base_config["paths"]["tables"]
    )

    reports_tables_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_name = (
        args.output_name
        if args.output_name is not None
        else f"model_summary_{evaluation_label}"
    )

    csv_path = (
        reports_tables_directory
        / f"{output_name}.csv"
    )

    excel_path = (
        reports_tables_directory
        / f"{output_name}.xlsx"
    )

    summary.to_csv(
        csv_path,
        index=False,
    )

    summary.to_excel(
        excel_path,
        index=False,
    )

    print_summary(summary)

    print(f"\nCSV summary saved to: {csv_path}")
    print(f"Excel summary saved to: {excel_path}")


if __name__ == "__main__":
    main()