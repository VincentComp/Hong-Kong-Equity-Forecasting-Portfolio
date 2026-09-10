"""Run missing backtests and then build one model-performance summary.

Examples:
    python -m scripts.run_experiment_suite

    python -m scripts.run_experiment_suite \
        --start 2025-01-01 \
        --end 2025-12-31 \
        --output-name model_summary_validation

This script:
1. Checks whether each configured model already has a backtest result.
2. Runs only missing backtests.
3. Builds one summary table from the resulting CSV files.

It does not overwrite existing model results unless --force is used.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

from src.hk_equity.utils.config import load_yaml


DEFAULT_MODEL_CONFIGS = [
    "configs/models/zero.yaml",
    "configs/models/moving_average.yaml",
    "configs/models/ewma.yaml",
]


def parse_arguments() -> argparse.Namespace:
    """Read experiment-suite command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Run missing model backtests and build a summary table."
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
            "Optional summary output name without file extension."
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Re-run all backtests even when output files already exist."
        ),
    )

    return parser.parse_args()


def backtest_result_exists(
    backtests_directory: str,
    model_run_id: str,
    evaluation_label: str,
) -> bool:
    """Check whether a model has already produced overall metrics."""

    expected_metrics_file = (
        Path(backtests_directory)
        / model_run_id
        / evaluation_label
        / "overall_metrics.csv"
    )

    return expected_metrics_file.exists()


def run_backtest(
    base_config_path: str,
    model_config_path: str,
    evaluation_start: str,
    evaluation_end: str,
) -> None:
    """Run one backtest through the existing module entry point."""

    command = [
        sys.executable,
        "-m",
        "scripts.run_backtest",
        "--base-config",
        base_config_path,
        "--model-config",
        model_config_path,
        "--start",
        evaluation_start,
        "--end",
        evaluation_end,
    ]

    print("\nRunning missing backtest:")
    print(" ".join(command))

    subprocess.run(
        command,
        check=True,
    )


def build_summary(
    base_config_path: str,
    evaluation_start: str,
    evaluation_end: str,
    output_name: str | None,
) -> None:
    """Build summary through the existing module entry point."""

    command = [
        sys.executable,
        "-m",
        "scripts.build_model_summary",
        "--base-config",
        base_config_path,
        "--start",
        evaluation_start,
        "--end",
        evaluation_end,
    ]

    if output_name is not None:
        command.extend([
            "--output-name",
            output_name,
        ])

    print("\nBuilding model summary:")
    print(" ".join(command))

    subprocess.run(
        command,
        check=True,
    )


def main() -> None:
    """Run missing backtests, then aggregate all result CSV files."""

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

    print(
        "\nExperiment Suite"
        f"\nEvaluation period: {evaluation_label}"
    )

    for model_config_path in DEFAULT_MODEL_CONFIGS:
        model_config = load_yaml(
            model_config_path
        )

        model_run_id = (
            model_config["model"]["run_id"]
        )

        result_exists = backtest_result_exists(
            backtests_directory=backtests_directory,
            model_run_id=model_run_id,
            evaluation_label=evaluation_label,
        )

        if result_exists and not args.force:
            print(
                f"\nSkipping {model_run_id}: "
                "existing backtest result found."
            )
            continue

        run_backtest(
            base_config_path=args.base_config,
            model_config_path=model_config_path,
            evaluation_start=evaluation_start,
            evaluation_end=evaluation_end,
        )

    build_summary(
        base_config_path=args.base_config,
        evaluation_start=evaluation_start,
        evaluation_end=evaluation_end,
        output_name=args.output_name,
    )

    print("\nExperiment suite completed successfully.")


if __name__ == "__main__":
    main()