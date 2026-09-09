from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

import pandas as pd


def save_forecast_run(
    forecast_df: pd.DataFrame,
    metadata: dict,
    forecast_date: str,
    model_key: str,
    forecasts_dir: str = "outputs/forecasts",
    allow_overwrite: bool = False,
) -> Path:
    """Save one forecast run under a date/model-specific directory.

    Output structure:
        outputs/forecasts/
        └── YYYY-MM-DD/
            └── model_key/
                ├── forecast.csv
                └── metadata.json

    Args:
        forecast_df: Final forecast table containing one row per stock.
        metadata: Dictionary containing data cutoff, model parameters, and
            forecast-target information.
        forecast_date: Forecast generation date in YYYY-MM-DD format.
        model_key: Machine-readable model name, for example
            ``moving_average`` or ``lightgbm``.
        forecasts_dir: Root directory for forecast artefacts.
        allow_overwrite: If False, stop execution when the same date/model
            output already exists. This protects official forecast records.

    Returns:
        Path to the saved model-specific forecast run directory.
    """

    # Create a model-specific folder so different models do not overwrite
    # each other when run on the same forecast date.
    run_dir = (
        Path(forecasts_dir)
        / forecast_date
        / model_key
    )

    forecast_path = run_dir / "forecast.csv"
    metadata_path = run_dir / "metadata.json"

    # Protect existing official forecast records unless overwrite is explicit.
    if run_dir.exists() and not allow_overwrite:
        raise FileExistsError(
            f"Forecast output already exists: {run_dir}\n"
            "Use --overwrite only if this is a draft/test run that you "
            "intentionally want to replace."
        )

    run_dir.mkdir(parents=True, exist_ok=True)

    # Save the forecasting result.
    forecast_df.to_csv(
        forecast_path,
        index=False,
    )

    # Save JSON metadata to make this model run reproducible.
    metadata = {
        **metadata,
        "saved_at_local": datetime.now().astimezone().isoformat(),
        "forecast_file": str(forecast_path),
    }

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False,
        )

    return run_dir


def export_submission_file(
    forecast_df: pd.DataFrame,
    forecast_date: str,
    model_key: str,
    submissions_dir: str = "outputs/submissions",
    allow_overwrite: bool = False,
) -> Path:
    """Export one model-specific CSV for internal review or Moodle submission.

    Args:
        forecast_df: Final cleaned forecast table.
        forecast_date: Forecast generation date in YYYY-MM-DD format.
        model_key: Machine-readable model name.
        submissions_dir: Directory for submission-format CSV files.
        allow_overwrite: Whether to allow replacing an existing file.

    Returns:
        Path to the exported CSV file.
    """

    output_dir = Path(submissions_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    submission_path = (
        output_dir
        / f"gp1_forecast_{forecast_date}_{model_key}.csv"
    )

    if submission_path.exists() and not allow_overwrite:
        raise FileExistsError(
            f"Submission-format file already exists: {submission_path}\n"
            "Use --overwrite only for draft/test outputs that you intend "
            "to replace."
        )

    # Save the clean forecast table for review or final submission.
    forecast_df.to_csv(
        submission_path,
        index=False,
    )

    return submission_path