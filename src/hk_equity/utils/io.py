from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

import pandas as pd


def save_forecast_run(
    forecast_df: pd.DataFrame,
    metadata: dict,
    forecast_date: str,
    forecasts_dir: str = "outputs/forecasts",
) -> Path:
    """Saves an immutable forecast execution run with associated metadata.

    Creates a timestamped execution directory under `outputs/forecasts/YYYY-MM-DD/`, 
    persists the output prediction DataFrame as a CSV file, and writes runtime 
    metadata (including model parameters and local ISO timestamp) to JSON.

    Args:
        forecast_df (pd.DataFrame): Forecast results DataFrame containing predicted 
            ticker returns, horizons, or prices.
        metadata (dict): Dictionary containing execution details, hyperparameter 
            configurations, git commits, or model pipeline specifications.
        forecast_date (str): Target forecast date string formatted as 'YYYY-MM-DD'.
        forecasts_dir (str, optional): Root directory where forecast execution runs 
            are archived. Defaults to "outputs/forecasts".

    Returns:
        Path: Path object pointing to the newly created forecast run directory.
    """
    
    #create the directory
    run_dir = Path(forecasts_dir) / forecast_date
    run_dir.mkdir(parents=True, exist_ok=True)

    #save the forcasting result
    forecast_df.to_csv(run_dir / "forecast.csv", index=False)

    #save the json meta data
    metadata = {
        **metadata,
        "saved_at_local": datetime.now().astimezone().isoformat(),
    }
    with open(run_dir / "metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)

    return run_dir


def export_submission_file(
    forecast_df: pd.DataFrame,
    forecast_date: str,
    submissions_dir: str = "outputs/submissions",
) -> Path:
    """Exports a formatted, deliverable forecast table for final submission.

    Writes the cleaned prediction table into the designated submission directory 
    using a standardized group artifact naming convention (`gp1_forecast_YYYY-MM-DD.csv`).

    Args:
        forecast_df (pd.DataFrame): Final cleaned forecast DataFrame ready for evaluation.
        forecast_date (str): Target submission/forecast date string formatted as 'YYYY-MM-DD'.
        submissions_dir (str, optional): Target output directory for coursework 
            submissions. Defaults to "outputs/submissions".

    Returns:
        Path: Path object pointing directly to the exported submission CSV file.
    """
    
    #save the file to submission
    output_dir = Path(submissions_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"gp1_forecast_{forecast_date}.csv"
    forecast_df.to_csv(path, index=False)
    return path