import numpy as np
import pandas as pd


def forecast_metrics(
    actual: pd.Series,
    predicted: pd.Series,
) -> dict:
    
    """Calculates forecast performance evaluation metrics.

    Evaluates time series predictions against actual ground-truth values by
    aligning series index labels, dropping missing entries, and computing 
    error-based (MAE, RMSE) and directional metrics (Directional Accuracy).

    Metrics calculated:
        - observations: Count of valid non-NaN prediction/actual pairs.
        - MAE (Mean Absolute Error): Mean magnitude of absolute forecast errors.
        - RMSE (Root Mean Squared Error): Square root of mean squared errors;
          penalizes larger outlier errors more heavily.
        - directional_accuracy: Proportion of periods where the forecasted 
          return direction (sign) matches the actual return direction.

    Args:
        actual (pd.Series): Historical target values (e.g., actual weekly returns)
            indexed by Date.
        predicted (pd.Series): Model forecast values (e.g., predicted weekly returns)
            indexed by Date.

    Returns:
        dict: A dictionary containing the summary performance evaluation metrics:
            {
                "observations": int,
                "MAE": float,
                "RMSE": float,
                "directional_accuracy": float
            }
    """

    valid = pd.concat( #side concate the df and then remove row with na
        [actual.rename("actual"), predicted.rename("predicted")],
        axis=1,
    ).dropna()

    error = valid["predicted"] - valid["actual"]

    return {
        "observations": len(valid),
        "MAE": error.abs().mean(),
        "RMSE": np.sqrt((error ** 2).mean()),
        "directional_accuracy": (
            np.sign(valid["predicted"])
            == np.sign(valid["actual"])
        ).mean(),
    }