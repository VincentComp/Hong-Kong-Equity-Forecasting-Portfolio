import numpy as np
import pandas as pd


def forecast_metrics(
    actual: pd.Series,
    predicted: pd.Series,
) -> dict:
    valid = pd.concat(
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