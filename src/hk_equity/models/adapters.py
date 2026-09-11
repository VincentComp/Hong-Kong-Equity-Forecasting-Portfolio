"""Shared callable interfaces for live forecasts and backtests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd


LiveForecastFunction = Callable[
    [
        pd.DataFrame,
        dict[str, Any],
        pd.Series | None,
    ],
    pd.Series,
]


BacktestForecastFunction = Callable[
    [
        pd.DataFrame,
        dict[str, Any],
        pd.Series | None,
    ],
    pd.DataFrame,
]


@dataclass(frozen=True)
class ModelAdapter:
    """Connect one model's live and backtest functions."""

    live_forecast: LiveForecastFunction
    backtest_forecast: BacktestForecastFunction