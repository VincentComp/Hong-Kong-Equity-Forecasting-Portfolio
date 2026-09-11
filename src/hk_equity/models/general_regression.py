"""Regression forecasting models using a processed point-in-time feature map."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from src.hk_equity.features.feature_config import (
    build_weekly_feature_spec,
    load_feature_set_config,
)
from src.hk_equity.features.weekly_features import (
    build_live_features,
)


def build_regression_estimator(
    parameters: dict[str, Any],
) -> Pipeline:
    """Create a configurable scikit-learn regression pipeline."""

    estimator_name = parameters.get(
        "estimator",
        "ridge",
    ).lower()

    alpha = float(
        parameters.get("alpha", 1.0)
    )

    if alpha <= 0:
        raise ValueError(
            "alpha must be greater than zero."
        )

    polynomial_degree = int(
        parameters.get("polynomial_degree", 1)
    )

    if polynomial_degree < 1:
        raise ValueError(
            "polynomial_degree must be at least 1."
        )

    fit_intercept = bool(
        parameters.get("fit_intercept", True)
    )

    max_iter = int(
        parameters.get("max_iter", 10000)
    )

    if estimator_name == "ridge":
        estimator = Ridge(
            alpha=alpha,
            fit_intercept=fit_intercept,
        )

    elif estimator_name == "lasso":
        estimator = Lasso(
            alpha=alpha,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
        )

    elif estimator_name in {
        "elastic_net",
        "elasticnet",
    }:
        l1_ratio = float(
            parameters.get("l1_ratio", 0.5)
        )

        if not 0 <= l1_ratio <= 1:
            raise ValueError(
                "l1_ratio must be between 0 and 1."
            )

        estimator = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
        )

    else:
        raise ValueError(
            f"Unknown regression estimator '{estimator_name}'. "
            "Available estimators: ridge, lasso, elastic_net"
        )

    steps = []

    if polynomial_degree > 1:
        steps.append(
            (
                "polynomial_features",
                PolynomialFeatures(
                    degree=polynomial_degree,
                    include_bias=False,
                ),
            )
        )

    if parameters.get("standardize", True):
        steps.append(
            (
                "standard_scaler",
                StandardScaler(),
            )
        )

    steps.append(
        (
            "regressor",
            estimator,
        )
    )

    return Pipeline(steps)


def _get_feature_set_name(
    parameters: dict[str, Any],
) -> str:
    """Return the configured feature-set name."""

    return str(
        parameters.get(
            "feature_set",
            "weekly_v1",
        )
    )


def _get_selected_feature_columns(
    parameters: dict[str, Any],
) -> list[str]:
    """Return the explicitly configured model input columns."""

    selected_columns = parameters.get(
        "selected_columns",
    )

    if not selected_columns:
        raise ValueError(
            "Regression model config must contain "
            "features.selected_columns."
        )

    if not isinstance(selected_columns, list):
        raise TypeError(
            "features.selected_columns must be a YAML list."
        )

    selected_columns = [
        str(column)
        for column in selected_columns
    ]

    if len(selected_columns) != len(
        set(selected_columns)
    ):
        raise ValueError(
            "features.selected_columns contains duplicates."
        )

    return selected_columns


def _get_feature_map_path(
    parameters: dict[str, Any],
) -> Path:
    """Return the configured processed feature-map file path."""

    feature_map_path = parameters.get(
        "feature_map_path",
    )

    if not feature_map_path:
        raise ValueError(
            "Regression model config must contain "
            "features.feature_map_path."
        )

    feature_map_path = Path(
        str(feature_map_path)
    )

    if not feature_map_path.exists():
        raise FileNotFoundError(
            "Cannot find processed regression feature map: "
            f"{feature_map_path}"
        )

    return feature_map_path


def load_regression_feature_map(
    parameters: dict[str, Any],
) -> pd.DataFrame:
    """Load and validate a processed regression feature map.

    The feature map is saved before model fitting, but each row must already
    be point-in-time safe: its feature values may only use information from
    before its target_week.
    """

    feature_map_path = _get_feature_map_path(
        parameters=parameters,
    )

    if feature_map_path.suffix.lower() == ".csv":
        feature_map = pd.read_csv(
            feature_map_path,
            parse_dates=["target_week"],
        )

    elif feature_map_path.suffix.lower() == ".parquet":
        feature_map = pd.read_parquet(
            feature_map_path,
        )

    else:
        raise ValueError(
            "Unsupported feature-map format. "
            "Use .csv or .parquet."
        )

    required_columns = {
        "target_week",
        "ticker",
        "target",
    }

    missing_required_columns = (
        required_columns
        - set(feature_map.columns)
    )

    if missing_required_columns:
        raise ValueError(
            "Feature map is missing required columns: "
            f"{sorted(missing_required_columns)}"
        )

    feature_map["target_week"] = pd.to_datetime(
        feature_map["target_week"],
    )

    feature_map["ticker"] = feature_map[
        "ticker"
    ].astype(str)

    feature_map = feature_map.sort_values(
        by=[
            "target_week",
            "ticker",
        ]
    ).reset_index(drop=True)

    duplicated_rows = feature_map.duplicated(
        subset=[
            "target_week",
            "ticker",
        ]
    )

    if duplicated_rows.any():
        duplicate_examples = feature_map.loc[
            duplicated_rows,
            [
                "target_week",
                "ticker",
            ],
        ].head().to_dict(
            orient="records"
        )

        raise ValueError(
            "Feature map contains duplicate "
            "(target_week, ticker) rows. Examples: "
            f"{duplicate_examples}"
        )

    return feature_map


def _validate_feature_columns(
    feature_table: pd.DataFrame,
    selected_columns: list[str],
) -> None:
    """Validate configured model features against a feature table."""

    missing_columns = [
        column
        for column in selected_columns
        if column not in feature_table.columns
    ]

    if missing_columns:
        raise ValueError(
            "Configured selected_columns are missing "
            "from the feature map: "
            f"{missing_columns}"
        )

    missing_values = feature_table[
        selected_columns
    ].isna().any()

    missing_value_columns = missing_values[
        missing_values
    ].index.tolist()

    if missing_value_columns:
        raise ValueError(
            "Feature map contains missing values in "
            "selected_columns: "
            f"{missing_value_columns}"
        )

    if feature_table["target"].isna().any():
        raise ValueError(
            "Feature map contains missing target values."
        )


def _validate_predictions(
    predictions: pd.Series,
    tickers: list[str],
) -> pd.Series:
    """Validate and order one prediction per portfolio ticker."""

    predictions = predictions.reindex(tickers)

    if predictions.isna().any():
        missing_tickers = predictions[
            predictions.isna()
        ].index.tolist()

        raise ValueError(
            "Regression model produced missing predictions for: "
            f"{missing_tickers}"
        )

    return predictions.astype(float)


def _load_feature_spec(
    parameters: dict[str, Any],
):
    """Load the YAML feature set used for live feature creation."""

    feature_set_name = _get_feature_set_name(
        parameters=parameters,
    )

    feature_config = load_feature_set_config(
        feature_set_name=feature_set_name,
    )

    return build_weekly_feature_spec(
        feature_config=feature_config,
    )


def get_regression_forecast(
    weekly_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    parameters: dict[str, Any],
) -> pd.Series:
    """Fit on the processed feature map and forecast next-week returns.

    Historical training rows come from the stored point-in-time feature map.
    Live rows are generated separately from the latest completed weekly data.
    The same selected_columns list is used for both training and prediction.
    """

    if benchmark_returns is None:
        raise ValueError(
            "benchmark_returns is required for regression forecasting."
        )

    selected_columns = _get_selected_feature_columns(
        parameters=parameters,
    )

    feature_map = load_regression_feature_map(
        parameters=parameters,
    )

    _validate_feature_columns(
        feature_table=feature_map,
        selected_columns=selected_columns,
    )

    feature_spec = _load_feature_spec(
        parameters=parameters,
    )

    live_features = build_live_features(
        weekly_returns=weekly_returns,
        benchmark_returns=benchmark_returns,
        spec=feature_spec,
    )

    required_live_columns = [
        "ticker",
        *selected_columns,
    ]

    missing_live_columns = [
        column
        for column in required_live_columns
        if column not in live_features.columns
    ]

    if missing_live_columns:
        raise ValueError(
            "Live feature table is missing configured columns: "
            f"{missing_live_columns}"
        )

    live_features = live_features[
        required_live_columns
    ].copy()

    missing_live_values = live_features[
        selected_columns
    ].isna().any()

    missing_live_value_columns = missing_live_values[
        missing_live_values
    ].index.tolist()

    if missing_live_value_columns:
        raise ValueError(
            "Live regression features contain missing values in: "
            f"{missing_live_value_columns}"
        )

    model = build_regression_estimator(
        parameters=parameters,
    )

    X_train = feature_map[selected_columns]
    y_train = feature_map["target"]
    X_live = live_features[selected_columns]

    model.fit(
        X_train,
        y_train,
    )

    raw_predictions = model.predict(
        X_live,
    )

    predictions = pd.Series(
        raw_predictions,
        index=live_features["ticker"],
        name="predicted_weekly_return",
    )

    return _validate_predictions(
        predictions=predictions,
        tickers=list(weekly_returns.columns),
    )


def get_regression_backtest_forecasts(
    weekly_returns: pd.DataFrame,
    parameters: dict[str, Any],
) -> pd.DataFrame:
    """Generate expanding-window regression backtest forecasts.

    For each target_week t:

        train_data = feature_map[target_week < t]
        test_data = feature_map[target_week == t]

        model.fit(train_data[selected_columns], train_data["target"])
        model.predict(test_data[selected_columns])

    The Pipeline fits StandardScaler using training rows only. Therefore,
    scaling statistics from the target week and future weeks never enter
    an earlier prediction.
    """

    selected_columns = _get_selected_feature_columns(
        parameters=parameters,
    )

    feature_map = load_regression_feature_map(
        parameters=parameters,
    )

    _validate_feature_columns(
        feature_table=feature_map,
        selected_columns=selected_columns,
    )

    minimum_training_weeks = int(
        parameters.get(
            "minimum_training_weeks",
            60,
        )
    )

    if minimum_training_weeks <= 0:
        raise ValueError(
            "minimum_training_weeks must be greater than zero."
        )

    portfolio_tickers = list(
        weekly_returns.columns
    )

    predictions = pd.DataFrame(
        index=weekly_returns.index,
        columns=portfolio_tickers,
        dtype=float,
    )

    for target_week in weekly_returns.index:
        target_week = pd.Timestamp(
            target_week
        )

        test_data = feature_map.loc[
            feature_map["target_week"] == target_week
        ].copy()

        if test_data.empty:
            continue

        train_data = feature_map.loc[
            feature_map["target_week"] < target_week
        ].copy()

        available_training_weeks = train_data[
            "target_week"
        ].nunique()

        if available_training_weeks < minimum_training_weeks:
            continue

        missing_test_tickers = [
            ticker
            for ticker in portfolio_tickers
            if ticker not in set(
                test_data["ticker"]
            )
        ]

        if missing_test_tickers:
            raise ValueError(
                "Feature map is missing target-week rows for: "
                f"{missing_test_tickers}; "
                f"target_week={target_week.date()}"
            )

        model = build_regression_estimator(
            parameters=parameters,
        )

        X_train = train_data[
            selected_columns
        ]

        y_train = train_data["target"]

        X_test = (
            test_data
            .set_index("ticker")
            .reindex(portfolio_tickers)[
                selected_columns
            ]
        )

        model.fit(
            X_train,
            y_train,
        )

        raw_predictions = model.predict(
            X_test,
        )

        weekly_predictions = pd.Series(
            raw_predictions,
            index=portfolio_tickers,
            name="predicted_weekly_return",
        )

        predictions.loc[
            target_week,
            portfolio_tickers,
        ] = _validate_predictions(
            predictions=weekly_predictions,
            tickers=portfolio_tickers,
        )

    return predictions