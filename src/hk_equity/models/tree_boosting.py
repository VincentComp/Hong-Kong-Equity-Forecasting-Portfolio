"""Tree-boosting models (XGBoost, LightGBM) for weekly return forecasting."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.hk_equity.features.feature_config import (
    build_weekly_feature_spec,
    load_feature_set_config,
)
from src.hk_equity.features.weekly_features import (
    build_live_features,
)


def _get_estimator_name(
    parameters: dict[str, Any],
) -> str:
    """Return and normalize the estimator name."""

    estimator_name = str(
        parameters.get(
            "estimator",
            "xgboost",
        )
    ).lower()

    if estimator_name not in {
        "xgboost",
        "xgb",
        "lightgbm",
        "lgbm",
        "lgb",
    }:
        raise ValueError(
            "Tree-boosting estimator must be one of: "
            "xgboost, lightgbm."
        )

    return estimator_name


def _get_selected_feature_columns(
    parameters: dict[str, Any],
) -> list[str]:
    """Return the explicitly configured model input columns."""

    selected_columns = parameters.get(
        "selected_columns",
    )

    if not selected_columns:
        raise ValueError(
            "Tree-boosting model config must contain "
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
            "Tree-boosting model config must contain "
            "features.feature_map_path."
        )

    feature_map_path = Path(
        str(feature_map_path)
    )

    if not feature_map_path.exists():
        raise FileNotFoundError(
            "Cannot find processed feature map: "
            f"{feature_map_path}"
        )

    return feature_map_path


def load_feature_map(
    parameters: dict[str, Any],
) -> pd.DataFrame:
    """Load and validate a processed regression feature map."""

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
            "Tree-boosting model produced missing predictions for: "
            f"{missing_tickers}"
        )

    return predictions.astype(float)


def _get_training_tickers(
    parameters: dict[str, Any],
    portfolio_tickers: list[str],
) -> list[str]:
    """Return validated tree-boosting training tickers from model YAML."""

    configured_tickers = parameters.get(
        "tickers",
        "all",
    )

    if configured_tickers == "all":
        return portfolio_tickers.copy()

    if not isinstance(configured_tickers, list):
        raise TypeError(
            "training.tickers must be 'all' "
            "or a YAML list."
        )

    training_tickers = [
        str(ticker)
        for ticker in configured_tickers
    ]

    if not training_tickers:
        raise ValueError(
            "training.tickers cannot be empty."
        )

    if len(training_tickers) != len(
        set(training_tickers)
    ):
        raise ValueError(
            "training.tickers contains duplicates."
        )

    invalid_tickers = [
        ticker
        for ticker in training_tickers
        if ticker not in portfolio_tickers
    ]

    if invalid_tickers:
        raise ValueError(
            "training.tickers must contain portfolio "
            "tickers only. Invalid tickers: "
            f"{invalid_tickers}"
        )

    return training_tickers


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


def _build_xgboost_estimator(
    parameters: dict[str, Any],
):
    """Build an XGBoost regressor from YAML parameters."""

    try:
        from xgboost import XGBRegressor
    except ImportError:
        raise ImportError(
            "XGBoost is not installed. "
            "Please install xgboost to use estimator='xgboost'."
        )

    n_estimators = int(
        parameters.get(
            "n_estimators",
            100,
        )
    )

    max_depth = int(
        parameters.get(
            "max_depth",
            3,
        )
    )

    learning_rate = float(
        parameters.get(
            "learning_rate",
            0.05,
        )
    )

    subsample = float(
        parameters.get(
            "subsample",
            0.8,
        )
    )

    colsample_bytree = float(
        parameters.get(
            "colsample_bytree",
            0.8,
        )
    )

    reg_alpha = float(
        parameters.get(
            "reg_alpha",
            0.0,
        )
    )

    reg_lambda = float(
        parameters.get(
            "reg_lambda",
            1.0,
        )
    )

    min_child_weight = float(
        parameters.get(
            "min_child_weight",
            1.0,
        )
    )

    random_state = int(
        parameters.get(
            "random_state",
            42,
        )
    )

    n_jobs = int(
        parameters.get(
            "n_jobs",
            -1,
        )
    )

    return XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        min_child_weight=min_child_weight,
        random_state=random_state,
        n_jobs=n_jobs,
        verbosity=0,
    )


def _build_lightgbm_estimator(
    parameters: dict[str, Any],
):
    """Build a LightGBM regressor from YAML parameters."""

    try:
        from lightgbm import LGBMRegressor
    except ImportError:
        raise ImportError(
            "LightGBM is not installed. "
            "Please install lightgbm to use estimator='lightgbm'."
        )

    n_estimators = int(
        parameters.get(
            "n_estimators",
            100,
        )
    )

    learning_rate = float(
        parameters.get(
            "learning_rate",
            0.05,
        )
    )

    num_leaves = int(
        parameters.get(
            "num_leaves",
            15,
        )
    )

    max_depth = int(
        parameters.get(
            "max_depth",
            -1,
        )
    )

    min_child_samples = int(
        parameters.get(
            "min_child_samples",
            20,
        )
    )

    subsample = float(
        parameters.get(
            "subsample",
            0.8,
        )
    )

    colsample_bytree = float(
        parameters.get(
            "colsample_bytree",
            0.8,
        )
    )

    reg_alpha = float(
        parameters.get(
            "reg_alpha",
            0.0,
        )
    )

    reg_lambda = float(
        parameters.get(
            "reg_lambda",
            1.0,
        )
    )

    random_state = int(
        parameters.get(
            "random_state",
            42,
        )
    )

    n_jobs = int(
        parameters.get(
            "n_jobs",
            -1,
        )
    )

    verbosity = int(
        parameters.get(
            "verbosity",
            -1,
        )
    )

    return LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        max_depth=max_depth,
        min_child_samples=min_child_samples,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        random_state=random_state,
        n_jobs=n_jobs,
        verbosity=verbosity,
    )


def _build_tree_estimator(
    parameters: dict[str, Any],
):
    """Build a tree-boosting estimator from YAML parameters."""

    estimator_name = _get_estimator_name(
        parameters=parameters,
    )

    if estimator_name in {
        "xgboost",
        "xgb",
    }:
        return _build_xgboost_estimator(
            parameters=parameters,
        )

    elif estimator_name in {
        "lightgbm",
        "lgbm",
        "lgb",
    }:
        return _build_lightgbm_estimator(
            parameters=parameters,
        )

    else:
        raise ValueError(
            f"Unknown tree-boosting estimator: '{estimator_name}'."
        )


def _get_tree_parameters(
    model_config: dict[str, Any],
) -> dict[str, Any]:
    """Merge tree-boosting parameters and feature settings from model YAML."""

    parameters = model_config.get(
        "parameters",
        {},
    )

    feature_settings = model_config.get(
        "features",
        {},
    )

    training_settings = model_config.get(
        "training",
        {},
    )

    return {
        **parameters,
        **feature_settings,
        **training_settings,
    }


def tree_boosting_live_forecast(
    weekly_returns: pd.DataFrame,
    model_config: dict[str, Any],
    benchmark_returns: pd.Series | None = None,
) -> pd.Series:
    """Adapt tree-boosting live forecasting to the common model interface."""

    if benchmark_returns is None:
        raise ValueError(
            "benchmark_returns is required for tree-boosting forecasting."
        )

    parameters = _get_tree_parameters(
        model_config=model_config,
    )

    selected_columns = _get_selected_feature_columns(
        parameters=parameters,
    )

    feature_map = load_feature_map(
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
            "Live tree-boosting features contain missing values in: "
            f"{missing_live_value_columns}"
        )

    portfolio_tickers = list(
        weekly_returns.columns
    )

    training_tickers = _get_training_tickers(
        parameters=parameters,
        portfolio_tickers=portfolio_tickers,
    )

    training_feature_map = feature_map.loc[
        feature_map["ticker"].isin(
            training_tickers
        )
    ].copy()

    if training_feature_map.empty:
        raise ValueError(
            "No feature-map rows were found for "
            f"training.tickers: {training_tickers}"
        )

    _validate_feature_columns(
        feature_table=training_feature_map,
        selected_columns=selected_columns,
    )

    model = _build_tree_estimator(
        parameters=parameters,
    )

    X_train = training_feature_map[
        selected_columns
    ]

    y_train = training_feature_map[
        "target"
    ]

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
        tickers=portfolio_tickers,
    )


def tree_boosting_backtest_forecast(
    weekly_returns: pd.DataFrame,
    model_config: dict[str, Any],
    benchmark_returns: pd.Series | None = None,
) -> pd.DataFrame:
    """Adapt tree-boosting backtesting to the common model interface."""

    parameters = _get_tree_parameters(
        model_config=model_config,
    )

    selected_columns = _get_selected_feature_columns(
        parameters=parameters,
    )

    feature_map = load_feature_map(
        parameters=parameters,
    )

    _validate_feature_columns(
        feature_table=feature_map,
        selected_columns=selected_columns,
    )

    minimum_training_weeks = int(
        parameters.get(
            "minimum_training_weeks",
            104,
        )
    )

    if minimum_training_weeks <= 0:
        raise ValueError(
            "minimum_training_weeks must be greater than zero."
        )

    portfolio_tickers = list(
        weekly_returns.columns
    )

    training_tickers = _get_training_tickers(
        parameters=parameters,
        portfolio_tickers=portfolio_tickers,
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

        training_data = train_data.loc[
            train_data["ticker"].isin(
                training_tickers
            )
        ].copy()

        available_training_weeks = training_data[
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

        model = _build_tree_estimator(
            parameters=parameters,
        )

        X_train = training_data[
            selected_columns
        ]

        y_train = training_data["target"]

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