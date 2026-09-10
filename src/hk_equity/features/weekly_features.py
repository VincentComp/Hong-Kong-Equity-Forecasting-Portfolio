# ==============================================================================
# ML FEATURE SCHEMA (44 FEATURES)
# ==============================================================================

# 1. Historical Lagged Returns (Short/Medium/Long-Term Momentum & Reversion)
# - return_lag_1 to 52: Individual stock weekly returns N weeks ago.
# - Used to capture momentum, short-term mean reversion, and yearly seasonality.
# 'return_lag_1', 'return_lag_2', 'return_lag_3', 'return_lag_4',
# 'return_lag_8', 'return_lag_12', 'return_lag_24', 'return_lag_52',

# 2. Rolling Trend & Risk Metrics (4-Week / 12-Week / 26-Week Windows)
# - rolling_mean_N: Moving average of weekly returns (trend indicator).
# - rolling_std_N: Total return volatility (total risk).
# - downside_deviation_N: Volatility of negative returns only (downside risk).
# - risk_adjusted_mean_N: Sharpe-like ratio (rolling_mean / rolling_std).
# - momentum_spread_4_12: Short vs medium-term momentum spread (MACD concept).
# 'rolling_mean_4', 'rolling_std_4', 'downside_deviation_4', 'risk_adjusted_mean_4',
# 'rolling_mean_12', 'rolling_std_12', 'downside_deviation_12', 'risk_adjusted_mean_12',
# 'rolling_mean_26', 'rolling_std_26', 'downside_deviation_26', 'risk_adjusted_mean_26',
# 'momentum_spread_4_12',

# 3. Market Benchmark Factors (Systematic Market Environment)
# - market_return_lag_1: Overall market (e.g., HSI) return from previous week.
# - market_rolling_mean_N: Market regime indicator over 4, 12, and 26-week horizons.
# 'market_return_lag_1', 'market_rolling_mean_4', 'market_rolling_mean_12', 'market_rolling_mean_26',

# 4. Stock Excess Returns (Pure Stock Alpha over Benchmark)
# - excess_return_lag_N: Stock return minus market return N weeks ago.
# - excess_rolling_mean_N: Rolling average of stock benchmark-outperformance.
# 'excess_return_lag_1', 'excess_return_lag_2', 'excess_return_lag_3', 'excess_return_lag_4',
# 'excess_return_lag_8', 'excess_return_lag_12', 'excess_return_lag_24', 'excess_return_lag_52',
# 'excess_rolling_mean_4', 'excess_rolling_mean_12', 'excess_rolling_mean_26',

# 5. Cross-Sectional Z-Scores (Peer Relative Ranking per Target Week)
# - *_cs_z: Normalized metrics relative to all stocks in the universe for that specific week.
# - Eliminates overall market direction bias and focuses purely on relative stock selection.
# 'return_lag_1_cs_z', 'return_lag_4_cs_z', 'return_lag_12_cs_z',
# 'rolling_mean_4_cs_z', 'rolling_mean_12_cs_z', 'rolling_mean_26_cs_z',
# 'risk_adjusted_mean_4_cs_z', 'excess_return_lag_1_cs_z'

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class WeeklyFeatureSpec:
    """Configuration for leakage-safe weekly return features.

    The defaults are intentionally conservative for the first improved
    feature set. HSI returns can be supplied through ``benchmark_returns``.
    """

    #lag and rolling (in week)
    lag_weeks: tuple[int, ...] = (1, 2, 3, 4, 8, 12, 24, 52)
    rolling_windows: tuple[int, ...] = (4, 12, 26)
    
    #Feature switches
    include_rolling_std: bool = True               
    include_downside_deviation: bool = True         #calculate the only the downside volatiliy 
    include_market_features: bool = True            #include the HSI
    include_excess_return_features: bool = True     #stock return - HSI return
    include_ratio_features: bool = False            #return per volatility
    include_cross_sectional_features: bool = True   #standardization
    
    cross_sectional_columns: tuple[str, ...] = (#Select subset of feature for standardization
        "return_lag_1",
        "return_lag_4",
        "return_lag_12",
        "rolling_mean_4",
        "rolling_mean_12",
        "rolling_mean_26",
        "risk_adjusted_mean_4",
        "excess_return_lag_1",
    )

    #winsorization (default turning off)
    winsorize_features: bool = False
    winsorize_lower: float = 0.05
    winsorize_upper: float = 0.95


PORTFOLIO_COLUMNS = {"target_week", "feature_week", "ticker", "target"}


def _validate_weekly_returns(
    weekly_returns: pd.DataFrame,
) -> pd.DataFrame:
    """Validate and return weekly returns in a sorted numeric format."""

    if not isinstance(weekly_returns.index, pd.DatetimeIndex): #check index col type correct
        raise TypeError(
            "weekly_returns must have a pandas DatetimeIndex."
        )

    if weekly_returns.empty: #check non-empty dataframe
        raise ValueError(
            "weekly_returns must contain at least one row."
        )

    if weekly_returns.columns.empty: # check at least one ticker column exists
        raise ValueError(
            "weekly_returns must contain at least one ticker column."
        )

    if weekly_returns.columns.has_duplicates: #sort and type casting
        raise ValueError(
            "weekly_returns must not contain duplicate ticker columns."
        )

    cleaned = weekly_returns.sort_index().astype(float)

    if cleaned.index.has_duplicates: #check if duplicate dates exists
        raise ValueError(
            "weekly_returns must not contain duplicate dates."
        )

    return cleaned


def _validate_benchmark_returns(
    benchmark_returns: pd.Series | None,
    weekly_index: pd.DatetimeIndex,
) -> pd.Series:
    """Align benchmark returns to the portfolio weekly index."""

    #Defensive Type Checking 1
    if benchmark_returns is None: 
        raise ValueError(
            "benchmark_returns is required for the improved feature set."
        )

    if not isinstance(benchmark_returns, pd.Series):
        raise TypeError(
            "benchmark_returns must be a pandas Series."
        )

    if not isinstance(benchmark_returns.index, pd.DatetimeIndex):
        raise TypeError(
            "benchmark_returns must have a pandas DatetimeIndex."
        )

    #Reindexing & Alignment
    aligned = benchmark_returns.sort_index().reindex(weekly_index)
    aligned = aligned.astype(float)
    aligned.name = benchmark_returns.name or "benchmark_return"

    return aligned


def _safe_cross_sectional_zscore(#standardized the row
    values: pd.Series,
) -> pd.Series:
    """Calculate a cross-sectional z-score for one target week."""

    valid = values.dropna()

    if len(valid) < 2:
        return pd.Series(float("nan"), index=values.index)

    standard_deviation = valid.std(ddof=1)

    if pd.isna(standard_deviation) or standard_deviation == 0:
        return pd.Series(0.0, index=values.index)

    return (values - valid.mean()) / standard_deviation


def _winsorize_cross_section( #windsor before cross-section
    feature_table: pd.DataFrame,
    feature_columns: list[str],
    lower: float,
    upper: float,
) -> pd.DataFrame:
    """Clip feature values within each target-week cross-section."""

    if not 0 <= lower < upper <= 1:
        raise ValueError(
            "Winsorization bounds must satisfy 0 <= lower < upper <= 1."
        )

    result = feature_table.copy()

    for column in feature_columns:
        result[column] = result.groupby("target_week")[column].transform(
            lambda values: values.clip(
                lower=values.quantile(lower),
                upper=values.quantile(upper),
            )
        )

    return result


def _add_cross_sectional_zscores(#batch standardization
    feature_table: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """Add date-by-date cross-sectional z-score columns."""

    result = feature_table.copy()

    available_columns = [
        column
        for column in columns
        if column in result.columns
    ]

    for column in available_columns:
        result[f"{column}_cs_z"] = (
            result.groupby("target_week")[column]
            .transform(_safe_cross_sectional_zscore)
        )

    return result

#===============================================================

def _add_stock_features(#process 1 stock, all week
    stock_returns: pd.Series,
    market_returns: pd.Series,
    ticker: str,
    spec: WeeklyFeatureSpec,
) -> pd.DataFrame:
    """Build historical features for one ticker."""

    features = pd.DataFrame(index=stock_returns.index)
    historical_returns = stock_returns.shift(1)
    historical_excess_returns = (
        stock_returns - market_returns
    ).shift(1)

    features["ticker"] = ticker

    for lag in spec.lag_weeks: #lag
        features[f"return_lag_{lag}"] = stock_returns.shift(lag)

    for window in spec.rolling_windows: #rolling history
        rolling_mean = historical_returns.rolling(
            window=window,
            min_periods=window,
        ).mean()

        rolling_std = historical_returns.rolling(
            window=window,
            min_periods=window,
        ).std()

        features[f"rolling_mean_{window}"] = rolling_mean

        if spec.include_rolling_std:
            features[f"rolling_std_{window}"] = rolling_std

        if spec.include_downside_deviation: #downside std
            downside_squared = historical_returns.clip(upper=0).pow(2)
            features[f"downside_deviation_{window}"] = (
                downside_squared
                .rolling(window=window, min_periods=window)
                .mean()
                .pow(0.5)
            )

        features[f"risk_adjusted_mean_{window}"] = (#risk adjusted rolling mean
            rolling_mean
            / rolling_std.replace(0, pd.NA)
        )

    if spec.include_ratio_features: # short/long mean ratio momentum (% change)
        short_mean = features["rolling_mean_4"]
        long_mean = features["rolling_mean_12"]
        features["momentum_ratio_4_12"] = (
            short_mean
            / long_mean.replace(0, pd.NA)
            - 1
        )

    if 4 in spec.rolling_windows and 12 in spec.rolling_windows: #momentum_spread_4_12
        features["momentum_spread_4_12"] = (
            features["rolling_mean_4"]
            - features["rolling_mean_12"]
        )

    if spec.include_market_features: #HSI
        features["market_return_lag_1"] = (
            market_returns.shift(1)
        )

        for window in spec.rolling_windows:
            features[f"market_rolling_mean_{window}"] = (
                market_returns
                .shift(1)
                .rolling(window=window, min_periods=window)
                .mean()
            )

    if spec.include_excess_return_features: #excess_return_features
        for lag in spec.lag_weeks:
            features[f"excess_return_lag_{lag}"] = (
                historical_excess_returns.shift(lag - 1)
            )

        for window in spec.rolling_windows:
            features[f"excess_rolling_mean_{window}"] = (
                historical_excess_returns
                .rolling(window=window, min_periods=window)
                .mean()
            )

    features["target"] = stock_returns
    features.index.name = "target_week"

    return features.reset_index()


def build_weekly_training_features(#process all stocks, all weeks
    weekly_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    spec: WeeklyFeatureSpec | None = None,
) -> pd.DataFrame:
    """Build leakage-safe long-format supervised-learning rows.

    ``target_week`` is the week whose realised return is the target. Every
    feature uses only information available before that target week.
    ``benchmark_returns`` should normally be HSI weekly returns.
    """

    spec = spec or WeeklyFeatureSpec()
    weekly_returns = _validate_weekly_returns(weekly_returns)
    benchmark_returns = _validate_benchmark_returns(
        benchmark_returns=benchmark_returns,
        weekly_index=weekly_returns.index,
    )

    market_returns = benchmark_returns

    ticker_frames = [
        _add_stock_features(
            stock_returns=weekly_returns[ticker],
            market_returns=market_returns,
            ticker=ticker,
            spec=spec,
        )
        for ticker in weekly_returns.columns
    ]

    feature_table = pd.concat(
        ticker_frames,
        ignore_index=True,
    )

    feature_columns = get_feature_columns(feature_table)

    if spec.winsorize_features:
        feature_table = _winsorize_cross_section(
            feature_table=feature_table,
            feature_columns=feature_columns,
            lower=spec.winsorize_lower,
            upper=spec.winsorize_upper,
        )

    if spec.include_cross_sectional_features:
        feature_table = _add_cross_sectional_zscores(
            feature_table=feature_table,
            columns=list(spec.cross_sectional_columns),
        )

    return feature_table

#================================================================

def get_feature_columns(#get only the feature column
    feature_table: pd.DataFrame,
) -> list[str]:
    """Return model input columns from a feature table."""

    excluded_columns = PORTFOLIO_COLUMNS

    return [
        column
        for column in feature_table.columns
        if column not in excluded_columns
    ]


def drop_incomplete_feature_rows(#clean the missing dataset
    feature_table: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Remove rows with missing feature values or missing target."""

    if feature_columns is None:
        feature_columns = get_feature_columns(feature_table)

    required_columns = [*feature_columns, "target"]

    return (
        feature_table
        .dropna(subset=required_columns)
        .reset_index(drop=True)
    )

#==============================

def _build_live_stock_row(#Process 1 stock for 1 week
    stock_returns: pd.Series,
    market_returns: pd.Series,
    ticker: str,
    latest_week: pd.Timestamp,
    spec: WeeklyFeatureSpec,
) -> dict[str, object]:
    """Build one current feature row for one ticker."""

    row: dict[str, object] = {
        "ticker": ticker,
        "feature_week": latest_week,
    }

    for lag in spec.lag_weeks:
        if len(stock_returns) < lag:
            row[f"return_lag_{lag}"] = float("nan")
        else:
            row[f"return_lag_{lag}"] = stock_returns.iloc[-lag]

    for window in spec.rolling_windows:
        recent_returns = stock_returns.tail(window)
        rolling_mean = recent_returns.mean()
        rolling_std = recent_returns.std()

        if len(recent_returns) < window:
            rolling_mean = float("nan")
            rolling_std = float("nan")

        row[f"rolling_mean_{window}"] = rolling_mean

        if spec.include_rolling_std:
            row[f"rolling_std_{window}"] = rolling_std

        if spec.include_downside_deviation:
            downside_deviation = (
                recent_returns.clip(upper=0)
                .pow(2)
                .mean()
                ** 0.5
            )
            row[f"downside_deviation_{window}"] = (
                downside_deviation
                if len(recent_returns) >= window
                else float("nan")
            )

        row[f"risk_adjusted_mean_{window}"] = (
            rolling_mean / rolling_std
            if pd.notna(rolling_mean)
            and pd.notna(rolling_std)
            and rolling_std != 0
            else float("nan")
        )

    if spec.include_ratio_features:
        mean_4 = row.get("rolling_mean_4", float("nan"))
        mean_12 = row.get("rolling_mean_12", float("nan"))
        row["momentum_ratio_4_12"] = (
            mean_4 / mean_12 - 1
            if pd.notna(mean_4)
            and pd.notna(mean_12)
            and mean_12 != 0
            else float("nan")
        )

    if 4 in spec.rolling_windows and 12 in spec.rolling_windows:
        row["momentum_spread_4_12"] = (
            row["rolling_mean_4"]
            - row["rolling_mean_12"]
        )

    if spec.include_market_features:
        row["market_return_lag_1"] = market_returns.iloc[-1]

        for window in spec.rolling_windows:
            recent_market_returns = market_returns.tail(window)
            row[f"market_rolling_mean_{window}"] = (
                recent_market_returns.mean()
                if len(recent_market_returns) >= window
                else float("nan")
            )

    if spec.include_excess_return_features:
        excess_returns = stock_returns - market_returns

        for lag in spec.lag_weeks:
            row[f"excess_return_lag_{lag}"] = (
                excess_returns.iloc[-lag]
                if len(excess_returns) >= lag
                else float("nan")
            )

        for window in spec.rolling_windows:
            recent_excess_returns = excess_returns.tail(window)
            row[f"excess_rolling_mean_{window}"] = (
                recent_excess_returns.mean()
                if len(recent_excess_returns) >= window
                else float("nan")
            )

    return row


def build_live_features(#Process multiple ticker, 1 Week
    weekly_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    spec: WeeklyFeatureSpec | None = None,
) -> pd.DataFrame:
    """Build one current feature row per ticker for next-week forecasting.

    The latest completed weekly return is known at live forecast time, so it
    is used as lag 1. The returned columns match the training feature schema.
    Cross-sectional z-scores are calculated across the current stock universe.
    """

    spec = spec or WeeklyFeatureSpec()
    weekly_returns = _validate_weekly_returns(weekly_returns)
    benchmark_returns = _validate_benchmark_returns(
        benchmark_returns=benchmark_returns,
        weekly_index=weekly_returns.index,
    )

    latest_week = weekly_returns.index.max()
    market_returns = benchmark_returns

    rows = [
        _build_live_stock_row(
            stock_returns=weekly_returns[ticker],
            market_returns=market_returns,
            ticker=ticker,
            latest_week=latest_week,
            spec=spec,
        )
        for ticker in weekly_returns.columns
    ]

    feature_table = pd.DataFrame(rows)

    feature_table["target_week"] = latest_week

    feature_columns = get_feature_columns(feature_table)

    if spec.winsorize_features:
        feature_table = _winsorize_cross_section(
            feature_table=feature_table,
            feature_columns=feature_columns,
            lower=spec.winsorize_lower,
            upper=spec.winsorize_upper,
        )

    if spec.include_cross_sectional_features:
        feature_table = _add_cross_sectional_zscores(
            feature_table=feature_table,
            columns=list(spec.cross_sectional_columns),
        )

    return feature_table.drop(columns=["target_week"])