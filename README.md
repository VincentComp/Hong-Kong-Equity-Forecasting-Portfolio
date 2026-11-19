# Optuna-Optimized Multi-Channel Machine Learning for Weekly Return Forecasting of Hong Kong Equities

> A configuration-driven forecasting system that combines stock-specific and pooled models to predict next Friday-ending weekly returns for ten Hong Kong equities.

![Language](https://img.shields.io/badge/Language-Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Optimisation](https://img.shields.io/badge/Optimisation-Optuna-3A78C2?style=flat-square)
![Model](https://img.shields.io/badge/Model-XGBoost-EA6A47?style=flat-square)
![Task](https://img.shields.io/badge/Task-Weekly_Return_Forecasting-6A5ACD?style=flat-square)

## Overview

Project 1 develops ten stock-specific forecasting channels and one global channel for the same equity universe. Each local channel produces a forecast for its designated stock. The global channel fits a single model to pooled stock–week observations and generates a separate forecast for each stock. The final output averages the local and global estimates with equal weights for each ticker.

The system uses adjusted stock closing prices and Hang Seng Index (HSI) closing levels. `^HSI` supplies market and benchmark-relative predictors but is not a training peer or prediction target. It is excluded from stock-level results, Rank IC and Top-3 diagnostics, and submission files.

## Key Results

| Method | Historical test MAE (pp) | Historical test DA (%) | Deployment MAE (pp) |
|---|---:|---:|---:|
| Local–global ensemble | **3.5215** | **53.03** | 2.9099 |
| Zero return | 3.5465 | N/A | 2.8161 |
| MA4 | 3.9349 | 49.57 | 2.8111 |
| EWMA (span 4) | 4.0361 | 49.86 | **2.7032** |

The ensemble reduced historical test MAE by about 0.7% relative to zero return. This advantage did not persist in deployment. The ensemble outperformed both moving-average baselines in two of five weeks but did not outperform zero return in any week. Deployment MAE pools 50 stock–week forecasts. Directional accuracy (DA) excludes pairs with zero realised or predicted returns.

## Methodology

```text
Yahoo Finance → daily adjusted stock closes and HSI closing levels
                             ↓
            Friday-ending weekly returns (W-FRI)
                             ↓
              Weekly feature map (44 predictors)
                             ↓
       Optuna configuration selection by validation MAE
                   ↙                     ↘
      Ten stock-specific channels      One pooled global channel
                   ↘                     ↙
              Align forecasts by portfolio ticker
                             ↓
              Equal-weight local–global average
                             ↓
                Ten next-week return forecasts
```

### Data and features

- **Universe:** `0005.HK`, `0016.HK`, `0066.HK`, `0700.HK`, `0981.HK`, `1113.HK`, `1398.HK`, `1810.HK`, `3988.HK`, and `9988.HK`. The authoritative ticker list is in [`configs/base.yaml`](configs/base.yaml).
- **Weekly target:** Percentage change in adjusted closing price between consecutive Friday-ending weeks, stored in decimal form. Use the last available trading-day price within each week.
- **Predictors:** [`weekly_v1`](configs/features/weekly_v1.yaml) contains 44 candidate inputs spanning lagged returns, rolling return and risk measures, momentum, lagged HSI returns, stock returns relative to the HSI, and cross-sectional z-scores across the ten stocks. A target week's predictors use information through the preceding completed week.

### Model selection and deployment

| Channel | Fitting sample | Validation criterion |
|---|---|---|
| Local | Target stock and any selected training peers | MAE for the target stock |
| Global | Pooled observations from all ten stocks | MAE pooled across stocks and validation weeks |

The candidate model pool includes Ridge, Lasso, Elastic Net and XGBoost. The selected report configurations use XGBoost in both streams. Weekly deployment keeps the selected configurations fixed and refits model parameters on an expanding history of completed weeks.

## Experiments

| Stage | Period | Purpose |
|---|---|---|
| Training | 2020–2024 | Fit candidate models |
| Validation | 2025 | Select configurations using MAE |
| Historical test | January–August 2026 | Compare the selected ensemble with zero return, MA4 and EWMA |
| Deployment | Weeks ending 25 September to 23 October 2026 | Evaluate five weekly forecast rounds |

The evaluation pipeline supports MAE, RMSE, directional accuracy, weekly Spearman Rank IC, and the realised equal-weight return of the three highest-ranked stocks relative to equal weighting across the stock universe. The Top-3 comparison is a ranking diagnostic, not a portfolio allocation backtest.

## Findings

- Performance varied substantially by stock. Deployment MAE ranged from 1.1889 pp for CK Asset to 6.5134 pp for SMIC.
- Deployment forecasts were more concentrated around zero than realised returns. Predicting the correct sign did not necessarily imply an accurate estimate of the return magnitude.
- The reported ensemble metrics do not isolate the contribution of forecast averaging. A matched comparison with local-only and global-only forecasts is needed to assess that contribution.

## Quick Start

Run the commands from the repository root. Create the environment from [`environment.yml`](environment.yml), then activate the environment specified by its `name:` field.

```bash
conda env create -f environment.yml
```

### 1. Prepare data and features

For report reproduction, use the archived price snapshot from the reported run. For a fresh run, refresh the data with the pipeline below and check the price convention before building features. The CSV filename alone does not establish that the stock prices are adjusted. Use the same price basis for features, realised targets and baselines.

```bash
python -m scripts.run_data_pipeline
python -m scripts.build_feature_map --feature-set weekly_v1
```

If using an archived snapshot, skip the download step and build the feature map from that snapshot. The standard input and feature-map paths are `data/raw/latest_daily_close.csv` and `data/processed/weekly_features_weekly_v1.csv`.

### 2. Run a historical backtest

The example uses the v1 ensemble YAML supplied in the repository tree. Use the configuration and data snapshot recorded for the reported run to reproduce the tables above.

```bash
ENSEMBLE=configs/models/ensembles/global/xgboost_group_global_v1_ensemble.yaml
python -m scripts.run_backtest --model-config "$ENSEMBLE" \
  --start 2026-01-01 --end 2026-08-31
```

To run a baseline over the same dates, substitute one of the YAML files in [`configs/models/universal/`](configs/models/universal/). Backtests export predictions, overall and stock-level metrics, weekly Rank IC and Top-3 diagnostics to the backtest path configured in `configs/base.yaml`.

### 3. Generate a weekly forecast

```bash
python -m scripts.run_weekly_forecast --model-config "$ENSEMBLE" \
  --as-of 2026-09-18
```

This example uses 18 September as the cutoff for the week ending 25 September. For a new forecast, update the input data and use the appropriate cutoff. Check that the exported forecast has exactly ten unique portfolio tickers and the expected target Friday.

## Repository Structure

```text
.
├── environment.yml
├── configs/
│   ├── base.yaml               # Universe, benchmark, dates and paths
│   ├── features/               # Weekly predictor definitions
│   ├── models/                 # Baselines, local/global models, routers and ensembles
│   └── tuning/                 # Optuna study configurations
├── data/                       # Raw, interim and processed inputs
├── models/                     # Model artefacts directory
├── notebooks/                  # Data exploration and evaluation
├── outputs/                    # Run output directory
├── reports/tables/             # Validation summaries in CSV and XLSX
├── scripts/                    # Data, tuning, backtest and forecast entry points
│   └── TempHelperScript/       # Batch tuning and configuration export helpers
└── src/hk_equity/
    ├── data/                   # Acquisition, preprocessing and forecast cutoffs
    ├── features/               # Weekly feature construction
    ├── models/                 # Estimators, routing and ensemble inference
    ├── tuning/                 # Optuna objectives and search routines
    ├── evaluation/             # Metrics and forecast tables
    └── utils/                  # Configuration and I/O
```

## Limitations and Future Work

- Compare local-only, global-only and equal-weight ensemble forecasts over identical stocks and dates.
- Test sector returns, volatility measures and interest-rate indicators for incremental value beyond HSI-based predictors.
- Compare the unweighted expanding fitting window against time-decay weighting and rolling windows under the same forecast cutoffs.
- Use nested walk-forward evaluation to assess configuration stability beyond the single validation year and five-week deployment period.

## Project Context

This repository documents Project 1, the weekly equity return forecasting component. Portfolio construction and allocation are outside the scope of this README.

## Author

Wai-Shing Ng  
School of Computing and Data Science, The University of Hong Kong

## Academic Disclaimer

This repository is intended for educational and research purposes. Forecasts and historical evaluations are not investment advice and do not establish future investment performance.
