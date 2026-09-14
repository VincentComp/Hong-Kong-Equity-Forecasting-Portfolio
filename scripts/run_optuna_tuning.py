"""Run one XGBoost Optuna tuning study from a YAML configuration file."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import optuna
from tqdm.auto import tqdm

from src.hk_equity.tuning.objectives import (
    get_objective_spec,
)
from src.hk_equity.tuning.xgboost_optuna import (
    evaluate_xgboost_trial,
    load_xgboost_tuning_config,
    prepare_tuning_weekly_returns,
)
from src.hk_equity.utils.config import (
    load_yaml,
)

import json
from typing import Any

import yaml


def dump_yaml(
    path: Path,
    data: Any,
) -> None:
    """Write a Python object to a YAML file."""

    with open(path, "w") as f:
        yaml.dump(
            data,
            f,
            default_flow_style=False,
            allow_unicode=True,
        )

def parse_arguments() -> argparse.Namespace:
    """Read command-line settings for one Optuna tuning run."""

    parser = argparse.ArgumentParser(
        description=(
            "Run an XGBoost Optuna tuning study for one "
            "base-stock objective."
        )
    )

    parser.add_argument(
        "--base-config",
        default="configs/base.yaml",
        help="Path to shared project configuration YAML.",
    )

    parser.add_argument(
        "--tuning-config",
        required=True,
        help="Path to one Optuna tuning YAML configuration.",
    )

    parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help=(
            "Override study.n_trials in tuning YAML. "
            "Use a small number for smoke tests."
        ),
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help=(
            "Maximum tuning duration in seconds. "
            "Default: no time limit."
        ),
    )

    parser.add_argument(
        "--storage",
        default=None,
        help=(
            "Optional Optuna SQLite database path. "
            "Default: output.storage_file in tuning YAML."
        ),
    )

    return parser.parse_args()


def build_progress_callback(
    progress_bar: tqdm,
):
    """Create an Optuna callback that updates terminal progress."""

    def callback(
        study: optuna.Study,
        trial: optuna.trial.FrozenTrial,
    ) -> None:
        """Update TQDM after each completed, pruned, or failed trial."""

        del trial

        completed_trials = len(
            study.trials
        )

        pruned_trials = sum(
            trial_result.state
            == optuna.trial.TrialState.PRUNED
            for trial_result in study.trials
        )

        complete_trials = [
            trial_result
            for trial_result in study.trials
            if trial_result.state
            == optuna.trial.TrialState.COMPLETE
        ]

        postfix = {
            "Complete": len(complete_trials),
            "Pruned": pruned_trials,
        }

        if complete_trials:
            postfix["Best"] = (
                f"{study.best_value:.6f}"
            )

        progress_bar.n = completed_trials
        progress_bar.set_postfix(
            postfix,
            refresh=True,
        )
        progress_bar.refresh()

    return callback


def print_study_summary(
    study: optuna.Study,
    elapsed_seconds: float,
    output_dir: Path,
    output_settings: dict,
    base_ticker: str,
    objective_spec,
    study_settings: dict,
    evaluation_start_date: str,
    evaluation_end_date: str,
    storage_url: str | None,
) -> None:
    """Print one readable summary after a tuning study and export artifacts."""

    completed_trials = [
        trial
        for trial in study.trials
        if trial.state
        == optuna.trial.TrialState.COMPLETE
    ]

    pruned_trials = [
        trial
        for trial in study.trials
        if trial.state
        == optuna.trial.TrialState.PRUNED
    ]

    failed_trials = [
        trial
        for trial in study.trials
        if trial.state
        == optuna.trial.TrialState.FAIL
    ]

    print("\n" + "=" * 70)
    print("OPTUNA TUNING COMPLETED")
    print("=" * 70)
    print(f"Study name: {study.study_name}")
    print(f"Total trials: {len(study.trials)}")
    print(f"Completed trials: {len(completed_trials)}")
    print(f"Pruned trials: {len(pruned_trials)}")
    print(f"Failed trials: {len(failed_trials)}")
    print(f"Elapsed seconds: {elapsed_seconds:.2f}")

    if not completed_trials:
        print(
            "\nNo completed trial was produced. "
            "Review the trial constraints and errors."
        )
        return

    best_trial = study.best_trial

    print("\nBest objective value:")
    print(f"{study.best_value:.6f}")

    print("\nBest trial number:")
    print(best_trial.number)

    print("\nBest training tickers:")
    print(
        best_trial.user_attrs[
            "training_tickers"
        ]
    )

    print("\nBest selected feature columns:")
    print(
        best_trial.user_attrs[
            "selected_columns"
        ]
    )

    print("\nBest XGBoost hyperparameters:")
    for parameter_name, parameter_value in (
        best_trial.params.items()
    ):
        if (
            parameter_name.startswith("use_peer_")
            or parameter_name.startswith("use_feature_")
        ):
            continue

        print(
            f"{parameter_name}: "
            f"{parameter_value}"
        )

    # Export artifacts
    trials_csv_name = output_settings.get(
        "trials_csv",
    )

    trials_csv_path = None

    if trials_csv_name:
        trials_csv_path = output_dir / trials_csv_name

        study.trials_dataframe().to_csv(
            trials_csv_path,
            index=False,
        )

        print(
            "Trials CSV: "
            f"{trials_csv_path}"
        )

    best_trial_json_name = output_settings.get(
        "best_trial_json",
    )

    best_trial_json_path = None

    if best_trial_json_name:
        best_trial_json_path = output_dir / best_trial_json_name

        best_trial_dict = {
            "number": best_trial.number,
            "value": best_trial.value,
            "params": best_trial.params,
        }

        with open(
            best_trial_json_path,
            "w",
        ) as f:
            json.dump(
                best_trial_dict,
                f,
                indent=2,
            )

        print(
            "Best trial JSON: "
            f"{best_trial_json_path}"
        )

    best_model_yaml_name = output_settings.get(
        "best_model_config_yaml",
    )

    best_model_yaml_path = None

    if best_model_yaml_name:
        best_model_yaml_path = output_dir / best_model_yaml_name

        best_training_tickers = best_trial.user_attrs[
            "training_tickers"
        ]

        best_selected_features = best_trial.user_attrs[
            "selected_columns"
        ]

        best_xgb_params = {
            key: value
            for key, value in best_trial.params.items()
            if not (
                key.startswith("use_peer_")
                or key.startswith("use_feature_")
            )
        }

        best_model_config = {
            "base_ticker": base_ticker,
            "objective": objective_spec.name,
            "training_tickers": best_training_tickers,
            "selected_feature_columns": best_selected_features,
            "xgboost_params": best_xgb_params,
        }

        dump_yaml(
            best_model_yaml_path,
            best_model_config,
        )

        print(
            "Best model config YAML: "
            f"{best_model_yaml_path}"
        )

    tuning_metadata_name = output_settings.get(
        "tuning_metadata_json",
    )

    tuning_metadata_path = None

    if tuning_metadata_name:
        tuning_metadata_path = output_dir / tuning_metadata_name

        tuning_metadata = {
            "study_name": study_settings["name"],
            "base_ticker": base_ticker,
            "objective": objective_spec.name,
            "evaluation_start_date": evaluation_start_date,
            "evaluation_end_date": evaluation_end_date,
            "storage_file": storage_url,
            "trials_csv": str(trials_csv_path) if trials_csv_path else None,
            "best_trial_json": str(best_trial_json_path) if best_trial_json_path else None,
            "best_model_config_yaml": str(best_model_yaml_path) if best_model_yaml_path else None,
            "total_trials": len(study.trials),
            "completed_trials": len(completed_trials),
            "pruned_trials": len(pruned_trials),
            "failed_trials": len(failed_trials),
            "best_trial_number": best_trial.number,
            "best_value": best_trial.value,
        }

        with open(
            tuning_metadata_path,
            "w",
        ) as f:
            json.dump(
                tuning_metadata,
                f,
                indent=2,
            )

        print(
            "Tuning metadata JSON: "
            f"{tuning_metadata_path}"
        )


def main() -> None:
    """Run one configured Optuna tuning study."""

    args = parse_arguments()

    optuna.logging.set_verbosity(
        optuna.logging.WARNING
    )

    base_config = load_yaml(
        args.base_config,
    )

    portfolio_tickers = list(
        base_config["tickers"].keys()
    )

    tuning_config = load_xgboost_tuning_config(
        config_path=args.tuning_config,
        portfolio_tickers=portfolio_tickers,
    )

    study_settings = tuning_config[
        "study"
    ]

    objective_spec = get_objective_spec(
        objective_name=study_settings[
            "objective"
        ],
    )

    configured_trials = int(
        study_settings["n_trials"]
    )

    n_trials = (
        args.n_trials
        if args.n_trials is not None
        else configured_trials
    )

    if n_trials <= 0:
        raise ValueError(
            "--n-trials must be greater than zero."
        )

    sampler_seed = int(
        study_settings.get(
            "sampler_seed",
            42,
        )
    )

    sampler = optuna.samplers.TPESampler(
        seed=sampler_seed,
    )

    output_settings = tuning_config["output"]

    output_dir = Path(
        output_settings["output_directory"]
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    configured_storage_file = output_settings.get(
        "storage_file",
    )

    storage_file = (
        args.storage
        if args.storage is not None
        else configured_storage_file
    )

    storage_url = None

    if storage_file:
        storage_path = Path(
            storage_file
        )

        storage_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        storage_url = (
            f"sqlite:///{storage_path}"
        )

    study = optuna.create_study(
        study_name=study_settings["name"],
        direction=objective_spec.direction,
        sampler=sampler,
        storage=storage_url,
        load_if_exists=bool(
            output_settings.get(
                "load_if_exists",
                False,
            )
        ),
    )

    weekly_returns, benchmark_returns = (
        prepare_tuning_weekly_returns(
            base_config=base_config,
        )
    )

    def objective(
        trial: optuna.Trial,
    ) -> float:
        """Evaluate one trial using the configured base-stock objective."""

        return evaluate_xgboost_trial(
            trial=trial,
            tuning_config=tuning_config,
            weekly_returns=weekly_returns,
            benchmark_returns=benchmark_returns,
        )

    print("\n" + "=" * 70)
    print("OPTUNA XGBOOST TUNING")
    print("=" * 70)
    print(f"Study: {study.study_name}")
    print(
        "Base ticker: "
        f"{tuning_config['target']['base_ticker']}"
    )
    print(
        "Objective: "
        f"{objective_spec.name} "
        f"({objective_spec.direction})"
    )
    print(f"Trials: {n_trials}")
    print(
        "Storage: "
        f"{storage_url or 'in-memory'}"
    )
    print(
        "Evaluation period: "
        f"{tuning_config['evaluation']['start']} to "
        f"{tuning_config['evaluation']['end']}"
    )

    start_time = time.perf_counter()

    with tqdm(
        total=n_trials,
        desc="Optuna trials",
        unit="trial",
    ) as progress_bar:
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=args.timeout,
            callbacks=[
                build_progress_callback(
                    progress_bar
                )
            ],
            catch=(
                ValueError,
            ),
        )

    elapsed_seconds = (
        time.perf_counter()
        - start_time
    )

    print_study_summary(
        study=study,
        elapsed_seconds=elapsed_seconds,
        output_dir=output_dir,
        output_settings=output_settings,
        base_ticker=tuning_config["target"]["base_ticker"],
        objective_spec=objective_spec,
        study_settings=study_settings,
        evaluation_start_date=tuning_config["evaluation"]["start"],
        evaluation_end_date=tuning_config["evaluation"]["end"],
        storage_url=storage_url,
    )


if __name__ == "__main__":
    main()