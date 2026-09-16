"""Run generated Optuna tuning configurations sequentially.

Examples:
    python -m scripts.run_all_optuna_tuning --dry-run

    python -m scripts.run_all_optuna_tuning \
        --n-trials 100 \
        --model-family xgboost \
        --version v1

    python -m scripts.run_all_optuna_tuning \
        --n-trials 100 \
        --model-family regression

    python -m scripts.run_all_optuna_tuning \
        --n-trials 100
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys
from typing import Any

import yaml


DEFAULT_CONFIG_DIRECTORY = Path(
    "configs/tuning/generated_stable"
)


def parse_arguments() -> argparse.Namespace:
    """Read command-line options for the Optuna batch runner."""

    parser = argparse.ArgumentParser(
        description=(
            "Run generated Optuna tuning configurations sequentially."
        ),
    )

    parser.add_argument(
        "--config-directory",
        default=str(DEFAULT_CONFIG_DIRECTORY),
        help=(
            "Directory containing generated Optuna YAML configs. "
            "Default: configs/tuning/generated_stable"
        ),
    )

    parser.add_argument(
        "--n-trials",
        type=int,
        default=100,
        help=(
            "Number of new Optuna trials to add per study. "
            "Default: 100"
        ),
    )

    parser.add_argument(
        "--model-family",
        choices=[
            "xgboost",
            "regression",
        ],
        default=None,
        help=(
            "Optional filter. Run only one model family."
        ),
    )

    parser.add_argument(
        "--version",
        choices=[
            "v1",
            "v2",
        ],
        default=None,
        help=(
            "Optional filter. Run only one tuning version."
        ),
    )

    parser.add_argument(
        "--ticker",
        default=None,
        help=(
            "Optional portfolio ticker filter, for example 0981.HK."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "List matching configs and commands without running Optuna."
        ),
    )

    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help=(
            "Continue to the next config after a failed study. "
            "By default, stop immediately on the first failure."
        ),
    )

    return parser.parse_args()


def load_yaml(
    yaml_path: Path,
) -> dict[str, Any]:
    """Load and validate one generated Optuna YAML config."""

    with yaml_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            f"YAML root must be a mapping: {yaml_path}"
        )

    required_sections = {
        "tuning",
        "study",
        "target",
        "evaluation",
        "output",
    }

    missing_sections = (
        required_sections - set(config.keys())
    )

    if missing_sections:
        raise KeyError(
            f"Config is missing sections {sorted(missing_sections)}: "
            f"{yaml_path}"
        )

    return config


def config_matches_filters(
    config: dict[str, Any],
    model_family: str | None,
    version: str | None,
    ticker: str | None,
) -> bool:
    """Return whether one config satisfies all optional filters."""

    configured_family = str(
        config["tuning"]["model_family"]
    ).lower()

    configured_ticker = str(
        config["target"]["base_ticker"]
    )

    configured_version = str(
        config.get("metadata", {}).get(
            "tuning_version",
            "",
        )
    ).lower()

    if model_family is not None:
        if configured_family != model_family:
            return False

    if ticker is not None:
        if configured_ticker != ticker:
            return False

    if version is not None:
        if configured_version != version:
            return False

    return True


def discover_configs(
    config_directory: Path,
    model_family: str | None,
    version: str | None,
    ticker: str | None,
) -> list[tuple[Path, dict[str, Any]]]:
    """Find generated YAML configs matching the chosen filters."""

    if not config_directory.exists():
        raise FileNotFoundError(
            "Cannot find config directory: "
            f"{config_directory}"
        )

    yaml_paths = sorted(
        list(config_directory.glob("*.yaml"))
        + list(config_directory.glob("*.yml"))
    )

    if not yaml_paths:
        raise FileNotFoundError(
            "No YAML files found in: "
            f"{config_directory}"
        )

    matches = []

    for yaml_path in yaml_paths:
        config = load_yaml(
            yaml_path=yaml_path,
        )

        if config_matches_filters(
            config=config,
            model_family=model_family,
            version=version,
            ticker=ticker,
        ):
            matches.append(
                (yaml_path, config)
            )

    if not matches:
        raise ValueError(
            "No generated configs matched the selected filters."
        )

    return matches


def build_command(
    config_path: Path,
    n_trials: int,
) -> list[str]:
    """Build the command used to run one Optuna study."""

    return [
        sys.executable,
        "-m",
        "scripts.run_optuna_tuning",
        "--tuning-config",
        str(config_path),
        "--n-trials",
        str(n_trials),
    ]


def format_study_summary(
    config_path: Path,
    config: dict[str, Any],
) -> str:
    """Return one readable study description."""

    study_name = config["study"]["name"]
    model_family = config["tuning"]["model_family"]
    base_ticker = config["target"]["base_ticker"]
    start = config["evaluation"]["start"]
    end = config["evaluation"]["end"]
    run_id = config["output"]["run_id"]

    return (
        f"study={study_name} | "
        f"family={model_family} | "
        f"ticker={base_ticker} | "
        f"evaluation={start} to {end} | "
        f"run_id={run_id} | "
        f"config={config_path}"
    )


def run_one_study(
    config_path: Path,
    n_trials: int,
) -> None:
    """Run one Optuna study through the existing project runner."""

    command = build_command(
        config_path=config_path,
        n_trials=n_trials,
    )

    subprocess.run(
        command,
        check=True,
    )


def main() -> None:
    """Run all matching Optuna configurations one at a time."""

    args = parse_arguments()

    if args.n_trials <= 0:
        raise ValueError(
            "--n-trials must be greater than zero."
        )

    config_directory = Path(
        args.config_directory
    )

    studies = discover_configs(
        config_directory=config_directory,
        model_family=args.model_family,
        version=args.version,
        ticker=args.ticker,
    )

    total_studies = len(studies)
    total_new_trials = (
        total_studies * args.n_trials
    )

    print("\n" + "=" * 70)
    print("OPTUNA BATCH RUNNER")
    print("=" * 70)
    print(f"Config directory: {config_directory}")
    print(f"Matching studies: {total_studies}")
    print(f"New trials per study: {args.n_trials}")
    print(f"Total requested new trials: {total_new_trials}")
    print(f"Dry run: {args.dry_run}")
    print("=" * 70)

    print("\nSelected studies:")

    for position, (config_path, config) in enumerate(
        studies,
        start=1,
    ):
        print(
            f"{position:02d}. "
            f"{format_study_summary(config_path, config)}"
        )

    if args.dry_run:
        print("\nCommands that would run:")

        for config_path, _ in studies:
            command = build_command(
                config_path=config_path,
                n_trials=args.n_trials,
            )

            print(" ".join(command))

        print("\nDry run completed. No studies were executed.")
        return

    started_at = datetime.now()

    completed = []
    failed = []

    for position, (config_path, config) in enumerate(
        studies,
        start=1,
    ):
        print("\n" + "=" * 70)
        print(
            f"RUNNING STUDY {position}/{total_studies}"
        )
        print("=" * 70)
        print(
            format_study_summary(
                config_path=config_path,
                config=config,
            )
        )

        command = build_command(
            config_path=config_path,
            n_trials=args.n_trials,
        )

        print("\nCommand:")
        print(" ".join(command))
        print()

        try:
            run_one_study(
                config_path=config_path,
                n_trials=args.n_trials,
            )

            completed.append(
                config["study"]["name"]
            )

        except subprocess.CalledProcessError as error:
            failed.append({
                "study_name": config["study"]["name"],
                "config_path": str(config_path),
                "return_code": error.returncode,
            })

            print("\nStudy failed:")
            print(
                f"study={config['study']['name']} | "
                f"return_code={error.returncode}"
            )

            if not args.continue_on_error:
                print(
                    "\nStopping because --continue-on-error "
                    "was not provided."
                )
                break

    elapsed_seconds = (
        datetime.now() - started_at
    ).total_seconds()

    print("\n" + "=" * 70)
    print("OPTUNA BATCH RUNNER COMPLETED")
    print("=" * 70)
    print(
        f"Completed studies: {len(completed)}"
    )
    print(f"Failed studies: {len(failed)}")
    print(
        f"Elapsed seconds: {elapsed_seconds:.2f}"
    )

    if completed:
        print("\nCompleted study names:")

        for study_name in completed:
            print(f"- {study_name}")

    if failed:
        print("\nFailed studies:")

        for failure in failed:
            print(
                f"- {failure['study_name']} | "
                f"{failure['config_path']} | "
                f"return_code={failure['return_code']}"
            )

        raise SystemExit(1)


if __name__ == "__main__":
    main()