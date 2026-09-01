from pathlib import Path
import argparse
import json
import random
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SINGLE_RUN_SCRIPT = (
    PROJECT_ROOT
    / "src"
    / "final_hpo_single.py"
)

RESULT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "raw"
    / "final_benchmark"
)

WORKERS = [
    1,
    2,
    4,
    8,
]

RANDOMIZATION_SEED = 4492026

COOLDOWN_SECONDS = 10


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--repetitions",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--label",
        type=str,
        default="dryrun",
    )

    return parser.parse_args()


def build_schedule(
    repetitions,
):
    rng = random.Random(
        RANDOMIZATION_SEED
    )

    schedule = []

    position = 1

    for repetition in range(
        1,
        repetitions + 1,
    ):
        block = WORKERS.copy()

        rng.shuffle(block)

        for workers in block:
            schedule.append(
                {
                    "run_position":
                        position,

                    "repetition":
                        repetition,

                    "workers":
                        workers,
                }
            )

            position += 1

    return schedule


def confidence_interval_95(
    values,
):
    values = np.asarray(
        values,
        dtype=float,
    )

    n = len(values)

    mean = float(
        np.mean(values)
    )

    if n < 2:
        return (
            mean,
            np.nan,
            np.nan,
            np.nan,
        )

    std = float(
        np.std(
            values,
            ddof=1,
        )
    )

    sem = (
        std
        / np.sqrt(n)
    )

    critical = stats.t.ppf(
        0.975,
        df=n - 1,
    )

    margin = (
        critical
        * sem
    )

    return (
        mean,
        std,
        mean - margin,
        mean + margin,
    )


def create_summary(
    all_runs,
):
    rows = []

    baseline = all_runs[
        all_runs["workers"] == 1
    ][
        "search_runtime_seconds"
    ].to_numpy()

    baseline_mean = float(
        np.mean(baseline)
    )

    for workers in WORKERS:

        subset = all_runs[
            all_runs["workers"]
            == workers
        ]

        runtimes = subset[
            "search_runtime_seconds"
        ].to_numpy()

        (
            mean_runtime,
            std_runtime,
            ci_low,
            ci_high,
        ) = confidence_interval_95(
            runtimes
        )

        speedup = (
            baseline_mean
            / mean_runtime
        )

        efficiency = (
            speedup
            / workers
        )

        rows.append(
            {
                "workers":
                    workers,

                "runs":
                    len(subset),

                "mean_search_runtime_seconds":
                    mean_runtime,

                "std_search_runtime_seconds":
                    std_runtime,

                "ci95_low_seconds":
                    ci_low,

                "ci95_high_seconds":
                    ci_high,

                "speedup":
                    speedup,

                "efficiency":
                    efficiency,

                "efficiency_percent":
                    efficiency * 100,

                "mean_throughput_fits_per_second":
                    subset[
                        "throughput_fits_per_second"
                    ].mean(),

                "mean_cpu_normalized_percent":
                    subset[
                        "mean_process_tree_cpu_normalized_percent"
                    ].mean(),

                "mean_peak_rss_mb":
                    subset[
                        "peak_process_tree_rss_mb"
                    ].mean(),
            }
        )

    return pd.DataFrame(
        rows
    )


def main():
    args = parse_arguments()

    experiment_dir = (
        RESULT_ROOT
        / args.label
    )

    run_dir = (
        experiment_dir
        / "runs"
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    schedule = build_schedule(
        args.repetitions
    )

    schedule_df = pd.DataFrame(
        schedule
    )

    schedule_file = (
        experiment_dir
        / "run_schedule.csv"
    )

    schedule_df.to_csv(
        schedule_file,
        index=False,
    )

    print(
        "\n=== BENCHMARK SCHEDULE ==="
    )

    print(
        schedule_df.to_string(
            index=False
        )
    )

    all_results = []

    for item in schedule:

        position = item[
            "run_position"
        ]

        repetition = item[
            "repetition"
        ]

        workers = item[
            "workers"
        ]

        print("\n" + "#" * 72)

        print(
            f"RUN {position} | "
            f"REP {repetition} | "
            f"WORKERS {workers}"
        )

        print("#" * 72)

        command = [
            sys.executable,
            str(
                SINGLE_RUN_SCRIPT
            ),
            "--workers",
            str(workers),
            "--repetition",
            str(repetition),
            "--run-position",
            str(position),
            "--output-dir",
            str(run_dir),
        ]

        subprocess.run(
            command,
            check=True,
        )

        stem = (
            f"run_{position:02d}"
            f"_rep_{repetition}"
            f"_workers_{workers}"
        )

        result_file = (
            run_dir
            / f"{stem}.json"
        )

        with open(
            result_file,
            "r",
            encoding="utf-8",
        ) as file:
            result = json.load(
                file
            )

        result[
            "best_parameters_json"
        ] = json.dumps(
            result[
                "best_parameters"
            ],
            sort_keys=True,
        )

        all_results.append(
            result
        )

        all_runs_df = pd.DataFrame(
            all_results
        )

        all_runs_df.to_csv(
            experiment_dir
            / "all_runs.csv",
            index=False,
        )

        print(
            f"\nCompleted "
            f"{position}/"
            f"{len(schedule)} runs."
        )

        if position < len(
            schedule
        ):
            print(
                f"Cooling down for "
                f"{COOLDOWN_SECONDS} seconds..."
            )

            time.sleep(
                COOLDOWN_SECONDS
            )

    all_runs_df = pd.DataFrame(
        all_results
    )

    summary_df = create_summary(
        all_runs_df
    )

    summary_file = (
        experiment_dir
        / "summary_by_workers.csv"
    )

    summary_df.to_csv(
        summary_file,
        index=False,
    )

    parameter_variants = (
        all_runs_df[
            "best_parameters_json"
        ].nunique()
    )

    score_range = (
        all_runs_df[
            "best_cv_macro_f1"
        ].max()
        -
        all_runs_df[
            "best_cv_macro_f1"
        ].min()
    )

    print(
        "\n=== FINAL SUMMARY ==="
    )

    print(
        summary_df.to_string(
            index=False
        )
    )

    print(
        "\n=== CORRECTNESS CHECK ==="
    )

    print(
        "Unique best-parameter sets:",
        parameter_variants,
    )

    print(
        "Best-score range:",
        f"{score_range:.12f}",
    )

    print(
        "\nSaved:"
    )

    print(
        experiment_dir
        / "all_runs.csv"
    )

    print(
        summary_file
    )

    print(
        schedule_file
    )


if __name__ == "__main__":
    main()