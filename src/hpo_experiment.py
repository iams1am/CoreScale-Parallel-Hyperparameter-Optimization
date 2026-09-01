from pathlib import Path
import argparse
import json
import time

import numpy as np
import pandas as pd

from joblib import parallel_config
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    GridSearchCV,
    ParameterGrid,
    StratifiedKFold,
)
from threadpoolctl import (
    threadpool_info,
    threadpool_limits,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "covertype.csv"
)

WORKLOAD_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "workload_indices.npz"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "raw"
)

TARGET_COLUMN = "Cover_Type"

RANDOM_STATE = 42
CV_FOLDS = 5


PARAM_GRID = {
    "n_estimators": [50, 100, 200],
    "max_depth": [10, None],
    "min_samples_leaf": [1, 4],
}


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--workers",
        type=int,
        required=True,
        choices=[1, 2, 4, 8],
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    workers = args.workers

    print("Loading dataset...")

    df = pd.read_csv(DATA_FILE)

    workloads = np.load(
        WORKLOAD_FILE
    )

    train_indices = workloads[
        "train_25000"
    ]

    X = df.drop(
        columns=[TARGET_COLUMN]
    )

    y = df[TARGET_COLUMN]

    X_train = X.iloc[
        train_indices
    ]

    y_train = y.iloc[
        train_indices
    ]

    candidate_count = len(
        list(ParameterGrid(PARAM_GRID))
    )

    total_cv_fits = (
        candidate_count
        * CV_FOLDS
    )

    print("\n=== EXPERIMENT ===")
    print("Workload: W25")

    print(
        "Training samples:",
        f"{len(X_train):,}"
    )

    print(
        "HPO workers:",
        workers
    )

    print(
        "Configurations:",
        candidate_count
    )

    print(
        "CV folds:",
        CV_FOLDS
    )

    print(
        "Total CV fits:",
        total_cv_fits
    )

    print(
        "Estimator n_jobs: 1"
    )

    print(
        "Inner thread limit: 1"
    )

    model = RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=1,
    )

    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    search = GridSearchCV(
        estimator=model,
        param_grid=PARAM_GRID,
        scoring="f1_macro",
        cv=cv,
        n_jobs=workers,
        pre_dispatch="n_jobs",
        refit=True,
        return_train_score=False,
        error_score="raise",
        verbose=1,
    )

    print(
        "\nParent thread pools before control:"
    )

    for pool in threadpool_info():
        print(
            pool.get("internal_api"),
            "| threads =",
            pool.get("num_threads"),
        )

    print(
        "\nStarting HPO..."
    )

    with threadpool_limits(limits=1):

        with parallel_config(
            backend="loky",
            inner_max_num_threads=1,
        ):

            start = time.perf_counter()

            search.fit(
                X_train,
                y_train,
            )

            total_runtime = (
                time.perf_counter()
                - start
            )

    refit_time = float(
        search.refit_time_
    )

    search_runtime = (
        total_runtime
        - refit_time
    )

    throughput = (
        total_cv_fits
        / search_runtime
    )

    print(
        "\n=== HPO RESULT ==="
    )

    print(
        f"Workers: {workers}"
    )

    print(
        f"Total runtime: "
        f"{total_runtime:.4f} s"
    )

    print(
        f"Refit time: "
        f"{refit_time:.4f} s"
    )

    print(
        f"Approx. search runtime: "
        f"{search_runtime:.4f} s"
    )

    print(
        f"CV-fit throughput: "
        f"{throughput:.4f} fits/s"
    )

    print(
        f"Best CV Macro F1: "
        f"{search.best_score_:.6f}"
    )

    print(
        "Best parameters:"
    )

    for key, value in (
        search.best_params_.items()
    ):
        print(
            f"  {key}: {value}"
        )

    output = {
        "workers": workers,
        "workload": "W25",
        "training_samples": int(
            len(X_train)
        ),
        "candidate_count": (
            candidate_count
        ),
        "cv_folds": CV_FOLDS,
        "total_cv_fits": (
            total_cv_fits
        ),
        "estimator_n_jobs": 1,
        "inner_max_num_threads": 1,
        "total_runtime_seconds": (
            total_runtime
        ),
        "refit_time_seconds": (
            refit_time
        ),
        "search_runtime_seconds": (
            search_runtime
        ),
        "throughput_fits_per_second": (
            throughput
        ),
        "best_cv_macro_f1": float(
            search.best_score_
        ),
        "best_parameters": (
            search.best_params_
        ),
        "random_state": (
            RANDOM_STATE
        ),
    }

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        OUTPUT_DIR
        / f"hpo_smoke_workers_{workers}.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output,
            file,
            indent=4,
        )

    print("\nSaved:")
    print(output_file)


if __name__ == "__main__":
    main()