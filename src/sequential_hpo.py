from pathlib import Path
import json
import time

import numpy as np
import pandas as pd

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

CANDIDATE_OUTPUT = (
    PROJECT_ROOT
    / "results"
    / "raw"
    / "sequential_hpo_rf_w25_candidates.csv"
)

SUMMARY_OUTPUT = (
    PROJECT_ROOT
    / "results"
    / "raw"
    / "sequential_hpo_rf_w25_summary.json"
)

TARGET_COLUMN = "Cover_Type"
RANDOM_STATE = 42
CV_FOLDS = 5


PARAM_GRID = {
    "n_estimators": [50, 100, 200],
    "max_depth": [10, None],
    "min_samples_leaf": [1, 4],
}


def main():
    print("Loading dataset...")

    df = pd.read_csv(DATA_FILE)

    print("Loading workload indices...")

    workloads = np.load(WORKLOAD_FILE)

    train_indices = workloads["train_25000"]

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train = X.iloc[train_indices]
    y_train = y.iloc[train_indices]

    candidate_count = len(
        list(ParameterGrid(PARAM_GRID))
    )

    total_cv_fits = (
        candidate_count
        * CV_FOLDS
    )

    print("\n=== EXPERIMENT CONFIGURATION ===")
    print("Model: RandomForestClassifier")
    print("Workload: W25")
    print(
        "Training samples:",
        f"{len(X_train):,}"
    )
    print("Scoring: f1_macro")
    print(
        "CV folds:",
        CV_FOLDS
    )
    print(
        "Hyperparameter configurations:",
        candidate_count
    )
    print(
        "Total CV fits:",
        total_cv_fits
    )
    print("HPO workers: 1")
    print("Estimator n_jobs: 1")

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
        n_jobs=1,
        refit=True,
        return_train_score=False,
        error_score="raise",
        verbose=1,
    )

    print(
        "\n=== THREAD POOLS BEFORE CONTROL ==="
    )

    for pool in threadpool_info():
        print(
            pool.get("internal_api"),
            "| threads =",
            pool.get("num_threads"),
        )

    print("\nStarting sequential HPO...")

    with threadpool_limits(limits=1):

        print(
            "\n=== THREAD POOLS DURING CONTROL ==="
        )

        for pool in threadpool_info():
            print(
                pool.get("internal_api"),
                "| threads =",
                pool.get("num_threads"),
            )

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
        getattr(
            search,
            "refit_time_",
            0.0,
        )
    )

    search_runtime = (
        total_runtime
        - refit_time
    )

    throughput = (
        total_cv_fits
        / search_runtime
        if search_runtime > 0
        else 0.0
    )

    print("\n=== SEQUENTIAL HPO RESULT ===")

    print(
        f"Total runtime: "
        f"{total_runtime:.4f} s"
    )

    print(
        f"Refit time: "
        f"{refit_time:.4f} s"
    )

    print(
        f"Approx. CV search runtime: "
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

    print("Best parameters:")

    for key, value in search.best_params_.items():
        print(
            f"  {key}: {value}"
        )

    # Save candidate-level results.
    cv_results = pd.DataFrame(
        search.cv_results_
    )

    candidate_columns = [
        "rank_test_score",
        "mean_test_score",
        "std_test_score",
        "mean_fit_time",
        "std_fit_time",
        "mean_score_time",
        "param_n_estimators",
        "param_max_depth",
        "param_min_samples_leaf",
    ]

    candidate_results = (
        cv_results[candidate_columns]
        .sort_values(
            "rank_test_score"
        )
    )

    CANDIDATE_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    candidate_results.to_csv(
        CANDIDATE_OUTPUT,
        index=False,
    )

    summary = {
        "experiment": (
            "Sequential Random Forest HPO - W25"
        ),
        "workload": "W25",
        "training_samples": int(
            len(X_train)
        ),
        "model": (
            "RandomForestClassifier"
        ),
        "scoring": "f1_macro",
        "cv_folds": CV_FOLDS,
        "candidate_count": candidate_count,
        "total_cv_fits": total_cv_fits,
        "hpo_workers": 1,
        "estimator_n_jobs": 1,
        "internal_threads": 1,
        "total_runtime_seconds": (
            total_runtime
        ),
        "refit_time_seconds": (
            refit_time
        ),
        "approx_search_runtime_seconds": (
            search_runtime
        ),
        "cv_fit_throughput_per_second": (
            throughput
        ),
        "best_cv_macro_f1": float(
            search.best_score_
        ),
        "best_parameters": (
            search.best_params_
        ),
        "random_state": RANDOM_STATE,
    }

    with open(
        SUMMARY_OUTPUT,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
        )

    print("\n=== TOP 5 CONFIGURATIONS ===")

    print(
        candidate_results
        .head(5)
        .to_string(index=False)
    )

    print("\nSaved:")
    print(CANDIDATE_OUTPUT)
    print(SUMMARY_OUTPUT)


if __name__ == "__main__":
    main()