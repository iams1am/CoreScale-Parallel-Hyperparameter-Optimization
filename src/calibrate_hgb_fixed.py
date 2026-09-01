from pathlib import Path
import time

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits


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

OUTPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "raw"
    / "hgb_fixed_iteration_calibration.csv"
)

TARGET_COLUMN = "Cover_Type"
RANDOM_STATE = 42

WORKLOADS = [
    ("W25", "train_25000"),
    ("W100", "train_100000"),
    ("W250", "train_250000"),
    ("WFULL", "train_full"),
]


def main():
    print("Loading dataset...")

    df = pd.read_csv(DATA_FILE)

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    workloads = np.load(WORKLOAD_FILE)

    results = []

    print(
        "\n=== FIXED-ITERATION "
        "HISTGRADIENTBOOSTING CALIBRATION ==="
    )

    for workload_name, workload_key in WORKLOADS:

        indices = workloads[workload_key]

        X_train = X.iloc[indices]
        y_train = y.iloc[indices]

        model = HistGradientBoostingClassifier(
            max_iter=100,
            early_stopping=False,
            random_state=RANDOM_STATE,
        )

        with threadpool_limits(limits=1):

            start = time.perf_counter()

            model.fit(
                X_train,
                y_train,
            )

            training_time = (
                time.perf_counter()
                - start
            )

        result = {
            "workload": workload_name,
            "training_samples": len(indices),
            "n_iter": model.n_iter_,
            "training_time_seconds": training_time,
        }

        results.append(result)

        print(
            f"{workload_name}: "
            f"samples={len(indices):,}, "
            f"n_iter_={model.n_iter_}, "
            f"time={training_time:.4f} s"
        )

    result_df = pd.DataFrame(results)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\n=== SUMMARY ===")
    print(result_df.to_string(index=False))

    print("\nSaved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()