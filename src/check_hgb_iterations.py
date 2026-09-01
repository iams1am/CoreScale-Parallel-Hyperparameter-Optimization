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

    print("\n=== HISTGRADIENTBOOSTING EARLY-STOPPING CHECK ===")

    for workload_name, workload_key in WORKLOADS:

        indices = workloads[workload_key]

        X_train = X.iloc[indices]
        y_train = y.iloc[indices]

        model = HistGradientBoostingClassifier(
            max_iter=100,
            random_state=RANDOM_STATE,
            early_stopping="auto",
        )

        with threadpool_limits(limits=1):

            start = time.perf_counter()

            model.fit(
                X_train,
                y_train,
            )

            elapsed = (
                time.perf_counter()
                - start
            )

        print(
            f"{workload_name}: "
            f"samples={len(indices):,}, "
            f"n_iter_={model.n_iter_}, "
            f"time={elapsed:.4f} s"
        )


if __name__ == "__main__":
    main()