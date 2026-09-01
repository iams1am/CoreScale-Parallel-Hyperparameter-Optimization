from pathlib import Path
import gc
import time

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
    / "workload_calibration.csv"
)

TARGET_COLUMN = "Cover_Type"
RANDOM_STATE = 42


WORKLOADS = [
    ("W25", "train_25000"),
    ("W100", "train_100000"),
    ("W250", "train_250000"),
    ("WFULL", "train_full"),
]


def build_models():
    return {
        "LogisticRegression": Pipeline([
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=500,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]),

        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),

        "HistGradientBoosting":
            HistGradientBoostingClassifier(
                max_iter=100,
                random_state=RANDOM_STATE,
            ),
    }


def evaluate(
    workload_name,
    model_name,
    model,
    X_train,
    y_train,
    X_test,
    y_test,
):
    print("\n" + "=" * 65)

    print(
        f"{workload_name} | "
        f"{model_name} | "
        f"{len(X_train):,} samples"
    )

    print("=" * 65)

    gc.collect()

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

        start = time.perf_counter()

        predictions = model.predict(
            X_test
        )

        prediction_time = (
            time.perf_counter()
            - start
        )

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )

    print(
        f"Training time: "
        f"{training_time:.4f} s"
    )

    print(
        f"Prediction time: "
        f"{prediction_time:.4f} s"
    )

    print(
        f"Accuracy: "
        f"{accuracy:.6f}"
    )

    print(
        f"Macro F1: "
        f"{macro_f1:.6f}"
    )

    print(
        f"Weighted F1: "
        f"{weighted_f1:.6f}"
    )

    return {
        "workload": workload_name,
        "training_samples": len(X_train),
        "model": model_name,
        "training_time_seconds": training_time,
        "prediction_time_seconds": prediction_time,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


def main():
    print("Loading dataset...")

    df = pd.read_csv(DATA_FILE)

    X = df.drop(
        columns=[TARGET_COLUMN]
    )

    y = df[TARGET_COLUMN]

    print(
        "Loading workload indices..."
    )

    workload_data = np.load(
        WORKLOAD_FILE
    )

    test_indices = workload_data[
        "test"
    ]

    X_test = X.iloc[
        test_indices
    ]

    y_test = y.iloc[
        test_indices
    ]

    results = []

    for workload_name, workload_key in WORKLOADS:

        train_indices = workload_data[
            workload_key
        ]

        X_train = X.iloc[
            train_indices
        ]

        y_train = y.iloc[
            train_indices
        ]

        models = build_models()

        for model_name, model in models.items():

            result = evaluate(
                workload_name,
                model_name,
                model,
                X_train,
                y_train,
                X_test,
                y_test,
            )

            results.append(
                result
            )

    result_df = pd.DataFrame(
        results
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\n")
    print("=" * 100)

    print(
        "WORKLOAD CALIBRATION SUMMARY"
    )

    print("=" * 100)

    print(
        result_df.to_string(
            index=False
        )
    )

    print("\nSaved to:")

    print(
        OUTPUT_FILE
    )


if __name__ == "__main__":
    main()