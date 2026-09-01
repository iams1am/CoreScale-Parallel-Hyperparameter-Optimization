from pathlib import Path
import json
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

SPLIT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "split_indices.npz"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "raw"
    / "baseline_model_results.csv"
)

TARGET_COLUMN = "Cover_Type"
RANDOM_STATE = 42


def evaluate_model(
    name,
    model,
    X_train,
    y_train,
    X_test,
    y_test,
):
    print(f"\n{'=' * 60}")
    print(f"MODEL: {name}")
    print(f"{'=' * 60}")

    with threadpool_limits(limits=1):

        print("\nThread pools inside controlled context:")

        for pool in threadpool_info():
            print(
                pool.get("internal_api"),
                "| threads =",
                pool.get("num_threads"),
            )

        print("\nTraining...")

        start = time.perf_counter()

        model.fit(
            X_train,
            y_train,
        )

        training_time = (
            time.perf_counter()
            - start
        )

        print("Predicting...")

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

    print("\nRESULTS")

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
        "model": name,
        "training_time_seconds": training_time,
        "prediction_time_seconds": prediction_time,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


def main():
    print("Loading dataset...")

    df = pd.read_csv(DATA_FILE)

    print("Loading fixed split...")

    split = np.load(SPLIT_FILE)

    train_indices = split[
        "train_indices"
    ]

    test_indices = split[
        "test_indices"
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

    X_test = X.iloc[
        test_indices
    ]

    y_test = y.iloc[
        test_indices
    ]

    print("\nDataset ready.")

    print(
        "Training samples:",
        f"{len(X_train):,}"
    )

    print(
        "Testing samples:",
        f"{len(X_test):,}"
    )

    models = {

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

    results = []

    for name, model in models.items():

        result = evaluate_model(
            name,
            model,
            X_train,
            y_train,
            X_test,
            y_test,
        )

        results.append(result)

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
    print("=" * 60)
    print("BASELINE COMPARISON")
    print("=" * 60)

    print(
        result_df.to_string(
            index=False
        )
    )

    print("\nSaved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()