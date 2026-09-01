from pathlib import Path
import json
import time

import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = PROJECT_ROOT / "data" / "raw" / "covertype.csv"
SPLIT_FILE = PROJECT_ROOT / "data" / "processed" / "split_indices.npz"

RESULT_FILE = (
    PROJECT_ROOT
    / "results"
    / "raw"
    / "dummy_baseline_metrics.json"
)

TARGET_COLUMN = "Cover_Type"


def main():
    print("Loading dataset...")
    df = pd.read_csv(DATA_FILE)

    print("Loading fixed train/test split...")
    split = np.load(SPLIT_FILE)

    train_indices = split["train_indices"]
    test_indices = split["test_indices"]

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train = X.iloc[train_indices]
    y_train = y.iloc[train_indices]

    X_test = X.iloc[test_indices]
    y_test = y.iloc[test_indices]

    print("\nTraining DummyClassifier...")

    model = DummyClassifier(
        strategy="most_frequent"
    )

    start_time = time.perf_counter()

    model.fit(X_train, y_train)

    training_time = time.perf_counter() - start_time

    print("Generating predictions...")

    start_time = time.perf_counter()

    predictions = model.predict(X_test)

    prediction_time = time.perf_counter() - start_time

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    report = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0
    )

    matrix = confusion_matrix(
        y_test,
        predictions
    )

    results = {
        "model": "DummyClassifier",
        "strategy": "most_frequent",
        "training_samples": len(X_train),
        "testing_samples": len(X_test),
        "training_time_seconds": training_time,
        "prediction_time_seconds": prediction_time,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }

    RESULT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        RESULT_FILE,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            results,
            file,
            indent=4
        )

    print("\n=== DUMMY BASELINE RESULTS ===")

    print(
        f"Training time: "
        f"{training_time:.6f} seconds"
    )

    print(
        f"Prediction time: "
        f"{prediction_time:.6f} seconds"
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

    print("\n=== CLASSIFICATION REPORT ===")

    report_df = pd.DataFrame(report).T
    print(report_df)

    print("\n=== CONFUSION MATRIX ===")
    print(matrix)

    print("\nSaved metrics to:")
    print(RESULT_FILE)


if __name__ == "__main__":
    main()