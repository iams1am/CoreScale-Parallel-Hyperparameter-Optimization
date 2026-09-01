from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


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
    / "data"
    / "processed"
    / "workload_indices.npz"
)

TARGET_COLUMN = "Cover_Type"
RANDOM_STATE = 42

WORKLOAD_SIZES = [
    25_000,
    100_000,
    250_000,
]


def make_stratified_subset(
    available_indices,
    targets,
    size,
    random_state,
):
    if size >= len(available_indices):
        return available_indices.copy()

    selected_indices, _ = train_test_split(
        available_indices,
        train_size=size,
        random_state=random_state,
        stratify=targets.iloc[available_indices],
    )

    return selected_indices


def print_distribution(
    name,
    indices,
    targets,
):
    y_subset = targets.iloc[indices]

    counts = (
        y_subset
        .value_counts()
        .sort_index()
    )

    percentages = (
        counts
        / len(y_subset)
        * 100
    )

    print(f"\n=== {name} ===")

    print(
        "Samples:",
        f"{len(indices):,}"
    )

    for class_id in counts.index:
        print(
            f"Class {class_id}: "
            f"{counts[class_id]:,} "
            f"({percentages[class_id]:.4f}%)"
        )


def main():
    print("Loading dataset...")

    df = pd.read_csv(DATA_FILE)

    y = df[TARGET_COLUMN]

    print("Loading fixed train/test split...")

    split = np.load(SPLIT_FILE)

    train_indices = split[
        "train_indices"
    ]

    test_indices = split[
        "test_indices"
    ]

    workloads = {}

    for size in WORKLOAD_SIZES:

        subset = make_stratified_subset(
            train_indices,
            y,
            size,
            RANDOM_STATE,
        )

        key = f"train_{size}"

        workloads[key] = subset

        print_distribution(
            key,
            subset,
            y,
        )

    workloads[
        "train_full"
    ] = train_indices

    workloads[
        "test"
    ] = test_indices

    print_distribution(
        "train_full",
        train_indices,
        y,
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        OUTPUT_FILE,
        **workloads,
    )

    print("\nSaved workloads to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()