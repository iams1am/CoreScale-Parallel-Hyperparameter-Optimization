from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = PROJECT_ROOT / "data" / "raw" / "covertype.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "split_indices.npz"

TARGET_COLUMN = "Cover_Type"
TEST_SIZE = 0.20
RANDOM_STATE = 42


def print_distribution(name, y):
    counts = y.value_counts().sort_index()
    percentages = counts / len(y) * 100

    print(f"\n=== {name} CLASS DISTRIBUTION ===")

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

    indices = np.arange(len(df))

    train_indices, test_indices = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        OUTPUT_FILE,
        train_indices=train_indices,
        test_indices=test_indices,
    )

    y_train = y.iloc[train_indices]
    y_test = y.iloc[test_indices]

    print("\n=== SPLIT SUMMARY ===")
    print(f"Total samples: {len(df):,}")
    print(f"Training samples: {len(train_indices):,}")
    print(f"Testing samples: {len(test_indices):,}")
    print(f"Training percentage: {len(train_indices) / len(df) * 100:.2f}%")
    print(f"Testing percentage: {len(test_indices) / len(df) * 100:.2f}%")
    print(f"Random state: {RANDOM_STATE}")
    print("Stratified: Yes")

    print_distribution("TRAIN", y_train)
    print_distribution("TEST", y_test)

    print("\nSaved split indices to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()