from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_covtype


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_FILE = DATA_DIR / "covertype.csv"


def main():
    print("Loading Covertype dataset...")

    dataset = fetch_covtype(as_frame=True)

    X = dataset.data
    y = dataset.target.rename("Cover_Type")

    df = pd.concat([X, y], axis=1)

    print("\n=== DATASET SUMMARY ===")
    print("Rows:", df.shape[0])
    print("Features:", X.shape[1])
    print("Target:", y.name)
    print("Classes:", sorted(y.unique().tolist()))
    print("Missing values:", int(df.isna().sum().sum()))

    memory_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    print(f"Memory usage: {memory_mb:.2f} MB")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print("\nSaved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()