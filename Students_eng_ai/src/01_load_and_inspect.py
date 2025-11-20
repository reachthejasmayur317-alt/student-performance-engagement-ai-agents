import os
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "student_engagement.csv")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

CLEAN_DATA_PATH = os.path.join(PROCESSED_DIR, "student_engagement.csv")


def main():
    print(f"Loading data from: {RAW_DATA_PATH}")
    df = pd.read_csv(RAW_DATA_PATH)

    print("\n Data loaded!")
    print("\nShape (rows, columns):", df.shape)

    print("\n First 5 rows:")
    print(df.head())

    print("\n Columns & types:")
    print(df.info())


    df = df.dropna(axis=1, how="all")


    df = df.drop_duplicates()

    print("\nAfter basic cleaning, shape:", df.shape)

    df.to_csv(CLEAN_DATA_PATH, index=False)
    print(f"\n Clean data saved to: {CLEAN_DATA_PATH}")


if __name__ == "__main__":
    main()
