from pathlib import Path
import pandas as pd


def load_bank_data(filepath: str) -> pd.DataFrame:
    """Load the Portuguese Bank Marketing dataset."""

    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    df = pd.read_csv(path, sep=";")

    return df


if __name__ == "__main__":

    data_path = "data/raw/bank-additional-full.csv"

    df = load_bank_data(data_path)

    print(f"Dataset shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())