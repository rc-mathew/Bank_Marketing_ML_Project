import pandas as pd

from load_data import load_bank_data


REQUIRED_COLUMNS = {
    "age",
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "duration",
    "campaign",
    "pdays",
    "previous",
    "poutcome",
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed",
    "y",
}


def validate_data(df: pd.DataFrame) -> dict:
    """Run basic quality checks on the Bank Marketing dataset."""

    missing_columns = REQUIRED_COLUMNS - set(df.columns)

    report = {
        "rows": len(df),
        "columns": len(df.columns),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_values": int(df.isna().sum().sum()),
        "missing_columns": sorted(missing_columns),
        "pdays_999_count": int((df["pdays"] == 999).sum()),
        "target_distribution": df["y"].value_counts().to_dict(),
    }

    return report


if __name__ == "__main__":

    data_path = "data/raw/bank-additional-full.csv"

    df = load_bank_data(data_path)

    report = validate_data(df)

    print("\nDATA QUALITY REPORT")
    print("=" * 50)

    for check, result in report.items():
        print(f"{check}: {result}")