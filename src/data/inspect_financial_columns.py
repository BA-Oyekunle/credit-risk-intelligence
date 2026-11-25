
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "loan_data.csv"

df = pd.read_csv(DATA_PATH)

print("\n" + "=" * 70)
print("CREDIT RISK DATASET INSPECTION")
print("=" * 70)

print("\nDATASET DIMENSIONS")
print(df.shape)

print("\nALL COLUMN NAMES")
for column in df.columns:
    print(column)

financial_keywords = [
    "loan",
    "principal",
    "payment",
    "recover",
    "collection",
    "interest",
    "outstanding",
    "funded",
    "settlement",
    "last_pymnt",
    "total_rec"
]

financial_columns = [
    column for column in df.columns
    if any(
        keyword in column.lower()
        for keyword in financial_keywords
    )
]

print("\n" + "=" * 70)
print("POTENTIAL FINANCIAL COLUMNS")
print("=" * 70)

for column in financial_columns:
    print(f"\nCOLUMN: {column}")
    print("DATA TYPE:", df[column].dtype)
    print("MISSING VALUES:", df[column].isna().sum())
    print("UNIQUE VALUES:", df[column].nunique())
    print("SAMPLE VALUES:")
    print(df[column].dropna().head(5).to_list())

    if pd.api.types.is_numeric_dtype(df[column]):
        print("\nDESCRIPTIVE STATISTICS:")
        print(df[column].describe())

print("\n" + "=" * 70)
print("LOAN STATUS DISTRIBUTION")
print("=" * 70)

if "loan_status" in df.columns:
    print(df["loan_status"].value_counts(dropna=False))

print("\n" + "=" * 70)
print("FINANCIAL COLUMN CORRELATIONS WITH LOAN STATUS")
print("=" * 70)

if "loan_status" in df.columns:
    encoded_status = df["loan_status"].map({
        "Fully Paid": 0,
        "Charged Off": 1
    })

    numeric_financial = df[
        financial_columns
    ].select_dtypes(include="number")

    correlations = numeric_financial.corrwith(
        encoded_status
    )

    print(
        correlations.sort_values(
            ascending=False
        ).to_string()
    )

print("\nInspection completed.")

