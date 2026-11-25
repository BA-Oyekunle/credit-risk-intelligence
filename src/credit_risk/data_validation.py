import pandas as pd


REQUIRED_COLUMNS = [
    "loan_amnt",
    "loan_status",
    "annual_inc",
    "dti"
]


def validate_required_columns(
    df: pd.DataFrame
) -> None:

    """
    Check whether all required columns exist.
    """

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )


def validate_dataset(
    df: pd.DataFrame
) -> dict:

    """
    Perform basic data quality checks.
    """

    validate_required_columns(df)

    report = {

        "number_of_rows": len(df),

        "number_of_columns": len(df.columns),

        "duplicate_rows": int(
            df.duplicated().sum()
        ),

        "missing_loan_status": int(
            df["loan_status"].isna().sum()
        ),

        "negative_loan_amounts": int(
            (df["loan_amnt"] < 0).sum()
        ),

        "negative_income": int(
            (df["annual_inc"] < 0).sum()
        ),

        "negative_dti": int(
            (df["dti"] < 0).sum()
        )

    }

    return report