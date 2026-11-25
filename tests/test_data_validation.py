import pandas as pd

import pytest

from credit_risk.data_validation import (
    validate_required_columns,
    validate_dataset
)


def test_required_columns_pass():

    df = pd.DataFrame({

        "loan_amnt": [1000, 2000],

        "loan_status": [
            "Fully Paid",
            "Charged Off"
        ],

        "annual_inc": [50000, 60000],

        "dti": [10, 15]

    })

    validate_required_columns(df)


def test_required_columns_fail():

    df = pd.DataFrame({

        "loan_amnt": [1000, 2000],

        "annual_inc": [50000, 60000]

    })

    with pytest.raises(ValueError):

        validate_required_columns(df)


def test_validation_report():

    df = pd.DataFrame({

        "loan_amnt": [1000, 2000],

        "loan_status": [
            "Fully Paid",
            "Charged Off"
        ],

        "annual_inc": [50000, 60000],

        "dti": [10, 15]

    })

    report = validate_dataset(df)

    assert report["number_of_rows"] == 2

    assert report["number_of_columns"] == 4

    assert report["duplicate_rows"] == 0

    assert report["missing_loan_status"] == 0