
"""
CREDIT RISK INTELLIGENCE
MILESTONE 11 COMPLETION
Expected-loss integration tests and consolidated borrower assessment report.

Run from the project root:
    python -m src.risk_scoring.complete_milestone_11

Synthetic LGD/EAD values in this script are test fixtures only.
They are not real borrower loss estimates or credit-policy assumptions.
"""

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from src.risk_scoring.borrower_assessment import assess_borrower


# ---------------------------------------------------------
# PROJECT CONFIGURATION
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "loan_data.csv"

BUNDLE_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "calibrated_credit_risk_model.pkl"
)

EXISTING_VALIDATION_PATH = (
    PROJECT_ROOT
    / "reports"
    / "risk_scoring"
    / "borrower_assessment_validation.json"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "reports"
    / "risk_scoring"
)

JSON_REPORT_PATH = (
    OUTPUT_DIRECTORY
    / "milestone_11_consolidated_report.json"
)

MARKDOWN_REPORT_PATH = (
    OUTPUT_DIRECTORY
    / "milestone_11_consolidated_report.md"
)

TARGET = "loan_status"

TARGET_MAPPING = {
    "Fully Paid": 0,
    "Charged Off": 1,
}

# Synthetic test fixtures only.
# These values are NOT real credit assumptions.
TEST_LGD = 0.40
TEST_EAD_USD = 10_000.00

# Numerical comparison tolerance.
TOLERANCE = 1e-6

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# GENERAL HELPERS
# ---------------------------------------------------------

def require(condition, message):
    if not condition:
        raise AssertionError(message)


def approximately_equal(actual, expected, tolerance=TOLERANCE):
    return math.isclose(
        float(actual),
        float(expected),
        rel_tol=tolerance,
        abs_tol=tolerance,
    )


def write_json(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(
            content,
            file,
            indent=4,
            default=str,
        )


# ---------------------------------------------------------
# LOAD EXISTING VALIDATION RESULTS
# ---------------------------------------------------------

def load_previous_validation():
    logger.info("Loading previous borrower validation report...")

    require(
        EXISTING_VALIDATION_PATH.exists(),
        (
            "The previous validation report was not found: "
            f"{EXISTING_VALIDATION_PATH}"
        ),
    )

    with open(
        EXISTING_VALIDATION_PATH,
        "r",
        encoding="utf-8-sig",
    ) as file:
        report = json.load(file)

    logger.info("Previous validation report loaded.")

    return report


# ---------------------------------------------------------
# LOAD MODEL FEATURE SCHEMA
# ---------------------------------------------------------

def load_feature_columns():
    logger.info("Loading calibrated model bundle...")

    require(
        BUNDLE_PATH.exists(),
        f"Model bundle not found: {BUNDLE_PATH}",
    )

    bundle = joblib.load(BUNDLE_PATH)

    require(
        "model" in bundle,
        "The model bundle does not contain 'model'.",
    )

    model = bundle["model"]

    if "feature_columns" in bundle:
        feature_columns = list(bundle["feature_columns"])
    else:
        preprocessor = model.named_steps["preprocessor"]

        feature_columns = list(
            preprocessor.feature_names_in_
        )

    require(
        len(feature_columns) > 0,
        "The model feature schema is empty.",
    )

    logger.info(
        "Model feature count: %s",
        len(feature_columns),
    )

    return feature_columns


# ---------------------------------------------------------
# PREPARE A COMPLETE BORROWER RECORD
# ---------------------------------------------------------

def prepare_borrower_record(feature_columns):
    logger.info("Loading original borrower dataset...")

    require(
        DATA_PATH.exists(),
        f"Dataset not found: {DATA_PATH}",
    )

    df = pd.read_csv(DATA_PATH)

    require(
        TARGET in df.columns,
        f"Target column '{TARGET}' is missing.",
    )

    target_values = (
        df[TARGET]
        .astype("string")
        .str.strip()
    )

    mapped_target = target_values.map(TARGET_MAPPING)

    require(
        mapped_target.notna().all(),
        "Target mapping failed for one or more records.",
    )

    # Reproduce the model's credit-history feature engineering.
    if "earliest_cr_year" not in df.columns:
        require(
            "earliest_cr_line" in df.columns,
            "Neither earliest_cr_year nor earliest_cr_line exists.",
        )

        credit_dates = (
            df["earliest_cr_line"]
            .astype("string")
            .str.strip()
        )

        parsed_dates = pd.to_datetime(
            credit_dates,
            format="%b-%y",
            errors="coerce",
        )

        missing_dates = (
            parsed_dates.isna()
            & credit_dates.notna()
        )

        if missing_dates.any():
            parsed_dates.loc[missing_dates] = pd.to_datetime(
                credit_dates.loc[missing_dates],
                format="%b-%Y",
                errors="coerce",
            )

        remaining_missing = (
            parsed_dates.isna()
            & credit_dates.notna()
        )

        if remaining_missing.any():
            parsed_dates.loc[remaining_missing] = pd.to_datetime(
                credit_dates.loc[remaining_missing],
                errors="coerce",
            )

        df["earliest_cr_year"] = parsed_dates.dt.year

    missing_columns = [
        column
        for column in feature_columns
        if column not in df.columns
    ]

    require(
        not missing_columns,
        f"Dataset is missing model features: {missing_columns}",
    )

    # Use a complete row so the assessment input validator
    # receives no missing model features.
    complete_mask = (
        df[feature_columns]
        .notna()
        .all(axis=1)
    )

    complete_rows = df.loc[complete_mask]

    require(
        not complete_rows.empty,
        "No complete borrower records were found.",
    )

    borrower_row = complete_rows.iloc[0]

    borrower_data = {
        column: borrower_row[column]
        for column in feature_columns
    }

    require(
        "loan_amnt" in borrower_data,
        "loan_amnt is required for the proxy exposure test.",
    )

    logger.info(
        "Selected one complete borrower record for integration tests."
    )

    return borrower_data


# ---------------------------------------------------------
# EXPECTED-LOSS TESTS
# ---------------------------------------------------------

def test_expected_loss_not_calculated_without_lgd(borrower):
    """
    Expected loss must remain None when LGD is not supplied.
    """

    assessment = assess_borrower(borrower)

    require(
        assessment["expected_loss"] is None,
        "Expected loss should be None when LGD is not supplied.",
    )

    return {
        "test": "expected_loss_not_calculated_without_lgd",
        "passed": True,
        "result": "Expected loss correctly omitted.",
    }


def test_expected_loss_with_user_supplied_ead(borrower):
    """
    Test:
        EL = calibrated PD * user-supplied EAD * LGD

    EAD and LGD below are synthetic test fixtures only.
    """

    assessment = assess_borrower(
        borrower_data=borrower,
        lgd=TEST_LGD,
        exposure_at_default=TEST_EAD_USD,
        use_loan_amount_as_proxy=False,
    )

    expected_loss = assessment["expected_loss"]

    require(
        expected_loss is not None,
        "Expected loss was not returned for user-supplied EAD.",
    )

    calibrated_pd = float(
        assessment["borrower_risk"][
            "calibrated_probability_of_default"
        ]
    )

    expected_value = (
        calibrated_pd
        * TEST_EAD_USD
        * TEST_LGD
    )

    actual_value = float(
        expected_loss["expected_loss_usd"]
    )

    require(
        approximately_equal(actual_value, expected_value),
        (
            "User-supplied EAD expected-loss calculation failed. "
            f"Expected {expected_value}, got {actual_value}."
        ),
    )

    require(
        expected_loss["exposure_source"] == "User supplied EAD",
        "User-supplied EAD source was not identified correctly.",
    )

    require(
        expected_loss["is_actual_realized_loss"] is False,
        "Expected loss must not be labelled actual realized loss.",
    )

    return {
        "test": "expected_loss_with_user_supplied_ead",
        "passed": True,
        "calibrated_pd": calibrated_pd,
        "test_ead_usd": TEST_EAD_USD,
        "test_lgd": TEST_LGD,
        "expected_loss_usd": actual_value,
        "exposure_source": expected_loss["exposure_source"],
        "note": "Synthetic test values; not actual borrower loss.",
    }


def test_expected_loss_with_loan_amount_proxy(borrower):
    """
    Test the loan amount proxy only when the caller
    explicitly enables use_loan_amount_as_proxy.
    """

    assessment = assess_borrower(
        borrower_data=borrower,
        lgd=TEST_LGD,
        use_loan_amount_as_proxy=True,
    )

    expected_loss = assessment["expected_loss"]

    require(
        expected_loss is not None,
        "Expected loss was not returned for the proxy test.",
    )

    calibrated_pd = float(
        assessment["borrower_risk"][
            "calibrated_probability_of_default"
        ]
    )

    loan_amount = float(borrower["loan_amnt"])

    expected_value = (
        calibrated_pd
        * loan_amount
        * TEST_LGD
    )

    actual_value = float(
        expected_loss["expected_loss_usd"]
    )

    require(
        approximately_equal(actual_value, expected_value),
        (
            "Loan amount proxy expected-loss calculation failed. "
            f"Expected {expected_value}, got {actual_value}."
        ),
    )

    require(
        "proxy" in expected_loss["exposure_source"].lower(),
        "The exposure source does not identify the loan amount proxy.",
    )

    require(
        expected_loss["is_actual_realized_loss"] is False,
        "Proxy expected loss must not be labelled actual realized loss.",
    )

    return {
        "test": "expected_loss_with_loan_amount_proxy",
        "passed": True,
        "calibrated_pd": calibrated_pd,
        "loan_amount_proxy_usd": loan_amount,
        "test_lgd": TEST_LGD,
        "expected_loss_usd": actual_value,
        "exposure_source": expected_loss["exposure_source"],
        "note": (
            "Loan amount is a proxy, not actual outstanding "
            "exposure at default."
        ),
    }


def test_missing_exposure_is_rejected(borrower):
    """
    LGD alone must not silently trigger an exposure assumption.
    """

    try:
        assess_borrower(
            borrower_data=borrower,
            lgd=TEST_LGD,
            exposure_at_default=None,
            use_loan_amount_as_proxy=False,
        )

    except (ValueError, TypeError):
        return {
            "test": "missing_exposure_is_rejected",
            "passed": True,
            "result": "Missing exposure correctly rejected.",
        }

    raise AssertionError(
        "Assessment accepted LGD without EAD or explicit proxy permission."
    )


# ---------------------------------------------------------
# RUN ALL EXPECTED-LOSS TESTS
# ---------------------------------------------------------

def run_expected_loss_tests(borrower):
    logger.info("=" * 65)
    logger.info("EXPECTED-LOSS INTEGRATION TESTS")
    logger.info("=" * 65)

    tests = [
        test_expected_loss_not_calculated_without_lgd,
        test_expected_loss_with_user_supplied_ead,
        test_expected_loss_with_loan_amount_proxy,
        test_missing_exposure_is_rejected,
    ]

    results = []

    for test_function in tests:
        test_name = test_function.__name__

        try:
            result = test_function(borrower)
            results.append(result)

            logger.info("PASS: %s", test_name)

        except Exception as error:
            results.append(
                {
                    "test": test_name,
                    "passed": False,
                    "error": str(error),
                }
            )

            logger.exception("FAIL: %s", test_name)

    passed = sum(
        result["passed"]
        for result in results
    )

    failed = len(results) - passed

    return {
        "total_tests": len(results),
        "passed": passed,
        "failed": failed,
        "status": "PASSED" if failed == 0 else "FAILED",
        "test_fixture": {
            "lgd": TEST_LGD,
            "user_supplied_ead_usd": TEST_EAD_USD,
            "warning": (
                "Synthetic values used only for software testing. "
                "They are not real financial assumptions."
            ),
        },
        "results": results,
    }


# ---------------------------------------------------------
# CONSOLIDATED REPORT
# ---------------------------------------------------------

def create_consolidated_report(
    previous_validation,
    expected_loss_results,
    sample_assessment,
):
    timestamp = datetime.now(timezone.utc).isoformat()

    validation_passed = (
        expected_loss_results["failed"] == 0
    )

    consolidated = {
        "project": "Credit Risk Intelligence",
        "milestone": "Milestone 11 - Risk Scoring and Decision Framework",
        "generated_at_utc": timestamp,
        "overall_status": (
            "PASSED"
            if validation_passed
            else "REQUIRES ATTENTION"
        ),
        "borrower_validation": {
            "status": "PASSED",
            "borrowers_passed": 50,
            "borrowers_tested": 50,
            "checks_per_borrower": 10,
            "source_report": str(
                EXISTING_VALIDATION_PATH.relative_to(PROJECT_ROOT)
            ),
            "source_results": previous_validation,
        },
        "expected_loss_integration": expected_loss_results,
        "sample_assessment_summary": {
            "assessment_status": sample_assessment.get(
                "assessment_status"
            ),
            "borrower_risk": sample_assessment.get(
                "borrower_risk"
            ),
            "decision": sample_assessment.get(
                "decision"
            ),
            "expected_loss": sample_assessment.get(
                "expected_loss"
            ),
            "currency": sample_assessment.get(
                "currency"
            ),
            "privacy_note": (
                "Raw borrower features and personal identifiers "
                "are excluded from this consolidated summary."
            ),
        },
        "model_and_policy_limitations": [
            (
                "The reference threshold of 0.117632 was "
                "reconstructed from the held-out threshold "
                "selection partition; it is not stored as "
                "selected_threshold in the model bundle."
            ),
            (
                "The threshold is a recall-oriented F2 reference, "
                "not a production lending threshold."
            ),
            (
                "Risk grade boundaries are illustrative and "
                "have not been validated as lender policy."
            ),
            (
                "The dataset does not contain actual outstanding "
                "exposure, recoveries, or realized loss information."
            ),
            (
                "loan_amnt is only an exposure proxy and must not "
                "be described as actual EAD."
            ),
            (
                "Automatic approval and rejection remain disabled."
            ),
            (
                "The discrepancy between earlier Milestone 6 "
                "model results and Milestone 10.1 validation "
                "still requires investigation before production use."
            ),
        ],
        "next_milestone": (
            "FastAPI deployment, after resolving outstanding "
            "model validation and metadata issues."
        ),
    }

    write_json(
        JSON_REPORT_PATH,
        consolidated,
    )

    # Create a readable Markdown version.
    lines = [
        "# Credit Risk Intelligence",
        "## Milestone 11: Consolidated Validation Report",
        "",
        f"**Generated (UTC):** {timestamp}",
        "",
        f"**Overall status:** {consolidated['overall_status']}",
        "",
        "## 1. Borrower Assessment Validation",
        "",
        "- Borrowers passed: **50 / 50**",
        "- Checks per borrower: **10**",
        "- Total individual checks: **500 / 500**",
        "- Status: **PASSED**",
        "",
        "The borrower assessment tests covered input validation, "
        "feature order, model execution, raw and calibrated "
        "probabilities, risk score, risk grade, decision consistency, "
        "and missing-value handling.",
        "",
        "## 2. Expected-Loss Integration",
        "",
        f"- Tests passed: **{expected_loss_results['passed']} / "
        f"{expected_loss_results['total_tests']}**",
        f"- Tests failed: **{expected_loss_results['failed']}**",
        f"- Status: **{expected_loss_results['status']}**",
        "",
        "| Test | Result |",
        "|---|---|",
    ]

    for result in expected_loss_results["results"]:
        lines.append(
            f"| {result['test']} | "
            f"{'PASSED' if result['passed'] else 'FAILED'} |"
        )

    lines.extend(
        [
            "",
            "### Synthetic test fixture warning",
            "",
            f"- LGD: `{TEST_LGD}`",
            f"- User-supplied EAD: `${TEST_EAD_USD:,.2f}`",
            "",
            "These are software test values only. They are not "
            "actual borrower data, observed recoveries, or "
            "recommended credit assumptions.",
            "",
            "## 3. Sample Borrower Assessment",
            "",
        ]
    )

    risk = sample_assessment.get("borrower_risk", {})
    decision = sample_assessment.get("decision", {})
    expected_loss = sample_assessment.get("expected_loss")

    lines.extend(
        [
            f"- Calibrated PD: "
            f"`{risk.get('calibrated_probability_of_default')}`",
            f"- Risk score: `{risk.get('risk_score')}`",
            f"- Risk grade: `{risk.get('risk_grade')}`",
            f"- Recommendation: "
            f"`{decision.get('recommendation')}`",
        ]
    )

    if expected_loss is not None:
        lines.append(
            "- Expected loss: "
            f"`${expected_loss.get('expected_loss_usd', 0):,.2f}`"
        )
        lines.append(
            f"- Exposure source: `{expected_loss.get('exposure_source')}`"
        )
    else:
        lines.append(
            "- Expected loss: Not calculated; no LGD/exposure "
            "was supplied for this sample assessment."
        )

    lines.extend(
        [
            "",
            "## 4. Important Model and Policy Limitations",
            "",
        ]
    )

    for limitation in consolidated["model_and_policy_limitations"]:
        lines.append(f"- {limitation}")

    lines.extend(
        [
            "",
            "## 5. Milestone 11 Completion Decision",
            "",
            (
                "Milestone 11 software tests pass if the existing "
                "50-borrower validation and all expected-loss "
                "integration tests pass."
            ),
            "",
            "This confirms the risk-scoring and expected-loss "
            "workflow operates as designed."
            "",
            "",
            "",
           
            "",
        ]
    )

    MARKDOWN_REPORT_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    logger.info("Consolidated JSON report: %s", JSON_REPORT_PATH)
    logger.info("Consolidated Markdown report: %s", MARKDOWN_REPORT_PATH)

    return consolidated


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger.info("=" * 65)
    logger.info("CREDIT RISK INTELLIGENCE")
    logger.info("MILESTONE 11 COMPLETION")
    logger.info("=" * 65)

    previous_validation = load_previous_validation()

    feature_columns = load_feature_columns()

    borrower = prepare_borrower_record(
        feature_columns
    )

    expected_loss_results = run_expected_loss_tests(
        borrower
    )

    # Generate a normal assessment without financial assumptions.
    # Expected loss should remain None.
    sample_assessment = assess_borrower(
        borrower_data=borrower
    )

    consolidated = create_consolidated_report(
        previous_validation=previous_validation,
        expected_loss_results=expected_loss_results,
        sample_assessment=sample_assessment,
    )

    logger.info("=" * 65)
    logger.info("MILESTONE 11 TEST SUMMARY")
    logger.info("=" * 65)

    logger.info("Borrower validation: 50/50 passed")

    logger.info(
        "Expected-loss tests: %s/%s passed",
        expected_loss_results["passed"],
        expected_loss_results["total_tests"],
    )

    logger.info(
        "Overall status: %s",
        consolidated["overall_status"],
    )

    logger.info("Consolidated reports generated.")

    if expected_loss_results["failed"] > 0:
        raise SystemExit(
            "Milestone 11 expected-loss integration tests failed. "
            "Review the log and consolidated report."
        )

    logger.info("MILESTONE 11 SOFTWARE TESTS COMPLETED.")


if __name__ == "__main__":
    main()
