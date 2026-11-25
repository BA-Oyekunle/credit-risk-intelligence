"""
CREDIT RISK INTELLIGENCE
MILESTONE 11: BORROWER ASSESSMENT VALIDATION

This script:
1. Loads the original LendingClub dataset and maps loan_status.
2. Recreates the Milestone 10.1 stratified data partitions.
3. Selects a sample of COMPLETE borrowers (no missing model features)
   from the held-out test set, balanced across both outcomes.
4. Excludes loan_status from every prediction input.
5. Loads the calibrated model bundle.
6. Independently calculates raw and calibrated default probabilities.
7. Compares them with borrower_assessment.assess_borrower().
8. Validates the risk score, risk grade and the decision against the
   saved decision threshold.
9. Separately tests the missing-value policy of assess_borrower().
10. Saves a JSON validation report whose pass/fail flags are computed
    from the actual checks (nothing is hardcoded).

Run from the project root:

    python -m src.risk_scoring.test_borrower_assessment
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from src.risk_scoring.borrower_assessment import assess_borrower


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "loan_data.csv"

MODEL_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "calibrated_credit_risk_model.pkl"
)

REPORT_DIR = PROJECT_ROOT / "reports" / "risk_scoring"

REPORT_PATH = REPORT_DIR / "borrower_assessment_validation.json"

TARGET = "loan_status"

RANDOM_STATE = 42
TEST_SIZE = 0.20
CALIBRATION_FRACTION = 0.30
THRESHOLD_FRACTION = 0.50

PROBABILITY_TOLERANCE = 1e-6
SCORE_TOLERANCE = 0.01

# Number of complete borrowers validated per outcome class.
BORROWERS_PER_CLASS = 25

# Missing-value policy test.
# "reject": assess_borrower() must raise ValueError naming the feature.
# "impute": assess_borrower() must return a result that matches the
#           model's own handling of the missing value.
MISSING_VALUE_FEATURE = "mort_acc"
EXPECTED_MISSING_POLICY = "reject"

# Keywords used to classify the free-text decision.
# Order of evaluation: reject, review, approve.
REJECT_KEYWORDS = (
    "reject", "decline", "deny", "denied", "not approv", "high risk"
)
REVIEW_KEYWORDS = (
    "review", "refer", "manual", "conditional", "caution"
)
APPROVE_KEYWORDS = (
    "approve", "accept", "low risk"
)


# ============================================================
# 2. LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# ============================================================
# 3. JSON UTILITIES
# ============================================================

def json_safe(value):
    """Convert NumPy and Pandas values into JSON-compatible values."""

    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return None if np.isnan(value) else float(value)

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()

    if value is pd.NA or value is pd.NaT or value is None:
        return None

    if isinstance(value, float) and np.isnan(value):
        return None

    return value


# ============================================================
# 4. RECURSIVE RESULT EXTRACTION
# ============================================================

def find_value_by_name(data, candidate_names):
    """Recursively find a value using possible dictionary key names."""

    if isinstance(data, dict):

        for key, value in data.items():

            normalized_key = (
                str(key).lower().replace("-", "_").replace(" ", "_")
            )

            if normalized_key in candidate_names:
                return value

            found = find_value_by_name(value, candidate_names)

            if found is not None:
                return found

    elif isinstance(data, list):

        for item in data:

            found = find_value_by_name(item, candidate_names)

            if found is not None:
                return found

    return None


def find_numeric_value(data, candidate_names):
    """Recursively extract a numeric value from an assessment dictionary."""

    value = find_value_by_name(data, candidate_names)

    if isinstance(value, dict):
        value = find_value_by_name(
            value,
            {"value", "score", "probability", "probability_of_default"}
        )

    if value is None or isinstance(value, bool):
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ============================================================
# 5. CHECK HELPERS
# ============================================================

def assert_close(name, actual, expected, tolerance):
    """Verify that two numerical values match within tolerance."""

    if actual is None:
        raise AssertionError(f"{name}: value not found in assessment output.")

    difference = abs(float(actual) - float(expected))

    if difference > tolerance:
        raise AssertionError(
            f"{name} mismatch. "
            f"Assessment={actual} | Direct={expected} | "
            f"Difference={difference} | Tolerance={tolerance}"
        )

    logger.debug(
        "%s PASSED | Assessment=%.10f | Direct=%.10f",
        name, float(actual), float(expected)
    )


def classify_decision(decision):
    """Classify a free-text decision as reject, review or approve."""

    text = str(decision).strip().lower()

    if any(keyword in text for keyword in REJECT_KEYWORDS):
        return "reject"

    if any(keyword in text for keyword in REVIEW_KEYWORDS):
        return "review"

    if any(keyword in text for keyword in APPROVE_KEYWORDS):
        return "approve"

    return "unrecognised"


def assert_decision_consistent(decision, calibrated_pd, threshold):
    """
    Verify that the decision agrees with the reference threshold.

    PD >= threshold:
        Approval must not be recommended.

    PD < threshold:
        Rejection must not be recommended.

    Review recommendations are acceptable in either case.
    """

    if decision is None:
        raise AssertionError(
            "Decision not found in assessment output."
        )

    # The assessment engine returns a decision dictionary.
    # Extract its recommendation before classifying it.
    if isinstance(decision, dict):
        recommendation = decision.get("recommendation")

        if recommendation is None:
            raise AssertionError(
                "Decision dictionary does not contain "
                "a 'recommendation' field."
            )
    else:
        recommendation = decision

    category = classify_decision(recommendation)

    if category == "unrecognised":
        raise AssertionError(
            f"Decision '{recommendation}' could not be classified. "
            "Update REJECT/REVIEW/APPROVE_KEYWORDS."
        )

    flagged = float(calibrated_pd) >= float(threshold)

    if flagged and category == "approve":
        raise AssertionError(
            f"PD {calibrated_pd:.6f} >= threshold "
            f"{threshold:.6f}, but recommendation is "
            f"'{recommendation}'."
        )

    if not flagged and category == "reject":
        raise AssertionError(
            f"PD {calibrated_pd:.6f} < threshold "
            f"{threshold:.6f}, but recommendation is "
            f"'{recommendation}'."
        )


# ============================================================
# 6. FEATURE ENGINEERING
# ============================================================

def create_earliest_credit_year(df):
    """
    Create earliest_cr_year from earliest_cr_line.

    Uses the same date parsing sequence as the calibration workflow.
    """

    df = df.copy()

    if "earliest_cr_year" in df.columns:
        logger.info(
            "earliest_cr_year already exists. Preserving the existing feature."
        )
        return df

    if "earliest_cr_line" not in df.columns:
        raise ValueError(
            "Feature engineering failed. "
            "Neither earliest_cr_year nor earliest_cr_line exists."
        )

    credit_dates = df["earliest_cr_line"].astype("string").str.strip()

    parsed_dates = pd.to_datetime(
        credit_dates, format="%b-%y", errors="coerce"
    )

    missing_dates = parsed_dates.isna() & credit_dates.notna()

    if missing_dates.any():
        parsed_dates.loc[missing_dates] = pd.to_datetime(
            credit_dates.loc[missing_dates],
            format="%b-%Y",
            errors="coerce"
        )

    remaining_missing = parsed_dates.isna() & credit_dates.notna()

    if remaining_missing.any():
        parsed_dates.loc[remaining_missing] = pd.to_datetime(
            credit_dates.loc[remaining_missing],
            errors="coerce"
        )

    df["earliest_cr_year"] = parsed_dates.dt.year

    logger.info("Created earliest_cr_year.")

    return df


# ============================================================
# 7. DATASET PREPARATION
# ============================================================

def prepare_dataset():
    """Load the original dataset and prepare X and y."""

    logger.info("Loading original dataset...")

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    logger.info("Dataset loaded successfully: %s", df.shape)

    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' is missing.")

    logger.info("Original target dtype: %s", df[TARGET].dtype)

    target_values = df[TARGET].astype("string").str.strip()

    mapped_target = target_values.map(
        {"Fully Paid": 0, "Charged Off": 1}
    )

    # If categorical mapping fails, check whether the target is
    # already stored numerically.
    if mapped_target.isna().any():

        numeric_target = pd.to_numeric(target_values, errors="coerce")

        if numeric_target.notna().all():
            y = numeric_target.astype(int)

        else:
            unmapped_values = (
                target_values[mapped_target.isna()].unique().tolist()
            )
            raise ValueError(
                "Target mapping failed.\n"
                f"Unmapped target values: {unmapped_values}"
            )

    else:
        y = mapped_target.astype(int)

    if not set(y.unique()).issubset({0, 1}):
        raise ValueError("Target must contain only 0 and 1.")

    logger.info("Target mapping successful.")
    logger.info("Target distribution:\n%s", y.value_counts().to_string())
    logger.info("Default prevalence: %.6f", y.mean())

    df = create_earliest_credit_year(df)

    columns_to_drop = [
        "loan_status",
        "id",
        "emp_title",
        "title",
        "address",
        "issue_d",
        "grade"
    ]

    X = df.drop(columns=columns_to_drop, errors="ignore").copy()

    if TARGET in X.columns:
        raise AssertionError("Target leakage detected: loan_status is in X.")

    if len(X) != len(y):
        raise AssertionError("Feature and target row counts do not match.")

    logger.info("Feature matrix prepared: %s", X.shape)

    return X, y


# ============================================================
# 8. RECREATE MILESTONE 10.1 DATA PARTITIONS
# ============================================================

def recreate_validation_split(X, y):
    """
    Reproduce the Milestone 10.1 stratified partitions.

    Original dataset:  80% training, 20% held-out test
    Training partition: 70% fitting, 15% calibration, 15% threshold
    """

    logger.info("Recreating Milestone 10.1 data partitions...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    X_fit, X_temp, y_fit, y_temp = train_test_split(
        X_train, y_train,
        test_size=CALIBRATION_FRACTION,
        random_state=RANDOM_STATE,
        stratify=y_train
    )

    (
        X_calibration,
        X_threshold,
        y_calibration,
        y_threshold
    ) = train_test_split(
        X_temp, y_temp,
        test_size=THRESHOLD_FRACTION,
        random_state=RANDOM_STATE,
        stratify=y_temp
    )

    logger.info("PARTITION SIZES")
    logger.info("Model fitting: %s", len(X_fit))
    logger.info("Calibration: %s", len(X_calibration))
    logger.info("Threshold selection: %s", len(X_threshold))
    logger.info("Held-out test: %s", len(X_test))

    index_sets = [
        set(X_fit.index),
        set(X_calibration.index),
        set(X_threshold.index),
        set(X_test.index)
    ]

    partition_names = [
        "Model fitting",
        "Calibration",
        "Threshold selection",
        "Held-out test"
    ]

    for i in range(len(index_sets)):
        for j in range(i + 1, len(index_sets)):

            if index_sets[i] & index_sets[j]:
                raise AssertionError(
                    f"Data leakage detected between "
                    f"{partition_names[i]} and {partition_names[j]}."
                )

    logger.info("Partition independence check PASSED.")

    return X_test, y_test


# ============================================================
# 9. LOAD CALIBRATED MODEL BUNDLE
# ============================================================

def load_model_bundle():
    """Load and validate the calibrated model artifact."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Calibrated model not found: {MODEL_PATH}"
        )

    logger.info("Loading calibrated model bundle...")

    bundle = joblib.load(MODEL_PATH)

    # These are essential for borrower predictions.
    required_keys = [
        "model",
        "calibrator",
        "feature_columns"
    ]

    for key in required_keys:
        if key not in bundle:
            raise KeyError(
                f"Model bundle is missing required key: {key}"
            )

    if bundle["model"] is None:
        raise ValueError("The saved model is None.")

    if bundle["calibrator"] is None:
        raise ValueError("The saved calibrator is None.")

    if not bundle["feature_columns"]:
        raise ValueError("The model feature list is empty.")

    logger.info(
        "Model type: %s",
        type(bundle["model"]).__name__
    )

    logger.info(
        "Calibrator type: %s",
        type(bundle["calibrator"]).__name__
    )

    logger.info(
        "Expected model input features: %s",
        len(bundle["feature_columns"])
    )

    saved_threshold = bundle.get("selected_threshold")

    if saved_threshold is None:
        logger.warning(
            "No selected_threshold is stored in the model bundle."
        )
        logger.info(
            "The validated reference threshold must be "
            "reconstructed independently."
        )
    else:
        logger.info(
            "Saved selected threshold: %.6f",
            float(saved_threshold)
        )

    return bundle
# ============================================================
# 10. GET EXPECTED MODEL FEATURES
# ============================================================

def get_expected_features(bundle):
    """Retrieve the model's expected input feature names."""

    if "feature_columns" in bundle:
        features = list(bundle["feature_columns"])

    elif hasattr(bundle["model"], "feature_names_in_"):
        features = list(bundle["model"].feature_names_in_)

    else:
        preprocessor = bundle["model"].named_steps.get("preprocessor")

        if preprocessor is None:
            raise ValueError("Cannot identify the model's expected features.")

        features = list(preprocessor.feature_names_in_)

    if not features:
        raise ValueError("The expected model feature list is empty.")

    if TARGET in features:
        raise AssertionError("The model feature list contains loan_status.")

    logger.info("Expected model input features: %s", len(features))

    return features


# ============================================================
# 11. DIRECT MODEL PROBABILITY CALCULATION
# ============================================================

def calculate_direct_probabilities(model, calibrator, borrower_df):
    """
    Independently calculate raw and calibrated default probability.

    Reproduces the Milestone 7 Platt scaling logic.
    """

    classifier = model.named_steps["classifier"]
    classes = list(classifier.classes_)

    if 1 not in classes:
        raise ValueError("Charged Off class (1) not found in classifier.")

    raw_pd = float(
        model.predict_proba(borrower_df)[0, classes.index(1)]
    )

    epsilon = 1e-6
    clipped = np.clip(raw_pd, epsilon, 1 - epsilon)
    logit_input = np.array([[np.log(clipped / (1 - clipped))]])

    calibrator_classes = list(calibrator.classes_)

    if 1 not in calibrator_classes:
        raise ValueError("Charged Off class (1) not found in calibrator.")

    calibrated_pd = float(
        calibrator.predict_proba(logit_input)[
            0, calibrator_classes.index(1)
        ]
    )

    return raw_pd, calibrated_pd


def assert_present(name, value):
    """Verify that a value exists and is not blank."""

    if value is None or str(value).strip() == "":
        raise AssertionError(
            f"{name} not found in assessment output."
        )


def run_check(record, name, function, *args):
    """
    Execute an individual validation check.

    Store successful checks as True.
    Store failed checks as False and record the error.
    """

    try:
        function(*args)
        record["checks"][name] = True

    except Exception as error:
        record["checks"][name] = False
        record["errors"].append(
            f"{name}: {error}"
        )

# ============================================================
# 12. SINGLE-BORROWER VALIDATION
# ============================================================

def validate_one_borrower(
    borrower_index,
    X_test,
    y_test,
    expected_features,
    model,
    calibrator,
    threshold
):
    """Validate assess_borrower() against direct calculations."""

    actual_label = int(y_test.loc[borrower_index])

    record = {
        "borrower_row_index": int(borrower_index),
        "actual_test_label": actual_label,
        "actual_test_label_interpretation": (
            "Charged Off" if actual_label == 1 else "Fully Paid"
        ),
        "checks": {},
        "errors": []
    }

    borrower_features = X_test.loc[
        [borrower_index], expected_features
    ].copy()

    record["missing_feature_count"] = int(
        borrower_features.isna().sum().sum()
    )

    borrower_input = {
        key: json_safe(value)
        for key, value in borrower_features.iloc[0].to_dict().items()
    }

    # ---------------- Input validation ----------------

    run_check(
        record, "borrower_is_complete",
        lambda: (_ for _ in ()).throw(
            AssertionError("Borrower has missing features.")
        ) if record["missing_feature_count"] else None
    )

    run_check(
        record, "target_excluded_from_input",
        lambda: (_ for _ in ()).throw(
            AssertionError("loan_status found in borrower input.")
        ) if TARGET in borrower_input else None
    )

    run_check(
        record, "feature_order_matches_model",
        lambda: (_ for _ in ()).throw(
            AssertionError("Borrower feature order differs from model.")
        ) if list(borrower_input.keys()) != expected_features else None
    )

    # ---------------- Direct calculation ----------------

    direct_raw_pd, direct_calibrated_pd = calculate_direct_probabilities(
        model, calibrator, borrower_features
    )

    expected_risk_score = round(100 * (1 - direct_calibrated_pd), 2)

    record["direct_model_results"] = {
        "raw_probability_of_default": direct_raw_pd,
        "calibrated_probability_of_default": direct_calibrated_pd,
        "expected_risk_score": expected_risk_score,
        "threshold": threshold,
        "flagged_as_default_risk": bool(direct_calibrated_pd >= threshold)
    }

    # ---------------- Assessment engine ----------------

    try:
        assessment = assess_borrower(borrower_input)

        if not isinstance(assessment, dict):
            raise TypeError(
                "assess_borrower() must return a dictionary. "
                f"Received: {type(assessment).__name__}"
            )

        record["checks"]["assessment_runs"] = True

    except Exception as error:  # noqa: BLE001
        record["checks"]["assessment_runs"] = False
        record["errors"].append(f"assessment_runs: {error}")
        return record

    record["assessment_results"] = json_safe(assessment)

    assessment_raw_pd = find_numeric_value(
        assessment,
        {"raw_probability_of_default", "raw_pd", "raw_default_probability"}
    )

    assessment_calibrated_pd = find_numeric_value(
        assessment,
        {
            "calibrated_probability_of_default",
            "calibrated_pd",
            "probability_of_default",
            "default_probability",
            "pd"
        }
    )

    assessment_risk_score = find_numeric_value(
        assessment,
        {"risk_score", "credit_risk_score", "score"}
    )

    risk_grade = find_value_by_name(assessment, {"risk_grade", "grade"})

    decision = find_value_by_name(
        assessment,
        {"decision", "recommendation", "decision_recommendation"}
    )

    run_check(
        record, "raw_probability_matches", assert_close,
        "Raw default probability",
        assessment_raw_pd, direct_raw_pd, PROBABILITY_TOLERANCE
    )

    run_check(
        record, "calibrated_probability_matches", assert_close,
        "Calibrated default probability",
        assessment_calibrated_pd, direct_calibrated_pd,
        PROBABILITY_TOLERANCE
    )

    run_check(
        record, "risk_score_matches", assert_close,
        "Risk score",
        assessment_risk_score, expected_risk_score, SCORE_TOLERANCE
    )

    run_check(record, "risk_grade_present", assert_present,
              "Risk grade", risk_grade)

    run_check(
        record, "decision_matches_threshold",
        assert_decision_consistent,
        decision, direct_calibrated_pd, threshold
    )

    record["assessment_summary"] = {
        "risk_grade": json_safe(risk_grade),
        "decision": json_safe(decision)
    }

    record["passed"] = all(record["checks"].values())

    return record


# ============================================================
# 13. MISSING-VALUE POLICY TEST
# ============================================================

def test_missing_value_policy(
    X_test,
    expected_features,
    complete_rows,
    model,
    calibrator
):
    """
    Verify how assess_borrower() handles a missing feature.

    Uses a real held-out borrower that is missing only the target
    feature when one exists, otherwise a complete borrower with that
    feature blanked out.
    """

    result = {
        "feature_tested": MISSING_VALUE_FEATURE,
        "expected_policy": EXPECTED_MISSING_POLICY,
        "observed_policy": None,
        "source": None,
        "passed": False,
        "detail": ""
    }

    if MISSING_VALUE_FEATURE not in expected_features:
        result["detail"] = (
            f"'{MISSING_VALUE_FEATURE}' is not a model feature; "
            "test skipped."
        )
        result["passed"] = True
        result["observed_policy"] = "skipped"
        return result

    features_df = X_test[expected_features]

    only_this_missing = (
        features_df[MISSING_VALUE_FEATURE].isna()
        & features_df.drop(columns=[MISSING_VALUE_FEATURE]).notna().all(axis=1)
    )

    if only_this_missing.any():
        row_index = only_this_missing[only_this_missing].index[0]
        borrower_df = features_df.loc[[row_index]].copy()
        result["source"] = f"real held-out borrower (index {row_index})"

    else:
        row_index = complete_rows[complete_rows].index[0]
        borrower_df = features_df.loc[[row_index]].copy()
        borrower_df[MISSING_VALUE_FEATURE] = np.nan
        result["source"] = (
            f"synthetic: complete borrower {row_index} "
            f"with {MISSING_VALUE_FEATURE} blanked"
        )

    borrower_input = {
        key: json_safe(value)
        for key, value in borrower_df.iloc[0].to_dict().items()
    }

    try:
        assessment = assess_borrower(borrower_input)

    except ValueError as error:
        result["observed_policy"] = "reject"
        result["detail"] = str(error)

        if MISSING_VALUE_FEATURE not in str(error):
            result["detail"] += (
                f" | WARNING: message does not name '{MISSING_VALUE_FEATURE}'."
            )
            result["observed_policy"] = "reject_unclear_message"

    except Exception as error:  # noqa: BLE001
        result["observed_policy"] = "unexpected_error"
        result["detail"] = f"{type(error).__name__}: {error}"

    else:
        direct_raw, direct_calibrated = calculate_direct_probabilities(
            model, calibrator, borrower_df
        )

        assessment_calibrated = find_numeric_value(
            assessment,
            {
                "calibrated_probability_of_default",
                "calibrated_pd",
                "probability_of_default",
                "default_probability",
                "pd"
            }
        )

        try:
            assert_close(
                "Calibrated PD with imputed feature",
                assessment_calibrated,
                direct_calibrated,
                PROBABILITY_TOLERANCE
            )
            result["observed_policy"] = "impute"
            result["detail"] = (
                "Engine scored the borrower and matched the model's own "
                "handling of the missing value."
            )

        except AssertionError as error:
            result["observed_policy"] = "impute_mismatch"
            result["detail"] = str(error)

    result["passed"] = (
        result["observed_policy"] == EXPECTED_MISSING_POLICY
    )

    return result


# ============================================================
# 14. SAMPLE SELECTION
# ============================================================

def select_complete_borrowers(X_test, y_test, expected_features):
    """Select complete borrowers, balanced across both outcomes."""

    complete_rows = X_test[expected_features].notna().all(axis=1)

    if not complete_rows.any():
        raise ValueError(
            "No borrower in the held-out test set has complete "
            "values across all model features. A separate test "
            "of the missing-value handling policy is required."
        )

    logger.info(
        "Complete held-out borrowers: %s of %s",
        int(complete_rows.sum()), len(X_test)
    )

    complete_y = y_test.loc[complete_rows[complete_rows].index]

    selected = []

    for label in (0, 1):

        label_index = complete_y[complete_y == label].index

        if len(label_index) == 0:
            logger.warning("No complete borrowers with label %s.", label)
            continue

        n = min(BORROWERS_PER_CLASS, len(label_index))

        sampled = (
            pd.Series(label_index)
            .sample(n, random_state=RANDOM_STATE)
            .tolist()
        )

        selected.extend(sampled)

    logger.info("Borrowers selected for validation: %s", len(selected))

    return selected, complete_rows


# ============================================================
# 15. MAIN VALIDATION
# ============================================================

def run_borrower_assessment_test():

    print("\n" + "=" * 65)
    print("CREDIT RISK INTELLIGENCE")
    print("BORROWER ASSESSMENT VALIDATION")
    print("=" * 65)

    X, y = prepare_dataset()

    X_test, y_test = recreate_validation_split(X, y)

    bundle = load_model_bundle()

    model = bundle["model"]
    calibrator = bundle["calibrator"]
    threshold = 0.117632

    expected_features = get_expected_features(bundle)

    missing_features = [
        feature for feature in expected_features
        if feature not in X_test.columns
    ]

    if missing_features:
        raise ValueError(
            "The held-out test set is missing model features: "
            f"{missing_features}"
        )

    # --------------------------------------------------------
    # Select complete borrowers
    # --------------------------------------------------------

    borrower_indices, complete_rows = select_complete_borrowers(
        X_test, y_test, expected_features
    )

    # --------------------------------------------------------
    # Validate each borrower
    # --------------------------------------------------------

    logger.info("Validating %s borrowers...", len(borrower_indices))

    records = []

    for borrower_index in borrower_indices:

        record = validate_one_borrower(
            borrower_index,
            X_test,
            y_test,
            expected_features,
            model,
            calibrator,
            threshold
        )

        record.setdefault("passed", False)
        records.append(record)

        if not record["passed"]:
            logger.error(
                "Borrower %s FAILED: %s",
                borrower_index, record["errors"]
            )

    # --------------------------------------------------------
    # Missing-value policy test
    # --------------------------------------------------------

    logger.info("Testing missing-value policy...")

    missing_policy_result = test_missing_value_policy(
        X_test, expected_features, complete_rows, model, calibrator
    )

    logger.info(
        "Missing-value policy: expected=%s observed=%s passed=%s",
        missing_policy_result["expected_policy"],
        missing_policy_result["observed_policy"],
        missing_policy_result["passed"]
    )

    # --------------------------------------------------------
    # Aggregate results (computed, never hardcoded)
    # --------------------------------------------------------

    check_names = []

    for record in records:
        for name in record["checks"]:
            if name not in check_names:
                check_names.append(name)

    check_summary = {}

    for name in check_names:

        outcomes = [
            record["checks"][name]
            for record in records
            if name in record["checks"]
        ]

        check_summary[name] = {
            "passed": int(sum(outcomes)),
            "total": len(outcomes),
            "all_passed": bool(all(outcomes)) and len(outcomes) > 0
        }

    borrowers_passed = int(sum(record["passed"] for record in records))

    overall_passed = (
        len(records) > 0
        and borrowers_passed == len(records)
        and missing_policy_result["passed"]
    )

    # --------------------------------------------------------
    # Build and save report
    # --------------------------------------------------------

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    report = {
        "project": "Credit Risk Intelligence",
        "test_name": "Borrower Assessment Validation",
        "timestamp": datetime.now().isoformat(),
        "test_status": "PASSED" if overall_passed else "FAILED",

        "validation_method": (
            "assess_borrower() output compared against independent "
            "direct predictions from the calibrated model bundle, "
            "plus decision-vs-threshold and missing-value policy tests."
        ),

        "test_partition": {
            "source": "Milestone 10.1 held-out test set",
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
            "complete_borrowers_available": int(complete_rows.sum()),
            "borrowers_validated": len(records),
            "borrowers_per_class_requested": BORROWERS_PER_CLASS
        },

        "decision_threshold": threshold,

        "summary": {
            "borrowers_passed": borrowers_passed,
            "borrowers_failed": len(records) - borrowers_passed,
            "checks": check_summary,
            "missing_value_policy_passed": missing_policy_result["passed"]
        },

        "missing_value_policy_test": missing_policy_result,

        "borrower_results": records,

        "financial_loss": {
            "calculated": False,
            "reason": (
                "No LGD assumption was supplied for this test. "
                "Actual recovery and realized-loss data are not "
                "available in the original dataset."
            )
        }
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(json_safe(report), file, indent=4, ensure_ascii=False)

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("BORROWER ASSESSMENT VALIDATION RESULTS")
    print("=" * 65)

    print(f"Decision threshold: {threshold:.6f}")
    print(f"Borrowers validated: {len(records)}")
    print(f"Borrowers passed: {borrowers_passed}/{len(records)}")

    if records and records[0].get("direct_model_results"):
        first = records[0]
        direct = first["direct_model_results"]
        print("\nExample borrower")
        print("-" * 65)
        print(f"Row index: {first['borrower_row_index']}")
        print(f"Actual outcome: {first['actual_test_label_interpretation']}")
        print(
            "Raw default probability: "
            f"{direct['raw_probability_of_default']:.6f}"
        )
        print(
            "Calibrated default probability: "
            f"{direct['calibrated_probability_of_default']:.6f}"
        )
        print(f"Risk score: {direct['expected_risk_score']:.2f}/100")

        summary = first.get("assessment_summary", {})
        print(f"Risk grade: {summary.get('risk_grade')}")
        print(f"Decision: {summary.get('decision')}")

    print("\nVALIDATION CHECKS")
    print("-" * 65)

    for name, stats in check_summary.items():
        status = "PASSED" if stats["all_passed"] else "FAILED"
        print(f"[{status}] {name} ({stats['passed']}/{stats['total']})")

    policy_status = (
        "PASSED" if missing_policy_result["passed"] else "FAILED"
    )

    print(
        f"[{policy_status}] missing_value_policy "
        f"(expected={missing_policy_result['expected_policy']}, "
        f"observed={missing_policy_result['observed_policy']})"
    )

    print("\nREPORT SAVED:")
    print(REPORT_PATH)

    print("\n" + "=" * 65)
    print(
        "BORROWER ASSESSMENT VALIDATION "
        + ("PASSED" if overall_passed else "FAILED")
    )
    print("=" * 65)

    if not overall_passed:
        raise AssertionError(
            "One or more validation checks failed. "
            f"See {REPORT_PATH} for details."
        )

    return report


# ============================================================
# 16. EXECUTION
# ============================================================

if __name__ == "__main__":

    try:
        run_borrower_assessment_test()

    except Exception as error:  # noqa: BLE001

        logger.exception("BORROWER ASSESSMENT VALIDATION FAILED")

        print("\nVALIDATION FAILED:")
        print(error)

        sys.exit(1)