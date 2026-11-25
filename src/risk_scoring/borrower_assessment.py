"""
CREDIT RISK INTELLIGENCE
Borrower Assessment Engine

Connects the calibrated XGBoost model to the risk-scoring framework.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .risk_score import calculate_risk_score
from .risk_grading import assign_risk_grade
from .decision_engine import generate_decision
from .expected_loss import calculate_expected_loss


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "artifacts" / "calibrated_credit_risk_model.pkl"

POLICY_PATH = (
    PROJECT_ROOT / "configs" / "risk_policy" / "risk_policy.json"
)

REPORT_DIR = PROJECT_ROOT / "reports" / "risk_scoring"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOGGING
# ============================================================

logger = logging.getLogger(__name__)

if not logger.handlers:
    logger.setLevel(logging.INFO)

    handler = logging.FileHandler(
        LOG_DIR / "borrower_assessment.log",
        encoding="utf-8"
    )

    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s"
        )
    )

    logger.addHandler(handler)


# ============================================================
# CONSTANTS
# ============================================================

DEFAULT_CLASS_OF_INTEREST = 1
TARGET_COLUMN = "loan_status"


# ============================================================
# LOAD POLICY
# ============================================================

def load_policy():
    """Load the JSON risk policy."""

    if not POLICY_PATH.exists():
        raise FileNotFoundError(
            f"Risk policy not found: {POLICY_PATH}"
        )

    with open(POLICY_PATH, "r", encoding="utf-8-sig") as file:
        return json.load(file)


# ============================================================
# LOAD CALIBRATED MODEL
# ============================================================

def load_calibrated_model():
    """
    Load the saved calibrated model bundle.

    Expected keys:
        model
        calibrator
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Calibrated model not found: {MODEL_PATH}"
        )

    bundle = joblib.load(MODEL_PATH)

    if not isinstance(bundle, dict):
        raise TypeError(
            "Expected a dictionary model bundle."
        )

    for key in ("model", "calibrator"):
        if key not in bundle:
            raise KeyError(
                f"Model bundle is missing '{key}'."
            )

    model = bundle["model"]
    calibrator = bundle["calibrator"]

    if not hasattr(model, "predict_proba"):
        raise TypeError(
            "Model does not support predict_proba."
        )

    if not hasattr(calibrator, "predict_proba"):
        raise TypeError(
            "Calibrator does not support predict_proba."
        )

    logger.info("Calibrated model loaded successfully.")

    return bundle


# ============================================================
# VALIDATE BORROWER FEATURES
# ============================================================

def validate_borrower_input(borrower_data, model):
    """
    Validate the supplied borrower data against the
    fitted preprocessing pipeline.
    """

    if not isinstance(borrower_data, dict):
        raise TypeError(
            "Borrower data must be a dictionary."
        )

    if not borrower_data:
        raise ValueError(
            "Borrower data cannot be empty."
        )

    if TARGET_COLUMN in borrower_data:
        raise ValueError(
            "loan_status must not be supplied for prediction."
        )

    preprocessor = model.named_steps.get("preprocessor")

    if preprocessor is None:
        raise ValueError(
            "Model preprocessor was not found."
        )

    expected_features = list(
        preprocessor.feature_names_in_
    )

    supplied_features = list(
        borrower_data.keys()
    )

    missing = [
        feature for feature in expected_features
        if feature not in supplied_features
    ]

    unexpected = [
        feature for feature in supplied_features
        if feature not in expected_features
    ]

    if missing:
        raise ValueError(
            f"Missing required features: {missing}"
        )

    if unexpected:
        raise ValueError(
            f"Unexpected features: {unexpected}"
        )

    borrower_df = pd.DataFrame(
        [borrower_data],
        columns=expected_features
    )

    if borrower_df.isnull().any().any():
        missing_columns = borrower_df.columns[
            borrower_df.isnull().any()
        ].tolist()

        raise ValueError(
            f"Missing values in features: {missing_columns}"
        )

    return borrower_df


# ============================================================
# GET RAW DEFAULT PROBABILITY
# ============================================================

def get_raw_default_probability(model, borrower_df):
    """
    Extract the probability for class 1: Charged Off.
    """

    classifier = model.named_steps.get("classifier")

    if classifier is None:
        raise ValueError(
            "Classifier was not found in model pipeline."
        )

    classes = list(classifier.classes_)

    if DEFAULT_CLASS_OF_INTEREST not in classes:
        raise ValueError(
            "Charged Off class 1 is not available."
        )

    class_index = classes.index(
        DEFAULT_CLASS_OF_INTEREST
    )

    probabilities = model.predict_proba(
        borrower_df
    )

    return float(
        probabilities[0, class_index]
    )


# ============================================================
# CALIBRATE DEFAULT PROBABILITY
# ============================================================

def calibrate_default_probability(
    calibrator,
    raw_probability
):
    """
    Apply the saved Platt scaling calibrator.

    The calibration model was trained on the logit
    transformation of the original XGBoost probability.
    """

    if not 0 <= raw_probability <= 1:
        raise ValueError(
            "Raw probability must be between 0 and 1."
        )

    epsilon = 1e-7

    clipped_probability = np.clip(
        raw_probability,
        epsilon,
        1 - epsilon
    )

    logit_probability = np.log(
        clipped_probability / (1 - clipped_probability)
    )

    calibrator_input = np.array(
        [[logit_probability]]
    )

    classes = list(calibrator.classes_)

    if DEFAULT_CLASS_OF_INTEREST not in classes:
        raise ValueError(
            "Calibrator does not contain class 1."
        )

    class_index = classes.index(
        DEFAULT_CLASS_OF_INTEREST
    )

    calibrated_probability = float(
        calibrator.predict_proba(
            calibrator_input
        )[0, class_index]
    )

    if not 0 <= calibrated_probability <= 1:
        raise ValueError(
            "Invalid calibrated probability."
        )

    return calibrated_probability


# ============================================================
# BORROWER ASSESSMENT
# ============================================================

def assess_borrower(
    borrower_data,
    lgd=None,
    exposure_at_default=None,
    use_loan_amount_as_proxy=False
):
    """
    Generate a complete borrower risk assessment.

    Parameters
    ----------
    borrower_data : dict
        All required model input features.

    lgd : float, optional
        User-supplied LGD between 0 and 1.

    exposure_at_default : float, optional
        User-supplied EAD in USD.

    use_loan_amount_as_proxy : bool
        Whether to use original loan amount as an
        explicitly labelled exposure proxy.
    """

    logger.info("Starting borrower assessment.")

    # Load model
    bundle = load_calibrated_model()

    model = bundle["model"]
    calibrator = bundle["calibrator"]

    # Validate borrower features
    borrower_df = validate_borrower_input(
        borrower_data,
        model
    )

    # Generate raw probability
    raw_pd = get_raw_default_probability(
        model,
        borrower_df
    )

    # Calibrate probability
    calibrated_pd = calibrate_default_probability(
        calibrator,
        raw_pd
    )

    # Risk score
    risk_score = calculate_risk_score(
        calibrated_pd
    )

    # Risk grade
    risk_grade = assign_risk_grade(
        risk_score
    )

    # Decision recommendation
    decision = generate_decision(
        calibrated_pd
    )

    # Expected loss is optional
    expected_loss = None

    if lgd is not None:

        if exposure_at_default is not None:

            selected_exposure = exposure_at_default
            exposure_source = "User supplied EAD"

        elif use_loan_amount_as_proxy:

            if "loan_amnt" not in borrower_df.columns:
                raise ValueError(
                    "loan_amnt is unavailable."
                )

            selected_exposure = float(
                borrower_df.iloc[0]["loan_amnt"]
            )

            exposure_source = (
                "Original loan amount proxy; not actual EAD"
            )

        else:
            raise ValueError(
                "Supply exposure_at_default or explicitly "
                "enable use_loan_amount_as_proxy."
            )

        expected_loss = calculate_expected_loss(
            probability_of_default=calibrated_pd,
            exposure_at_default=selected_exposure,
            loss_given_default=lgd
        )

        expected_loss["exposure_source"] = exposure_source

    # Build result
    assessment = {
        "project": "Credit Risk Intelligence",

        "assessment_timestamp": datetime.now(
            timezone.utc
        ).isoformat(),

        "model_information": {
            "model_type": type(
                model.named_steps["classifier"]
            ).__name__,
            "calibration_method": "Platt Scaling",
            "prediction_target": "Charged Off",
            "prediction_class": 1
        },

        "borrower_risk": {
            "raw_probability_of_default": round(
                raw_pd, 6
            ),

            "calibrated_probability_of_default": round(
                calibrated_pd, 6
            ),

            "risk_score": risk_score,
            "risk_grade": risk_grade
        },

        "decision": decision,

        "expected_loss": expected_loss,

        "currency": "USD",

        "assessment_status": "SUCCESS"
    }

    logger.info(
        "Assessment completed. PD=%.6f, Score=%.2f",
        calibrated_pd,
        risk_score
    )

    return assessment


# ============================================================
# SAVE ASSESSMENT REPORT
# ============================================================

def save_assessment(
    assessment,
    filename=None
):
    """Save a borrower assessment as JSON."""

    if not isinstance(assessment, dict):
        raise TypeError(
            "Assessment must be a dictionary."
        )

    if filename is None:

        timestamp = datetime.now(
            timezone.utc
        ).strftime("%Y%m%d_%H%M%S_%f")

        filename = (
            f"borrower_assessment_{timestamp}.json"
        )

    if Path(filename).name != filename:
        raise ValueError(
            "Filename must not contain directory paths."
        )

    if not filename.lower().endswith(".json"):
        raise ValueError(
            "Filename must end with .json."
        )

    output_path = REPORT_DIR / filename

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            assessment,
            file,
            indent=4,
            ensure_ascii=False
        )

    logger.info(
        "Assessment report saved: %s",
        output_path
    )

    return output_path


# ============================================================
# INITIALIZATION TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("CREDIT RISK INTELLIGENCE")
    print("BORROWER ASSESSMENT ENGINE")
    print("=" * 60)

    try:

        bundle = load_calibrated_model()
        policy = load_policy()

        print("\nMODEL INITIALIZATION SUCCESSFUL")

        print(
            "Model type:",
            type(bundle["model"]).__name__
        )

        print(
            "Classifier:",
            type(
                bundle["model"].named_steps["classifier"]
            ).__name__
        )

        print(
            "Calibrator:",
            type(bundle["calibrator"]).__name__
        )

        print(
            "Policy:",
            policy["project_name"]
        )

        print("\nEXPECTED MODEL INPUT FEATURES:")

        preprocessor = bundle[
            "model"
        ].named_steps["preprocessor"]

        for index, feature in enumerate(
            preprocessor.feature_names_in_,
            start=1
        ):
            print(f"{index}. {feature}")

        print("\nINITIALIZATION CHECK COMPLETED")

    except Exception:

        logger.exception(
            "Borrower assessment initialization failed."
        )

        raise

