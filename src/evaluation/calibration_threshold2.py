
# ============================================================
# CREDIT RISK INTELLIGENCE
# MILESTONE 7: PROBABILITY CALIBRATION
# MILESTONE 7.1: FINANCIAL LOSS OPTIMIZATION
#
# Currency: USD
# Dataset: LendingClub loan_data.csv
#
# IMPORTANT:
# This dataset does not contain actual repayment or recovery
# amounts. Financial losses are therefore estimated using:
#
# Estimated Loss = Loan Exposure * LGD
#
# False-positive costs and LGD are supplied by the user.
# No financial cost assumptions are hardcoded.
# ============================================================

from pathlib import Path
import json
import logging
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve
)

from sklearn.calibration import calibration_curve

from sklearn.model_selection import train_test_split


warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "loan_data.csv"
)

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
LOGS_DIR = PROJECT_ROOT / "logs"

for directory in [
    ARTIFACTS_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
    LOGS_DIR
]:
    directory.mkdir(parents=True, exist_ok=True)


RANDOM_STATE = 42

TARGET_COLUMN = "loan_status"

EXPOSURE_COLUMN = "loan_amnt"

CLASS_MAPPING = {
    "Fully Paid": 0,
    "Charged Off": 1
}


# ============================================================
# 2. LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(
            LOGS_DIR / "milestone7_financial_optimization.log",
            encoding="utf-8"
        ),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


# ============================================================
# 3. USER-DEFINED FINANCIAL COST SCENARIOS
# ============================================================

def get_financial_scenarios():
    """
    Collect financial assumptions from the user.

    False-positive cost:
        User-supplied cost of incorrectly flagging a
        borrower who would repay.

    LGD:
        User-supplied loss-given-default rate.

    False-negative cost:
        Loan exposure multiplied by LGD.

    No default monetary assumptions are used.
    """

    print("\n" + "=" * 70)
    print("FINANCIAL COST CONFIGURATION")
    print("=" * 70)

    print(
        "\nAll financial amounts must be entered in USD."
    )

    print(
        "\nFalse-positive cost is the estimated cost of "
        "flagging a borrower who would have repaid."
    )

    print(
        "\nLGD is the percentage of loan exposure expected "
        "to be lost if a borrower defaults."
    )

    while True:

        try:
            number_of_scenarios = int(
                input(
                    "\nHow many financial scenarios do you "
                    "want to compare? "
                )
            )

            if number_of_scenarios < 1:
                raise ValueError

            break

        except ValueError:
            print("Enter a whole number greater than zero.")

    scenarios = {}

    for index in range(number_of_scenarios):

        print(
            f"\n--- Financial Scenario {index + 1} ---"
        )

        while True:

            scenario_name = input(
                "Scenario name: "
            ).strip()

            if scenario_name and scenario_name not in scenarios:
                break

            print(
                "Enter a non-empty unique scenario name."
            )

        while True:

            try:
                fp_cost = float(
                    input(
                        "False-positive cost per flagged "
                        "good loan (USD): "
                    )
                )

                if not np.isfinite(fp_cost) or fp_cost < 0:
                    raise ValueError

                break

            except ValueError:
                print(
                    "Enter a valid non-negative USD amount."
                )

        while True:

            try:
                lgd = float(
                    input(
                        "Loss Given Default (enter a "
                        "percentage, e.g. 40 for 40%): "
                    )
                )

                if not np.isfinite(lgd) or not 0 <= lgd <= 100:
                    raise ValueError

                break

            except ValueError:
                print(
                    "Enter an LGD percentage between 0 and 100."
                )

        scenarios[scenario_name] = {
            "false_positive_cost_usd": fp_cost,
            "lgd_rate": lgd / 100
        }

    return scenarios


# ============================================================
# 4. LOAD AND PREPARE DATA
# ============================================================

def load_and_prepare_data():

    logger.info("Loading dataset: %s", DATA_PATH)

    if not DATA_PATH.exists():

        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    logger.info(
        "Dataset shape: %s",
        df.shape
    )

    required_columns = [
        TARGET_COLUMN,
        EXPOSURE_COLUMN
    ]

    for column in required_columns:

        if column not in df.columns:

            raise ValueError(
                f"Required column missing: {column}"
            )

    # --------------------------------------------------------
    # TARGET ENCODING
    # --------------------------------------------------------

    df[TARGET_COLUMN] = (
        df[TARGET_COLUMN]
        .astype(str)
        .str.strip()
        .map(CLASS_MAPPING)
    )

    if df[TARGET_COLUMN].isna().any():

        raise ValueError(
            "Unrecognized loan_status values found."
        )

    df[TARGET_COLUMN] = (
        df[TARGET_COLUMN].astype(int)
    )

    # --------------------------------------------------------
    # EXPOSURE VALIDATION
    # --------------------------------------------------------

    df[EXPOSURE_COLUMN] = pd.to_numeric(
        df[EXPOSURE_COLUMN],
        errors="coerce"
    )

    if df[EXPOSURE_COLUMN].isna().any():

        raise ValueError(
            "Missing or invalid loan exposure values found."
        )

    if (df[EXPOSURE_COLUMN] < 0).any():

        raise ValueError(
            "Negative loan exposure values found."
        )

    # --------------------------------------------------------
    # DATE FEATURE ENGINEERING
    # --------------------------------------------------------

    if "earliest_cr_line" in df.columns:

        parsed_dates = pd.to_datetime(
            df["earliest_cr_line"],
            format="%b-%y",
            errors="coerce"
        )

        missing_dates = parsed_dates.isna()

        if missing_dates.any():

            parsed_dates.loc[missing_dates] = (
                pd.to_datetime(
                    df.loc[
                        missing_dates,
                        "earliest_cr_line"
                    ],
                    errors="coerce"
                )
            )

        df["earliest_cr_year"] = (
            parsed_dates.dt.year
        )

    # --------------------------------------------------------
    # SEPARATE EXPOSURE FROM PREDICTIVE FEATURES
    # --------------------------------------------------------

    # Exposure is retained separately for financial cost
    # calculation. It remains a predictive feature only if
    # it was included in the original model pipeline.

    exposure = df[EXPOSURE_COLUMN].copy()

    columns_to_drop = [
        TARGET_COLUMN,
        "emp_title",
        "title",
        "address",
        "issue_d",
        "grade",
        "id",
        "member_id"
    ]

    X = df.drop(
        columns=columns_to_drop,
        errors="ignore"
    )

    y = df[TARGET_COLUMN].copy()

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    feature_columns = X.columns.tolist()

    logger.info(
        "Modeling features: %d",
        len(feature_columns)
    )

    logger.info(
        "Default rate: %.4f",
        y.mean()
    )

    logger.info(
        "Mean loan exposure: USD %.2f",
        exposure.mean()
    )

    return X, y, exposure, feature_columns


# ============================================================
# 5. CREATE DATA SPLITS
# ============================================================

def create_data_splits(X, y, exposure):

    logger.info("Creating stratified data splits.")

    # Original 80% training / 20% test split

    (
        X_train,
        X_test,
        y_train,
        y_test,
        exposure_train,
        exposure_test
    ) = train_test_split(
        X,
        y,
        exposure,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE
    )

    # Divide original training data:
    # 70% model fitting
    # 15% calibration
    # 15% threshold selection

    (
        X_fit,
        X_remaining,
        y_fit,
        y_remaining,
        exposure_fit,
        exposure_remaining
    ) = train_test_split(
        X_train,
        y_train,
        exposure_train,
        test_size=0.30,
        stratify=y_train,
        random_state=RANDOM_STATE
    )

    (
        X_calibration,
        X_threshold,
        y_calibration,
        y_threshold,
        exposure_calibration,
        exposure_threshold
    ) = train_test_split(
        X_remaining,
        y_remaining,
        exposure_remaining,
        test_size=0.50,
        stratify=y_remaining,
        random_state=RANDOM_STATE
    )

    logger.info(
        "Model fitting rows: %d",
        len(X_fit)
    )

    logger.info(
        "Calibration rows: %d",
        len(X_calibration)
    )

    logger.info(
        "Threshold rows: %d",
        len(X_threshold)
    )

    logger.info(
        "Test rows: %d",
        len(X_test)
    )

    return {
        "X_fit": X_fit,
        "X_calibration": X_calibration,
        "X_threshold": X_threshold,
        "X_test": X_test,
        "y_fit": y_fit,
        "y_calibration": y_calibration,
        "y_threshold": y_threshold,
        "y_test": y_test,
        "exposure_threshold": exposure_threshold,
        "exposure_test": exposure_test
    }


# ============================================================
# 6. LOAD AND REFIT SELECTED MODEL
# ============================================================

def fit_selected_model(X_fit, y_fit):

    model_path = (
        ARTIFACTS_DIR / "best_model.pkl"
    )

    if not model_path.exists():

        raise FileNotFoundError(
            "artifacts/best_model.pkl not found. "
            "Run the advanced modeling milestone first."
        )

    logger.info("Loading selected model.")

    saved_model = joblib.load(model_path)

    model = clone(saved_model)

    logger.info(
        "Refitting selected model on fitting partition."
    )

    model.fit(
        X_fit,
        y_fit
    )

    logger.info("Model fitting completed.")

    return model


# ============================================================
# 7. PROBABILITY CALIBRATION
# ============================================================

def fit_probability_calibrator(
    model,
    X_calibration,
    y_calibration
):

    logger.info("Fitting Platt probability calibrator.")

    raw_probabilities = model.predict_proba(
        X_calibration
    )[:, 1]

    epsilon = 1e-6

    clipped = np.clip(
        raw_probabilities,
        epsilon,
        1 - epsilon
    )

    logits = np.log(
        clipped / (1 - clipped)
    ).reshape(-1, 1)

    calibrator = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE
    )

    calibrator.fit(
        logits,
        y_calibration
    )

    logger.info("Probability calibration completed.")

    return calibrator


def calibrate_probabilities(
    model,
    calibrator,
    X
):

    raw_probabilities = model.predict_proba(
        X
    )[:, 1]

    epsilon = 1e-6

    clipped = np.clip(
        raw_probabilities,
        epsilon,
        1 - epsilon
    )

    logits = np.log(
        clipped / (1 - clipped)
    ).reshape(-1, 1)

    return calibrator.predict_proba(
        logits
    )[:, 1]


# ============================================================
# 8. PROBABILITY QUALITY METRICS
# ============================================================

def probability_quality_metrics(
    y_true,
    probabilities,
    model_name
):

    return {
        "model": model_name,
        "roc_auc": roc_auc_score(
            y_true,
            probabilities
        ),
        "pr_auc": average_precision_score(
            y_true,
            probabilities
        ),
        "brier_score": brier_score_loss(
            y_true,
            probabilities
        ),
        "log_loss": log_loss(
            y_true,
            probabilities,
            labels=[0, 1]
        )
    }


# ============================================================
# 9. LOAN-LEVEL FINANCIAL COST
# ============================================================

def calculate_financial_cost(
    y_true,
    probabilities,
    exposure,
    threshold,
    false_positive_cost_usd,
    lgd_rate
):
    """
    Class 0 = Fully Paid
    Class 1 = Charged Off

    False positive:
        Good borrower flagged as risky.

    False negative:
        Defaulting borrower classified as safe.

    False-negative estimated loss:
        Loan exposure * LGD.

    No recovery information is available in this dataset.
    """

    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    exposure = np.asarray(exposure)

    if not (
        len(y_true)
        == len(probabilities)
        == len(exposure)
    ):
        raise ValueError(
            "Target, probabilities and exposure "
            "must have identical lengths."
        )

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    ).ravel()

    false_positive_mask = (
        (y_true == 0)
        & (predictions == 1)
    )

    false_negative_mask = (
        (y_true == 1)
        & (predictions == 0)
    )

    # False-positive financial cost

    total_false_positive_cost = (
        fp * false_positive_cost_usd
    )

    # False-negative exposure

    missed_default_exposure = exposure[
        false_negative_mask
    ].sum()

    # Estimated credit loss

    estimated_missed_default_loss = (
        missed_default_exposure * lgd_rate
    )

    # Total estimated financial cost

    total_cost = (
        total_false_positive_cost
        + estimated_missed_default_loss
    )

    return {
        "threshold": float(threshold),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "missed_default_exposure_usd": float(
            missed_default_exposure
        ),
        "estimated_missed_default_loss_usd": float(
            estimated_missed_default_loss
        ),
        "false_positive_cost_usd": float(
            total_false_positive_cost
        ),
        "total_cost_usd": float(total_cost),
        "cost_per_loan_usd": float(
            total_cost / len(y_true)
        )
    }


# ============================================================
# 10. COST-OPTIMAL THRESHOLD SELECTION
# ============================================================

def find_optimal_threshold(
    y_true,
    probabilities,
    exposure,
    false_positive_cost_usd,
    lgd_rate
):

    # Search a consistent grid of thresholds.
    thresholds = np.linspace(
        0.01,
        0.99,
        99
    )

    results = []

    for threshold in thresholds:

        result = calculate_financial_cost(
            y_true=y_true,
            probabilities=probabilities,
            exposure=exposure,
            threshold=threshold,
            false_positive_cost_usd=(
                false_positive_cost_usd
            ),
            lgd_rate=lgd_rate
        )

        results.append(result)

    results_df = pd.DataFrame(results)

    best_index = results_df[
        "total_cost_usd"
    ].idxmin()

    best_result = results_df.loc[
        best_index
    ].to_dict()

    return results_df, best_result


# ============================================================
# 11. FINAL TEST EVALUATION
# ============================================================

def evaluate_test_set(
    y_test,
    test_probabilities,
    exposure_test,
    threshold,
    scenario_name,
    false_positive_cost_usd,
    lgd_rate
):

    predictions = (
        np.asarray(test_probabilities)
        >= threshold
    ).astype(int)

    financial_result = calculate_financial_cost(
        y_true=y_test,
        probabilities=test_probabilities,
        exposure=exposure_test,
        threshold=threshold,
        false_positive_cost_usd=(
            false_positive_cost_usd
        ),
        lgd_rate=lgd_rate
    )

    result = {
        "scenario": scenario_name,
        "accuracy": accuracy_score(
            y_test,
            predictions
        ),
        "roc_auc": roc_auc_score(
            y_test,
            test_probabilities
        ),
        "pr_auc": average_precision_score(
            y_test,
            test_probabilities
        ),
        "precision_default": precision_score(
            y_test,
            predictions,
            zero_division=0
        ),
        "recall_default": recall_score(
            y_test,
            predictions,
            zero_division=0
        ),
        "f1_default": f1_score(
            y_test,
            predictions,
            zero_division=0
        )
    }

    result.update(financial_result)

    return result


# ============================================================
# 12. PROBABILITY EVALUATION PLOTS
# ============================================================

def save_probability_plots(
    y_test,
    raw_probabilities,
    calibrated_probabilities
):

    # --------------------------------------------------------
    # CALIBRATION CURVE
    # --------------------------------------------------------

    plt.figure(figsize=(8, 6))

    for probabilities, label in [
        (raw_probabilities, "Uncalibrated XGBoost"),
        (calibrated_probabilities, "Calibrated XGBoost")
    ]:

        fraction_positive, mean_predicted = (
            calibration_curve(
                y_test,
                probabilities,
                n_bins=10,
                strategy="quantile"
            )
        )

        plt.plot(
            mean_predicted,
            fraction_positive,
            marker="o",
            label=label
        )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect calibration"
    )

    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed default frequency")
    plt.title("Probability Calibration Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "calibration_curve.png",
        dpi=300
    )

    plt.close()

    # --------------------------------------------------------
    # ROC CURVE
    # --------------------------------------------------------

    plt.figure(figsize=(8, 6))

    for probabilities, label in [
        (raw_probabilities, "Uncalibrated XGBoost"),
        (calibrated_probabilities, "Calibrated XGBoost")
    ]:

        fpr, tpr, _ = roc_curve(
            y_test,
            probabilities
        )

        auc_value = roc_auc_score(
            y_test,
            probabilities
        )

        plt.plot(
            fpr,
            tpr,
            label=f"{label} (AUC={auc_value:.3f})"
        )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "roc_curve.png",
        dpi=300
    )

    plt.close()

    # --------------------------------------------------------
    # PRECISION-RECALL CURVE
    # --------------------------------------------------------

    plt.figure(figsize=(8, 6))

    for probabilities, label in [
        (raw_probabilities, "Uncalibrated XGBoost"),
        (calibrated_probabilities, "Calibrated XGBoost")
    ]:

        precision, recall, _ = (
            precision_recall_curve(
                y_test,
                probabilities
            )
        )

        plt.plot(
            recall,
            precision,
            label=label
        )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "precision_recall_curve.png",
        dpi=300
    )

    plt.close()


# ============================================================
# 13. FINANCIAL COST CURVE
# ============================================================

def save_cost_curve(all_results):

    plt.figure(figsize=(10, 7))

    for scenario, group in (
        all_results.groupby("scenario")
    ):

        plt.plot(
            group["threshold"],
            group["cost_per_loan_usd"],
            label=scenario
        )

    plt.xlabel("Classification Threshold")
    plt.ylabel("Estimated Financial Cost Per Loan (USD)")
    plt.title("Financial Cost vs Classification Threshold")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "financial_cost_threshold_curve.png",
        dpi=300
    )

    plt.close()


# ============================================================
# 14. MAIN EXECUTION
# ============================================================

def main():

    logger.info("=" * 70)
    logger.info("CREDIT RISK INTELLIGENCE")
    logger.info("MILESTONE 7: CALIBRATION")
    logger.info("MILESTONE 7.1: FINANCIAL COST OPTIMIZATION")
    logger.info("=" * 70)

    # --------------------------------------------------------
    # USER FINANCIAL ASSUMPTIONS
    # --------------------------------------------------------

    financial_scenarios = get_financial_scenarios()

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    X, y, exposure, feature_columns = (
        load_and_prepare_data()
    )

    # --------------------------------------------------------
    # SPLIT DATA
    # --------------------------------------------------------

    splits = create_data_splits(
        X,
        y,
        exposure
    )

    # --------------------------------------------------------
    # FIT MODEL
    # --------------------------------------------------------

    model = fit_selected_model(
        splits["X_fit"],
        splits["y_fit"]
    )

    # --------------------------------------------------------
    # CALIBRATION
    # --------------------------------------------------------

    calibrator = fit_probability_calibrator(
        model,
        splits["X_calibration"],
        splits["y_calibration"]
    )

    # --------------------------------------------------------
    # GENERATE PROBABILITIES
    # --------------------------------------------------------

    raw_test_probabilities = (
        model.predict_proba(
            splits["X_test"]
        )[:, 1]
    )

    threshold_probabilities = (
        calibrate_probabilities(
            model,
            calibrator,
            splits["X_threshold"]
        )
    )

    test_probabilities = (
        calibrate_probabilities(
            model,
            calibrator,
            splits["X_test"]
        )
    )

    # --------------------------------------------------------
    # PROBABILITY QUALITY
    # --------------------------------------------------------

    probability_results = pd.DataFrame([
        probability_quality_metrics(
            splits["y_test"],
            raw_test_probabilities,
            "Uncalibrated XGBoost"
        ),
        probability_quality_metrics(
            splits["y_test"],
            test_probabilities,
            "Calibrated XGBoost"
        )
    ])

    probability_results.to_csv(
        REPORTS_DIR / "probability_calibration_comparison.csv",
        index=False
    )

    # --------------------------------------------------------
    # COST OPTIMIZATION
    # --------------------------------------------------------

    all_threshold_results = []
    selected_threshold_results = []
    test_results = []

    selected_thresholds = {}

    for scenario_name, assumptions in (
        financial_scenarios.items()
    ):

        fp_cost = assumptions[
            "false_positive_cost_usd"
        ]

        lgd_rate = assumptions[
            "lgd_rate"
        ]

        logger.info(
            "Optimizing scenario: %s",
            scenario_name
        )

        # ----------------------------------------------------
        # SELECT THRESHOLD ON HOLDOUT DATA
        # ----------------------------------------------------

        threshold_results, best_result = (
            find_optimal_threshold(
                y_true=splits["y_threshold"],
                probabilities=threshold_probabilities,
                exposure=splits["exposure_threshold"],
                false_positive_cost_usd=fp_cost,
                lgd_rate=lgd_rate
            )
        )

        threshold_results["scenario"] = (
            scenario_name
        )

        all_threshold_results.append(
            threshold_results
        )

        selected_threshold = float(
            best_result["threshold"]
        )

        selected_thresholds[
            scenario_name
        ] = selected_threshold

        selected_threshold_results.append({
            "scenario": scenario_name,
            "false_positive_cost_usd": fp_cost,
            "lgd_rate": lgd_rate,
            "selected_threshold": selected_threshold,
            "threshold_set_total_cost_usd": (
                best_result["total_cost_usd"]
            ),
            "threshold_set_cost_per_loan_usd": (
                best_result["cost_per_loan_usd"]
            ),
            "missed_default_exposure_usd": (
                best_result["missed_default_exposure_usd"]
            ),
            "estimated_missed_default_loss_usd": (
                best_result[
                    "estimated_missed_default_loss_usd"
                ]
            ),
            "false_positive_count": (
                best_result["false_positive"]
            ),
            "false_negative_count": (
                best_result["false_negative"]
            )
        })

        # ----------------------------------------------------
        # FINAL TEST EVALUATION
        # ----------------------------------------------------

        test_result = evaluate_test_set(
            y_test=splits["y_test"],
            test_probabilities=test_probabilities,
            exposure_test=splits["exposure_test"],
            threshold=selected_threshold,
            scenario_name=scenario_name,
            false_positive_cost_usd=fp_cost,
            lgd_rate=lgd_rate
        )

        test_results.append(
            test_result
        )

        logger.info(
            "%s optimal threshold: %.4f",
            scenario_name,
            selected_threshold
        )

        logger.info(
            "%s estimated test cost: USD %.2f",
            scenario_name,
            test_result["total_cost_usd"]
        )

    # --------------------------------------------------------
    # COMBINE RESULTS
    # --------------------------------------------------------

    all_threshold_df = pd.concat(
        all_threshold_results,
        ignore_index=True
    )

    selected_thresholds_df = pd.DataFrame(
        selected_threshold_results
    )

    test_results_df = pd.DataFrame(
        test_results
    )

    # --------------------------------------------------------
    # SAVE REPORTS
    # --------------------------------------------------------

    all_threshold_df.to_csv(
        REPORTS_DIR / "cost_threshold_analysis.csv",
        index=False
    )

    selected_thresholds_df.to_csv(
        REPORTS_DIR / "selected_cost_thresholds.csv",
        index=False
    )

    test_results_df.to_csv(
        REPORTS_DIR / "cost_based_test_comparison.csv",
        index=False
    )

    probability_results.to_csv(
        REPORTS_DIR / "probability_calibration_comparison.csv",
        index=False
    )

    # --------------------------------------------------------
    # SAVE CLASSIFICATION REPORTS
    # --------------------------------------------------------

    for scenario_name, threshold in (
        selected_thresholds.items()
    ):

        predictions = (
            test_probabilities >= threshold
        ).astype(int)

        report = classification_report(
            splits["y_test"],
            predictions,
            target_names=[
                "Fully Paid",
                "Charged Off"
            ],
            zero_division=0
        )

        with open(
            REPORTS_DIR
            / f"classification_report_{scenario_name}.txt",
            "w",
            encoding="utf-8"
        ) as file:

            file.write(report)

    # --------------------------------------------------------
    # SAVE FIGURES
    # --------------------------------------------------------

    save_probability_plots(
        y_test=splits["y_test"],
        raw_probabilities=raw_test_probabilities,
        calibrated_probabilities=test_probabilities
    )

    save_cost_curve(
        all_threshold_df
    )

    # --------------------------------------------------------
    # SAVE MODEL BUNDLE
    # --------------------------------------------------------

    model_bundle = {
        "model": model,
        "calibrator": calibrator,

        # No scenario is silently selected as the final policy.
        "selected_threshold": None,

        "cost_scenario_thresholds": selected_thresholds,

        "feature_columns": feature_columns,

        "target_mapping": CLASS_MAPPING,

        "financial_scenarios": financial_scenarios,

        "prediction_class_0": "Fully Paid",
        "prediction_class_1": "Charged Off",

        "currency": "USD",

        "loss_method": (
            "Loan exposure multiplied by user-supplied LGD"
        ),

        "actual_recovery_data_available": False
    }

    joblib.dump(
        model_bundle,
        ARTIFACTS_DIR / "calibrated_credit_risk_model.pkl"
    )

    # --------------------------------------------------------
    # SAVE CONFIGURATION
    # --------------------------------------------------------

    configuration = {
        "currency": "USD",
        "financial_scenarios": financial_scenarios,
        "selected_thresholds": selected_thresholds,
        "threshold_selection_method": (
            "Minimum estimated financial cost"
        ),
        "loss_calculation": (
            "Missed default exposure multiplied by LGD"
        ),
        "test_set_used_for_threshold_selection": False,
        "actual_recoveries_available": False
    }

    with open(
        REPORTS_DIR / "financial_cost_configuration.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            configuration,
            file,
            indent=4
        )

    # --------------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("SELECTED FINANCIAL COST THRESHOLDS")
    print("=" * 80)

    print(
        selected_thresholds_df.to_string(
            index=False
        )
    )

    print("\n" + "=" * 80)
    print("FINAL TEST SET PERFORMANCE")
    print("=" * 80)

    print(
        test_results_df.to_string(
            index=False
        )
    )

    print("\n" + "=" * 80)
    print("PROBABILITY CALIBRATION")
    print("=" * 80)

    print(
        probability_results.to_string(
            index=False
        )
    )

    logger.info(
        "Milestone 7 and 7.1 completed successfully."
    )

    logger.info(
        "Reports saved to: %s",
        REPORTS_DIR
    )

    logger.info(
        "Model saved to: %s",
        ARTIFACTS_DIR
    )


if __name__ == "__main__":
    main()
