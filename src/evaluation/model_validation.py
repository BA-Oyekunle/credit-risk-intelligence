
"""
CREDIT RISK INTELLIGENCE
MILESTONE 10.1: OUT-OF-SAMPLE MODEL VALIDATION

Purpose:
1. Recreate the original stratified data partitions.
2. Evaluate the original XGBoost pipeline.
3. Evaluate the saved calibrated model bundle.
4. Select an F2 threshold using the threshold-selection partition.
5. Evaluate both models on the untouched test partition.
6. Generate performance metrics, CSV reports, plots and Markdown documentation.

Target:
    0 = Fully Paid
    1 = Charged Off

Important:
    The test set is never used for model fitting, calibration,
    or threshold selection.

    The calibrated artifact may have been overwritten by the
    supplementary financial-cost experiment. Its saved threshold
    is therefore not trusted. The F2 threshold is independently
    reconstructed using the original threshold-selection split.

    This script does not modify existing model artifacts.
"""

# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

from pathlib import Path
import json
import logging
import warnings

import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve
)

from sklearn.calibration import calibration_curve

from scipy.special import expit, logit


warnings.filterwarnings("ignore")


# ============================================================
# 2. PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "loan_data.csv"
)

ORIGINAL_MODEL_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "best_model.pkl"
)

CALIBRATED_MODEL_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "calibrated_credit_risk_model.pkl"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "model_validation"
)

PERFORMANCE_DIR = (
    REPORT_DIR
    / "performance"
)

LOG_DIR = (
    PROJECT_ROOT
    / "logs"
)

for directory in [
    REPORT_DIR,
    PERFORMANCE_DIR,
    LOG_DIR
]:
    directory.mkdir(
        parents=True,
        exist_ok=True
    )


# ------------------------------------------------------------
# TARGET CONFIGURATION
# ------------------------------------------------------------

TARGET = "loan_status"

TARGET_MAPPING = {
    "Fully Paid": 0,
    "Charged Off": 1
}

CLASS_NAMES = [
    "Fully Paid",
    "Charged Off"
]


# ------------------------------------------------------------
# DATA SPLIT CONFIGURATION
# ------------------------------------------------------------

RANDOM_STATE = 42

TEST_SIZE = 0.20

CALIBRATION_FRACTION = 0.30

THRESHOLD_FRACTION = 0.50

BETA = 2

CALIBRATION_BINS = 10


# ------------------------------------------------------------
# FEATURE CONFIGURATION
# ------------------------------------------------------------

DROP_COLUMNS = [
    "emp_title",
    "title",
    "address",
    "issue_d",
    "grade"
]

ID_COLUMNS = [
    "id",
    "loan_id",
    "customer_id",
    "application_id"
]


# ------------------------------------------------------------
# OUTPUT CONFIGURATION
# ------------------------------------------------------------

SAVE_PLOTS = True

SAVE_CSV = True

SAVE_JSON = True


# ============================================================
# 3. LOGGING CONFIGURATION
# ============================================================

LOG_PATH = (
    LOG_DIR
    / "model_validation.log"
)

logger = logging.getLogger(
    "credit_risk_model_validation"
)

logger.setLevel(logging.INFO)

logger.handlers.clear()

formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(message)s"
)

file_handler = logging.FileHandler(
    LOG_PATH,
    encoding="utf-8"
)

file_handler.setFormatter(formatter)

console_handler = logging.StreamHandler()

console_handler.setFormatter(formatter)

logger.addHandler(file_handler)

logger.addHandler(console_handler)


# ============================================================
# 4. GENERAL UTILITIES
# ============================================================

def print_section(title):

    print("\n")
    print("=" * 75)
    print(title)
    print("=" * 75)


def save_json(data, path):

    def convert(value):

        if isinstance(value, (np.integer,)):
            return int(value)

        if isinstance(value, (np.floating,)):
            return float(value)

        if isinstance(value, np.ndarray):
            return value.tolist()

        if isinstance(value, Path):
            return str(value)

        return str(value)

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=4,
            default=convert
        )


def safe_probability(probabilities):

    probabilities = np.asarray(
        probabilities,
        dtype=float
    )

    return np.clip(
        probabilities,
        1e-10,
        1 - 1e-10
    )


# ============================================================
# 5. FEATURE ENGINEERING
# ============================================================

def create_earliest_credit_year(df):

    """
    Reproduce the feature engineering used in
    calibration_threshold.py.
    """

    if "earliest_cr_year" in df.columns:

        return df

    if "earliest_cr_line" not in df.columns:

        raise ValueError(
            "Neither earliest_cr_year nor "
            "earliest_cr_line exists."
        )

    logger.info(
        "Creating earliest_cr_year from earliest_cr_line."
    )

    credit_dates = (
        df["earliest_cr_line"]
        .astype("string")
        .str.strip()
    )

    parsed_dates = pd.to_datetime(
        credit_dates,
        format="%b-%y",
        errors="coerce"
    )

    missing_dates = (
        parsed_dates.isna()
        & credit_dates.notna()
    )

    if missing_dates.any():

        parsed_dates.loc[missing_dates] = (
            pd.to_datetime(
                credit_dates.loc[missing_dates],
                format="%b-%Y",
                errors="coerce"
            )
        )

    remaining_missing = (
        parsed_dates.isna()
        & credit_dates.notna()
    )

    if remaining_missing.any():

        parsed_dates.loc[remaining_missing] = (
            pd.to_datetime(
                credit_dates.loc[remaining_missing],
                errors="coerce"
            )
        )

    df["earliest_cr_year"] = (
        parsed_dates.dt.year
    )

    logger.info(
        "earliest_cr_year created."
    )

    logger.info(
        "Missing earliest_cr_year values: %s",
        df["earliest_cr_year"].isna().sum()
    )

    return df


# ============================================================
# 6. LOAD AND PREPARE DATASET
# ============================================================

def load_and_prepare_data():

    print_section(
        "LOADING CREDIT RISK DATASET"
    )

    if not DATA_PATH.exists():

        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    logger.info(
        "Dataset dimensions: %s",
        df.shape
    )

    if TARGET not in df.columns:

        raise ValueError(
            f"Target column '{TARGET}' not found."
        )

    df = create_earliest_credit_year(df)

    # --------------------------------------------------------
    # TARGET MAPPING
    # --------------------------------------------------------

    if not pd.api.types.is_numeric_dtype(
        df[TARGET]
    ):

        df[TARGET] = (
            df[TARGET]
            .astype(str)
            .str.strip()
            .map(TARGET_MAPPING)
        )

    if df[TARGET].isna().any():

        raise ValueError(
            "Target contains missing or unmapped values."
        )

    df[TARGET] = df[TARGET].astype(int)

    unique_targets = set(
        df[TARGET].unique()
    )

    if not unique_targets.issubset({0, 1}):

        raise ValueError(
            f"Unexpected target classes: {unique_targets}"
        )

    # --------------------------------------------------------
    # FEATURE MATRIX
    # --------------------------------------------------------

    columns_to_drop = (
        [TARGET]
        + DROP_COLUMNS
        + ID_COLUMNS
    )

    X = df.drop(
        columns=columns_to_drop,
        errors="ignore"
    ).copy()

    y = df[TARGET].copy()

    logger.info(
        "Feature matrix dimensions: %s",
        X.shape
    )

    logger.info(
        "Target distribution:\n%s",
        y.value_counts().sort_index()
    )

    logger.info(
        "Default prevalence: %.6f",
        y.mean()
    )

    return X, y


# ============================================================
# 7. RECREATE OFFICIAL DATA PARTITIONS
# ============================================================

def create_data_splits(X, y):

    print_section(
        "RECREATING OFFICIAL DATA PARTITIONS"
    )

    # --------------------------------------------------------
    # 7.1 ORIGINAL TRAIN / TEST SPLIT
    # --------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # --------------------------------------------------------
    # 7.2 MODEL FITTING / REMAINING DATA
    # --------------------------------------------------------

    (
        X_fit,
        X_temp,
        y_fit,
        y_temp
    ) = train_test_split(
        X_train,
        y_train,
        test_size=CALIBRATION_FRACTION,
        random_state=RANDOM_STATE,
        stratify=y_train
    )

    # --------------------------------------------------------
    # 7.3 CALIBRATION / THRESHOLD PARTITIONS
    # --------------------------------------------------------

    (
        X_calibration,
        X_threshold,
        y_calibration,
        y_threshold
    ) = train_test_split(
        X_temp,
        y_temp,
        test_size=THRESHOLD_FRACTION,
        random_state=RANDOM_STATE,
        stratify=y_temp
    )

    split_information = {
        "total_observations": len(X),
        "model_fitting": len(X_fit),
        "calibration": len(X_calibration),
        "threshold_selection": len(X_threshold),
        "test": len(X_test),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "stratified": True
    }

    print("\nPARTITION SIZES")

    for name, size in split_information.items():

        print(f"{name}: {size}")

    logger.info(
        "Data partitions recreated successfully."
    )

    logger.info(
        "Split information: %s",
        split_information
    )

    # --------------------------------------------------------
    # ASSERTIONS
    # --------------------------------------------------------

    assert len(X_fit) + len(X_calibration) + len(
        X_threshold
    ) + len(X_test) == len(X)

    assert len(
        set(X_test.index).intersection(
            set(X_fit.index)
        )
    ) == 0

    assert len(
        set(X_test.index).intersection(
            set(X_calibration.index)
        )
    ) == 0

    assert len(
        set(X_test.index).intersection(
            set(X_threshold.index)
        )
    ) == 0

    logger.info(
        "Partition independence checks passed."
    )

    return {
        "X_fit": X_fit,
        "y_fit": y_fit,
        "X_calibration": X_calibration,
        "y_calibration": y_calibration,
        "X_threshold": X_threshold,
        "y_threshold": y_threshold,
        "X_test": X_test,
        "y_test": y_test,
        "split_information": split_information
    }


# ============================================================
# 8. LOAD ORIGINAL MODEL
# ============================================================

def load_original_model():

    print_section(
        "LOADING ORIGINAL XGBOOST MODEL"
    )

    if not ORIGINAL_MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Original model not found: {ORIGINAL_MODEL_PATH}"
        )

    model = joblib.load(
        ORIGINAL_MODEL_PATH
    )

    if not hasattr(model, "predict_proba"):

        raise TypeError(
            "Original model does not support predict_proba."
        )

    logger.info(
        "Original model loaded successfully."
    )

    logger.info(
        "Model type: %s",
        type(model).__name__
    )

    return model


# ============================================================
# 9. LOAD CALIBRATED MODEL BUNDLE
# ============================================================

def load_calibrated_bundle():

    print_section(
        "LOADING CALIBRATED MODEL BUNDLE"
    )

    if not CALIBRATED_MODEL_PATH.exists():

        logger.warning(
            "Calibrated model artifact not found."
        )

        return None

    bundle = joblib.load(
        CALIBRATED_MODEL_PATH
    )

    if not isinstance(bundle, dict):

        logger.warning(
            "Calibrated artifact is not a dictionary."
        )

        return None

    required_keys = [
        "model",
        "calibrator"
    ]

    for key in required_keys:

        if key not in bundle:

            logger.warning(
                "Calibrated bundle missing key: %s",
                key
            )

            return None

    if not hasattr(
        bundle["model"],
        "predict_proba"
    ):

        logger.warning(
            "Calibrated model does not support predict_proba."
        )

        return None

    if not hasattr(
        bundle["calibrator"],
        "predict_proba"
    ):

        logger.warning(
            "Calibrator does not support predict_proba."
        )

        return None

    logger.info(
        "Calibrated bundle loaded successfully."
    )

    logger.warning(
        "Saved selected_threshold is: %s",
        bundle.get("selected_threshold")
    )

    logger.warning(
        "Threshold will be independently reconstructed "
        "using the threshold-selection partition."
    )

    return bundle


# ============================================================
# 10. VALIDATE FEATURE COMPATIBILITY
# ============================================================

def align_features(X, model, bundle=None):

    """
    Align dataset columns with the features expected
    by the saved model.
    """

    expected_features = None

    if bundle is not None:

        expected_features = bundle.get(
            "feature_columns"
        )

    if not expected_features:

        preprocessor = None

        if hasattr(model, "named_steps"):

            for step in model.named_steps.values():

                if hasattr(
                    step,
                    "feature_names_in_"
                ):

                    preprocessor = step
                    break

        if preprocessor is not None:

            expected_features = list(
                preprocessor.feature_names_in_
            )

    if expected_features is None:

        if hasattr(model, "feature_names_in_"):

            expected_features = list(
                model.feature_names_in_
            )

    if expected_features is None:

        raise ValueError(
            "Unable to determine model input feature columns."
        )

    missing_features = [
        feature
        for feature in expected_features
        if feature not in X.columns
    ]

    if missing_features:

        raise ValueError(
            "Missing model input features: "
            + str(missing_features)
        )

    X_aligned = X.loc[
        :,
        expected_features
    ].copy()

    logger.info(
        "Feature compatibility confirmed."
    )

    logger.info(
        "Model input feature count: %s",
        len(expected_features)
    )

    return X_aligned


# ============================================================
# 11. GET POSITIVE-CLASS PROBABILITIES
# ============================================================

def get_positive_class_probabilities(
    model,
    X
):

    """
    Extract probabilities for class 1 = Charged Off.
    """

    probabilities = model.predict_proba(X)

    classes = list(model.classes_)

    if 1 not in classes:

        raise ValueError(
            "Positive class 1 is missing from model classes."
        )

    positive_index = classes.index(1)

    return safe_probability(
        probabilities[:, positive_index]
    )


# ============================================================
# 12. PLATT CALIBRATION
# ============================================================

def get_calibrated_probabilities(
    bundle,
    X
):

    """
    Apply the saved LogisticRegression calibrator.

    The original calibration workflow applies Platt scaling
    to the logit of the underlying model probabilities.
    """

    base_model = bundle["model"]

    calibrator = bundle["calibrator"]

    raw_probabilities = (
        get_positive_class_probabilities(
            base_model,
            X
        )
    )

    raw_logits = logit(
        safe_probability(raw_probabilities)
    )

    calibrated_matrix = calibrator.predict_proba(
        raw_logits.reshape(-1, 1)
    )

    calibrator_classes = list(
        calibrator.classes_
    )

    if 1 not in calibrator_classes:

        raise ValueError(
            "Calibrator does not contain positive class 1."
        )

    positive_index = calibrator_classes.index(1)

    return safe_probability(
        calibrated_matrix[:, positive_index]
    )


# ============================================================
# 13. SELECT THRESHOLD USING F2
# ============================================================

def select_f2_threshold(
    y_true,
    probabilities
):

    """
    Select a threshold using the threshold-selection partition.

    The final test set is never used for threshold selection.
    """

    precision, recall, thresholds = (
        precision_recall_curve(
            y_true,
            probabilities
        )
    )

    if len(thresholds) == 0:

        raise ValueError(
            "Unable to generate candidate thresholds."
        )

    precision = precision[:-1]

    recall = recall[:-1]

    f2_values = (
        (1 + BETA ** 2)
        * precision
        * recall
        / (
            (BETA ** 2 * precision)
            + recall
            + 1e-12
        )
    )

    best_index = int(
        np.nanargmax(f2_values)
    )

    best_threshold = float(
        thresholds[best_index]
    )

    logger.info(
        "F2 threshold selected: %.6f",
        best_threshold
    )

    logger.info(
        "Threshold-selection F2: %.6f",
        f2_values[best_index]
    )

    return best_threshold, float(
        f2_values[best_index]
    )


# ============================================================
# 14. CALCULATE CLASSIFICATION METRICS
# ============================================================

def calculate_metrics(
    y_true,
    probabilities,
    threshold,
    model_name
):

    probabilities = safe_probability(
        probabilities
    )

    predictions = (
        probabilities >= threshold
    ).astype(int)

    metrics = {
        "model": model_name,
        "threshold": float(threshold),
        "accuracy": float(
            accuracy_score(
                y_true,
                predictions
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_true,
                probabilities
            )
        ),
        "pr_auc": float(
            average_precision_score(
                y_true,
                probabilities
            )
        ),
        "precision_default": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0
            )
        ),
        "recall_default": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0
            )
        ),
        "f2": float(
            fbeta_score(
                y_true,
                predictions,
                beta=BETA,
                zero_division=0
            )
        ),
        "brier_score": float(
            brier_score_loss(
                y_true,
                probabilities
            )
        ),
        "log_loss": float(
            log_loss(
                y_true,
                probabilities,
                labels=[0, 1]
            )
        )
    }

    matrix = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    )

    tn, fp, fn, tp = matrix.ravel()

    metrics.update({
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp)
    })

    metrics["specificity"] = float(
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
    )

    metrics["default_prevalence"] = float(
        np.mean(y_true)
    )

    return metrics, predictions, matrix


# ============================================================
# 15. SAVE CLASSIFICATION REPORT
# ============================================================

def save_classification_report(
    y_true,
    predictions,
    model_name
):

    report = classification_report(
        y_true,
        predictions,
        labels=[0, 1],
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(
        report
    ).transpose()

    filename = (
        model_name.lower().replace(" ", "_")
        + "_classification_report.csv"
    )

    if SAVE_CSV:

        report_df.to_csv(
            PERFORMANCE_DIR / filename
        )

    return report_df


# ============================================================
# 16. SAVE CONFUSION MATRIX
# ============================================================

def save_confusion_matrix(
    matrix,
    model_name
):

    matrix_df = pd.DataFrame(
        matrix,
        index=CLASS_NAMES,
        columns=CLASS_NAMES
    )

    matrix_df.index.name = "Actual"

    matrix_df.columns.name = "Predicted"

    if SAVE_CSV:

        filename = (
            model_name.lower().replace(" ", "_")
            + "_confusion_matrix.csv"
        )

        matrix_df.to_csv(
            PERFORMANCE_DIR / filename
        )

    return matrix_df


# ============================================================
# 17. GENERATE ROC AND PRECISION-RECALL PLOTS
# ============================================================

def generate_curve_plots(
    y_true,
    model_results
):

    if not SAVE_PLOTS:
        return

    # --------------------------------------------------------
    # ROC CURVE
    # --------------------------------------------------------

    plt.figure(figsize=(9, 7))

    for name, result in model_results.items():

        fpr, tpr, _ = roc_curve(
            y_true,
            result["probabilities"]
        )

        auc_value = roc_auc_score(
            y_true,
            result["probabilities"]
        )

        plt.plot(
            fpr,
            tpr,
            label=f"{name} (AUC={auc_value:.4f})"
        )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random classifier"
    )

    plt.xlabel("False Positive Rate")

    plt.ylabel("True Positive Rate")

    plt.title(
        "Credit Risk Model ROC Comparison"
    )

    plt.legend()

    plt.grid(alpha=0.25)

    plt.tight_layout()

    plt.savefig(
        PERFORMANCE_DIR / "roc_curve_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # PRECISION-RECALL CURVE
    # --------------------------------------------------------

    plt.figure(figsize=(9, 7))

    for name, result in model_results.items():

        precision, recall, _ = (
            precision_recall_curve(
                y_true,
                result["probabilities"]
            )
        )

        auc_value = average_precision_score(
            y_true,
            result["probabilities"]
        )

        plt.plot(
            recall,
            precision,
            label=f"{name} (PR-AUC={auc_value:.4f})"
        )

    prevalence = np.mean(y_true)

    plt.axhline(
        prevalence,
        linestyle="--",
        label=f"Default prevalence={prevalence:.4f}"
    )

    plt.xlabel("Recall")

    plt.ylabel("Precision")

    plt.title(
        "Credit Risk Model Precision-Recall Comparison"
    )

    plt.legend()

    plt.grid(alpha=0.25)

    plt.tight_layout()

    plt.savefig(
        PERFORMANCE_DIR
        / "precision_recall_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 18. GENERATE CONFUSION MATRIX PLOTS
# ============================================================

def generate_confusion_plots(
    model_results
):

    if not SAVE_PLOTS:
        return

    for name, result in model_results.items():

        plt.figure(figsize=(8, 6))

        sns.heatmap(
            result["confusion_matrix"],
            annot=True,
            fmt=",d",
            cmap="Blues",
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES
        )

        plt.xlabel("Predicted Class")

        plt.ylabel("Actual Class")

        plt.title(
            f"{name} - Confusion Matrix"
        )

        plt.tight_layout()

        filename = (
            name.lower().replace(" ", "_")
            + "_confusion_matrix.png"
        )

        plt.savefig(
            PERFORMANCE_DIR / filename,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()


# ============================================================
# 19. GENERATE CALIBRATION PLOTS
# ============================================================

def generate_calibration_plots(
    y_true,
    model_results
):

    if not SAVE_PLOTS:
        return

    plt.figure(figsize=(9, 7))

    for name, result in model_results.items():

        fraction_positive, mean_predicted = (
            calibration_curve(
                y_true,
                result["probabilities"],
                n_bins=CALIBRATION_BINS,
                strategy="quantile"
            )
        )

        brier = brier_score_loss(
            y_true,
            result["probabilities"]
        )

        plt.plot(
            mean_predicted,
            fraction_positive,
            marker="o",
            label=f"{name} (Brier={brier:.4f})"
        )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect calibration"
    )

    plt.xlabel("Mean Predicted Probability")

    plt.ylabel("Observed Default Frequency")

    plt.title(
        "Credit Risk Probability Calibration"
    )

    plt.legend()

    plt.grid(alpha=0.25)

    plt.tight_layout()

    plt.savefig(
        PERFORMANCE_DIR / "calibration_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 20. CREATE MODEL COMPARISON REPORT
# ============================================================

def create_model_comparison(
    model_results
):

    comparison = pd.DataFrame([
        result["metrics"]
        for result in model_results.values()
    ])

    comparison = comparison.set_index(
        "model"
    )

    comparison = comparison.round(6)

    if SAVE_CSV:

        comparison.to_csv(
            PERFORMANCE_DIR / "model_comparison.csv"
        )

    print_section(
        "HELD-OUT TEST PERFORMANCE COMPARISON"
    )

    print(
        comparison.to_string()
    )

    return comparison


# ============================================================
# 21. CREATE VALIDATION REPORT
# ============================================================

def create_validation_report(
    comparison,
    split_information,
    threshold_information,
    model_results
):

    report_path = (
        PERFORMANCE_DIR
        / "validation_report.md"
    )

    lines = []

    lines.append(
        "# Credit Risk Intelligence"
    )

    lines.append(
        "## Milestone 10.1: Out-of-Sample Model Validation"
    )

    lines.append("")

    lines.append(
        "### 1. Validation methodology"
    )

    lines.append("")

    lines.append(
        "- Dataset: LendingClub-style loan dataset."
    )

    lines.append(
        "- Target: loan_status."
    )

    lines.append(
        "- Positive class: Charged Off (1)."
    )

    lines.append(
        "- Negative class: Fully Paid (0)."
    )

    lines.append(
        "- Random state: 42."
    )

    lines.append(
        "- Test proportion: 20%."
    )

    lines.append(
        "- Stratified train/test split."
    )

    lines.append(
        "- Test set excluded from threshold selection."
    )

    lines.append("")

    lines.append(
        "### 2. Data partition sizes"
    )

    lines.append("")

    for name, value in split_information.items():

        lines.append(
            f"- {name}: {value}"
        )

    lines.append("")

    lines.append(
        "### 3. Model performance"
    )

    lines.append("")

    lines.append(
        comparison.to_markdown()
    )

    lines.append("")

    lines.append(
        "### 4. Threshold selection"
    )

    lines.append("")

    for name, value in threshold_information.items():

        lines.append(
            f"- {name}: {value}"
        )

    lines.append("")

    lines.append(
        "### 5. Validation observations"
    )

    lines.append("")

    lines.append(
        "- ROC-AUC measures ranking discrimination."
    )

    lines.append(
        "- PR-AUC measures precision-recall performance "
        "for the default class."
    )

    lines.append(
        "- Brier score and log loss evaluate probability quality."
    )

    lines.append(
        "- Precision, recall and F-scores depend on the "
        "selected decision threshold."
    )

    lines.append(
        "- Lower Brier score and log loss indicate better "
        "probability predictions."
    )

    lines.append("")

    lines.append(
        "### 6. Important limitations"
    )

    lines.append("")

    lines.append(
        "- This evaluation does not establish causal relationships."
    )

    lines.append(
        "- This evaluation does not establish fairness."
    )

    lines.append(
        "- This evaluation does not estimate actual realized "
        "credit losses."
    )

    lines.append(
        "- The dataset does not provide sufficient recorded "
        "recovery information to calculate actual loss given default."
    )

    lines.append(
        "- Performance on this historical test partition does "
        "not guarantee future performance."
    )

    lines.append(
        "- Calibration artifact provenance must be verified "
        "against the original Milestone 7.0 run."
    )

    lines.append("")

    lines.append(
        "### 7. Output files"
    )

    lines.append("")

    for path in sorted(
        PERFORMANCE_DIR.iterdir()
    ):

        if path.is_file() and path.name != report_path.name:

            lines.append(
                f"- `{path.name}`"
            )

    lines.append("")

    lines.append(
        "End of Milestone 10.1 validation report."
    )

    report_path.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )

    logger.info(
        "Validation report saved: %s",
        report_path
    )


# ============================================================
# 22. MAIN VALIDATION WORKFLOW
# ============================================================

def main():

    print_section(
        "CREDIT RISK INTELLIGENCE"
    )

    print(
        "MILESTONE 10.1: OUT-OF-SAMPLE MODEL VALIDATION"
    )

    logger.info(
        "Starting model validation workflow."
    )

    # --------------------------------------------------------
    # STEP 1: LOAD DATA
    # --------------------------------------------------------

    X, y = load_and_prepare_data()

    # --------------------------------------------------------
    # STEP 2: RECREATE DATA PARTITIONS
    # --------------------------------------------------------

    splits = create_data_splits(
        X,
        y
    )

    X_test = splits["X_test"]

    y_test = splits["y_test"]

    X_threshold = splits["X_threshold"]

    y_threshold = splits["y_threshold"]

    # --------------------------------------------------------
    # STEP 3: LOAD ORIGINAL MODEL
    # --------------------------------------------------------

    original_model = load_original_model()

    X_test_original = align_features(
        X_test,
        original_model
    )

    # --------------------------------------------------------
    # STEP 4: ORIGINAL MODEL PROBABILITIES
    # --------------------------------------------------------

    original_probabilities = (
        get_positive_class_probabilities(
            original_model,
            X_test_original
        )
    )

    # --------------------------------------------------------
    # STEP 5: ORIGINAL MODEL EVALUATION
    # --------------------------------------------------------

    original_metrics, original_predictions, original_matrix = (
        calculate_metrics(
            y_test,
            original_probabilities,
            threshold=0.50,
            model_name="Original XGBoost"
        )
    )

    original_report = save_classification_report(
        y_test,
        original_predictions,
        "original_xgboost"
    )

    original_matrix_df = save_confusion_matrix(
        original_matrix,
        "original_xgboost"
    )

    model_results = {
        "Original XGBoost": {
            "metrics": original_metrics,
            "probabilities": original_probabilities,
            "predictions": original_predictions,
            "confusion_matrix": original_matrix
        }
    }

    threshold_information = {
        "Original XGBoost": 0.50
    }

    # --------------------------------------------------------
    # STEP 6: LOAD CALIBRATED MODEL BUNDLE
    # --------------------------------------------------------

    calibrated_bundle = load_calibrated_bundle()

    if calibrated_bundle is not None:

        calibrated_model = calibrated_bundle["model"]

        X_test_calibrated = align_features(
            X_test,
            calibrated_model,
            calibrated_bundle
        )

        X_threshold_calibrated = align_features(
            X_threshold,
            calibrated_model,
            calibrated_bundle
        )

        # ----------------------------------------------------
        # CALIBRATION ARTIFACT FEATURE CHECK
        # ----------------------------------------------------

        artifact_features = calibrated_bundle.get(
            "feature_columns"
        )

        if artifact_features:

            if list(artifact_features) != list(
                X_test_calibrated.columns
            ):

                raise ValueError(
                    "Calibrated artifact feature order "
                    "does not match validation features."
                )

        # ----------------------------------------------------
        # THRESHOLD PARTITION CALIBRATED PROBABILITIES
        # ----------------------------------------------------

        threshold_probabilities = (
            get_calibrated_probabilities(
                calibrated_bundle,
                X_threshold_calibrated
            )
        )

        # ----------------------------------------------------
        # RECONSTRUCT F2 THRESHOLD
        # ----------------------------------------------------

        selected_threshold, threshold_f2 = (
            select_f2_threshold(
                y_threshold,
                threshold_probabilities
            )
        )

        threshold_information[
            "Calibrated model F2 threshold"
        ] = selected_threshold

        threshold_information[
            "Threshold partition F2"
        ] = threshold_f2

        # ----------------------------------------------------
        # CALIBRATED TEST PROBABILITIES
        # ----------------------------------------------------

        calibrated_probabilities = (
            get_calibrated_probabilities(
                calibrated_bundle,
                X_test_calibrated
            )
        )

        # ----------------------------------------------------
        # CALIBRATED MODEL EVALUATION AT 0.50
        # ----------------------------------------------------

        (
            calibrated_default_metrics,
            calibrated_default_predictions,
            calibrated_default_matrix
        ) = calculate_metrics(
            y_test,
            calibrated_probabilities,
            threshold=0.50,
            model_name="Calibrated Model at 0.50"
        )

        model_results[
            "Calibrated Model at 0.50"
        ] = {
            "metrics": calibrated_default_metrics,
            "probabilities": calibrated_probabilities,
            "predictions": calibrated_default_predictions,
            "confusion_matrix": calibrated_default_matrix
        }

        save_classification_report(
            y_test,
            calibrated_default_predictions,
            "calibrated_model_at_050"
        )

        save_confusion_matrix(
            calibrated_default_matrix,
            "calibrated_model_at_050"
        )

        # ----------------------------------------------------
        # CALIBRATED MODEL AT F2 THRESHOLD
        # ----------------------------------------------------

        (
            calibrated_f2_metrics,
            calibrated_f2_predictions,
            calibrated_f2_matrix
        ) = calculate_metrics(
            y_test,
            calibrated_probabilities,
            threshold=selected_threshold,
            model_name="Calibrated Model at F2 Threshold"
        )

        model_results[
            "Calibrated Model at F2 Threshold"
        ] = {
            "metrics": calibrated_f2_metrics,
            "probabilities": calibrated_probabilities,
            "predictions": calibrated_f2_predictions,
            "confusion_matrix": calibrated_f2_matrix
        }

        save_classification_report(
            y_test,
            calibrated_f2_predictions,
            "calibrated_model_at_f2_threshold"
        )

        save_confusion_matrix(
            calibrated_f2_matrix,
            "calibrated_model_at_f2_threshold"
        )

        # ----------------------------------------------------
        # CALIBRATION ARTIFACT DIAGNOSTICS
        # ----------------------------------------------------

        artifact_threshold = calibrated_bundle.get(
            "selected_threshold"
        )

        logger.warning(
            "Saved artifact threshold: %s",
            artifact_threshold
        )

        logger.info(
            "Reconstructed F2 threshold: %.6f",
            selected_threshold
        )

        if artifact_threshold is None:

            logger.warning(
                "Saved threshold is None. "
                "Reconstructed threshold is used for evaluation."
            )

        else:

            logger.info(
                "Saved threshold exists. "
                "It is reported for comparison only."
            )

        # ----------------------------------------------------
        # CALIBRATED PROBABILITY SUMMARY
        # ----------------------------------------------------

        threshold_information[
            "Calibrated test mean probability"
        ] = float(
            np.mean(calibrated_probabilities)
        )

        threshold_information[
            "Calibrated test Brier score"
        ] = float(
            brier_score_loss(
                y_test,
                calibrated_probabilities
            )
        )

    else:

        logger.warning(
            "Calibrated model evaluation skipped."
        )

    # --------------------------------------------------------
    # STEP 7: MODEL COMPARISON
    # --------------------------------------------------------

    comparison = create_model_comparison(
        model_results
    )

    # --------------------------------------------------------
    # STEP 8: GENERATE PLOTS
    # --------------------------------------------------------

    generate_curve_plots(
        y_test,
        model_results
    )

    generate_confusion_plots(
        model_results
    )

    generate_calibration_plots(
        y_test,
        model_results
    )

    # --------------------------------------------------------
    # STEP 9: SAVE METRICS JSON
    # --------------------------------------------------------

    validation_metrics = {
        "project": "Credit Risk Intelligence",
        "milestone": "10.1",
        "validation_date": pd.Timestamp.now().isoformat(),
        "target": TARGET,
        "positive_class": "Charged Off",
        "negative_class": "Fully Paid",
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "beta": BETA,
        "split_information": splits[
            "split_information"
        ],
        "threshold_information": threshold_information,
        "models": {
            name: result["metrics"]
            for name, result in model_results.items()
        }
    }

    if SAVE_JSON:

        save_json(
            validation_metrics,
            PERFORMANCE_DIR
            / "validation_metrics.json"
        )

    # --------------------------------------------------------
    # STEP 10: SAVE VALIDATION REPORT
    # --------------------------------------------------------

    create_validation_report(
        comparison,
        splits["split_information"],
        threshold_information,
        model_results
    )

    # --------------------------------------------------------
    # STEP 11: FINAL SUMMARY
    # --------------------------------------------------------

    print_section(
        "MILESTONE 10.1 COMPLETED"
    )

    print(
        comparison.to_string()
    )

    print("\nREPORT DIRECTORY:")

    print(PERFORMANCE_DIR)

    logger.info(
        "Model validation workflow completed successfully."
    )


# ============================================================
# 23. EXECUTION
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except Exception:

        logger.exception(
            "Model validation workflow failed."
        )

        raise
