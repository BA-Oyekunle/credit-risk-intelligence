# 1. IMPORT LIBRARIES
from pathlib import Path
import json
import logging
import warnings

import joblib
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import clone
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
from sklearn.linear_model import LogisticRegression


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

MODEL_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "best_model.pkl"
)

ARTIFACT_DIR = (
    PROJECT_ROOT
    / "artifacts"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
)

FIGURE_DIR = (
    REPORT_DIR
    / "figures"
)

# Automatically create directories if they do not exist.
ARTIFACT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# Target column
TARGET = "loan_status"


# Random seed
# MUST match the random_state used in advanced_models.py
RANDOM_STATE = 42


# Columns excluded in the original modelling workflow.
DROP_COLUMNS = [
    "emp_title",
    "title",
    "address",
    "issue_d",
    "grade"
]


# Possible identifier columns.
ID_COLUMNS = [
    "id",
    "loan_id",
    "customer_id",
    "application_id"
]


# Calibration settings.
CALIBRATION_METHOD = "platt_scaling"


# Threshold optimization metric.
THRESHOLD_METRIC = "F2"


# F-beta parameter.
BETA = 2


# Number of bins used in calibration plots.
CALIBRATION_BINS = 10


# ============================================================
# 3. LOGGING CONFIGURATION
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


warnings.filterwarnings(
    "ignore",
    category=FutureWarning
)


# ============================================================
# 4. LOAD AND PREPARE DATA
# ============================================================

def load_data():

    logger.info("=" * 60)
    logger.info("LOADING DATA")
    logger.info("=" * 60)

    if not DATA_PATH.exists():

        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    logger.info(
        "Original dataset shape: %s",
        df.shape
    )


    # --------------------------------------------------------
    # 4.1 CHECK TARGET
    # --------------------------------------------------------

    if TARGET not in df.columns:

        raise ValueError(
            f"Target column '{TARGET}' "
            "does not exist in the dataset."
        )


    # --------------------------------------------------------
    # 4.2 RECREATE EARLIEST CREDIT YEAR FEATURE
    # --------------------------------------------------------

    if "earliest_cr_year" not in df.columns:

        if "earliest_cr_line" not in df.columns:

            raise ValueError(
                "Feature engineering failed. "
                "Neither 'earliest_cr_year' nor "
                "'earliest_cr_line' exists."
            )

        logger.info(
            "Creating earliest_cr_year "
            "from earliest_cr_line..."
        )

        credit_dates = (
            df["earliest_cr_line"]
            .astype("string")
            .str.strip()
        )

        # First attempt: abbreviated month and two-digit year.
        parsed_dates = pd.to_datetime(
            credit_dates,
            format="%b-%y",
            errors="coerce"
        )

        # Second attempt: abbreviated month and four-digit year.
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

        # Third attempt: general datetime parsing.
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
            "earliest_cr_year successfully created."
        )

        logger.info(
            "Missing earliest_cr_year values: %s",
            df["earliest_cr_year"].isna().sum()
        )


    # --------------------------------------------------------
    # 4.3 PREPARE TARGET
    # --------------------------------------------------------

    if not pd.api.types.is_numeric_dtype(
        df[TARGET]
    ):

        target_mapping = {
            "Fully Paid": 0,
            "Charged Off": 1
        }

        df[TARGET] = df[TARGET].map(
            target_mapping
        )


    if df[TARGET].isna().any():

        raise ValueError(
            "The target contains missing values "
            "or unmapped target labels."
        )


    df[TARGET] = df[TARGET].astype(int)


    # Confirm target is binary.
    unique_target_values = set(
        df[TARGET].unique()
    )

    if not unique_target_values.issubset(
        {0, 1}
    ):

        raise ValueError(
            f"Unexpected target values: "
            f"{unique_target_values}"
        )


    logger.info(
        "Target distribution:\n%s",
        df[TARGET].value_counts()
    )


    logger.info(
        "Default prevalence: %.4f",
        df[TARGET].mean()
    )


    # --------------------------------------------------------
    # 4.4 CREATE FEATURE MATRIX
    # --------------------------------------------------------

    columns_to_drop = (
        [TARGET]
        + DROP_COLUMNS
        + ID_COLUMNS
    )

    X = df.drop(
        columns=columns_to_drop,
        errors="ignore"
    )

    y = df[TARGET]


    # --------------------------------------------------------
    # 4.5 VALIDATE REQUIRED FEATURE
    # --------------------------------------------------------

    if "earliest_cr_year" not in X.columns:

        raise ValueError(
            "earliest_cr_year is missing "
            "from the feature matrix."
        )


    logger.info(
        "Feature matrix shape: %s",
        X.shape
    )

    logger.info(
        "Number of features: %s",
        X.shape[1]
    )

    logger.info(
        "Features:\n%s",
        X.columns.tolist()
    )


    return X, y


# ============================================================
# 5. CREATE TRAINING, CALIBRATION, THRESHOLD AND TEST SETS
# ============================================================

def create_data_splits(X, y):

    logger.info("=" * 60)
    logger.info("CREATING DATA SPLITS")
    logger.info("=" * 60)


    # --------------------------------------------------------
    # 5.1 ORIGINAL TRAIN/TEST SPLIT
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=RANDOM_STATE,
            stratify=y
        )
    )


    logger.info(
        "Original training observations: %s",
        len(X_train)
    )

    logger.info(
        "Original test observations: %s",
        len(X_test)
    )


    # --------------------------------------------------------
    # 5.2 DIVIDE TRAINING DATA
    # --------------------------------------------------------

    # 70% of original training data:
    # model fitting
    #
    # 30% of original training data:
    # calibration and threshold selection

    X_fit, X_temp, y_fit, y_temp = (
        train_test_split(
            X_train,
            y_train,
            test_size=0.30,
            random_state=RANDOM_STATE,
            stratify=y_train
        )
    )


    # Split the remaining 30% equally:
    # 15% calibration
    # 15% threshold selection

    X_calibration, X_threshold, y_calibration, y_threshold = (
        train_test_split(
            X_temp,
            y_temp,
            test_size=0.50,
            random_state=RANDOM_STATE,
            stratify=y_temp
        )
    )


    # --------------------------------------------------------
    # 5.3 REPORT SPLIT SIZES
    # --------------------------------------------------------

    logger.info(
        "Model fitting set: %s",
        len(X_fit)
    )

    logger.info(
        "Calibration set: %s",
        len(X_calibration)
    )

    logger.info(
        "Threshold selection set: %s",
        len(X_threshold)
    )

    logger.info(
        "Final test set: %s",
        len(X_test)
    )


    # --------------------------------------------------------
    # 5.4 REPORT CLASS DISTRIBUTIONS
    # --------------------------------------------------------

    logger.info(
        "Model fitting default rate: %.4f",
        y_fit.mean()
    )

    logger.info(
        "Calibration default rate: %.4f",
        y_calibration.mean()
    )

    logger.info(
        "Threshold set default rate: %.4f",
        y_threshold.mean()
    )

    logger.info(
        "Test default rate: %.4f",
        y_test.mean()
    )


    return (
        X_fit,
        X_calibration,
        X_threshold,
        X_test,
        y_fit,
        y_calibration,
        y_threshold,
        y_test
    )


# ============================================================
# 6. LOAD AND REFIT SELECTED MODEL
# ============================================================

def train_selected_model(
    X_fit,
    y_fit
):

    logger.info("=" * 60)
    logger.info("REFITTING SELECTED MODEL")
    logger.info("=" * 60)


    # --------------------------------------------------------
    # 6.1 CHECK SAVED MODEL
    # --------------------------------------------------------

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            "The saved tuned model does not exist:\n"
            f"{MODEL_PATH}\n\n"
            "Run advanced_models.py first."
        )


    # --------------------------------------------------------
    # 6.2 LOAD SAVED MODEL
    # --------------------------------------------------------

    saved_model = joblib.load(
        MODEL_PATH
    )


    logger.info(
        "Saved model loaded successfully."
    )


    # --------------------------------------------------------
    # 6.3 CLONE THE MODEL
    # --------------------------------------------------------

    model = clone(
        saved_model
    )


    # --------------------------------------------------------
    # 6.4 VALIDATE FEATURES EXPECTED BY PIPELINE
    # --------------------------------------------------------

    if hasattr(
        model,
        "feature_names_in_"
    ):

        expected_features = list(
            model.feature_names_in_
        )

        missing_features = [
            feature
            for feature in expected_features
            if feature not in X_fit.columns
        ]

        if missing_features:

            raise ValueError(
                "The training data is missing "
                "features expected by the saved model:\n"
                f"{missing_features}"
            )


    # --------------------------------------------------------
    # 6.5 REFIT MODEL
    # --------------------------------------------------------

    logger.info(
        "Fitting model on model-fitting subset..."
    )

    model.fit(
        X_fit,
        y_fit
    )


    logger.info(
        "Model fitting completed successfully."
    )


    return model


# ============================================================
# 7. PROBABILITY CALIBRATION
# ============================================================

def logit_transform(
    probabilities
):

    probabilities = np.asarray(
        probabilities
    )

    # Avoid infinite log odds.
    probabilities = np.clip(
        probabilities,
        1e-6,
        1 - 1e-6
    )

    logits = np.log(
        probabilities
        /
        (1 - probabilities)
    )

    return logits.reshape(
        -1,
        1
    )


# ------------------------------------------------------------
# 7.1 FIT PLATT SCALING
# ------------------------------------------------------------

def fit_calibrator(
    model,
    X_calibration,
    y_calibration
):

    logger.info("=" * 60)
    logger.info("PROBABILITY CALIBRATION")
    logger.info("=" * 60)


    # --------------------------------------------------------
    # GET RAW MODEL PROBABILITIES
    # --------------------------------------------------------

    raw_probabilities = (
        model.predict_proba(
            X_calibration
        )[:, 1]
    )


    logger.info(
        "Raw calibration probabilities generated."
    )


    # --------------------------------------------------------
    # PLATT SCALING
    # --------------------------------------------------------

    calibrator = LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE
    )


    calibrator.fit(
        logit_transform(
            raw_probabilities
        ),
        y_calibration
    )


    # --------------------------------------------------------
    # GENERATE CALIBRATED PROBABILITIES
    # --------------------------------------------------------

    calibrated_probabilities = (
        calibrator.predict_proba(
            logit_transform(
                raw_probabilities
            )
        )[:, 1]
    )


    logger.info(
        "Platt scaling fitted successfully."
    )


    return (
        calibrator,
        raw_probabilities,
        calibrated_probabilities
    )


# ------------------------------------------------------------
# 7.2 APPLY CALIBRATOR
# ------------------------------------------------------------

def calibrate_probabilities(
    model,
    calibrator,
    X
):

    # Raw probabilities from the original model.
    raw_probabilities = (
        model.predict_proba(
            X
        )[:, 1]
    )


    # Convert raw probabilities to logits.
    raw_logits = logit_transform(
        raw_probabilities
    )


    # Apply calibration.
    calibrated_probabilities = (
        calibrator.predict_proba(
            raw_logits
        )[:, 1]
    )


    return (
        raw_probabilities,
        calibrated_probabilities
    )


# ============================================================
# 8. THRESHOLD OPTIMIZATION USING F2-SCORE
# ============================================================

def select_threshold(
    y_true,
    probabilities
):

    logger.info("=" * 60)
    logger.info("THRESHOLD OPTIMIZATION")
    logger.info("=" * 60)


    # --------------------------------------------------------
    # 8.1 COMPUTE PRECISION AND RECALL
    # --------------------------------------------------------

    precision_values, recall_values, thresholds = (
        precision_recall_curve(
            y_true,
            probabilities
        )
    )


    # Precision and recall have one additional element
    # compared with the threshold array.

    precision_values = (
        precision_values[:-1]
    )

    recall_values = (
        recall_values[:-1]
    )


    # --------------------------------------------------------
    # 8.2 COMPUTE F2-SCORE
    # --------------------------------------------------------

    beta = BETA

    numerator = (
        (1 + beta ** 2)
        * precision_values
        * recall_values
    )


    denominator = (
        (beta ** 2)
        * precision_values
        + recall_values
    )


    f2_scores = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(
            numerator,
            dtype=float
        ),
        where=denominator != 0
    )


    # --------------------------------------------------------
    # 8.3 SELECT BEST THRESHOLD
    # --------------------------------------------------------

    best_index = np.argmax(
        f2_scores
    )


    best_threshold = float(
        thresholds[best_index]
    )


    best_precision = float(
        precision_values[best_index]
    )


    best_recall = float(
        recall_values[best_index]
    )


    best_f2 = float(
        f2_scores[best_index]
    )


    # --------------------------------------------------------
    # 8.4 REPORT RESULTS
    # --------------------------------------------------------

    logger.info(
        "Selected threshold: %.6f",
        best_threshold
    )

    logger.info(
        "Threshold selection precision: %.4f",
        best_precision
    )

    logger.info(
        "Threshold selection recall: %.4f",
        best_recall
    )

    logger.info(
        "Threshold selection F2-score: %.4f",
        best_f2
    )


    threshold_result = {
        "threshold": best_threshold,
        "precision": best_precision,
        "recall": best_recall,
        "f2_score": best_f2,
        "selection_metric": "F2-score",
        "beta": BETA
    }


    return threshold_result


# ============================================================
# 9. EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    y_true,
    probabilities,
    threshold,
    model_name
):

    logger.info("=" * 60)
    logger.info(
        "EVALUATING: %s",
        model_name
    )
    logger.info("=" * 60)


    # --------------------------------------------------------
    # 9.1 GENERATE CLASS PREDICTIONS
    # --------------------------------------------------------

    predictions = (
        probabilities >= threshold
    ).astype(int)


    # --------------------------------------------------------
    # 9.2 COMPUTE METRICS
    # --------------------------------------------------------

    metrics = {

        "model": model_name,

        "threshold": float(
            threshold
        ),

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

        "f1_default": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0
            )
        ),

        "f2_default": float(
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


    # --------------------------------------------------------
    # 9.3 CONFUSION MATRIX
    # --------------------------------------------------------

    matrix = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    )


    # Matrix layout:
    #
    #                Predicted
    #              0          1
    # Actual  0    TN         FP
    #         1    FN         TP


    tn, fp, fn, tp = (
        matrix.ravel()
    )


    logger.info(
        "True Negatives: %s",
        tn
    )

    logger.info(
        "False Positives: %s",
        fp
    )

    logger.info(
        "False Negatives: %s",
        fn
    )

    logger.info(
        "True Positives: %s",
        tp
    )


    # --------------------------------------------------------
    # 9.4 CLASSIFICATION REPORT
    # --------------------------------------------------------

    report = classification_report(
        y_true,
        predictions,
        labels=[0, 1],
        target_names=[
            "Fully Paid",
            "Charged Off"
        ],
        zero_division=0
    )


    # --------------------------------------------------------
    # 9.5 DISPLAY RESULTS
    # --------------------------------------------------------

    logger.info(
        "\nMETRICS:\n%s",
        pd.Series(
            metrics
        ).to_string()
    )


    logger.info(
        "\nCONFUSION MATRIX:\n%s",
        matrix
    )


    logger.info(
        "\nCLASSIFICATION REPORT:\n%s",
        report
    )


    return (
        metrics,
        matrix,
        report,
        predictions
    )


# ============================================================
# 10. PROBABILITY METRIC COMPARISON
# ============================================================

def compare_probability_quality(
    y_true,
    raw_probabilities,
    calibrated_probabilities
):

    logger.info(
        "Comparing raw and calibrated probability quality..."
    )


    results = []


    # --------------------------------------------------------
    # RAW MODEL
    # --------------------------------------------------------

    raw_brier = brier_score_loss(
        y_true,
        raw_probabilities
    )

    raw_log_loss = log_loss(
        y_true,
        raw_probabilities,
        labels=[0, 1]
    )


    results.append({

        "model": "Uncalibrated XGBoost",

        "brier_score": float(
            raw_brier
        ),

        "log_loss": float(
            raw_log_loss
        )

    })


    # --------------------------------------------------------
    # CALIBRATED MODEL
    # --------------------------------------------------------

    calibrated_brier = brier_score_loss(
        y_true,
        calibrated_probabilities
    )

    calibrated_log_loss = log_loss(
        y_true,
        calibrated_probabilities,
        labels=[0, 1]
    )


    results.append({

        "model": "Calibrated XGBoost",

        "brier_score": float(
            calibrated_brier
        ),

        "log_loss": float(
            calibrated_log_loss
        )

    })


    comparison_df = pd.DataFrame(
        results
    )


    logger.info(
        "\nPROBABILITY QUALITY COMPARISON:\n%s",
        comparison_df.to_string(
            index=False
        )
    )


    return comparison_df


# ============================================================
# 11. CALIBRATION CURVE
# ============================================================

def plot_calibration_curve(
    y_true,
    raw_probabilities,
    calibrated_probabilities
):

    logger.info(
        "Creating calibration curve..."
    )


    # --------------------------------------------------------
    # RAW CALIBRATION
    # --------------------------------------------------------

    fraction_raw, mean_raw = (
        calibration_curve(
            y_true,
            raw_probabilities,
            n_bins=CALIBRATION_BINS,
            strategy="quantile"
        )
    )


    # --------------------------------------------------------
    # CALIBRATED CALIBRATION
    # --------------------------------------------------------

    fraction_calibrated, mean_calibrated = (
        calibration_curve(
            y_true,
            calibrated_probabilities,
            n_bins=CALIBRATION_BINS,
            strategy="quantile"
        )
    )


    # --------------------------------------------------------
    # PLOT
    # --------------------------------------------------------

    plt.figure(
        figsize=(9, 7)
    )


    plt.plot(
        mean_raw,
        fraction_raw,
        marker="o",
        linewidth=2,
        label="Before calibration"
    )


    plt.plot(
        mean_calibrated,
        fraction_calibrated,
        marker="s",
        linewidth=2,
        label="After calibration"
    )


    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="black",
        label="Perfect calibration"
    )


    plt.xlabel(
        "Mean predicted probability"
    )

    plt.ylabel(
        "Observed default frequency"
    )


    plt.title(
        "Probability Calibration Curve"
    )


    plt.legend()

    plt.grid(
        alpha=0.3
    )


    plt.tight_layout()


    plt.savefig(
        FIGURE_DIR
        / "calibration_curve.png",
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


    logger.info(
        "Calibration curve saved."
    )


# ============================================================
# 12. CONFUSION MATRIX VISUALIZATION
# ============================================================

def plot_confusion_matrix(
    matrix
):

    logger.info(
        "Creating confusion matrix plot..."
    )


    plt.figure(
        figsize=(8, 6)
    )


    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[
            "Fully Paid",
            "Charged Off"
        ],
        yticklabels=[
            "Fully Paid",
            "Charged Off"
        ]
    )


    plt.xlabel(
        "Predicted class"
    )


    plt.ylabel(
        "Actual class"
    )


    plt.title(
        "Calibrated XGBoost Confusion Matrix"
    )


    plt.tight_layout()


    plt.savefig(
        FIGURE_DIR
        / "calibrated_confusion_matrix.png",
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


    logger.info(
        "Confusion matrix plot saved."
    )


# ============================================================
# 13. ROC CURVE
# ============================================================

def plot_roc_curve(
    y_true,
    probabilities
):

    logger.info(
        "Creating ROC curve..."
    )


    fpr, tpr, thresholds = (
        roc_curve(
            y_true,
            probabilities
        )
    )


    roc_auc = roc_auc_score(
        y_true,
        probabilities
    )


    plt.figure(
        figsize=(8, 6)
    )


    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"ROC-AUC = {roc_auc:.4f}"
    )


    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="black"
    )


    plt.xlabel(
        "False Positive Rate"
    )


    plt.ylabel(
        "True Positive Rate"
    )


    plt.title(
        "Receiver Operating Characteristic Curve"
    )


    plt.legend()

    plt.grid(
        alpha=0.3
    )


    plt.tight_layout()


    plt.savefig(
        FIGURE_DIR
        / "calibrated_roc_curve.png",
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


    logger.info(
        "ROC curve saved."
    )


# ============================================================
# 14. PRECISION-RECALL CURVE
# ============================================================

def plot_precision_recall_curve(
    y_true,
    probabilities
):

    logger.info(
        "Creating Precision-Recall curve..."
    )


    precision_values, recall_values, thresholds = (
        precision_recall_curve(
            y_true,
            probabilities
        )
    )


    pr_auc = average_precision_score(
        y_true,
        probabilities
    )


    plt.figure(
        figsize=(8, 6)
    )


    plt.plot(
        recall_values,
        precision_values,
        linewidth=2,
        label=f"PR-AUC = {pr_auc:.4f}"
    )


    plt.xlabel(
        "Recall"
    )


    plt.ylabel(
        "Precision"
    )


    plt.title(
        "Precision-Recall Curve"
    )


    plt.legend()

    plt.grid(
        alpha=0.3
    )


    plt.tight_layout()


    plt.savefig(
        FIGURE_DIR
        / "calibrated_precision_recall_curve.png",
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


    logger.info(
        "Precision-Recall curve saved."
    )


# ============================================================
# 15. THRESHOLD PERFORMANCE CURVE
# ============================================================

def plot_threshold_performance(
    y_true,
    probabilities
):

    logger.info(
        "Creating threshold performance plot..."
    )


    thresholds = np.linspace(
        0.01,
        0.99,
        100
    )


    precision_list = []

    recall_list = []

    f1_list = []

    f2_list = []


    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)


        precision_list.append(
            precision_score(
                y_true,
                predictions,
                zero_division=0
            )
        )


        recall_list.append(
            recall_score(
                y_true,
                predictions,
                zero_division=0
            )
        )


        f1_list.append(
            f1_score(
                y_true,
                predictions,
                zero_division=0
            )
        )


        f2_list.append(
            fbeta_score(
                y_true,
                predictions,
                beta=BETA,
                zero_division=0
            )
        )


    plt.figure(
        figsize=(10, 7)
    )


    plt.plot(
        thresholds,
        precision_list,
        label="Precision"
    )


    plt.plot(
        thresholds,
        recall_list,
        label="Recall"
    )


    plt.plot(
        thresholds,
        f1_list,
        label="F1-score"
    )


    plt.plot(
        thresholds,
        f2_list,
        label="F2-score"
    )


    plt.xlabel(
        "Decision threshold"
    )


    plt.ylabel(
        "Metric value"
    )


    plt.title(
        "Threshold Optimization Performance"
    )


    plt.legend()

    plt.grid(
        alpha=0.3
    )


    plt.tight_layout()


    plt.savefig(
        FIGURE_DIR
        / "threshold_performance_curve.png",
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


    logger.info(
        "Threshold performance plot saved."
    )


# ============================================================
# 16. SAVE RESULTS
# ============================================================

def save_results(
    raw_metrics,
    calibrated_metrics,
    probability_comparison,
    threshold_result,
    matrix,
    report,
    bundle
):

    logger.info(
        "Saving Milestone 7 results..."
    )


    # --------------------------------------------------------
    # 16.1 MODEL METRICS
    # --------------------------------------------------------

    metrics_df = pd.DataFrame([
        raw_metrics,
        calibrated_metrics
    ])


    metrics_path = (
        REPORT_DIR
        / "milestone7_test_metrics.csv"
    )


    metrics_df.to_csv(
        metrics_path,
        index=False
    )


    # --------------------------------------------------------
    # 16.2 PROBABILITY COMPARISON
    # --------------------------------------------------------

    probability_path = (
        REPORT_DIR
        / "probability_calibration_comparison.csv"
    )


    probability_comparison.to_csv(
        probability_path,
        index=False
    )


    # --------------------------------------------------------
    # 16.3 THRESHOLD RESULTS
    # --------------------------------------------------------

    threshold_path = (
        REPORT_DIR
        / "selected_threshold.json"
    )


    with open(
        threshold_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            threshold_result,
            file,
            indent=4
        )


    # --------------------------------------------------------
    # 16.4 CONFUSION MATRIX
    # --------------------------------------------------------

    confusion_df = pd.DataFrame(
        matrix,
        index=[
            "Actual Fully Paid",
            "Actual Charged Off"
        ],
        columns=[
            "Predicted Fully Paid",
            "Predicted Charged Off"
        ]
    )


    confusion_df.to_csv(
        REPORT_DIR
        / "milestone7_confusion_matrix.csv"
    )


    # --------------------------------------------------------
    # 16.5 CLASSIFICATION REPORT
    # --------------------------------------------------------

    with open(
        REPORT_DIR
        / "milestone7_classification_report.txt",
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            report
        )


    # --------------------------------------------------------
    # 16.6 MODEL BUNDLE
    # --------------------------------------------------------

    model_bundle_path = (
        ARTIFACT_DIR
        / "calibrated_credit_risk_model.pkl"
    )


    joblib.dump(
        bundle,
        model_bundle_path
    )


    logger.info(
        "Saved metrics: %s",
        metrics_path
    )


    logger.info(
        "Saved probability comparison: %s",
        probability_path
    )


    logger.info(
        "Saved threshold: %s",
        threshold_path
    )


    logger.info(
        "Saved model bundle: %s",
        model_bundle_path
    )


# ============================================================
# 17. MAIN EXECUTION
# ============================================================

def main():

    logger.info("=" * 70)

    logger.info(
        "MILESTONE 7: EVALUATION, CALIBRATION "
        "AND THRESHOLD OPTIMIZATION"
    )

    logger.info("=" * 70)


    # --------------------------------------------------------
    # STEP 1: LOAD DATA
    # --------------------------------------------------------

    X, y = load_data()


    # --------------------------------------------------------
    # STEP 2: CREATE DATA SPLITS
    # --------------------------------------------------------

    (
        X_fit,
        X_calibration,
        X_threshold,
        X_test,
        y_fit,
        y_calibration,
        y_threshold,
        y_test
    ) = create_data_splits(
        X,
        y
    )


    # --------------------------------------------------------
    # STEP 3: REFIT SELECTED MODEL
    # --------------------------------------------------------

    model = train_selected_model(
        X_fit,
        y_fit
    )


    # --------------------------------------------------------
    # STEP 4: FIT CALIBRATOR
    # --------------------------------------------------------

    (
        calibrator,
        calibration_raw_probabilities,
        calibration_calibrated_probabilities
    ) = fit_calibrator(
        model,
        X_calibration,
        y_calibration
    )


    # --------------------------------------------------------
    # STEP 5: SELECT THRESHOLD
    # --------------------------------------------------------

    (
        threshold_raw_probabilities,
        threshold_calibrated_probabilities
    ) = calibrate_probabilities(
        model,
        calibrator,
        X_threshold
    )


    threshold_result = select_threshold(
        y_threshold,
        threshold_calibrated_probabilities
    )


    selected_threshold = (
        threshold_result["threshold"]
    )


    # --------------------------------------------------------
    # STEP 6: FINAL TEST PREDICTIONS
    # --------------------------------------------------------

    logger.info("=" * 70)

    logger.info(
        "FINAL TEST EVALUATION"
    )

    logger.info("=" * 70)


    (
        test_raw_probabilities,
        test_calibrated_probabilities
    ) = calibrate_probabilities(
        model,
        calibrator,
        X_test
    )


    # --------------------------------------------------------
    # STEP 7: EVALUATE RAW MODEL
    # --------------------------------------------------------

    (
        raw_metrics,
        raw_matrix,
        raw_report,
        raw_predictions
    ) = evaluate_model(
        y_test,
        test_raw_probabilities,
        threshold=0.50,
        model_name="Uncalibrated XGBoost - Threshold 0.50"
    )


    # --------------------------------------------------------
    # STEP 8: EVALUATE CALIBRATED MODEL
    # --------------------------------------------------------

    (
        calibrated_metrics,
        calibrated_matrix,
        calibrated_report,
        calibrated_predictions
    ) = evaluate_model(
        y_test,
        test_calibrated_probabilities,
        threshold=selected_threshold,
        model_name="Calibrated XGBoost - Optimized F2 Threshold"
    )


    # --------------------------------------------------------
    # STEP 9: COMPARE PROBABILITY QUALITY
    # --------------------------------------------------------

    probability_comparison = (
        compare_probability_quality(
            y_test,
            test_raw_probabilities,
            test_calibrated_probabilities
        )
    )


    # --------------------------------------------------------
    # STEP 10: CREATE VISUALIZATIONS
    # --------------------------------------------------------

    plot_calibration_curve(
        y_test,
        test_raw_probabilities,
        test_calibrated_probabilities
    )


    plot_confusion_matrix(
        calibrated_matrix
    )


    plot_roc_curve(
        y_test,
        test_calibrated_probabilities
    )


    plot_precision_recall_curve(
        y_test,
        test_calibrated_probabilities
    )


    plot_threshold_performance(
        y_test,
        test_calibrated_probabilities
    )


    # --------------------------------------------------------
    # STEP 11: SAVE CALIBRATED MODEL BUNDLE
    # --------------------------------------------------------

    bundle = {

        "model": model,

        "calibrator": calibrator,

        "threshold": selected_threshold,

        "threshold_selection_metrics": threshold_result,

        "feature_columns": list(
            X.columns
        ),

        "target_name": TARGET,

        "positive_class": 1,

        "negative_class": 0,

        "calibration_method": CALIBRATION_METHOD,

        "threshold_selection_method": (
            THRESHOLD_METRIC
        ),

        "random_state": RANDOM_STATE,

        "model_name": "Tuned XGBoost",

        "project_name": "Credit Risk Intelligence"

    }


    # --------------------------------------------------------
    # STEP 12: SAVE REPORTS
    # --------------------------------------------------------

    save_results(
        raw_metrics=raw_metrics,
        calibrated_metrics=calibrated_metrics,
        probability_comparison=probability_comparison,
        threshold_result=threshold_result,
        matrix=calibrated_matrix,
        report=calibrated_report,
        bundle=bundle
    )


    # --------------------------------------------------------
    # STEP 13: FINAL SUMMARY
    # --------------------------------------------------------

    logger.info("=" * 70)

    logger.info(
        "MILESTONE 7 COMPLETED SUCCESSFULLY"
    )

    logger.info("=" * 70)


    logger.info(
        "Selected threshold: %.6f",
        selected_threshold
    )


    logger.info(
        "Test ROC-AUC: %.4f",
        calibrated_metrics["roc_auc"]
    )


    logger.info(
        "Test PR-AUC: %.4f",
        calibrated_metrics["pr_auc"]
    )


    logger.info(
        "Test Precision: %.4f",
        calibrated_metrics["precision_default"]
    )


    logger.info(
        "Test Recall: %.4f",
        calibrated_metrics["recall_default"]
    )


    logger.info(
        "Test F1-score: %.4f",
        calibrated_metrics["f1_default"]
    )


    logger.info(
        "Test F2-score: %.4f",
        calibrated_metrics["f2_default"]
    )


    logger.info(
        "Test Brier Score: %.4f",
        calibrated_metrics["brier_score"]
    )


    logger.info(
        "Test Log Loss: %.4f",
        calibrated_metrics["log_loss"]
    )


    logger.info(
        "Model saved successfully."
    )


    logger.info(
        "Reports saved to reports/."
    )


    logger.info(
        "Figures saved to reports/figures/."
    )


    logger.info("=" * 70)


# ============================================================
# 18. SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
