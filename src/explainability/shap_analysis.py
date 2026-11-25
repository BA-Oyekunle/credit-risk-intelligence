
"""
======================================================================
CREDIT RISK INTELLIGENCE
MILESTONE 9: SHAP MODEL EXPLAINABILITY
======================================================================

Model:
    XGBoost Classifier inside Scikit-learn Pipeline

Explainability:
    SHAP TreeExplainer

Outputs:
    1. Global SHAP feature importance
    2. Native XGBoost feature importance
    3. SHAP beeswarm summary plot
    4. SHAP global bar plot
    5. SHAP dependence plots
    6. Individual borrower waterfall explanations
    7. Prediction-versus-SHAP diagnostics
    8. Explainability report

Validation:
    - Positive-class SHAP output alignment
    - SHAP raw-margin reconstruction
    - Pipeline probability consistency

IMPORTANT:
    SHAP values explain the XGBoost raw model margin.
    They are not calibrated default probabilities.

    This script explains best_model.pkl, not the
    separately calibrated model bundle.
======================================================================
"""

import logging
import re
import warnings

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from scipy.special import expit
from sklearn.model_selection import train_test_split


# ==============================================================
# 1. PROJECT CONFIGURATION
# ==============================================================

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

# NEW OUTPUT DIRECTORY.
# Previous Milestone 9 charts remain untouched.

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "explainability_validated"
)

INDIVIDUAL_DIR = (
    REPORT_DIR
    / "individual_explanations"
)

LOG_DIR = PROJECT_ROOT / "logs"

RANDOM_STATE = 42

MAX_EXPLANATION_ROWS = 5000

TOP_FEATURES = 20

SHAP_MARGIN_TOLERANCE = 1e-3

PROBABILITY_TOLERANCE = 1e-4


# Create directories.

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

INDIVIDUAL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

LOG_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==============================================================
# 2. LOGGING CONFIGURATION
# ==============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(
            LOG_DIR / "shap_analysis_validated.log",
            encoding="utf-8"
        ),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

warnings.filterwarnings(
    "ignore",
    category=FutureWarning
)


# ==============================================================
# 3. SAFE PLOT SAVING
# ==============================================================

def save_current_plot(filename):

    """
    Save the current Matplotlib figure.

    Each plot is saved into the new validated output folder.
    """

    output_path = REPORT_DIR / filename

    if output_path.exists() and output_path.is_dir():

        raise IsADirectoryError(
            f"Output filename is already a directory: {output_path}"
        )

    if not output_path.parent.exists():

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

    fig = plt.gcf()

    try:

        fig.tight_layout()

        fig.savefig(
            output_path,
            format="png",
            dpi=300,
            bbox_inches="tight"
        )

        if not output_path.exists():

            raise IOError(
                f"Plot was not created: {output_path}"
            )

        if output_path.stat().st_size == 0:

            raise IOError(
                f"Plot file is empty: {output_path}"
            )

        logger.info(
            "Plot saved successfully: %s",
            output_path
        )

    finally:

        plt.close(fig)


# ==============================================================
# 4. LOAD SAVED MODEL
# ==============================================================

def load_model():

    logger.info(
        "Loading saved XGBoost pipeline."
    )

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    if not hasattr(model, "named_steps"):

        raise TypeError(
            "The saved model is not a Scikit-learn Pipeline."
        )

    if "preprocessor" not in model.named_steps:

        raise ValueError(
            "Pipeline does not contain a preprocessor step."
        )

    if "classifier" not in model.named_steps:

        raise ValueError(
            "Pipeline does not contain a classifier step."
        )

    preprocessor = model.named_steps["preprocessor"]

    classifier = model.named_steps["classifier"]

    logger.info(
        "Model loaded successfully."
    )

    logger.info(
        "Preprocessor: %s",
        type(preprocessor).__name__
    )

    logger.info(
        "Classifier: %s",
        type(classifier).__name__
    )

    logger.info(
        "Classifier classes: %s",
        classifier.classes_
    )

    return model, preprocessor, classifier


# ==============================================================
# 5. LOAD AND PREPARE DATASET
# ==============================================================

def load_and_prepare_data(preprocessor):

    logger.info(
        "Loading credit risk dataset."
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

    # ----------------------------------------------------------
    # TARGET ENCODING
    # ----------------------------------------------------------

    target_mapping = {
        "Fully Paid": 0,
        "Charged Off": 1
    }

    df["loan_status"] = df["loan_status"].map(
        target_mapping
    )

    df = df.dropna(
        subset=["loan_status"]
    )

    df["loan_status"] = df[
        "loan_status"
    ].astype(int)

    # ----------------------------------------------------------
    # FEATURE ENGINEERING
    # ----------------------------------------------------------

    if "earliest_cr_line" in df.columns:

        earliest_date = pd.to_datetime(
            df["earliest_cr_line"],
            format="%b-%Y",
            errors="coerce"
        )

        df["earliest_cr_year"] = (
            earliest_date.dt.year
        )

    # ----------------------------------------------------------
    # RECOVER ORIGINAL MODEL INPUT FEATURES
    # ----------------------------------------------------------

    numeric_columns = []

    categorical_columns = []

    for transformer_name, transformer, columns in (
        preprocessor.transformers_
    ):

        if transformer_name == "remainder":

            continue

        if isinstance(columns, str):

            columns = [columns]

        columns = list(columns)

        if transformer_name == "numeric":

            numeric_columns.extend(columns)

        elif transformer_name == "categorical":

            categorical_columns.extend(columns)

    feature_columns = (
        numeric_columns
        + categorical_columns
    )

    missing_columns = [
        column
        for column in feature_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"Missing required model features: {missing_columns}"
        )

    X = df[feature_columns].copy()

    y = df["loan_status"].copy()

    logger.info(
        "Numerical input features: %s",
        len(numeric_columns)
    )

    logger.info(
        "Categorical input features: %s",
        len(categorical_columns)
    )

    logger.info(
        "Total original input features: %s",
        len(feature_columns)
    )

    logger.info(
        "Default rate: %.4f",
        y.mean()
    )

    return X, y, feature_columns


# ==============================================================
# 6. SELECT STRATIFIED EXPLANATION SAMPLE
# ==============================================================

def select_explanation_sample(X, y):

    logger.info(
        "Selecting stratified explanation sample."
    )

    if len(X) > MAX_EXPLANATION_ROWS:

        X_sample, _, y_sample, _ = train_test_split(
            X,
            y,
            train_size=MAX_EXPLANATION_ROWS,
            stratify=y,
            random_state=RANDOM_STATE
        )

    else:

        X_sample = X.copy()

        y_sample = y.copy()

    logger.info(
        "Explanation sample size: %s",
        len(X_sample)
    )

    return X_sample, y_sample


# ==============================================================
# 7. TRANSFORM FEATURES
# ==============================================================

def transform_features(preprocessor, X):

    logger.info(
        "Transforming input features."
    )

    transformed = preprocessor.transform(X)

    if hasattr(transformed, "toarray"):

        transformed = transformed.toarray()

    feature_names = (
        preprocessor.get_feature_names_out()
    )

    if transformed.shape[1] != len(feature_names):

        raise ValueError(
            "Transformed feature count does not match "
            "the feature-name count."
        )

    transformed_df = pd.DataFrame(
        transformed,
        columns=feature_names,
        index=X.index
    )

    logger.info(
        "Transformed feature dimensions: %s",
        transformed_df.shape
    )

    return transformed_df


# ==============================================================
# 8. EXTRACT SHAP OUTPUT CANDIDATES
# ==============================================================

def extract_shap_candidates(
    shap_output,
    expected_value,
    n_rows,
    n_features
):

    """
    Normalize SHAP output formats across SHAP versions.

    Supports:
        - List of class-specific arrays
        - Two-dimensional arrays
        - Three-dimensional arrays with classes last
        - Three-dimensional arrays with classes first
    """

    if isinstance(shap_output, list):

        raw_candidates = [
            np.asarray(item)
            for item in shap_output
        ]

    else:

        values = np.asarray(shap_output)

        if values.ndim == 2:

            raw_candidates = [values]

        elif values.ndim == 3:

            if (
                values.shape[0] == n_rows
                and values.shape[1] == n_features
            ):

                raw_candidates = [
                    values[:, :, i]
                    for i in range(values.shape[2])
                ]

            elif (
                values.shape[1] == n_rows
                and values.shape[2] == n_features
            ):

                raw_candidates = [
                    values[i, :, :]
                    for i in range(values.shape[0])
                ]

            else:

                raise ValueError(
                    f"Unsupported SHAP shape: {values.shape}"
                )

        else:

            raise ValueError(
                f"Unsupported SHAP dimensions: {values.shape}"
            )

    expected_array = np.asarray(
        expected_value
    )

    candidates = []

    for i, candidate in enumerate(raw_candidates):

        if candidate.shape != (n_rows, n_features):

            raise ValueError(
                f"Unexpected candidate shape: {candidate.shape}"
            )

        if expected_array.ndim == 0:

            base_value = float(expected_array)

        else:

            expected_flat = expected_array.reshape(-1)

            if len(expected_flat) == 1:

                base_value = float(expected_flat[0])

            elif len(expected_flat) == len(raw_candidates):

                base_value = float(expected_flat[i])

            else:

                raise ValueError(
                    "Expected values cannot be aligned "
                    "with SHAP output candidates."
                )

        candidates.append(
            (candidate, base_value)
        )

    return candidates


# ==============================================================
# 9. CALCULATE AND VALIDATE SHAP VALUES
# ==============================================================

def calculate_shap_values(
    classifier,
    transformed_df
):

    logger.info(
        "Initializing SHAP TreeExplainer."
    )

    X_array = transformed_df.to_numpy(
        dtype=np.float32
    )

    feature_names = transformed_df.columns.tolist()

    n_rows, n_features = X_array.shape

    # ----------------------------------------------------------
    # VERIFY CLASS ORDER
    # ----------------------------------------------------------

    classes = np.asarray(
        classifier.classes_
    )

    if not np.array_equal(
        classes,
        np.array([0, 1])
    ):

        raise ValueError(
            "Expected classifier classes [0, 1]. "
            f"Received: {classes}"
        )

    # ----------------------------------------------------------
    # DIRECT XGBOOST RAW MARGINS
    # ----------------------------------------------------------

    logger.info(
        "Calculating XGBoost positive-class raw margins."
    )

    raw_margins = classifier.predict(
        X_array,
        output_margin=True
    )

    raw_margins = np.asarray(
        raw_margins
    ).reshape(-1)

    # ----------------------------------------------------------
    # TREE EXPLAINER
    # ----------------------------------------------------------

    explainer = shap.TreeExplainer(
        classifier,
        model_output="raw",
        feature_perturbation="tree_path_dependent"
    )

    logger.info(
        "Calculating SHAP values."
    )

    shap_output = explainer.shap_values(
        X_array,
        check_additivity=False
    )

    candidates = extract_shap_candidates(
        shap_output,
        explainer.expected_value,
        n_rows,
        n_features
    )

    # ----------------------------------------------------------
    # SELECT POSITIVE-CLASS OUTPUT BY RAW-MARGIN AGREEMENT
    # ----------------------------------------------------------

    best_candidate = None

    best_base_value = None

    best_error = np.inf

    for candidate_values, base_value in candidates:

        reconstructed = (
            base_value
            + candidate_values.sum(axis=1)
        )

        error = np.max(
            np.abs(
                raw_margins - reconstructed
            )
        )

        if error < best_error:

            best_error = error

            best_candidate = candidate_values

            best_base_value = base_value

    if best_candidate is None:

        raise RuntimeError(
            "No valid SHAP output candidate was found."
        )

    logger.info(
        "Maximum SHAP reconstruction error: %.10f",
        best_error
    )

    if best_error > SHAP_MARGIN_TOLERANCE:

        raise ValueError(
            "SHAP reconstruction validation failed. "
            f"Maximum error: {best_error:.8f}"
        )

    # ----------------------------------------------------------
    # CREATE EXPLANATION OBJECT
    # ----------------------------------------------------------

    explanation = shap.Explanation(
        values=best_candidate,
        base_values=np.full(
            n_rows,
            best_base_value
        ),
        data=X_array,
        feature_names=feature_names
    )

    logger.info(
        "SHAP calculation and validation completed."
    )

    logger.info(
        "SHAP dimensions: %s",
        best_candidate.shape
    )

    logger.info(
        "SHAP base value: %.8f",
        best_base_value
    )

    return (
        explainer,
        explanation,
        raw_margins
    )


# ==============================================================
# 10. GLOBAL FEATURE IMPORTANCE
# ==============================================================

def generate_feature_importance(
    classifier,
    explanation,
    feature_names
):

    logger.info(
        "Generating global feature importance."
    )

    mean_absolute_shap = np.abs(
        explanation.values
    ).mean(axis=0)

    importance_df = pd.DataFrame({

        "feature": feature_names,

        "mean_absolute_shap": mean_absolute_shap

    })

    importance_df = importance_df.sort_values(
        by="mean_absolute_shap",
        ascending=False
    ).reset_index(drop=True)

    importance_df.insert(
        0,
        "rank",
        range(1, len(importance_df) + 1)
    )

    importance_df.to_csv(
        REPORT_DIR / "feature_importance.csv",
        index=False
    )

    # ----------------------------------------------------------
    # NATIVE XGBOOST IMPORTANCE
    # ----------------------------------------------------------

    native_importance = (
        classifier.feature_importances_
    )

    if len(native_importance) != len(feature_names):

        raise ValueError(
            "Native XGBoost importance does not match "
            "the transformed feature count."
        )

    native_df = pd.DataFrame({

        "feature": feature_names,

        "xgboost_importance": native_importance

    })

    native_df = native_df.sort_values(
        by="xgboost_importance",
        ascending=False
    ).reset_index(drop=True)

    native_df.insert(
        0,
        "rank",
        range(1, len(native_df) + 1)
    )

    native_df.to_csv(
        REPORT_DIR / "xgboost_feature_importance.csv",
        index=False
    )

    print("\n" + "=" * 70)
    print("TOP 20 SHAP FEATURES")
    print("=" * 70)

    print(
        importance_df.head(TOP_FEATURES).to_string(
            index=False
        )
    )

    print("\n" + "=" * 70)
    print("TOP 20 NATIVE XGBOOST FEATURES")
    print("=" * 70)

    print(
        native_df.head(TOP_FEATURES).to_string(
            index=False
        )
    )

    logger.info(
        "Feature importance reports saved."
    )

    return importance_df


# ==============================================================
# 11. SHAP SUMMARY PLOTS
# ==============================================================

def generate_summary_plots(explanation):

    logger.info(
        "Generating SHAP summary plots."
    )

    # ----------------------------------------------------------
    # BEESWARM SUMMARY
    # ----------------------------------------------------------

    try:

        plt.close("all")

        shap.plots.beeswarm(
            explanation,
            max_display=TOP_FEATURES,
            show=False
        )

        plt.title(
            "SHAP Summary: Credit Default Prediction"
        )

        save_current_plot(
            "shap_summary_validated.png"
        )

    except Exception:

        plt.close("all")

        logger.exception(
            "Failed to save SHAP beeswarm plot."
        )

        raise

    # ----------------------------------------------------------
    # GLOBAL BAR PLOT
    # ----------------------------------------------------------

    try:

        plt.close("all")

        shap.plots.bar(
            explanation,
            max_display=TOP_FEATURES,
            show=False
        )

        plt.title(
            "Global Feature Importance Using SHAP"
        )

        save_current_plot(
            "shap_bar_validated.png"
        )

    except Exception:

        plt.close("all")

        logger.exception(
            "Failed to save SHAP bar plot."
        )

        raise

    logger.info(
        "SHAP summary plots saved successfully."
    )


# ==============================================================
# 12. SHAP DEPENDENCE PLOTS
# ==============================================================

def generate_dependence_plots(
    explanation,
    importance_df
):

    logger.info(
        "Generating SHAP dependence plots."
    )

    top_features = (
        importance_df["feature"]
        .head(3)
        .tolist()
    )

    for position, feature in enumerate(
        top_features,
        start=1
    ):

        try:

            plt.close("all")

            shap.plots.scatter(
                explanation[:, feature],
                show=False
            )

            plt.title(
                f"SHAP Dependence: {feature}"
            )

            safe_name = re.sub(
                r'[<>:"/\\|?*]',
                "_",
                feature
            )

            safe_name = safe_name.strip(" .")

            filename = (
                f"dependence_{position}_{safe_name}_validated.png"
            )

            save_current_plot(filename)

        except Exception:

            plt.close("all")

            logger.exception(
                "Dependence plot failed for feature: %s",
                feature
            )

            raise

    logger.info(
        "SHAP dependence plots completed."
    )


# ==============================================================
# 13. INDIVIDUAL BORROWER EXPLANATIONS
# ==============================================================

def generate_individual_explanations(
    model,
    classifier,
    X_sample,
    y_sample,
    explanation,
    raw_margins
):

    logger.info(
        "Generating individual borrower explanations."
    )

    # ----------------------------------------------------------
    # PIPELINE DEFAULT PROBABILITIES
    # ----------------------------------------------------------

    probabilities = model.predict_proba(
        X_sample
    )[:, 1]

    # ----------------------------------------------------------
    # RAW MARGIN PROBABILITIES
    # ----------------------------------------------------------

    margin_probabilities = expit(
        raw_margins
    )

    # ----------------------------------------------------------
    # SHAP RECONSTRUCTION
    # ----------------------------------------------------------

    shap_reconstructed_margins = (
        explanation.base_values
        + explanation.values.sum(axis=1)
    )

    shap_probabilities = expit(
        shap_reconstructed_margins
    )

    # ----------------------------------------------------------
    # DIAGNOSTIC TABLE
    # ----------------------------------------------------------

    diagnostic_df = pd.DataFrame({

        "actual_default": y_sample.to_numpy(),

        "pipeline_probability": probabilities,

        "xgboost_raw_margin": raw_margins,

        "probability_from_raw_margin": margin_probabilities,

        "shap_reconstructed_margin": (
            shap_reconstructed_margins
        ),

        "probability_from_shap_margin": (
            shap_probabilities
        ),

        "margin_difference": (
            raw_margins - shap_reconstructed_margins
        ),

        "probability_difference": (
            probabilities - shap_probabilities
        )

    }, index=X_sample.index)

    diagnostic_path = (
        INDIVIDUAL_DIR
        / "shap_prediction_diagnostics.csv"
    )

    diagnostic_df.to_csv(
        diagnostic_path,
        index_label="row_index"
    )

    # ----------------------------------------------------------
    # CHECK ALIGNMENT
    # ----------------------------------------------------------

    max_margin_error = np.max(
        np.abs(
            diagnostic_df["margin_difference"]
        )
    )

    max_probability_error = np.max(
        np.abs(
            diagnostic_df["probability_difference"]
        )
    )

    print("\n" + "=" * 70)
    print("SHAP AND MODEL PREDICTION ALIGNMENT")
    print("=" * 70)

    print(
        "Classifier classes:",
        classifier.classes_
    )

    print(
        "Maximum raw-margin difference:",
        max_margin_error
    )

    print(
        "Maximum probability difference:",
        max_probability_error
    )

    if max_margin_error > SHAP_MARGIN_TOLERANCE:

        raise ValueError(
            "SHAP raw-margin validation failed."
        )

    if max_probability_error > PROBABILITY_TOLERANCE:

        raise ValueError(
            "Pipeline probabilities do not agree with "
            "SHAP-reconstructed probabilities. "
            "Individual plots have been stopped."
        )

    # ----------------------------------------------------------
    # SELECT HIGHEST AND LOWEST RISK BORROWERS
    # ----------------------------------------------------------

    high_risk_index = diagnostic_df[
        "pipeline_probability"
    ].idxmax()

    low_risk_index = diagnostic_df[
        "pipeline_probability"
    ].idxmin()

    selected_examples = {

        "highest_risk": high_risk_index,

        "lowest_risk": low_risk_index

    }

    # ----------------------------------------------------------
    # GENERATE WATERFALL PLOTS
    # ----------------------------------------------------------

    for label, row_index in selected_examples.items():

        row_position = X_sample.index.get_loc(
            row_index
        )

        individual_explanation = explanation[
            row_position
        ]

        borrower_diagnostic = diagnostic_df.loc[
            row_index
        ]

        print("\n" + "-" * 70)
        print(label.upper())
        print("-" * 70)

        print(
            "Actual default:",
            borrower_diagnostic["actual_default"]
        )

        print(
            "Pipeline probability:",
            borrower_diagnostic["pipeline_probability"]
        )

        print(
            "XGBoost raw margin:",
            borrower_diagnostic["xgboost_raw_margin"]
        )

        print(
            "Probability from raw margin:",
            borrower_diagnostic["probability_from_raw_margin"]
        )

        print(
            "SHAP reconstructed margin:",
            borrower_diagnostic["shap_reconstructed_margin"]
        )

        print(
            "Probability from SHAP:",
            borrower_diagnostic["probability_from_shap_margin"]
        )

        # ------------------------------------------------------
        # WATERFALL
        # ------------------------------------------------------

        plt.close("all")

        shap.plots.waterfall(
            individual_explanation,
            max_display=15,
            show=False
        )

        plt.title(
            f"Individual Borrower Explanation: {label}"
        )

        output_path = (
            INDIVIDUAL_DIR
            / f"{label}_waterfall_validated.png"
        )

        fig = plt.gcf()

        try:

            fig.tight_layout()

            fig.savefig(
                output_path,
                format="png",
                dpi=300,
                bbox_inches="tight"
            )

            if not output_path.exists():

                raise IOError(
                    f"Waterfall plot was not saved: {output_path}"
                )

            logger.info(
                "Waterfall saved: %s",
                output_path
            )

        finally:

            plt.close(fig)

        # ------------------------------------------------------
        # SAVE BORROWER INPUTS
        # ------------------------------------------------------

        borrower_record = X_sample.loc[
            row_index
        ].to_frame().T

        borrower_record["actual_default"] = (
            y_sample.loc[row_index]
        )

        borrower_record[
            "predicted_default_probability"
        ] = probabilities[row_position]

        borrower_record[
            "xgboost_raw_margin"
        ] = raw_margins[row_position]

        borrower_record[
            "shap_reconstructed_margin"
        ] = shap_reconstructed_margins[row_position]

        borrower_record.to_csv(
            INDIVIDUAL_DIR
            / f"{label}_borrower_validated.csv",
            index=False
        )

        logger.info(
            "%s borrower probability: %.6f",
            label,
            probabilities[row_position]
        )

    logger.info(
        "Individual explanations completed."
    )


# ==============================================================
# 14. GENERATE EXPLAINABILITY REPORT
# ==============================================================

def generate_report(
    importance_df,
    explanation,
    X_sample,
    raw_margins
):

    logger.info(
        "Generating explainability report."
    )

    report_path = (
        REPORT_DIR / "explainability_report.md"
    )

    top_features = importance_df.head(10)

    reconstructed_margins = (
        explanation.base_values
        + explanation.values.sum(axis=1)
    )

    max_error = np.max(
        np.abs(
            raw_margins - reconstructed_margins
        )
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "# CREDIT RISK INTELLIGENCE\n\n"
        )

        file.write(
            "## Milestone 9: Model Explainability Report\n\n"
        )

        file.write(
            "### Model Information\n\n"
        )

        file.write(
            "- Model: XGBoost Classifier\n"
        )

        file.write(
            "- Model container: Scikit-learn Pipeline\n"
        )

        file.write(
            "- Explainability: SHAP TreeExplainer\n"
        )

        file.write(
            "- Explained output: Positive-class raw margin\n"
        )

        file.write(
            f"- Explanation sample: {len(X_sample):,}\n"
        )

        file.write(
            f"- Transformed features: "
            f"{explanation.values.shape[1]}\n"
        )

        file.write(
            f"- Maximum SHAP reconstruction error: "
            f"{max_error:.10f}\n\n"
        )

        file.write(
            "### Top 10 SHAP Features\n\n"
        )

        file.write(
            "| Rank | Feature | Mean Absolute SHAP |\n"
        )

        file.write(
            "|---:|---|---:|\n"
        )

        for _, row in top_features.iterrows():

            file.write(
                f"| {int(row['rank'])} "
                f"| {row['feature']} "
                f"| {row['mean_absolute_shap']:.6f} |\n"
            )

        file.write(
            "\n### Interpretation\n\n"
        )

        file.write(
            "Mean absolute SHAP values measure the average "
            "magnitude of each feature's contribution to "
            "the model output. They do not indicate direction.\n\n"
        )

        file.write(
            "Positive SHAP values increase the positive-class "
            "raw model margin, while negative SHAP values "
            "decrease it.\n\n"
        )

        file.write(
            "The raw model margin is not a calibrated "
            "default probability.\n\n"
        )

        file.write(
            "The individual borrower explanations were "
            "validated against direct XGBoost raw margins "
            "and pipeline probabilities.\n\n"
        )

        file.write(
            "These explanations describe model behaviour "
            "and statistical associations. They do not "
            "establish causal relationships.\n\n"
        )

        file.write(
            "### Limitations\n\n"
        )

        file.write(
            "- The explanation sample is stratified and "
            "contains observations from the entire dataset.\n"
        )

        file.write(
            "- This is not a test-set-only explanation audit.\n"
        )

        file.write(
            "- Correlated features can share SHAP contributions.\n"
        )

        file.write(
            "- Feature importance can vary across samples.\n"
        )

        file.write(
            "- SHAP does not establish causality or fairness.\n"
        )

        file.write(
            "- This report explains best_model.pkl, not the "
            "separately calibrated model bundle.\n"
        )

    logger.info(
        "Explainability report saved: %s",
        report_path
    )


# ==============================================================
# 15. MAIN EXECUTION
# ==============================================================

def main():

    print("\n" + "=" * 70)

    print("CREDIT RISK INTELLIGENCE")

    print("MILESTONE 9: SHAP MODEL EXPLAINABILITY")

    print("=" * 70)

    # ----------------------------------------------------------
    # STEP 1: LOAD MODEL
    # ----------------------------------------------------------

    model, preprocessor, classifier = load_model()

    # ----------------------------------------------------------
    # STEP 2: LOAD DATA
    # ----------------------------------------------------------

    X, y, feature_columns = load_and_prepare_data(
        preprocessor
    )

    # ----------------------------------------------------------
    # STEP 3: SELECT SAMPLE
    # ----------------------------------------------------------

    X_sample, y_sample = select_explanation_sample(
        X,
        y
    )

    # ----------------------------------------------------------
    # STEP 4: TRANSFORM FEATURES
    # ----------------------------------------------------------

    transformed_df = transform_features(
        preprocessor,
        X_sample
    )

    # ----------------------------------------------------------
    # STEP 5: CALCULATE SHAP
    # ----------------------------------------------------------

    (
        explainer,
        explanation,
        raw_margins
    ) = calculate_shap_values(
        classifier,
        transformed_df
    )

    # ----------------------------------------------------------
    # STEP 6: GLOBAL FEATURE IMPORTANCE
    # ----------------------------------------------------------

    importance_df = generate_feature_importance(
        classifier,
        explanation,
        transformed_df.columns.tolist()
    )

    # ----------------------------------------------------------
    # STEP 7: SUMMARY PLOTS
    # ----------------------------------------------------------

    generate_summary_plots(
        explanation
    )

    # ----------------------------------------------------------
    # STEP 8: DEPENDENCE PLOTS
    # ----------------------------------------------------------

    generate_dependence_plots(
        explanation,
        importance_df
    )

    # ----------------------------------------------------------
    # STEP 9: INDIVIDUAL EXPLANATIONS
    # ----------------------------------------------------------

    generate_individual_explanations(
        model=model,
        classifier=classifier,
        X_sample=X_sample,
        y_sample=y_sample,
        explanation=explanation,
        raw_margins=raw_margins
    )

    # ----------------------------------------------------------
    # STEP 10: WRITTEN REPORT
    # ----------------------------------------------------------

    generate_report(
        importance_df=importance_df,
        explanation=explanation,
        X_sample=X_sample,
        raw_margins=raw_margins
    )

    # ----------------------------------------------------------
    # COMPLETION
    # ----------------------------------------------------------

    print("\n" + "=" * 70)

    print("MILESTONE 9 COMPLETED SUCCESSFULLY")

    print("=" * 70)

    print("\nValidated reports directory:")

    print(REPORT_DIR)

    print("\nGenerated files:")

    for file_path in sorted(
        REPORT_DIR.rglob("*")
    ):

        if file_path.is_file():

            print(
                " -",
                file_path.relative_to(REPORT_DIR)
            )

    logger.info(
        "SHAP explainability workflow completed successfully."
    )


# ==============================================================
# 16. EXECUTE SCRIPT
# ==============================================================

if __name__ == "__main__":

    try:

        main()

    except Exception:

        logger.exception(
            "SHAP explainability workflow failed."
        )

        raise


