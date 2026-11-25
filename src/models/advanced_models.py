# ============================================================
# CREDIT RISK INTELLIGENCE
# ADVANCED MODELING AND HYPERPARAMETER TUNING
# ============================================================

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBClassifier


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

try:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
except (NameError, IndexError):
    # Notebook or shallow folder: fall back to the working directory
    PROJECT_ROOT = Path.cwd()

# First existing file in this list is used
DATA_CANDIDATES = [
    PROJECT_ROOT / "data" / "raw" / "loan_data.csv",
    PROJECT_ROOT / "data" / "raw" / "lending_club_loan_two.csv",
    PROJECT_ROOT / "data" / "raw" / "credit_risk_dataset.csv",
]

ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
REPORT_DIR = PROJECT_ROOT / "reports"

ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "loan_status"

RANDOM_STATE = 42
TEST_SIZE = 0.20

# Tuning controls
SEARCH_SAMPLE_SIZE = 100_000   # rows used for the hyperparameter search
SEARCH_CV_FOLDS = 3
SEARCH_ITERATIONS = 12
THRESHOLD_CV_FOLDS = 3         # folds used to choose the decision threshold

# High-cardinality text, raw dates and redundant columns
DROP_COLUMNS = ["emp_title", "title", "address", "issue_d", "grade"]

# Columns only known after a loan is issued (leakage). Dropped if present.
LEAKAGE_COLUMNS = [
    "total_pymnt",
    "total_pymnt_inv",
    "total_rec_prncp",
    "total_rec_int",
    "total_rec_late_fee",
    "recoveries",
    "collection_recovery_fee",
    "last_pymnt_d",
    "last_pymnt_amnt",
    "next_pymnt_d",
    "out_prncp",
    "out_prncp_inv",
    "last_credit_pull_d",
]

IDENTIFIER_COLUMNS = ["id", "loan_id", "customer_id", "application_id"]


# ============================================================
# 2. LOGGING CONFIGURATION
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# 3. LOAD AND CLEAN DATA
# ============================================================

def load_data():

    logger.info("Loading dataset...")

    data_path = next((p for p in DATA_CANDIDATES if p.exists()), None)

    if data_path is None:
        searched = "\n".join(str(p) for p in DATA_CANDIDATES)
        raise FileNotFoundError(
            f"Dataset not found. Looked in:\n{searched}"
        )

    df = pd.read_csv(data_path)

    logger.info(f"Dataset loaded from {data_path}: {df.shape}")

    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' does not exist.")

    return df


def clean_data(df):

    df = df.copy()

    before = len(df)
    df = df.drop_duplicates()
    logger.info(f"Removed {before - len(df)} duplicate rows.")

    # Convert "Jan-1990" style strings to a numeric year
    if "earliest_cr_line" in df.columns:
        df["earliest_cr_year"] = pd.to_datetime(
            df["earliest_cr_line"],
            format="%b-%Y",
            errors="coerce",
        ).dt.year
        df = df.drop(columns=["earliest_cr_line"])

    to_drop = [
        col for col in (
            DROP_COLUMNS + LEAKAGE_COLUMNS + IDENTIFIER_COLUMNS
        )
        if col in df.columns
    ]

    if to_drop:
        logger.info(f"Dropping columns: {to_drop}")
        df = df.drop(columns=to_drop)

    return df.reset_index(drop=True)


# ============================================================
# 4. PREPARE TARGET VARIABLE
# ============================================================

def prepare_target(df):

    if pd.api.types.is_numeric_dtype(df[TARGET]):

        logger.warning(
            "Target is numeric. Confirm that 1 means Charged Off."
        )

        y = pd.to_numeric(df[TARGET], errors="coerce")

    else:

        target_mapping = {
            "charged off": 1,
            "fully paid": 0,
        }

        y = (
            df[TARGET]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(target_mapping)
        )

    if y.isna().any():
        raise ValueError(
            "Target contains missing or unrecognized values."
        )

    if set(y.unique()) != {0, 1}:
        raise ValueError("Expected binary target values 0 and 1.")

    # 1 = Charged Off / Default
    # 0 = Fully Paid / Non-default

    return y.astype(int)


# ============================================================
# 5. PREPARE FEATURES
# ============================================================

def prepare_features(df):

    X = df.drop(columns=[TARGET]).copy()

    numeric_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    if not numeric_features and not categorical_features:
        raise ValueError("No usable features were found.")

    logger.info(f"Numeric features: {numeric_features}")
    logger.info(f"Categorical features: {categorical_features}")

    return X, numeric_features, categorical_features


# ============================================================
# 6. PREPROCESSING PIPELINE
# ============================================================

def create_preprocessor(numeric_features, categorical_features):

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=50,   # collapse rare categories
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, categorical_features),
        ]
    )


# ============================================================
# 7. EVALUATION
# ============================================================

def evaluate_model(model, X_test, y_test, model_name, threshold=0.5):

    logger.info(f"Evaluating {model_name} at threshold {threshold:.4f}...")

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    results = {
        "model": model_name,
        "threshold": float(threshold),
        "accuracy": accuracy_score(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "pr_auc": average_precision_score(y_test, probabilities),
        "precision_default": precision_score(
            y_test, predictions, pos_label=1, zero_division=0
        ),
        "recall_default": recall_score(
            y_test, predictions, pos_label=1, zero_division=0
        ),
        "f1_default": f1_score(
            y_test, predictions, pos_label=1, zero_division=0
        ),
    }

    print("\n" + "=" * 60)
    print(f"MODEL: {model_name}  (threshold {threshold:.4f})")
    print("=" * 60)

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[1, 0],
            target_names=["Charged Off", "Fully Paid"],
            zero_division=0,
        )
    )

    print("\nConfusion Matrix (rows = actual, columns = predicted):")
    print(confusion_matrix(y_test, predictions, labels=[1, 0]))

    print("\nEvaluation Metrics:")
    for metric, value in results.items():
        if metric != "model":
            print(f"{metric}: {value:.4f}")

    return results


def find_threshold(model, X_train, y_train):
    """Pick the F1 maximizing threshold from out of fold training predictions."""

    logger.info("Selecting decision threshold with cross validation...")

    cv = StratifiedKFold(
        n_splits=THRESHOLD_CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    probabilities = cross_val_predict(
        clone(model),
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]

    precision, recall, thresholds = precision_recall_curve(
        y_train, probabilities
    )

    f1_scores = (
        2 * precision[:-1] * recall[:-1]
        / (precision[:-1] + recall[:-1] + 1e-9)
    )

    return float(thresholds[np.argmax(f1_scores)])


# ============================================================
# 8. MODELS
# ============================================================

def build_random_forest(preprocessor):

    rf_classifier = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", rf_classifier),
        ]
    )


def build_xgboost(preprocessor, scale_pos_weight):

    xgb_classifier = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        objective="binary:logistic",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", xgb_classifier),
        ]
    )


# ============================================================
# 9. HYPERPARAMETER TUNING
# ============================================================

def stratified_subsample(X, y, size):

    if len(X) <= size:
        return X, y

    X_sub, _, y_sub, _ = train_test_split(
        X,
        y,
        train_size=size,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    return X_sub, y_sub


def tune_model(model, parameter_distributions, X_train, y_train, model_name):

    logger.info(f"Starting hyperparameter tuning: {model_name}")

    X_search, y_search = stratified_subsample(
        X_train, y_train, SEARCH_SAMPLE_SIZE
    )

    logger.info(f"Search sample size: {len(X_search)}")

    # The search parallelizes over candidates, so the model itself runs on one core
    search_model = clone(model)
    search_model.set_params(classifier__n_jobs=1)

    cv = StratifiedKFold(
        n_splits=SEARCH_CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    search = RandomizedSearchCV(
        estimator=search_model,
        param_distributions=parameter_distributions,
        n_iter=SEARCH_ITERATIONS,
        scoring="average_precision",
        cv=cv,
        verbose=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=False,   # the winner is refit on the full training set later
    )

    search.fit(X_search, y_search)

    logger.info(
        f"Best CV PR-AUC for {model_name}: {search.best_score_:.4f}"
    )
    logger.info(
        f"Best parameters for {model_name}: {search.best_params_}"
    )

    return search


def refit_best(model, best_params, X_train, y_train):
    """Rebuild the model with the best parameters and fit it on all training data."""

    final_model = clone(model)
    final_model.set_params(**best_params)
    return final_model


# ============================================================
# 10. JSON HELPER
# ============================================================

def to_serializable(value):

    if isinstance(value, np.generic):
        return value.item()

    return str(value)


# ============================================================
# 11. MAIN WORKFLOW
# ============================================================

def main():

    # --------------------------------------------------------
    # LOAD, CLEAN, PREPARE
    # --------------------------------------------------------

    df = clean_data(load_data())

    y = prepare_target(df)

    X, numeric_features, categorical_features = prepare_features(df)

    print("\nTARGET DISTRIBUTION")
    print(y.value_counts())

    print("\nTARGET PROPORTIONS")
    print(y.value_counts(normalize=True))

    # --------------------------------------------------------
    # TRAIN TEST SPLIT
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    logger.info(f"Training observations: {len(X_train)}")
    logger.info(f"Testing observations: {len(X_test)}")

    # Class imbalance from training data only
    scale_pos_weight = float(
        (y_train == 0).sum() / (y_train == 1).sum()
    )

    logger.info(f"Class imbalance ratio: {scale_pos_weight:.4f}")

    preprocessor = create_preprocessor(
        numeric_features,
        categorical_features,
    )

    # --------------------------------------------------------
    # BASELINE MODELS (default 0.5 threshold)
    # --------------------------------------------------------

    logger.info("Training baseline Random Forest...")
    rf_pipeline = build_random_forest(preprocessor)
    rf_pipeline.fit(X_train, y_train)
    rf_results = evaluate_model(
        rf_pipeline, X_test, y_test, "Random Forest"
    )

    logger.info("Training baseline XGBoost...")
    xgb_pipeline = build_xgboost(preprocessor, scale_pos_weight)
    xgb_pipeline.fit(X_train, y_train)
    xgb_results = evaluate_model(
        xgb_pipeline, X_test, y_test, "XGBoost"
    )

    pd.DataFrame([rf_results, xgb_results]).to_csv(
        REPORT_DIR / "initial_model_comparison.csv",
        index=False,
    )

    # --------------------------------------------------------
    # SEARCH SPACES
    # --------------------------------------------------------

    rf_parameters = {
        "classifier__n_estimators": randint(200, 401),
        "classifier__max_depth": [10, 15, 20, 25],
        "classifier__min_samples_split": randint(2, 30),
        "classifier__min_samples_leaf": randint(10, 60),
        "classifier__max_features": ["sqrt", 0.3],
        "classifier__class_weight": ["balanced", "balanced_subsample"],
    }

    xgb_parameters = {
        "classifier__n_estimators": randint(200, 601),
        "classifier__max_depth": randint(3, 9),
        "classifier__learning_rate": uniform(0.02, 0.18),
        "classifier__subsample": uniform(0.65, 0.35),
        "classifier__colsample_bytree": uniform(0.65, 0.35),
        "classifier__min_child_weight": randint(1, 10),
        "classifier__reg_lambda": uniform(0.5, 5.0),
        "classifier__scale_pos_weight": [
            1.0,
            scale_pos_weight * 0.75,
            scale_pos_weight,
            scale_pos_weight * 1.25,
        ],
    }

    # --------------------------------------------------------
    # TUNE ON A SUBSAMPLE
    # --------------------------------------------------------

    rf_search = tune_model(
        rf_pipeline, rf_parameters, X_train, y_train, "Random Forest"
    )

    xgb_search = tune_model(
        xgb_pipeline, xgb_parameters, X_train, y_train, "XGBoost"
    )

    # --------------------------------------------------------
    # REFIT BEST PARAMETERS ON FULL TRAINING DATA
    # --------------------------------------------------------

    best_rf = refit_best(
        rf_pipeline, rf_search.best_params_, X_train, y_train
    )
    best_xgb = refit_best(
        xgb_pipeline, xgb_search.best_params_, X_train, y_train
    )

    # Choose thresholds on training data only (single core estimators,
    # parallelism comes from cross_val_predict)
    best_rf.set_params(classifier__n_jobs=1)
    best_xgb.set_params(classifier__n_jobs=1)

    rf_threshold = find_threshold(best_rf, X_train, y_train)
    xgb_threshold = find_threshold(best_xgb, X_train, y_train)

    best_rf.set_params(classifier__n_jobs=-1)
    best_xgb.set_params(classifier__n_jobs=-1)

    logger.info("Fitting tuned Random Forest on full training data...")
    best_rf.fit(X_train, y_train)

    logger.info("Fitting tuned XGBoost on full training data...")
    best_xgb.fit(X_train, y_train)

    # --------------------------------------------------------
    # EVALUATE TUNED MODELS ON THE UNTOUCHED TEST SET
    # --------------------------------------------------------

    tuned_rf_results = evaluate_model(
        best_rf, X_test, y_test, "Tuned Random Forest", rf_threshold
    )

    tuned_xgb_results = evaluate_model(
        best_xgb, X_test, y_test, "Tuned XGBoost", xgb_threshold
    )

    # --------------------------------------------------------
    # SAVE COMPARISON
    # --------------------------------------------------------

    comparison = pd.DataFrame(
        [rf_results, xgb_results, tuned_rf_results, tuned_xgb_results]
    ).sort_values(by="pr_auc", ascending=False)

    comparison.to_csv(REPORT_DIR / "model_comparison.csv", index=False)

    print("\nFINAL MODEL COMPARISON")
    print(comparison.to_string(index=False))

    # --------------------------------------------------------
    # SAVE HYPERPARAMETER RESULTS
    # --------------------------------------------------------

    tuning_results = {
        "random_forest": {
            "best_cv_pr_auc": float(rf_search.best_score_),
            "best_parameters": rf_search.best_params_,
            "threshold": rf_threshold,
        },
        "xgboost": {
            "best_cv_pr_auc": float(xgb_search.best_score_),
            "best_parameters": xgb_search.best_params_,
            "threshold": xgb_threshold,
        },
    }

    with open(REPORT_DIR / "hyperparameter_results.json", "w") as file:
        json.dump(
            tuning_results,
            file,
            indent=4,
            default=to_serializable,
        )

    # --------------------------------------------------------
    # SELECT BEST MODEL USING CV PR-AUC
    # --------------------------------------------------------

    candidates = {
        "random_forest": (best_rf, rf_threshold),
        "xgboost": (best_xgb, xgb_threshold),
    }

    cv_scores = {
        "random_forest": rf_search.best_score_,
        "xgboost": xgb_search.best_score_,
    }

    best_name = max(cv_scores, key=cv_scores.get)
    best_model, best_threshold = candidates[best_name]

    # --------------------------------------------------------
    # SAVE SELECTED MODEL AND METADATA
    # --------------------------------------------------------

    model_path = ARTIFACT_DIR / "best_model.pkl"

    joblib.dump(best_model, model_path)

    metadata = {
        "model": best_name,
        "threshold": best_threshold,
        "cv_pr_auc": float(cv_scores[best_name]),
        "target_positive_class": "Charged Off",
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
    }

    with open(ARTIFACT_DIR / "model_metadata.json", "w") as file:
        json.dump(metadata, file, indent=4, default=to_serializable)

    print("\nSELECTED MODEL")
    print("------------------------------")
    print(f"Model: {best_name}")
    print(f"Cross validation PR-AUC: {cv_scores[best_name]:.4f}")
    print(f"Decision threshold: {best_threshold:.4f}")
    print(f"Saved model: {model_path}")

    logger.info("Advanced modeling workflow completed successfully.")


# ============================================================
# 12. EXECUTE MAIN FUNCTION
# ============================================================

if __name__ == "__main__":
    main()