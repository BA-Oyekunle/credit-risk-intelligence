import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# --------------------------------
# 1. Load dataset
# --------------------------------

df = pd.read_csv("data/raw/loan_data.csv")


# --------------------------------
# 2. Clean data
# --------------------------------

df = df.drop_duplicates()

# High-cardinality text, raw dates and redundant columns
columns_to_drop = ["emp_title", "title", "address", "issue_d", "grade"]
df = df.drop(columns=columns_to_drop, errors="ignore")

# Convert "Jan-1990" style strings to a numeric year
df["earliest_cr_year"] = pd.to_datetime(
    df["earliest_cr_line"], format="%b-%Y", errors="coerce"
).dt.year
df = df.drop(columns=["earliest_cr_line"])


# --------------------------------
# 3. Define target (1 = Charged Off)
# --------------------------------

y = (df["loan_status"] == "Charged Off").astype(int)
X = df.drop(columns=["loan_status"])


# --------------------------------
# 4. Identify feature types
# --------------------------------

numerical_features = X.select_dtypes(include="number").columns.tolist()
categorical_features = X.select_dtypes(include="object").columns.tolist()


# --------------------------------
# 5. Preprocessing pipelines
# --------------------------------

numerical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_pipeline, numerical_features),
        ("cat", categorical_pipeline, categorical_features),
    ]
)


# --------------------------------
# 6. Train/test split
# --------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)


# --------------------------------
# 7. Model pipeline
# --------------------------------

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(max_iter=1000, class_weight="balanced"),
        ),
    ]
)


# --------------------------------
# 8. Choose threshold on training data only
# --------------------------------

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

train_probabilities = cross_val_predict(
    model,
    X_train,
    y_train,
    cv=cv,
    method="predict_proba",
    n_jobs=-1,
)[:, 1]

precision, recall, thresholds = precision_recall_curve(y_train, train_probabilities)

f1_scores = (
    2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-9)
)
best_threshold = thresholds[np.argmax(f1_scores)]


# --------------------------------
# 9. Final fit and test predictions
# --------------------------------

model.fit(X_train, y_train)

y_probability = model.predict_proba(X_test)[:, 1]
y_pred = (y_probability >= best_threshold).astype(int)


# --------------------------------
# 10. Evaluate
# --------------------------------

print("ROC-AUC:", roc_auc_score(y_test, y_probability))
print("PR-AUC:", average_precision_score(y_test, y_probability))
print("Threshold:", best_threshold)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Fully Paid", "Charged Off"],
    )
)

print("\nConfusion Matrix (rows = actual, columns = predicted):")
print(confusion_matrix(y_test, y_pred))


# --------------------------------
# 11. Optional: cost-based threshold
# --------------------------------

COST_MISSED_DEFAULT = 5   # cost of approving a loan that defaults
COST_REJECTED_GOOD = 1    # cost of rejecting a loan that would be repaid

candidate_thresholds = np.linspace(0.05, 0.95, 91)
costs = []

for t in candidate_thresholds:
    pred = (train_probabilities >= t).astype(int)
    false_negatives = ((pred == 0) & (y_train == 1)).sum()
    false_positives = ((pred == 1) & (y_train == 0)).sum()
    costs.append(
        false_negatives * COST_MISSED_DEFAULT
        + false_positives * COST_REJECTED_GOOD
    )

cost_threshold = candidate_thresholds[int(np.argmin(costs))]
cost_pred = (y_probability >= cost_threshold).astype(int)

print(f"\nCost-optimal threshold: {cost_threshold:.2f}")
print(confusion_matrix(y_test, cost_pred))