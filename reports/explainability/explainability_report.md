# CREDIT RISK INTELLIGENCE

## Model Explainability Report

### Model Information

- Model: XGBoost Classifier
- Explainability method: SHAP TreeExplainer
- Explanation observations: 5,000
- Transformed features: 88

### Top 10 Features

| Rank | Feature | Mean Absolute SHAP |
|---:|---|---:|
| 1 | categorical__application_type_JOINT | 0.544003 |
| 2 | categorical__purpose_small_business | 0.482346 |
| 3 | numeric__int_rate | 0.353366 |
| 4 | categorical__purpose_wedding | 0.284090 |
| 5 | numeric__annual_inc | 0.258830 |
| 6 | numeric__revol_util | 0.226297 |
| 7 | categorical__sub_grade_F5 | 0.211785 |
| 8 | categorical__purpose_renewable_energy | 0.173548 |
| 9 | numeric__pub_rec | 0.145680 |
| 10 | categorical__sub_grade_A1 | 0.144601 |

### Interpretation

Mean absolute SHAP values measure the average magnitude of each feature's contribution to the model output. They do not indicate whether a feature increases or decreases default risk.

The SHAP beeswarm plot provides information about the direction of feature contributions. Positive SHAP values push the model output toward the Charged Off class, while negative values push it away from that class.

The SHAP values from this TreeExplainer represent contributions to the XGBoost raw model output (margin), not calibrated default probabilities.

These explanations describe model behaviour and statistical associations. They do not establish causal relationships or prove that a feature caused a borrower to default.

### Limitations

- Explanations were calculated on a stratified sample of the dataset.
- Feature importance can vary across samples.
- Correlated predictors can share or redistribute SHAP contributions.
- SHAP explanations do not establish causality.
- These explanations do not replace independent model validation or fairness testing.
