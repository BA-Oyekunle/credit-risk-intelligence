# Credit Risk Intelligence
## Milestone 10.1: Out-of-Sample Model Validation

### 1. Validation methodology

- Dataset: LendingClub-style loan dataset.
- Target: loan_status.
- Positive class: Charged Off (1).
- Negative class: Fully Paid (0).
- Random state: 42.
- Test proportion: 20%.
- Stratified train/test split.
- Test set excluded from threshold selection.

### 2. Data partition sizes

- total_observations: 396030
- model_fitting: 221776
- calibration: 47524
- threshold_selection: 47524
- test: 79206
- random_state: 42
- test_size: 0.2
- stratified: True

### 3. Model performance

| model                            |   threshold |   accuracy |   roc_auc |   pr_auc |   precision_default |   recall_default |       f1 |       f2 |   brier_score |   log_loss |   true_negatives |   false_positives |   false_negatives |   true_positives |   specificity |   default_prevalence |
|:---------------------------------|------------:|-----------:|----------:|---------:|--------------------:|-----------------:|---------:|---------:|--------------:|-----------:|-----------------:|------------------:|------------------:|-----------------:|--------------:|---------------------:|
| Original XGBoost                 |    0.5      |   0.600927 |  0.723253 |  0.39232 |            0.297179 |         0.75803  | 0.426968 | 0.578582 |      0.238832 |   0.671312 |            35821 |             27850 |              3759 |            11776 |      0.562595 |             0.196134 |
| Calibrated Model at 0.50         |    0.5      |   0.807704 |  0.722827 |  0.39135 |            0.561638 |         0.089154 | 0.15388  | 0.107188 |      0.140942 |   0.443596 |            62590 |              1081 |             14150 |             1385 |      0.983022 |             0.196134 |
| Calibrated Model at F2 Threshold |    0.117632 |   0.480254 |  0.722827 |  0.39135 |            0.256526 |         0.869199 | 0.396139 | 0.588222 |      0.140942 |   0.443596 |            24536 |             39135 |              2032 |            13503 |      0.385356 |             0.196134 |

### 4. Threshold selection

- Original XGBoost: 0.5
- Calibrated model F2 threshold: 0.11763191030133913
- Threshold partition F2: 0.5906171257115619
- Calibrated test mean probability: 0.19634298154825974
- Calibrated test Brier score: 0.14094163705363405

### 5. Validation observations

- ROC-AUC measures ranking discrimination.
- PR-AUC measures precision-recall performance for the default class.
- Brier score and log loss evaluate probability quality.
- Precision, recall and F-scores depend on the selected decision threshold.
- Lower Brier score and log loss indicate better probability predictions.

### 6. Important limitations

- This evaluation does not establish causal relationships.
- This evaluation does not establish fairness.
- This evaluation does not estimate actual realized credit losses.
- The dataset does not provide sufficient recorded recovery information to calculate actual loss given default.
- Performance on this historical test partition does not guarantee future performance.
- Calibration artifact provenance must be verified against the original Milestone 7.0 run.

### 7. Output files

- `calibrated_model_at_0.50_confusion_matrix.png`
- `calibrated_model_at_050_classification_report.csv`
- `calibrated_model_at_050_confusion_matrix.csv`
- `calibrated_model_at_f2_threshold_classification_report.csv`
- `calibrated_model_at_f2_threshold_confusion_matrix.csv`
- `calibrated_model_at_f2_threshold_confusion_matrix.png`
- `calibration_comparison.png`
- `model_comparison.csv`
- `original_xgboost_classification_report.csv`
- `original_xgboost_confusion_matrix.csv`
- `original_xgboost_confusion_matrix.png`
- `precision_recall_comparison.png`
- `roc_curve_comparison.png`
- `validation_metrics.json`

End of Milestone 10.1 validation report.