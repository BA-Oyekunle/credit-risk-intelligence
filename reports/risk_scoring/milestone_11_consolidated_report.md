# Credit Risk Intelligence
## Milestone 11: Consolidated Validation Report

**Generated (UTC):** 2026-11-25T07:22:08.785126+00:00

**Overall status:** PASSED

## 1. Borrower Assessment Validation

- Borrowers passed: **50 / 50**
- Checks per borrower: **10**
- Total individual checks: **500 / 500**
- Status: **PASSED**

The borrower assessment tests covered input validation, feature order, model execution, raw and calibrated probabilities, risk score, risk grade, decision consistency, and missing-value handling.

## 2. Expected-Loss Integration

- Tests passed: **4 / 4**
- Tests failed: **0**
- Status: **PASSED**

| Test | Result |
|---|---|
| expected_loss_not_calculated_without_lgd | PASSED |
| expected_loss_with_user_supplied_ead | PASSED |
| expected_loss_with_loan_amount_proxy | PASSED |
| missing_exposure_is_rejected | PASSED |

### Synthetic test fixture warning

- LGD: `0.4`
- User-supplied EAD: `$10,000.00`

These are software test values only. They are not actual borrower data, observed recoveries, or recommended credit assumptions.

## 3. Sample Borrower Assessment

- Calibrated PD: `0.197425`
- Risk score: `80.26`
- Risk grade: `{'grade': 'B', 'description': 'Low predicted default risk', 'risk_score': 80.26}`
- Recommendation: `ENHANCED_REVIEW`
- Expected loss: Not calculated; no LGD/exposure was supplied for this sample assessment.

## 4. Important Model and Policy Limitations

- The reference threshold of 0.117632 was reconstructed from the held-out threshold selection partition; it is not stored as selected_threshold in the model bundle.
- The threshold is a recall-oriented F2 reference, not a production lending threshold.
- Risk grade boundaries are illustrative and have not been validated as lender policy.
- The dataset does not contain actual outstanding exposure, recoveries, or realized loss information.
- loan_amnt is only an exposure proxy and must not be described as actual EAD.
- Automatic approval and rejection remain disabled.
- The discrepancy between earlier Milestone 6 model results and Milestone 10.1 validation still requires investigation before production use.

## 5. Milestone 11 Completion Decision

Milestone 11 software tests pass if the existing 50-borrower validation and all expected-loss integration tests pass.

This confirms the risk-scoring and expected-loss workflow operates as designed. It does not establish production credit-policy validity or production readiness.

## 6. Next Step

Resolve the outstanding model artifact, threshold metadata, and performance reconciliation issues before proceeding to FastAPI deployment.
