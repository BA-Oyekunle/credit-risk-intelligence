
"""
Credit Risk Intelligence
Configurable lending decision recommendation engine.

This module produces risk-review recommendations.
It does not make legally binding lending decisions.
"""

from .risk_grading import load_risk_policy


def generate_decision(
    probability_of_default: float
) -> dict:
    """
    Generate a risk-review recommendation using the
    configured probability threshold.
    """

    if isinstance(probability_of_default, bool):
        raise ValueError("PD must be numeric.")

    try:
        pd = float(probability_of_default)
    except (TypeError, ValueError):
        raise ValueError("PD must be numeric.")

    if not 0 <= pd <= 1:
        raise ValueError("PD must be between 0 and 1.")

    policy = load_risk_policy()
    decision_policy = policy["decision_policy"]

    threshold = float(
        decision_policy["reference_threshold"]
    )

    if pd >= threshold:
        recommendation = "ENHANCED_REVIEW"
        explanation = (
            "Predicted default probability meets or exceeds "
            "the recall-oriented reference threshold."
        )
    else:
        recommendation = "STANDARD_REVIEW"
        explanation = (
            "Predicted default probability is below the "
            "recall-oriented reference threshold."
        )

    return {
        "recommendation": recommendation,
        "probability_of_default": round(pd, 6),
        "reference_threshold": threshold,
        "explanation": explanation,
        "automatic_approval_enabled": (
            decision_policy["automatic_approval_enabled"]
        ),
        "automatic_rejection_enabled": (
            decision_policy["automatic_rejection_enabled"]
        )
    }
