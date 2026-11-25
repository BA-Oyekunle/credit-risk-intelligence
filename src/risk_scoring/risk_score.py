
"""
Credit Risk Intelligence
Risk score calculation.

A higher score represents lower predicted default risk.
"""

from typing import Union


def calculate_risk_score(
    probability_of_default: Union[float, int]
) -> float:
    """
    Convert probability of default into a 0-100 risk score.

    Formula:
        Risk Score = 100 * (1 - PD)

    Args:
        probability_of_default: Calibrated probability between 0 and 1.

    Returns:
        Risk score between 0 and 100.
    """

    if isinstance(probability_of_default, bool):
        raise ValueError("Probability of default must be numeric.")

    try:
        pd = float(probability_of_default)
    except (TypeError, ValueError):
        raise ValueError("Probability of default must be numeric.")

    if not 0 <= pd <= 1:
        raise ValueError(
            "Probability of default must be between 0 and 1."
        )

    risk_score = 100 * (1 - pd)

    return round(risk_score, 2)
