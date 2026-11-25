
"""
Credit Risk Intelligence
Configurable borrower risk grading.
"""

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

POLICY_PATH = (
    PROJECT_ROOT
    / "configs"
    / "risk_policy"
    / "risk_policy.json"
)


def load_risk_policy():
    """Load risk policy configuration."""

    if not POLICY_PATH.exists():
        raise FileNotFoundError(
            f"Risk policy not found: {POLICY_PATH}"
        )

    with open(POLICY_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def assign_risk_grade(risk_score: float) -> dict:
    """
    Assign a risk grade based on configured score boundaries.
    """

    if isinstance(risk_score, bool):
        raise ValueError("Risk score must be numeric.")

    try:
        score = float(risk_score)
    except (TypeError, ValueError):
        raise ValueError("Risk score must be numeric.")

    if not 0 <= score <= 100:
        raise ValueError("Risk score must be between 0 and 100.")

    policy = load_risk_policy()

    for grade in policy["risk_grades"]:
        if (
            grade["minimum_score"]
            <= score
            <= grade["maximum_score"]
        ):
            return {
                "grade": grade["grade"],
                "description": grade["description"],
                "risk_score": score
            }

    raise ValueError(
        f"No risk grade configured for score {score}."
    )
