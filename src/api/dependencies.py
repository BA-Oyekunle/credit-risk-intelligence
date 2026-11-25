
"""
Credit Risk Intelligence
FastAPI dependencies and model initialization.
"""

from functools import lru_cache

from src.risk_scoring.borrower_assessment import (
    load_calibrated_model,
    load_policy,
)


@lru_cache(maxsize=1)
def get_model_bundle():
    """Load and cache the calibrated model bundle."""
    return load_calibrated_model()


@lru_cache(maxsize=1)
def get_risk_policy():
    """Load and cache the risk policy."""
    return load_policy()


def check_model_readiness():
    """
    Confirm that the calibrated model and policy
    can be loaded successfully.
    """

    bundle = get_model_bundle()
    policy = get_risk_policy()

    required_model_keys = [
        "model",
        "calibrator",
    ]

    for key in required_model_keys:
        if key not in bundle:
            raise RuntimeError(
                f"Model bundle is missing required key: {key}"
            )

    if not isinstance(policy, dict):
        raise RuntimeError(
            "Risk policy must be a dictionary."
        )

    return True
