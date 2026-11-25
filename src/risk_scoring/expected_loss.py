
"""
Credit Risk Intelligence
Expected credit loss estimation.

EL = PD * EAD * LGD

The original loan amount may be used as an exposure proxy,
but must not be represented as actual outstanding exposure.
"""

def calculate_expected_loss(
    probability_of_default: float,
    exposure_at_default: float,
    loss_given_default: float
) -> dict:
    """
    Calculate estimated expected loss.

    LGD must be supplied as a decimal between 0 and 1.
    """

    values = {
        "probability_of_default": probability_of_default,
        "exposure_at_default": exposure_at_default,
        "loss_given_default": loss_given_default
    }

    for name, value in values.items():
        if isinstance(value, bool):
            raise ValueError(f"{name} must be numeric.")

        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be numeric.")

        if numeric_value < 0:
            raise ValueError(f"{name} cannot be negative.")

        values[name] = numeric_value

    pd = values["probability_of_default"]
    ead = values["exposure_at_default"]
    lgd = values["loss_given_default"]

    if pd > 1:
        raise ValueError("PD must be between 0 and 1.")

    if lgd > 1:
        raise ValueError("LGD must be between 0 and 1.")

    expected_loss = pd * ead * lgd

    return {
        "probability_of_default": round(pd, 6),
        "exposure_at_default_proxy_usd": round(ead, 2),
        "loss_given_default": round(lgd, 6),
        "expected_loss_usd": round(expected_loss, 2),
        "exposure_type": "Loan amount proxy, not actual EAD",
        "method": "PD * EAD * LGD",
        "is_actual_realized_loss": False
    }
