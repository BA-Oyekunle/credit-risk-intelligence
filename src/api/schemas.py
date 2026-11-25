
"""
Credit Risk Intelligence
FastAPI request and response schemas.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class BorrowerAssessmentRequest(BaseModel):
    """Request body for an individual borrower assessment."""

    model_config = ConfigDict(extra="forbid")

    borrower_data: dict[str, Any] = Field(
        ...,
        description=(
            "Borrower features required by the calibrated "
            "credit risk model."
        )
    )

    lgd: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description="User-supplied loss given default, between 0 and 1."
    )

    exposure_at_default: float | None = Field(
        default=None,
        ge=0,
        description="User-supplied exposure at default in USD."
    )

    use_loan_amount_as_proxy: bool = Field(
        default=False,
        description=(
            "Explicitly allow original loan amount to be used "
            "as an exposure proxy. This is not actual EAD."
        )
    )

    save_report: bool = Field(
        default=False,
        description="Whether to save the assessment as a JSON report."
    )

    save_to_database: bool = Field(
        default=False,
        description="Whether to save the assessment to PostgreSQL."
    )


class APIHealthResponse(BaseModel):
    status: str
    application: str
    model_available: bool
    database_available: bool


class APIErrorResponse(BaseModel):
    detail: str

