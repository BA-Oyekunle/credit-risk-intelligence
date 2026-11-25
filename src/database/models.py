
from datetime import datetime, timezone

from sqlalchemy import (
    Integer,
    String,
    Float,
    DateTime,
    JSON,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from src.database.connection import Base


class BorrowerAssessment(Base):

    __tablename__ = "borrower_assessments"

    # Primary key
    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )

    # Assessment timestamp
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Model information
    model_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    # Probability predictions
    raw_probability_of_default: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    calibrated_probability_of_default: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    # Risk scoring
    risk_score: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    risk_grade: Mapped[str] = mapped_column(
        String(10),
        nullable=False,
    )

    # Decision recommendation
    recommendation: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    reference_threshold: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    # Optional expected-loss information
    expected_loss: Mapped[dict | None] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"),
        nullable=True,
    )

    # Currency
    currency: Mapped[str] = mapped_column(
        String(10),
        default="USD",
        nullable=False,
    )

    # Assessment status
    assessment_status: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
    )

    # Complete assessment response (not raw borrower inputs)
    assessment_data: Mapped[dict] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"),
        nullable=False,
    )
