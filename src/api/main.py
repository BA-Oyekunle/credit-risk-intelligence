
"""
Credit Risk Intelligence
FastAPI application.

Provides borrower risk assessment, PostgreSQL persistence,
and retrieval of previous borrower assessments.
"""

import json
import logging
from typing import Any

from fastapi import FastAPI, HTTPException, Query, status
from sqlalchemy import select, text
from sqlalchemy.exc import SQLAlchemyError

from src.api.schemas import (
    BorrowerAssessmentRequest,
    APIHealthResponse,
)

from src.api.dependencies import (
    check_model_readiness,
    get_model_bundle,
    get_risk_policy,
)

from src.risk_scoring.borrower_assessment import (
    assess_borrower,
    save_assessment,
)

from src.database.connection import (
    engine,
    SessionLocal,
    Base,
)

from src.database.models import BorrowerAssessment


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | %(levelname)s | "
        "%(name)s | %(message)s"
    )
)

logger = logging.getLogger(__name__)


# ============================================================
# APPLICATION INITIALIZATION
# ============================================================

app = FastAPI(
    title="Credit Risk Intelligence API",
    description=(
        "An API for calibrated loan default prediction, "
        "borrower risk scoring, risk grading, decision "
        "recommendations, expected-loss estimation and "
        "PostgreSQL assessment storage."
    ),
    version="1.1.0",
)



@app.on_event("startup")
def initialize_application():
    """
    Validate model and policy readiness,
    establish database connectivity,
    and create database tables if they do not exist.
    """

    logger.info(
        "Initializing Credit Risk Intelligence API..."
    )

    try:
        # 1. Validate model and policy.
        check_model_readiness()

        logger.info(
            "Model and policy initialized successfully."
        )

        # 2. Verify database connectivity.
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))

        logger.info(
            "PostgreSQL connection verified successfully."
        )

        # 3. Create database tables if they do not exist.
        Base.metadata.create_all(bind=engine)

        logger.info(
            "Database tables initialized successfully."
        )

    except Exception:
        logger.exception(
            "API initialization failed."
        )
        raise



@app.on_event("shutdown")
def shutdown_application():
    """Dispose of database connection pool."""

    engine.dispose()

    logger.info(
        "Database connection pool disposed."
    )


# ============================================================
# DATABASE SERIALIZATION
# ============================================================

def serialize_assessment(assessment: dict) -> dict:
    """
    Convert assessment values into JSON-compatible values.

    This ensures the JSONB columns receive serializable data.
    """

    return json.loads(
        json.dumps(
            assessment,
            default=str,
            allow_nan=False,
        )
    )


# ============================================================
# DATABASE PERSISTENCE
# ============================================================

def save_assessment_to_database(
    assessment: dict,
) -> int:
    """
    Save a borrower assessment to PostgreSQL.

    Returns:
        Newly created database record ID.
    """

    session = SessionLocal()

    try:

        borrower_risk = assessment["borrower_risk"]
        risk_grade = borrower_risk["risk_grade"]
        decision = assessment["decision"]
        model_information = assessment["model_information"]

        assessment_data = serialize_assessment(
            assessment
        )

        database_record = BorrowerAssessment(

            model_type=model_information["model_type"],

            raw_probability_of_default=float(
                borrower_risk["raw_probability_of_default"]
            ),

            calibrated_probability_of_default=float(
                borrower_risk["calibrated_probability_of_default"]
            ),

            risk_score=float(
                borrower_risk["risk_score"]
            ),

            risk_grade=str(
                risk_grade["grade"]
            ),

            recommendation=str(
                decision["recommendation"]
            ),

            reference_threshold=float(
                decision["reference_threshold"]
            ),

            expected_loss=(
                serialize_assessment(
                    assessment["expected_loss"]
                )
                if assessment.get("expected_loss") is not None
                else None
            ),

            currency=str(
                assessment.get("currency", "USD")
            ),

            assessment_status=str(
                assessment["assessment_status"]
            ),

            assessment_data=assessment_data,
        )

        session.add(database_record)

        session.commit()

        session.refresh(database_record)

        record_id = database_record.id

        logger.info(
            "Assessment saved to PostgreSQL. Record ID: %s",
            record_id,
        )

        return record_id

    except Exception:

        session.rollback()

        logger.exception(
            "Failed to save assessment to PostgreSQL."
        )

        raise

    finally:
        session.close()


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():
    """Return basic application information."""

    return {
        "application": "Credit Risk Intelligence",
        "message": "Credit Risk Intelligence API is running.",
        "version": "1.1.0",
        "documentation": "/docs",
        "health_endpoint": "/health",
        "assessment_endpoint": "/assess",
        "assessment_history_endpoint": "/assessments",
    }


# ============================================================
# HEALTH ENDPOINT
# ============================================================

@app.get(
    "/health",
    response_model=APIHealthResponse,
)
def health_check():
    """Check API, model and database availability."""

    model_available = False
    database_available = False

    try:
        check_model_readiness()
        model_available = True

    except Exception:
        logger.exception(
            "Model health check failed."
        )

    try:

        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))

        database_available = True

    except Exception:
        logger.exception(
            "Database health check failed."
        )

    overall_status = (
        "HEALTHY"
        if model_available and database_available
        else "DEGRADED"
    )

    result = {
        "status": overall_status,
        "application": "Credit Risk Intelligence",
        "model_available": model_available,
        "database_available": database_available,
    }

    if not model_available or not database_available:

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=result,
        )

    return result


# ============================================================
# MODEL INFORMATION ENDPOINT
# ============================================================

@app.get("/model-info")
def model_information():
    """Return information about the loaded model and policy."""

    try:

        bundle = get_model_bundle()
        policy = get_risk_policy()

        model = bundle["model"]
        classifier = model.named_steps["classifier"]

        feature_names = getattr(
            model,
            "feature_names_in_",
            None,
        )

        if feature_names is None:

            feature_names = (
                model.named_steps["preprocessor"]
                .feature_names_in_
            )

        return {
            "application": "Credit Risk Intelligence",
            "model_type": type(classifier).__name__,
            "calibration_method": "Platt Scaling",
            "prediction_target": "Charged Off",
            "prediction_class": 1,
            "number_of_input_features": len(feature_names),
            "input_features": list(feature_names),
            "currency": policy.get("currency", "USD"),
            "decision_policy": policy.get(
                "decision_policy",
                {},
            ),
            "automatic_approval_enabled": False,
            "automatic_rejection_enabled": False,
        }

    except Exception:

        logger.exception(
            "Could not retrieve model information."
        )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model information is unavailable.",
        )


# ============================================================
# BORROWER ASSESSMENT ENDPOINT
# ============================================================

@app.post("/assess")
def assess_borrower_endpoint(
    request: BorrowerAssessmentRequest,
) -> dict[str, Any]:
    """
    Assess an individual borrower.

    Optionally save the JSON report and/or PostgreSQL record.
    """

    logger.info(
        "Received borrower assessment request."
    )

    try:

        # ----------------------------------------------------
        # Validate expected-loss input combinations
        # ----------------------------------------------------

        if (
            request.lgd is None
            and request.exposure_at_default is not None
        ):

            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    "LGD must be supplied when "
                    "exposure_at_default is provided."
                ),
            )

        # ----------------------------------------------------
        # Run existing borrower assessment engine
        # ----------------------------------------------------

        assessment = assess_borrower(
            borrower_data=request.borrower_data,
            lgd=request.lgd,
            exposure_at_default=request.exposure_at_default,
            use_loan_amount_as_proxy=(
                request.use_loan_amount_as_proxy
            ),
        )

        # ----------------------------------------------------
        # Optionally save local JSON report
        # ----------------------------------------------------

        if request.save_report:

            report_path = save_assessment(
                assessment
            )

            assessment["report_saved"] = True

            assessment["report_filename"] = (
                report_path.name
            )

        else:

            assessment["report_saved"] = False

        # ----------------------------------------------------
        # Optionally save PostgreSQL record
        # ----------------------------------------------------

        if request.save_to_database:

            try:

                record_id = save_assessment_to_database(
                    assessment
                )

                assessment["database_saved"] = True
                assessment["database_record_id"] = record_id

            except Exception:

                logger.exception(
                    "PostgreSQL persistence failed."
                )

                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=(
                        "Borrower assessment was completed, "
                        "but saving it to PostgreSQL failed. "
                        "Please check database availability."
                    ),
                )

        else:

            assessment["database_saved"] = False
            assessment["database_record_id"] = None

        logger.info(
            "Borrower assessment completed successfully."
        )

        return assessment

    except HTTPException:
        raise

    except ValueError as error:

        logger.warning(
            "Borrower assessment validation failed: %s",
            str(error),
        )

        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(error),
        )

    except Exception:

        logger.exception(
            "Unexpected borrower assessment error."
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "An unexpected error occurred while "
                "processing the borrower assessment."
            ),
        )


# ============================================================
# ASSESSMENT HISTORY ENDPOINT
# ============================================================

@app.get("/assessments")
def get_assessments(
    limit: int = Query(
        default=20,
        ge=1,
        le=100,
        description="Maximum number of records to return.",
    ),
    offset: int = Query(
        default=0,
        ge=0,
        description="Number of records to skip.",
    ),
):
    """
    Retrieve previously saved borrower assessments.

    Supports pagination through limit and offset.
    """

    session = SessionLocal()

    try:

        total_records = session.query(
            BorrowerAssessment
        ).count()

        statement = (
            select(BorrowerAssessment)
            .order_by(
                BorrowerAssessment.created_at.desc(),
                BorrowerAssessment.id.desc(),
            )
            .limit(limit)
            .offset(offset)
        )

        records = session.scalars(
            statement
        ).all()

        results = []

        for record in records:

            results.append({

                "id": record.id,

                "created_at": (
                    record.created_at.isoformat()
                    if record.created_at
                    else None
                ),

                "model_type": record.model_type,

                "raw_probability_of_default": (
                    record.raw_probability_of_default
                ),

                "calibrated_probability_of_default": (
                    record.calibrated_probability_of_default
                ),

                "risk_score": record.risk_score,

                "risk_grade": record.risk_grade,

                "recommendation": record.recommendation,

                "reference_threshold": (
                    record.reference_threshold
                ),

                "expected_loss": record.expected_loss,

                "currency": record.currency,

                "assessment_status": (
                    record.assessment_status
                ),
            })

        return {
            "total_records": total_records,
            "returned_records": len(results),
            "limit": limit,
            "offset": offset,
            "assessments": results,
        }

    except SQLAlchemyError:

        logger.exception(
            "Could not retrieve assessment history."
        )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Could not retrieve assessment history.",
        )

    finally:
        session.close()


# ============================================================
# INDIVIDUAL ASSESSMENT RETRIEVAL
# ============================================================

@app.get("/assessments/{assessment_id}")
def get_assessment_by_id(
    assessment_id: int,
):
    """Retrieve one saved borrower assessment."""

    session = SessionLocal()

    try:

        record = session.get(
            BorrowerAssessment,
            assessment_id,
        )

        if record is None:

            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=(
                    f"Assessment with ID {assessment_id} "
                    "was not found."
                ),
            )

        return {
            "id": record.id,

            "created_at": (
                record.created_at.isoformat()
                if record.created_at
                else None
            ),

            "model_type": record.model_type,

            "raw_probability_of_default": (
                record.raw_probability_of_default
            ),

            "calibrated_probability_of_default": (
                record.calibrated_probability_of_default
            ),

            "risk_score": record.risk_score,

            "risk_grade": record.risk_grade,

            "recommendation": record.recommendation,

            "reference_threshold": (
                record.reference_threshold
            ),

            "expected_loss": record.expected_loss,

            "currency": record.currency,

            "assessment_status": (
                record.assessment_status
            ),

            "assessment_data": record.assessment_data,
        }

    except HTTPException:
        raise

    except SQLAlchemyError:

        logger.exception(
            "Could not retrieve assessment ID %s.",
            assessment_id,
        )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Could not retrieve the requested assessment.",
        )

    finally:
        session.close()

     
