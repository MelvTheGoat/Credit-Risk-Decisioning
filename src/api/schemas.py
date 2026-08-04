"""Request and response schemas for the decision service.

Validation is deliberately strict. A credit decision service that silently
coerces a malformed value into something plausible will decline people for
reasons that trace back to a typo in a caller's payload, and the audit log will
faithfully record the wrong reason. Rejecting the request is the safer failure.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ApplicantFeatures(BaseModel):
    """Features for a single credit application.

    Fields mirror the synthetic loan book. Every one carries bounds, so
    out-of-range values are rejected at the edge rather than silently scored.

    Attributes:
        income_recorded: Annual income as recorded on the application.
        debt_to_income: Total debt repayments over income.
        utilisation: Balances over available credit limits.
        n_delinq_24m: Delinquencies in the last 24 months.
        credit_history_months: Length of credit history in months.
        n_inquiries_6m: Credit applications in the last 6 months.
        employment_years: Years in current employment.
        loan_amount: Amount requested, also used as exposure at default.
        age: Applicant age in years.
        housing_status: One of own, mortgage, rent, other.
        product_type: One of card, personal_loan, auto.
    """

    model_config = ConfigDict(extra="forbid")

    income_recorded: float = Field(gt=0, le=10_000_000, description="Annual income as recorded")
    debt_to_income: float = Field(ge=0, le=10, description="Debt repayments over income")
    utilisation: float = Field(ge=0, le=5, description="Balances over available limits")
    n_delinq_24m: int = Field(ge=0, le=50, description="Delinquencies in last 24 months")
    credit_history_months: float = Field(ge=0, le=900, description="Credit history length")
    n_inquiries_6m: int = Field(ge=0, le=50, description="Credit applications in last 6 months")
    employment_years: float = Field(ge=0, le=70, description="Years in current employment")
    loan_amount: float = Field(gt=0, le=5_000_000, description="Amount requested")
    age: float = Field(ge=18, le=120, description="Applicant age")
    housing_status: str = Field(description="own, mortgage, rent or other")
    product_type: str = Field(description="card, personal_loan or auto")


class DecisionRequest(BaseModel):
    """A decision request.

    Attributes:
        application_id: Caller's identifier, recorded in the audit log so a
            decision can later be retrieved by it.
        features: The applicant's features.
        exposure_at_default: Override the exposure used in the expected-loss
            arithmetic. Defaults to ``loan_amount``.
    """

    model_config = ConfigDict(extra="forbid")

    application_id: str = Field(description="Caller's application identifier")
    features: ApplicantFeatures
    exposure_at_default: float | None = Field(
        default=None, gt=0, description="Override exposure; defaults to loan_amount"
    )


class ReasonCodeResponse(BaseModel):
    """One principal reason contributing to an adverse decision.

    Attributes:
        rank: 1 is the strongest reason.
        feature: Characteristic the reason derives from.
        statement: Plain-language sentence for the applicant.
        contribution: Size of the adverse contribution.
        protected_basis: Whether this reason needs legal review before use in
            an adverse action notice.
    """

    rank: int
    feature: str
    statement: str
    contribution: float
    protected_basis: bool


class DecisionResponse(BaseModel):
    """The decision returned to the caller.

    Attributes:
        application_id: Echo of the caller's identifier.
        decision: ``"approve"``, ``"refer"`` or ``"decline"``.
        probability_of_default: Calibrated PD.
        score: Scorecard points, if a scorecard is deployed.
        score_band: Reporting band ``A`` to ``E``.
        expected_loss: ``PD x LGD x EAD``.
        expected_value: Expected margin less expected loss.
        threshold: PD cutoff in force at decision time.
        model_version: Artifact version that produced the decision.
        reason_codes: Principal reasons, populated for adverse decisions only.
        input_hash: Hash of the inputs, matching the audit log entry.
        decided_at: UTC timestamp.
    """

    application_id: str
    decision: str
    probability_of_default: float
    score: float | None
    score_band: str | None
    expected_loss: float
    expected_value: float
    threshold: float
    model_version: str
    reason_codes: list[ReasonCodeResponse]
    input_hash: str
    decided_at: str


class HealthResponse(BaseModel):
    """Service liveness and readiness.

    Attributes:
        status: ``"ok"`` when a validated bundle is loaded.
        model_version: Loaded artifact version.
        model_loaded: Whether a bundle is loaded and validated.
        uptime_seconds: Seconds since process start.
    """

    status: str
    model_version: str
    model_loaded: bool
    uptime_seconds: float


class ModelInfoResponse(BaseModel):
    """Live description of what is deployed and on what assumptions.

    Attributes:
        model_version: Artifact version.
        training_run_id: Run identifier shared by the model and calibrator.
        model_name: Model family.
        calibrator_name: Calibration strategy.
        created_at: Artifact creation timestamp.
        feature_names: Ordered model inputs.
        threshold: PD cutoff in force.
        costs: Cost assumptions the cutoff was derived from.
        calibration_metrics: Calibration measured at build time.
        performance_metrics: Discrimination measured at build time.
        training_rows: Rows the model was fitted on.
        training_period: Training window description.
        notes: Free-text provenance.
    """

    model_config = ConfigDict(protected_namespaces=())

    model_version: str
    training_run_id: str
    model_name: str
    calibrator_name: str
    created_at: str
    feature_names: list[str]
    threshold: float
    costs: dict[str, float]
    calibration_metrics: dict[str, float]
    performance_metrics: dict[str, float]
    training_rows: int
    training_period: str
    notes: str


class MetricsResponse(BaseModel):
    """Score distribution and drift since the reference population.

    Attributes:
        n_decisions: Decisions in the audit log.
        by_decision: Counts by outcome.
        by_model_version: Counts by deciding model version.
        score_psi: PSI of scored probabilities against the reference.
        psi_status: ``stable``, ``watch``, ``review`` or ``unknown``.
        score_distribution: Per-bin reference and current shares.
        approval_rate: Share of decisions that were approvals.
        mean_probability_of_default: Mean PD across logged decisions.
        reference_available: Whether a reference distribution is loaded.
        first_timestamp: Earliest logged decision.
        last_timestamp: Most recent logged decision.
    """

    n_decisions: int
    by_decision: dict[str, int]
    by_model_version: dict[str, int]
    score_psi: float | None
    psi_status: str
    score_distribution: list[dict[str, Any]]
    approval_rate: float | None
    mean_probability_of_default: float | None
    reference_available: bool
    first_timestamp: str | None
    last_timestamp: str | None
