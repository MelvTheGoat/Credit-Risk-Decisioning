"""FastAPI decision service.

Endpoints
---------
``POST /decision``    score an applicant and return an auditable decision
``GET  /health``      liveness and readiness
``GET  /model-info``  what is deployed, on what assumptions
``GET  /metrics``     PSI and score distribution since the reference

Startup behaviour
-----------------
The service loads its bundle at startup through
:func:`src.artifacts.load_bundle`, which **refuses a model and calibrator from
different training runs** and refuses files whose content hashes do not match
the manifest. If that check fails the service does not start.

That is the intended behaviour, not a rough edge. A service that is down is an
incident somebody notices in minutes. A service quietly serving a model against
the wrong calibrator returns confident, well-ranked, wrongly-priced
probabilities, and nothing in the ordinary monitoring pack will reveal it —
AUC is untouched by the mismatch. Failing loudly is the cheaper failure.

Every decision is written to the append-only audit log **before** the response
is returned, so the institution can always evidence a decision the applicant
has received.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from src.api.audit import AuditLog
from src.api.schemas import (
    DecisionRequest,
    DecisionResponse,
    HealthResponse,
    MetricsResponse,
    ModelInfoResponse,
)
from src.artifacts import ArtifactMismatchError, ArtifactNotFoundError, resolve_artifact_root
from src.config import DEFAULT_COSTS
from src.engine import DecisionEngine
from src.monitoring import classify_psi, population_stability_index, score_distribution_report

logger = logging.getLogger(__name__)

REFERENCE_SCORES_FILENAME = "reference_scores.npy"


class ServiceState:
    """Process-wide state: the engine, the audit log and the reference scores."""

    def __init__(self) -> None:
        """Initialise empty state."""
        self.engine: DecisionEngine | None = None
        self.audit: AuditLog = AuditLog()
        self.reference_scores: np.ndarray | None = None
        self.started_at: float = time.time()
        self.startup_error: str | None = None

    def load(self, version: str | None = None) -> None:
        """Load and validate the model bundle.

        Args:
            version: Artifact version, or ``None`` for the current pointer.

        Raises:
            ArtifactMismatchError: If the model and calibrator disagree.
            ArtifactNotFoundError: If no artifact is available.
        """
        self.engine = DecisionEngine.from_artifacts(version, costs=DEFAULT_COSTS)
        manifest = self.engine.bundle.manifest
        logger.info(
            "Loaded model %s (run %s), threshold %.4f",
            manifest.version,
            manifest.training_run_id,
            manifest.threshold,
        )

        # Always reassign, including to None. Leaving a previously loaded
        # reference in place would mean that reloading or rolling back to a
        # version without reference scores kept comparing against a different
        # model's distribution, and reporting the result as this model's PSI.
        reference_path = resolve_artifact_root(None) / manifest.version / REFERENCE_SCORES_FILENAME
        if reference_path.exists():
            self.reference_scores = np.load(reference_path)
            logger.info("Loaded %d reference scores for drift monitoring", len(self.reference_scores))
        else:
            self.reference_scores = None
            logger.warning(
                "No reference scores at %s; /metrics will report PSI as unavailable", reference_path
            )


state = ServiceState()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:  # noqa: ARG001
    """Load the model at startup, refusing to serve a mismatched bundle.

    Args:
        app: The FastAPI application.

    Yields:
        Control back to the server once startup is complete.

    Raises:
        ArtifactMismatchError: Propagated so the process exits rather than
            serving miscalibrated probabilities.
    """
    try:
        state.load()
    except ArtifactMismatchError as error:
        # Deliberately fatal. See the module docstring.
        state.startup_error = str(error)
        logger.critical("REFUSING TO START: %s", error)
        raise
    except ArtifactNotFoundError as error:
        state.startup_error = str(error)
        logger.critical("REFUSING TO START: no usable model artifact: %s", error)
        raise
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Credit Risk Decision Service",
    description=(
        "Calibrated probability of default, cost-optimal approve/decline decisions, "
        "adverse action reason codes, and an append-only audit trail."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


def _require_engine() -> DecisionEngine:
    """Return the loaded engine or fail with a 503.

    Returns:
        The decision engine.

    Raises:
        HTTPException: If no model is loaded.
    """
    if state.engine is None:
        raise HTTPException(
            status_code=503,
            detail=state.startup_error or "No model loaded",
        )
    return state.engine


@app.post("/decision", response_model=DecisionResponse)
def decide(request: DecisionRequest) -> DecisionResponse:
    """Score an applicant and return an auditable decision.

    Args:
        request: The decision request.

    Returns:
        The decision, including calibrated PD, score band, expected loss,
        adverse action reason codes and the deciding model version.

    Raises:
        HTTPException: 503 if no model is loaded, 422 if the features do not
            match what the deployed model expects.
    """
    engine = _require_engine()
    frame = pd.DataFrame([request.features.model_dump()])

    try:
        decisions = engine.decide(
            frame, exposure=None if request.exposure_at_default is None else np.array([request.exposure_at_default])
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    decision = decisions[0].to_dict()

    # Write before responding: a decision the applicant has received must
    # always be evidenceable.
    state.audit.record(decision, application_id=request.application_id, channel="api")

    return DecisionResponse(
        application_id=request.application_id,
        decided_at=datetime.now(UTC).isoformat(),
        **decision,
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Report liveness and whether a validated model is loaded.

    Returns:
        Health status.
    """
    loaded = state.engine is not None
    return HealthResponse(
        status="ok" if loaded else "unavailable",
        model_version=state.engine.model_version if state.engine else "",
        model_loaded=loaded,
        uptime_seconds=time.time() - state.started_at,
    )


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    """Describe the deployed model, its cutoff and the assumptions behind it.

    Returns the live values read from the loaded artifact's manifest, not
    documentation that can drift away from what is actually running.

    Returns:
        Model metadata, calibration and performance metrics, the cutoff in
        force, and the cost assumptions it was derived from.

    Raises:
        HTTPException: 503 if no model is loaded.
    """
    manifest = _require_engine().bundle.manifest
    return ModelInfoResponse(
        model_version=manifest.version,
        training_run_id=manifest.training_run_id,
        model_name=manifest.model_name,
        calibrator_name=manifest.calibrator_name,
        created_at=manifest.created_at,
        feature_names=manifest.feature_names,
        threshold=manifest.threshold,
        costs=manifest.costs,
        calibration_metrics=manifest.calibration_metrics,
        performance_metrics=manifest.performance_metrics,
        training_rows=manifest.training_rows,
        training_period=manifest.training_period,
        notes=manifest.notes,
    )


@app.get("/metrics", response_model=MetricsResponse)
def metrics(n_bins: int = Query(default=10, ge=2, le=50)) -> MetricsResponse:
    """Report the score distribution and drift since the reference population.

    PSI compares the probabilities this service has actually issued against the
    reference distribution captured at training time. It answers "are the
    applicants arriving now like the ones the model was fitted on?" and nothing
    more — see :mod:`src.monitoring` for why that is a prompt to investigate
    rather than a verdict.

    Args:
        n_bins: Number of bins for the distribution comparison.

    Returns:
        Decision counts, approval rate, score PSI and its status.
    """
    summary = state.audit.summary()
    probabilities = summary.pop("probabilities", [])

    score_psi: float | None = None
    distribution: list[dict[str, Any]] = []
    status = "unknown"

    if state.reference_scores is not None and len(probabilities) >= n_bins:
        current = np.asarray(probabilities, dtype=float)
        score_psi = population_stability_index(state.reference_scores, current, n_bins=n_bins)
        status = classify_psi(score_psi)
        distribution = [
            {str(key): value for key, value in row.items()}
            for row in score_distribution_report(
                state.reference_scores, current, n_bins=n_bins
            ).to_dict(orient="records")
        ]
    elif state.reference_scores is None:
        status = "no_reference"
    else:
        status = "insufficient_decisions"

    n_decisions = int(summary.get("n_decisions", 0))
    approvals = int(summary.get("by_decision", {}).get("approve", 0))

    return MetricsResponse(
        n_decisions=n_decisions,
        by_decision=summary.get("by_decision", {}),
        by_model_version=summary.get("by_model_version", {}),
        score_psi=score_psi,
        psi_status=status,
        score_distribution=distribution,
        approval_rate=approvals / n_decisions if n_decisions else None,
        mean_probability_of_default=float(np.mean(probabilities)) if probabilities else None,
        reference_available=state.reference_scores is not None,
        first_timestamp=summary.get("first_timestamp"),
        last_timestamp=summary.get("last_timestamp"),
    )
