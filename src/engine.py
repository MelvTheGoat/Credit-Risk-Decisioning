"""The decision engine: one code path for every decision this system makes.

Both the real-time API and the batch scorer call :meth:`DecisionEngine.decide`.
Neither implements any scoring, thresholding, banding or reason-code logic of
its own.

That is deliberate and it is not a tidiness preference. Where real-time and
batch scoring are implemented separately they drift, usually in the small
things — a feature clipped in one and not the other, a threshold updated in one
config and not the other, a null handled differently. The result is that the
same applicant receives different decisions depending on which door they came
through, which is indefensible to a regulator and very hard to detect, because
each path is individually self-consistent.

Here the only difference between the two is how rows arrive and where the
output goes. ``tests/test_engine.py`` asserts the two produce byte-identical
decisions on the same input, so the drift is not merely discouraged but
checked.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.artifacts import ModelBundle, load_bundle
from src.config import DEFAULT_COSTS, CostParameters
from src.explain import (
    ReasonCode,
    ShapExplainer,
    scorecard_reason_codes,
    shap_reason_codes,
)
from src.policy import assign_score_band

logger = logging.getLogger(__name__)

# Decimal places retained when hashing applicant inputs. Six is finer than any
# feature this system takes (the most precise are ratios recorded to four) while
# being coarse enough to be insensitive to float representation.
HASH_FLOAT_PRECISION = 6


def hash_features(features: dict[str, Any]) -> str:
    """Hash an applicant's inputs for the audit trail.

    The audit log records this rather than the raw feature values, so a
    decision can be tied to its exact inputs months later without the log
    itself becoming a copy of the applicant database. Anyone holding the
    original application can recompute the hash and prove which inputs produced
    which decision.

    Floats are rounded to :data:`HASH_FLOAT_PRECISION` decimals before
    serialising. Without that, the digest depends on the exact repr of a float,
    and a value arriving as ``0.1 + 0.2`` hashes differently from ``0.3``
    despite describing the same application. For a record that has to be
    reconstructible years later, tying the hash to floating-point trailing
    digits is a needless fragility.

    Args:
        features: Applicant feature values.

    Returns:
        Hexadecimal SHA-256 digest of the canonicalised inputs.
    """
    rounded = {
        key: round(value, HASH_FLOAT_PRECISION) if isinstance(value, float) else value
        for key, value in features.items()
    }
    canonical = json.dumps(rounded, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


@dataclass
class Decision:
    """A single credit decision and everything needed to defend it.

    Attributes:
        decision: ``"approve"``, ``"refer"`` or ``"decline"``.
        probability_of_default: Calibrated PD.
        score: Scorecard points, if a scorecard is bundled.
        score_band: Reporting band ``A`` to ``E``.
        expected_loss: ``PD x LGD x EAD``.
        expected_margin: Expected margin if the applicant repays.
        expected_value: Expected margin less expected loss. Positive means the
            application is worth writing.
        exposure_at_default: Exposure used in the calculation.
        threshold: PD cutoff in force at decision time.
        model_version: Artifact version that decided.
        reason_codes: Principal reasons, populated for declines only.
        input_hash: Hash of the applicant inputs.
    """

    decision: str
    probability_of_default: float
    score: float | None
    score_band: str | None
    expected_loss: float
    expected_margin: float
    expected_value: float
    exposure_at_default: float
    threshold: float
    model_version: str
    reason_codes: list[dict[str, Any]]
    input_hash: str

    def to_dict(self) -> dict[str, Any]:
        """Return the decision as a plain dictionary."""
        return {
            "decision": self.decision,
            "probability_of_default": self.probability_of_default,
            "score": self.score,
            "score_band": self.score_band,
            "expected_loss": self.expected_loss,
            "expected_margin": self.expected_margin,
            "expected_value": self.expected_value,
            "exposure_at_default": self.exposure_at_default,
            "threshold": self.threshold,
            "model_version": self.model_version,
            "reason_codes": self.reason_codes,
            "input_hash": self.input_hash,
        }


class DecisionEngine:
    """Scores applicants and turns probabilities into defensible decisions."""

    def __init__(
        self,
        bundle: ModelBundle,
        costs: CostParameters = DEFAULT_COSTS,
        threshold: float | None = None,
        referral_width: float = 0.0,
        explainer_background: pd.DataFrame | None = None,
        top_k_reasons: int = 4,
    ) -> None:
        """Initialise the engine from a loaded bundle.

        Args:
            bundle: A validated model bundle.
            costs: Cost assumptions for the expected-loss arithmetic.
            threshold: Override the manifest's cutoff. Normally left as
                ``None`` so the cutoff travels with the artifact.
            referral_width: Half-width of an optional manual review band.
            explainer_background: Background rows enabling SHAP reason codes
                when no scorecard is bundled. Optional because building a SHAP
                explainer is expensive and the scorecard path does not need it.
            top_k_reasons: Number of principal reasons to return on a decline.
        """
        self.bundle = bundle
        self.costs = costs
        self.threshold = float(bundle.manifest.threshold if threshold is None else threshold)
        self.referral_width = referral_width
        self.top_k_reasons = top_k_reasons
        self.feature_names = list(bundle.manifest.feature_names)

        self.explainer: ShapExplainer | None = None
        if bundle.scorecard is None and explainer_background is not None:
            self.explainer = ShapExplainer(bundle.model, explainer_background)

    @classmethod
    def from_artifacts(
        cls,
        version: str | None = None,
        costs: CostParameters = DEFAULT_COSTS,
        **kwargs: Any,
    ) -> DecisionEngine:
        """Load the current (or a named) bundle and build an engine from it.

        Args:
            version: Artifact version, or ``None`` for the current pointer.
            costs: Cost assumptions.
            **kwargs: Forwarded to the constructor.

        Returns:
            The engine.
        """
        return cls(load_bundle(version), costs=costs, **kwargs)

    @property
    def model_version(self) -> str:
        """Version string of the loaded artifact."""
        return self.bundle.manifest.version

    def _validate_columns(self, frame: pd.DataFrame) -> None:
        """Fail loudly when required features are absent.

        Args:
            frame: Incoming applicant frame.

        Raises:
            ValueError: If any model input is missing.
        """
        missing = [name for name in self.feature_names if name not in frame.columns]
        if missing:
            raise ValueError(
                f"Missing required features: {missing}. "
                f"Model {self.model_version} expects: {self.feature_names}"
            )

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        """Return calibrated P(default) for a frame of applicants.

        Args:
            frame: Applicant features.

        Returns:
            Array of calibrated probabilities.
        """
        self._validate_columns(frame)
        return np.asarray(self.bundle.predict_proba(frame[self.feature_names]), dtype=float)

    def scores(self, frame: pd.DataFrame) -> np.ndarray | None:
        """Return scorecard points if a scorecard is bundled, else ``None``.

        Args:
            frame: Applicant features.

        Returns:
            Array of points, or ``None``.
        """
        if self.bundle.scorecard is None:
            return None
        return np.asarray(self.bundle.scorecard.score(frame[self.feature_names]), dtype=float)

    def _reason_codes(self, row: pd.DataFrame) -> list[ReasonCode]:
        """Compute principal reasons for one applicant."""
        if self.bundle.scorecard is not None:
            return scorecard_reason_codes(self.bundle.scorecard, row, top_k=self.top_k_reasons)
        if self.explainer is not None:
            return shap_reason_codes(self.explainer, row, top_k=self.top_k_reasons)
        return []

    def decide(
        self,
        frame: pd.DataFrame,
        exposure: np.ndarray | pd.Series | None = None,
        with_reasons: bool = True,
    ) -> list[Decision]:
        """Score applicants and produce full decision records.

        The single code path. Both the API and the batch scorer enter here.

        Args:
            frame: Applicant features, one row per applicant.
            exposure: Per-applicant exposure at default. Falls back to
                ``loan_amount`` if present, then to the flat cost assumption.
            with_reasons: Compute reason codes for declines. Batch runs over
                large portfolios may switch this off, since reason codes are
                computed per row and are only needed for adverse decisions.

        Returns:
            One :class:`Decision` per input row, in input order.
        """
        self._validate_columns(frame)
        probabilities = self.predict_proba(frame)
        points = self.scores(frame)

        if exposure is not None:
            exposures = np.asarray(exposure, dtype=float)
        elif "loan_amount" in frame.columns:
            exposures = frame["loan_amount"].to_numpy(dtype=float)
        else:
            exposures = np.full(len(frame), self.costs.exposure_at_default, dtype=float)

        decisions: list[Decision] = []
        for position in range(len(frame)):
            probability = float(probabilities[position])
            exposure_value = float(exposures[position])

            if self.referral_width > 0.0 and abs(probability - self.threshold) <= self.referral_width:
                outcome = "refer"
            elif probability < self.threshold:
                outcome = "approve"
            else:
                outcome = "decline"

            row = frame.iloc[[position]]
            features = {
                name: _jsonable(row[name].iloc[0]) for name in self.feature_names
            }

            # Reasons are computed for adverse outcomes only. They are the
            # basis of the adverse action notice; an approved applicant has no
            # principal reasons for a decline.
            reasons: list[dict[str, Any]] = []
            if with_reasons and outcome != "approve":
                reasons = [code.to_dict() for code in self._reason_codes(row[self.feature_names])]

            expected_loss = probability * self.costs.lgd * exposure_value
            expected_margin = (1.0 - probability) * self.costs.margin_rate * exposure_value

            decisions.append(
                Decision(
                    decision=outcome,
                    probability_of_default=probability,
                    score=None if points is None else float(points[position]),
                    score_band=None if points is None else assign_score_band(float(points[position])),
                    expected_loss=expected_loss,
                    expected_margin=expected_margin,
                    expected_value=expected_margin - expected_loss,
                    exposure_at_default=exposure_value,
                    threshold=self.threshold,
                    model_version=self.model_version,
                    reason_codes=reasons,
                    input_hash=hash_features(features),
                )
            )
        return decisions

    def decide_frame(
        self,
        frame: pd.DataFrame,
        exposure: np.ndarray | pd.Series | None = None,
        with_reasons: bool = False,
    ) -> pd.DataFrame:
        """Decide a portfolio and return the results as a dataframe.

        Args:
            frame: Applicant features.
            exposure: Per-applicant exposure at default.
            with_reasons: Include reason codes as a JSON string column.

        Returns:
            One row per applicant, aligned to the input index.
        """
        decisions = self.decide(frame, exposure=exposure, with_reasons=with_reasons)
        records = []
        for decision in decisions:
            record = decision.to_dict()
            record["reason_codes"] = json.dumps(record["reason_codes"])
            records.append(record)
        return pd.DataFrame(records, index=frame.index)


def _jsonable(value: object) -> object:
    """Coerce numpy and pandas scalars to JSON-serialisable Python types."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, str | int | float | bool):
        return None if isinstance(value, float) and np.isnan(value) else value
    if value is None or value is pd.NaT:
        return None
    return str(value)
