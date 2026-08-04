"""Probability calibration, treated as a first-class requirement.

Why calibration is not an add-on
--------------------------------
Expected credit loss is ``EL = PD x LGD x EAD``. That identity consumes a
*probability*, not a ranking. A model with an excellent AUC and a badly
calibrated output will rank applicants perfectly and then price every one of
them wrong — under-provisioning the book, mispricing risk-based interest, and
producing a cost-optimal cutoff that is optimal for nothing.

AUC is invariant to any monotone transformation of the score. You can multiply
every predicted probability by 0.5 and the AUC will not move by a thousandth.
Every number the business actually acts on will be wrong. This is why
calibration gets its own module and its own regression gate in CI.

What is measured
----------------
* **Brier score** — mean squared error on probabilities. A proper scoring rule:
  it is minimised only by the true probability, so it cannot be gamed by
  sharpening or hedging. Decomposed here into reliability, resolution and
  uncertainty.
* **Expected calibration error (ECE)** — mean absolute gap between predicted
  and observed frequency, averaged over bins and weighted by bin population.
  Reported with **quantile** bins by default: on an imbalanced credit book,
  uniform-width bins put almost every applicant in the first two bins and leave
  the rest nearly empty, which flatters the result. Both are reported.
* **Calibration slope and intercept** — regress the outcome on the logit of the
  prediction. Perfect calibration gives intercept 0 and slope 1. This is the
  standard supervisory diagnostic and it says *how* a model is wrong: a slope
  below 1 means predictions are too spread out, an intercept away from 0 means
  the overall level is off. ECE alone cannot distinguish those.
* **Reliability diagrams** — the picture that makes the failure obvious.

Three calibrators are compared: none (raw), Platt scaling, and isotonic
regression. See ``README.md`` for which is recommended and why.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)

# Clip probabilities before taking logits. 1e-6 keeps logits inside +/- 13.8,
# which is far outside any realistic PD and avoids infinities from a model that
# returns a hard 0 or 1.
PROBABILITY_EPSILON = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    """Log-odds of a probability, clipped away from 0 and 1."""
    clipped = np.clip(np.asarray(p, dtype=float), PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON)
    return np.log(clipped / (1.0 - clipped))


@runtime_checkable
class Calibrator(Protocol):
    """Maps raw model scores onto calibrated probabilities."""

    name: str

    def fit(self, scores: np.ndarray, y: np.ndarray) -> Calibrator:
        """Fit the calibration map on held-out scores."""
        ...

    def transform(self, scores: np.ndarray) -> np.ndarray:
        """Apply the calibration map."""
        ...


class IdentityCalibrator:
    """Pass-through. The honest baseline that Platt and isotonic must beat."""

    name = "raw"

    def fit(self, scores: np.ndarray, y: np.ndarray) -> IdentityCalibrator:  # noqa: ARG002
        """No-op. Present so the calibrators share an interface.

        Args:
            scores: Ignored.
            y: Ignored.

        Returns:
            Self.
        """
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        """Return the scores unchanged, clipped into the open unit interval."""
        return np.clip(np.asarray(scores, dtype=float), PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON)


class PlattCalibrator:
    """Logistic recalibration of the log-odds.

    Fits ``P(default) = sigmoid(a + b * logit(score))``. Two parameters only,
    which is the attraction: it is nearly impossible to overfit, it extrapolates
    sensibly into the sparse high-risk tail where isotonic runs out of data, and
    ``a = 0, b = 1`` recovers the identity, so the fitted values are directly
    interpretable as the calibration intercept and slope.

    The cost is rigidity. If a model's miscalibration is not sigmoidal, Platt
    cannot fix it.
    """

    name = "platt"

    def __init__(self) -> None:
        """Initialise with an unfitted logistic regression."""
        self.model = LogisticRegression(C=1e6, max_iter=2000, solver="lbfgs")

    def fit(self, scores: np.ndarray, y: np.ndarray) -> PlattCalibrator:
        """Fit on held-out scores.

        Args:
            scores: Raw model probabilities from data the model did not train on.
            y: Binary default flag.

        Returns:
            Self.
        """
        self.model.fit(_logit(scores).reshape(-1, 1), np.asarray(y).astype(int))
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        """Apply the fitted sigmoid."""
        return self.model.predict_proba(_logit(scores).reshape(-1, 1))[:, 1]

    @property
    def parameters(self) -> tuple[float, float]:
        """Fitted ``(intercept, slope)``. Identity calibration is ``(0, 1)``."""
        return float(self.model.intercept_[0]), float(self.model.coef_[0][0])


class IsotonicCalibrator:
    """Non-parametric monotone calibration.

    Fits an arbitrary non-decreasing step function, so it can correct any
    monotone distortion, not just a sigmoidal one. It will usually win on
    in-sample-period Brier score.

    Two real costs, both of which matter more in credit than in most settings:
    it produces a step function, so many applicants receive identical
    probabilities and the fine-grained ranking within a step is destroyed; and
    it cannot extrapolate, so it clamps at the highest and lowest calibration
    values it saw. In the sparse high-PD tail — precisely where the cutoff sits
    — that is where the data runs thinnest.
    """

    name = "isotonic"

    def __init__(self) -> None:
        """Initialise with an unfitted isotonic regression clipped to [0, 1]."""
        self.model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)

    def fit(self, scores: np.ndarray, y: np.ndarray) -> IsotonicCalibrator:
        """Fit on held-out scores.

        Args:
            scores: Raw model probabilities from data the model did not train on.
            y: Binary default flag.

        Returns:
            Self.
        """
        self.model.fit(np.asarray(scores, dtype=float), np.asarray(y).astype(float))
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        """Apply the fitted step function."""
        return np.clip(
            self.model.predict(np.asarray(scores, dtype=float)),
            PROBABILITY_EPSILON,
            1.0 - PROBABILITY_EPSILON,
        )


def build_calibrators() -> dict[str, Calibrator]:
    """Return one unfitted instance of each calibration strategy."""
    return {
        "raw": IdentityCalibrator(),
        "platt": PlattCalibrator(),
        "isotonic": IsotonicCalibrator(),
    }


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def brier_score(y: np.ndarray, p: np.ndarray) -> float:
    """Mean squared error between predicted probability and outcome.

    Args:
        y: Binary default flag.
        p: Predicted probabilities.

    Returns:
        Brier score. Lower is better; 0 is perfect.
    """
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2))


@dataclass(frozen=True)
class BrierDecomposition:
    """Murphy decomposition of the Brier score.

    ``Brier = reliability - resolution + uncertainty``.

    Attributes:
        reliability: Calibration error. Lower is better, 0 is perfect.
        resolution: How far bin outcomes deviate from the base rate. Higher is
            better — it is the discriminating power.
        uncertainty: Base-rate variance ``p(1-p)``. A property of the data, not
            the model; nothing can reduce it.
        brier: The reconstructed score.
    """

    reliability: float
    resolution: float
    uncertainty: float
    brier: float


def brier_decomposition(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> BrierDecomposition:
    """Split a Brier score into calibration and discrimination components.

    Useful because two models with identical Brier scores can be wrong in
    opposite ways: one poorly calibrated but sharp, the other well calibrated
    but uninformative. Only the first is fixable by recalibration.

    Args:
        y: Binary default flag.
        p: Predicted probabilities.
        n_bins: Number of equal-mass bins.

    Returns:
        The decomposition.
    """
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    base_rate = float(y.mean())

    edges = np.unique(np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1)))
    bin_index = np.clip(np.digitize(p, edges[1:-1], right=True), 0, len(edges) - 2)

    reliability = 0.0
    resolution = 0.0
    for b in np.unique(bin_index):
        mask = bin_index == b
        n_b = int(mask.sum())
        if n_b == 0:
            continue
        mean_p = float(p[mask].mean())
        observed = float(y[mask].mean())
        reliability += n_b * (mean_p - observed) ** 2
        resolution += n_b * (observed - base_rate) ** 2

    reliability /= len(y)
    resolution /= len(y)
    uncertainty = base_rate * (1.0 - base_rate)
    return BrierDecomposition(
        reliability=reliability,
        resolution=resolution,
        uncertainty=uncertainty,
        brier=reliability - resolution + uncertainty,
    )


def _bin_edges(p: np.ndarray, n_bins: int, strategy: str) -> np.ndarray:
    """Compute calibration bin edges under the requested strategy."""
    if strategy == "quantile":
        return np.unique(np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1)))
    if strategy == "uniform":
        return np.linspace(0.0, 1.0, n_bins + 1)
    raise ValueError(f"Unknown binning strategy: {strategy!r}")


def reliability_table(
    y: np.ndarray, p: np.ndarray, n_bins: int = 10, strategy: str = "quantile"
) -> pd.DataFrame:
    """Observed versus predicted default rate, by score bin.

    Args:
        y: Binary default flag.
        p: Predicted probabilities.
        n_bins: Number of bins.
        strategy: ``"quantile"`` for equal-mass bins or ``"uniform"`` for
            equal-width bins.

    Returns:
        Frame with one row per non-empty bin: count, mean prediction, observed
        rate, and the signed gap between them.
    """
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = _bin_edges(p, n_bins, strategy)
    bin_index = np.clip(np.digitize(p, edges[1:-1], right=True), 0, len(edges) - 2)

    rows = []
    for b in range(len(edges) - 1):
        mask = bin_index == b
        n_b = int(mask.sum())
        if n_b == 0:
            continue
        mean_p = float(p[mask].mean())
        observed = float(y[mask].mean())
        rows.append(
            {
                "bin": b,
                "lower": float(edges[b]),
                "upper": float(edges[b + 1]),
                "count": n_b,
                "mean_predicted": mean_p,
                "observed_rate": observed,
                "gap": mean_p - observed,
            }
        )
    return pd.DataFrame(rows)


def expected_calibration_error(
    y: np.ndarray, p: np.ndarray, n_bins: int = 10, strategy: str = "quantile"
) -> float:
    """Population-weighted mean absolute calibration gap.

    Args:
        y: Binary default flag.
        p: Predicted probabilities.
        n_bins: Number of bins.
        strategy: ``"quantile"`` or ``"uniform"``.

    Returns:
        ECE in probability units. 0.02 means the average applicant's predicted
        PD is two percentage points away from the realised rate for applicants
        who look like them.
    """
    table = reliability_table(y, p, n_bins=n_bins, strategy=strategy)
    if table.empty:
        return float("nan")
    weights = table["count"].to_numpy(dtype=float)
    return float(np.sum(weights * table["gap"].abs().to_numpy()) / weights.sum())


def maximum_calibration_error(
    y: np.ndarray, p: np.ndarray, n_bins: int = 10, strategy: str = "quantile"
) -> float:
    """Worst absolute calibration gap over all bins.

    ECE can hide a single catastrophic bin behind nine good ones. In credit the
    catastrophic bin is usually the high-PD tail, which is exactly where the
    cutoff sits, so the maximum matters as well as the mean.

    Args:
        y: Binary default flag.
        p: Predicted probabilities.
        n_bins: Number of bins.
        strategy: ``"quantile"`` or ``"uniform"``.

    Returns:
        Maximum absolute gap across bins.
    """
    table = reliability_table(y, p, n_bins=n_bins, strategy=strategy)
    return float("nan") if table.empty else float(table["gap"].abs().max())


def calibration_slope_intercept(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Regress the outcome on the logit of the prediction.

    Fits ``logit(P(y=1)) = intercept + slope * logit(p)``. Perfect calibration
    gives ``(0, 1)``.

    Interpretation, which ECE cannot give you:

    * **slope < 1** — predictions are too spread out; the model is
      overconfident at both ends. The usual symptom of an over-fitted booster.
    * **slope > 1** — predictions are too compressed; the model is
      underconfident. Common after heavy regularisation.
    * **intercept != 0** — the overall level is wrong, usually a base-rate shift
      between the calibration period and the scoring period.

    Args:
        y: Binary default flag.
        p: Predicted probabilities.

    Returns:
        Tuple of ``(intercept, slope)``.
    """
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    model = LogisticRegression(C=1e6, max_iter=2000, solver="lbfgs")
    model.fit(_logit(p).reshape(-1, 1), y)
    return float(model.intercept_[0]), float(model.coef_[0][0])


def calibration_report(
    y: np.ndarray, p: np.ndarray, n_bins: int = 10, label: str = ""
) -> dict[str, float | str]:
    """Full calibration summary for one set of predictions.

    Args:
        y: Binary default flag.
        p: Predicted probabilities.
        n_bins: Number of calibration bins.
        label: Optional identifier carried through into the output.

    Returns:
        Mapping of metric name to value.
    """
    intercept, slope = calibration_slope_intercept(y, p)
    decomposition = brier_decomposition(y, p, n_bins=n_bins)
    return {
        "label": label,
        "n": float(len(y)),
        "observed_rate": float(np.mean(y)),
        "mean_predicted": float(np.mean(p)),
        "bias": float(np.mean(p) - np.mean(y)),
        "brier": brier_score(y, p),
        "brier_reliability": decomposition.reliability,
        "brier_resolution": decomposition.resolution,
        "ece_quantile": expected_calibration_error(y, p, n_bins, "quantile"),
        "ece_uniform": expected_calibration_error(y, p, n_bins, "uniform"),
        "mce_quantile": maximum_calibration_error(y, p, n_bins, "quantile"),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }


# --------------------------------------------------------------------------- #
# Fitting and comparison
# --------------------------------------------------------------------------- #


class CalibratedModel:
    """A fitted model bound to a fitted calibrator.

    This pairing is the deployable unit. Serving a model without its calibrator
    — or with the wrong one — silently produces well-ranked, wrongly-priced
    decisions, which is why :mod:`src.artifacts` refuses to load a mismatched
    pair and the API refuses to start on one.
    """

    def __init__(self, model: object, calibrator: Calibrator, name: str = "") -> None:
        """Bind a model to a calibrator.

        Args:
            model: Any object exposing ``predict_proba(X) -> np.ndarray``.
            calibrator: A fitted calibrator.
            name: Identifier used in reports.
        """
        self.model = model
        self.calibrator = calibrator
        self.name = name or f"{type(model).__name__}+{calibrator.name}"

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return calibrated P(default)."""
        return self.calibrator.transform(self.model.predict_proba(X))  # type: ignore[attr-defined]

    def predict_raw(self, X: pd.DataFrame) -> np.ndarray:
        """Return uncalibrated P(default), for diagnostics."""
        return self.model.predict_proba(X)  # type: ignore[attr-defined]


def fit_calibrators(scores: np.ndarray, y: np.ndarray) -> dict[str, Calibrator]:
    """Fit every calibration strategy on a held-out calibration set.

    The calibration set must be data the *model* did not train on. Fitting a
    calibrator on in-sample scores produces a calibrator that corrects the
    model's in-sample optimism and nothing else — it will look excellent in
    validation and fail in production.

    Args:
        scores: Raw model probabilities on the calibration set.
        y: Binary default flag on the calibration set.

    Returns:
        Mapping from calibrator name to fitted calibrator.
    """
    fitted: dict[str, Calibrator] = {}
    for name, calibrator in build_calibrators().items():
        fitted[name] = calibrator.fit(scores, y)
    return fitted


def compare_calibrators(
    calibrators: dict[str, Calibrator],
    scores: np.ndarray,
    y: np.ndarray,
    n_bins: int = 10,
    model_name: str = "",
) -> pd.DataFrame:
    """Score every calibrator on an evaluation set.

    Args:
        calibrators: Fitted calibrators.
        scores: Raw model probabilities on the evaluation set.
        y: Binary default flag on the evaluation set.
        n_bins: Number of calibration bins.
        model_name: Carried into the output for joining.

    Returns:
        One row per calibrator, sorted by Brier score.
    """
    rows = []
    for name, calibrator in calibrators.items():
        report = calibration_report(y, calibrator.transform(scores), n_bins=n_bins, label=name)
        report["model"] = model_name
        report["calibrator"] = name
        rows.append(report)
    return pd.DataFrame(rows).sort_values("brier", ignore_index=True)


def select_calibrator(
    comparison: pd.DataFrame,
    slope_tolerance: float = 0.15,
) -> str:
    """Choose a calibrator from a comparison table, with a stated rule.

    The rule, applied in order:

    1. Prefer a calibrator whose calibration slope is within ``slope_tolerance``
       of 1.0. A good Brier score with a slope of 0.7 means the model is still
       systematically overconfident, which no amount of average-case accuracy
       makes acceptable for pricing.
    2. Among those, take the lowest ECE.
    3. If none qualifies, fall back to the lowest Brier score and let the
       caller see it in the report.

    This is a documented policy, not a search. It is written down so that a
    reviewer can disagree with it explicitly.

    Args:
        comparison: Output of :func:`compare_calibrators`.
        slope_tolerance: Allowed absolute deviation of the slope from 1.0.

    Returns:
        The chosen calibrator name.
    """
    eligible = comparison[(comparison["calibration_slope"] - 1.0).abs() <= slope_tolerance]
    if not eligible.empty:
        return str(eligible.sort_values("ece_quantile").iloc[0]["calibrator"])
    logger.warning(
        "No calibrator achieved a slope within %.2f of 1.0; falling back to lowest Brier",
        slope_tolerance,
    )
    return str(comparison.sort_values("brier").iloc[0]["calibrator"])
