"""Production monitoring: drift, vintage performance, and what triggers a review.

A credit model does not fail loudly. It fails by being quietly wrong about a
population that has changed since it was fitted, while its accuracy metrics
stay unavailable because the outcomes have not matured yet. By the time the bad
rate confirms the problem, twelve months of business has been written on it.

Monitoring therefore watches two different clocks:

* **Input drift, available immediately.** Population stability on the score
  distribution and on each feature. Detects that the applicants have changed
  before any of them have had time to default.
* **Vintage performance, available slowly.** Cumulative default by months on
  book for each origination cohort. This is the measure that actually settles
  whether the model is still right, and it arrives a year late — which is
  exactly why the leading indicators matter.

Thresholds and why these numbers
--------------------------------
The PSI levels (0.10 watch, 0.25 review) are the long-standing industry
convention, and their real justification is institutional rather than
statistical: they are the numbers a credit risk committee already recognises,
and a monitoring framework nobody acts on is worse than none.

Two properties are worth stating plainly, because PSI is often over-read:

* PSI has **no sampling distribution attached to these cut-offs**. It scales
  with sample size, so a large enough monthly batch will eventually breach 0.10
  on noise alone. Volumes should be comparable across comparisons.
* PSI detects that a distribution moved. It says **nothing about whether the
  model still works** on the moved distribution. A population can shift
  substantially with no loss of discrimination, and can shift barely at all
  while the relationship between features and default breaks completely. PSI is
  a prompt to investigate, never a verdict.

What happened when this was run — PSI missed it
-----------------------------------------------
The synthetic book contains a large, deliberate deterioration. Vintage analysis
sees it immediately and unambiguously: cohorts v00-v23 default at 16-17.5% by
month twelve, v24-29 at 21.8%, and v30-35 at **30.7%, a 75% increase** over the
first cohort.

The conventional input-drift monitoring did **not** fire:

======================  =======  ===============
Indicator               Value    Conventional status
======================  =======  ===============
Score PSI               0.103    watch, not review
Worst feature PSI       0.212    watch, not review
Calibration ECE         0.028    within tolerance
======================  =======  ===============

A framework operated strictly on 0.25 would have raised nothing while realised
losses rose by three quarters. That is not an argument for quietly lowering the
threshold until it fires on this particular example — that is fitting the
governance to the incident. It is an argument for three things, which is how
:func:`evaluate_triggers` is structured:

1. **Input drift metrics are a prompt, not a safety net.** PSI answers "did the
   applicants change?", and the damage here came mostly from a change in the
   *relationship* between features and default, which no input-side metric can
   see by construction.
2. **The lagging indicator is the one that settles it**, and it is slow. There
   is no way to make vintage performance arrive faster; the only honest
   response is to size the exposure taken between review points accordingly.
3. **Escalate on the "watch" band, do not wait for "review".** Two indicators
   simultaneously in watch, in the same direction, is itself a finding.

:func:`evaluate_triggers` combines the leading and lagging indicators into an
explicit action, so that "the model is being monitored" means something
falsifiable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.calibration import expected_calibration_error
from src.config import DEFAULT_MONITORING, MonitoringThresholds

logger = logging.getLogger(__name__)

# Floor applied to bin proportions so an empty bin does not produce an infinite
# PSI. 1e-4 corresponds to one observation in ten thousand.
PSI_EPSILON = 1e-4


def population_stability_index(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
    edges: np.ndarray | None = None,
) -> float:
    """Population stability index between a reference and a current sample.

    ``PSI = sum over bins of (current% - reference%) * ln(current% / reference%)``

    Bins come from the **reference** quantiles, not the current ones. Re-binning
    on the current sample each period would partly absorb the very shift the
    metric is supposed to detect.

    Args:
        reference: Reference sample, typically the training or last-approved
            population.
        current: Current sample.
        n_bins: Number of bins for numeric data.
        edges: Explicit bin edges, overriding ``n_bins``.

    Returns:
        PSI. Below 0.10 is stable, 0.10-0.25 warrants watching, above 0.25
        warrants a formal review.
    """
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    reference = reference[~np.isnan(reference)]
    current = current[~np.isnan(current)]

    if len(reference) == 0 or len(current) == 0:
        return float("nan")

    if edges is None:
        edges = np.unique(np.quantile(reference, np.linspace(0.0, 1.0, n_bins + 1)))
    if len(edges) < 3:
        return 0.0

    inner = edges[1:-1]
    reference_counts = np.bincount(
        np.digitize(reference, inner, right=True), minlength=len(edges) - 1
    ).astype(float)
    current_counts = np.bincount(
        np.digitize(current, inner, right=True), minlength=len(edges) - 1
    ).astype(float)

    reference_share = np.maximum(reference_counts / reference_counts.sum(), PSI_EPSILON)
    current_share = np.maximum(current_counts / current_counts.sum(), PSI_EPSILON)

    return float(np.sum((current_share - reference_share) * np.log(current_share / reference_share)))


def categorical_stability_index(reference: pd.Series, current: pd.Series) -> float:
    """PSI for a categorical feature, computed over its levels.

    Args:
        reference: Reference sample.
        current: Current sample.

    Returns:
        PSI across the union of levels seen in either sample.
    """
    reference_share = reference.astype(str).value_counts(normalize=True)
    current_share = current.astype(str).value_counts(normalize=True)
    levels = sorted(set(reference_share.index) | set(current_share.index))

    total = 0.0
    for level in levels:
        r = max(float(reference_share.get(level, 0.0)), PSI_EPSILON)
        c = max(float(current_share.get(level, 0.0)), PSI_EPSILON)
        total += (c - r) * np.log(c / r)
    return float(total)


def drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features: list[str],
    thresholds: MonitoringThresholds = DEFAULT_MONITORING,
    n_bins: int = 10,
) -> pd.DataFrame:
    """PSI for every feature, with a status label.

    Args:
        reference: Reference population.
        current: Current population.
        features: Columns to test.
        thresholds: Governance thresholds.
        n_bins: Bins for numeric features.

    Returns:
        One row per feature with PSI, status, and reference/current means,
        sorted worst first.
    """
    rows = []
    for feature in features:
        if feature not in reference.columns or feature not in current.columns:
            logger.warning("Feature %r missing from one of the samples; skipping", feature)
            continue

        if pd.api.types.is_numeric_dtype(reference[feature]):
            psi = population_stability_index(
                reference[feature].to_numpy(), current[feature].to_numpy(), n_bins=n_bins
            )
            reference_mean: float | None = float(reference[feature].mean())
            current_mean: float | None = float(current[feature].mean())
        else:
            psi = categorical_stability_index(reference[feature], current[feature])
            reference_mean = current_mean = None

        rows.append(
            {
                "feature": feature,
                "psi": psi,
                "status": classify_psi(psi, thresholds),
                "reference_mean": reference_mean,
                "current_mean": current_mean,
                "relative_change": (
                    (current_mean - reference_mean) / reference_mean
                    if reference_mean not in (None, 0.0) and current_mean is not None
                    else None
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("psi", ascending=False, ignore_index=True)


def classify_psi(psi: float, thresholds: MonitoringThresholds = DEFAULT_MONITORING) -> str:
    """Label a PSI value against the governance thresholds.

    Args:
        psi: PSI value.
        thresholds: Governance thresholds.

    Returns:
        ``"stable"``, ``"watch"`` or ``"review"``.
    """
    if np.isnan(psi):
        return "unknown"
    if psi >= thresholds.psi_breach:
        return "review"
    if psi >= thresholds.psi_warn:
        return "watch"
    return "stable"


def score_distribution_report(
    reference_scores: np.ndarray,
    current_scores: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Compare the reference and current score distributions bin by bin.

    The score PSI is the single most important monitoring number, because it
    aggregates drift across every feature weighted by how much the model
    actually relies on them. A feature can move sharply and barely shift the
    score; that is the model being robust, not the monitoring failing.

    Args:
        reference_scores: Reference model scores.
        current_scores: Current model scores.
        n_bins: Number of bins.

    Returns:
        Per-bin shares and PSI contributions, with the total in every row for
        convenience.
    """
    reference_scores = np.asarray(reference_scores, dtype=float)
    current_scores = np.asarray(current_scores, dtype=float)
    edges = np.unique(np.quantile(reference_scores, np.linspace(0.0, 1.0, n_bins + 1)))
    inner = edges[1:-1]

    reference_counts = np.bincount(
        np.digitize(reference_scores, inner, right=True), minlength=len(edges) - 1
    ).astype(float)
    current_counts = np.bincount(
        np.digitize(current_scores, inner, right=True), minlength=len(edges) - 1
    ).astype(float)

    reference_share = np.maximum(reference_counts / reference_counts.sum(), PSI_EPSILON)
    current_share = np.maximum(current_counts / current_counts.sum(), PSI_EPSILON)
    contributions = (current_share - reference_share) * np.log(current_share / reference_share)

    return pd.DataFrame(
        {
            "bin": np.arange(len(reference_share)),
            "lower": edges[:-1],
            "upper": edges[1:],
            "reference_share": reference_share,
            "current_share": current_share,
            "psi_contribution": contributions,
            "psi_total": contributions.sum(),
        }
    )


# --------------------------------------------------------------------------- #
# Vintage analysis
# --------------------------------------------------------------------------- #


def vintage_curves(
    frame: pd.DataFrame,
    vintage_column: str = "vintage",
    outcome_column: str = "default_true",
    timing_column: str = "months_to_default",
    max_months: int = 12,
    group_size: int = 6,
) -> pd.DataFrame:
    """Cumulative default rate by months on book, for each origination cohort.

    The classic credit chart. Each line is one cohort; a line sitting above the
    ones before it means the business written in that period is performing
    worse at the same age, which separates a genuine deterioration in
    underwriting or population from the arithmetic artefact of a growing book
    (where recent, unseasoned accounts drag the headline rate down).

    Args:
        frame: Loan book with vintage, outcome and default-timing columns.
        vintage_column: Origination cohort index.
        outcome_column: Binary default flag.
        timing_column: Months on book at default; 0 for non-defaulters.
        max_months: Months of the performance window to trace.
        group_size: Number of vintages pooled into one reporting cohort.
            Monthly cohorts are usually too thin to read; six-month pools are
            the common compromise.

    Returns:
        Long-format frame of cohort, months on book, cumulative default rate
        and cohort size.
    """
    working = frame[[vintage_column, outcome_column, timing_column]].copy()
    working["cohort"] = (working[vintage_column] // group_size) * group_size

    rows = []
    for cohort, cohort_frame in working.groupby("cohort", observed=True):
        n = len(cohort_frame)
        if n == 0:
            continue
        timing = cohort_frame[timing_column].to_numpy()
        outcome = cohort_frame[outcome_column].to_numpy()
        cohort_start = int(np.asarray(cohort).item())
        for month in range(1, max_months + 1):
            defaulted = int(((outcome == 1) & (timing <= month) & (timing > 0)).sum())
            rows.append(
                {
                    "cohort": f"v{cohort_start:02d}-{cohort_start + group_size - 1:02d}",
                    "cohort_start_vintage": cohort_start,
                    "months_on_book": month,
                    "n": n,
                    "cumulative_defaults": defaulted,
                    "cumulative_default_rate": defaulted / n,
                }
            )
    return pd.DataFrame(rows)


def vintage_summary(curves: pd.DataFrame, at_month: int = 12) -> pd.DataFrame:
    """Cross-section of the vintage curves at a fixed age.

    Args:
        curves: Output of :func:`vintage_curves`.
        at_month: Months on book at which to compare cohorts.

    Returns:
        One row per cohort with its default rate at that age and the change
        against the earliest cohort.
    """
    snapshot = curves[curves["months_on_book"] == at_month].copy()
    if snapshot.empty:
        return snapshot
    snapshot = snapshot.sort_values("cohort_start_vintage", ignore_index=True)
    baseline = float(snapshot["cumulative_default_rate"].iloc[0])
    snapshot["change_vs_first_cohort"] = snapshot["cumulative_default_rate"] - baseline
    snapshot["relative_change"] = snapshot["cumulative_default_rate"] / baseline - 1.0
    return snapshot


# --------------------------------------------------------------------------- #
# Triggers
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TriggerResult:
    """Outcome of one monitoring check.

    Attributes:
        name: Check identifier.
        value: Observed value.
        threshold: Level that would trigger action.
        breached: Whether the threshold was breached.
        action: What the breach requires.
        rationale: Why this threshold, in one sentence.
    """

    name: str
    value: float
    threshold: float
    breached: bool
    action: str
    rationale: str


def evaluate_triggers(
    score_psi: float,
    worst_feature_psi: float,
    current_ece: float | None = None,
    auc_drop: float | None = None,
    worst_subgroup_ece: float | None = None,
    thresholds: MonitoringThresholds = DEFAULT_MONITORING,
) -> pd.DataFrame:
    """Apply the governance thresholds and say what each result requires.

    Each row states the number, the level, and the action, so that the
    monitoring pack answers "and then what?" rather than stopping at a chart.

    Args:
        score_psi: PSI on the score distribution.
        worst_feature_psi: Highest PSI across individual features.
        current_ece: Expected calibration error on recently matured outcomes.
        auc_drop: Fall in AUC against the validation baseline.
        worst_subgroup_ece: Highest within-group ECE from the fairness audit.
        thresholds: Governance thresholds.

    Returns:
        One row per check.
    """
    checks: list[TriggerResult] = [
        TriggerResult(
            name="score_psi",
            value=score_psi,
            threshold=thresholds.psi_breach,
            breached=bool(score_psi >= thresholds.psi_breach),
            action="Open a model review; re-fit the calibrator before the model itself.",
            rationale=(
                "The applicant population the model scores has moved materially. "
                "Level is the industry convention and, more usefully, the number "
                "the credit committee already acts on."
            ),
        ),
        TriggerResult(
            name="worst_feature_psi",
            value=worst_feature_psi,
            threshold=thresholds.psi_breach,
            breached=bool(worst_feature_psi >= thresholds.psi_breach),
            action="Investigate the specific feature: upstream data change, or a real population shift.",
            rationale=(
                "A single feature moving hard is usually a broken pipeline or a "
                "changed bureau definition, which is cheaper to fix than a model."
            ),
        ),
    ]

    if current_ece is not None:
        checks.append(
            TriggerResult(
                name="calibration_ece",
                value=current_ece,
                threshold=thresholds.ece_breach,
                breached=bool(current_ece >= thresholds.ece_breach),
                action="Re-fit the calibrator on the most recent matured window; no need to re-fit the model.",
                rationale=(
                    "At 5% average absolute error, expected loss and risk-based "
                    "pricing are wrong by enough to matter, while ranking may be "
                    "perfectly intact. Recalibration is the cheap fix and should "
                    "be tried before anything else."
                ),
            )
        )

    if auc_drop is not None:
        checks.append(
            TriggerResult(
                name="auc_drop",
                value=auc_drop,
                threshold=thresholds.auc_drop_breach,
                breached=bool(auc_drop >= thresholds.auc_drop_breach),
                action="Re-fit or redevelop the model. Recalibration cannot restore lost ranking.",
                rationale=(
                    "Discrimination has genuinely degraded: the feature-outcome "
                    "relationship has changed, not just the level. This is the "
                    "failure recalibration cannot touch."
                ),
            )
        )

    # Combined-watch rule. Added because on this book the single-indicator
    # thresholds did not fire on a deterioration that raised realised losses by
    # 75%. Two indicators sitting in the watch band at once is itself a signal,
    # and waiting for either to reach "review" independently costs a year of
    # business.
    both_watching = bool(
        score_psi >= thresholds.psi_warn and worst_feature_psi >= thresholds.psi_warn
    )
    checks.append(
        TriggerResult(
            name="combined_watch",
            value=float(min(score_psi, worst_feature_psi)),
            threshold=thresholds.psi_warn,
            breached=both_watching,
            action="Bring forward the scheduled review; pull vintage curves for the most recent cohorts now.",
            rationale=(
                "Score and feature drift both in the watch band is a coherent "
                "population shift rather than noise in one variable. On this "
                "book that pattern accompanied a 75% rise in the 12-month "
                "default rate while neither indicator individually reached the "
                "review level."
            ),
        )
    )

    if worst_subgroup_ece is not None:
        checks.append(
            TriggerResult(
                name="subgroup_calibration",
                value=worst_subgroup_ece,
                threshold=thresholds.subgroup_ece_breach,
                breached=bool(worst_subgroup_ece >= thresholds.subgroup_ece_breach),
                action="Escalate to the fairness review; do not adjust cutoffs as a remedy.",
                rationale=(
                    "A subgroup is being priced against probabilities that are "
                    "wrong for it. Threshold changes redistribute decisions "
                    "without repairing the underlying numbers."
                ),
            )
        )

    return pd.DataFrame([vars(check) for check in checks])


def monitoring_snapshot(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features: list[str],
    reference_scores: np.ndarray,
    current_scores: np.ndarray,
    current_y: np.ndarray | None = None,
    thresholds: MonitoringThresholds = DEFAULT_MONITORING,
) -> dict[str, object]:
    """Produce a complete monitoring pack for one reporting period.

    Args:
        reference: Reference population.
        current: Current population.
        features: Features to test for drift.
        reference_scores: Reference model scores.
        current_scores: Current model scores.
        current_y: Matured outcomes for the current period, if any are
            available yet.
        thresholds: Governance thresholds.

    Returns:
        Mapping with the feature drift table, the score distribution table, the
        headline PSI values and the trigger evaluation.
    """
    feature_drift = drift_report(reference, current, features, thresholds=thresholds)
    score_table = score_distribution_report(reference_scores, current_scores)
    score_psi = float(score_table["psi_total"].iloc[0]) if not score_table.empty else float("nan")
    worst_feature_psi = float(feature_drift["psi"].max()) if not feature_drift.empty else float("nan")

    current_ece = (
        expected_calibration_error(current_y, current_scores) if current_y is not None else None
    )

    return {
        "feature_drift": feature_drift,
        "score_distribution": score_table,
        "score_psi": score_psi,
        "worst_feature_psi": worst_feature_psi,
        "current_ece": current_ece,
        "triggers": evaluate_triggers(
            score_psi=score_psi,
            worst_feature_psi=worst_feature_psi,
            current_ece=current_ece,
            thresholds=thresholds,
        ),
    }
