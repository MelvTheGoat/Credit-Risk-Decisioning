"""Fairness audit: measure the disparity, show the trade, do not quietly patch it.

This module does not "de-bias" anything. It measures, and it presents the
choice. Which point on the accuracy/fairness frontier a lender operates at is a
**policy decision** with legal, commercial and ethical content — it is not a
hyperparameter, and a model author silently picking one and shipping it has
made a policy decision on someone else's behalf without telling them.

What is measured, and why each one is here
------------------------------------------
* **Approval rate by group, and the adverse impact ratio.** The four-fifths
  rule of thumb: if the lowest group's approval rate is below 80% of the
  highest's, that is the conventional trigger for scrutiny. It is a screening
  device, not a legal standard, and it is reported as such.
* **Demographic parity difference** — the raw gap in approval rates. Note what
  it cannot do: if two groups genuinely differ in risk, demographic parity can
  only be reached by approving worse applicants from one group or declining
  better ones from another. On this book the groups *do* genuinely differ, so
  the gap is not by itself evidence of wrongdoing.
* **Equal opportunity difference** — the gap in ``P(approve | would not
  default)``. This one has real moral weight: it counts applicants who *would
  have repaid* and were declined anyway. Unlike demographic parity it is not
  automatically violated by a true base-rate difference, so a large value is
  much harder to explain away.
* **Equalised odds** — the above plus ``P(approve | would default)``, reported
  because a method that equalises one rate by degrading the other should not be
  able to hide.
* **Calibration within each group.** The failure that harms people, and the one
  aggregate metrics conceal. A model can be beautifully calibrated overall and
  systematically over-predict risk for one group — every member of that group
  is then priced, provisioned and declined against a probability that is simply
  wrong for them. Overall ECE will not show it. This is the metric that catches
  the measurement bias injected into the synthetic book, where recorded income
  understates group B's true income and the model, seeing only what it is
  given, marks them as riskier than they are.

Why dropping the protected attribute is not a defence
-----------------------------------------------------
No model here receives ``group``, ``sex``, ``age_band``, ``education`` or
``marriage`` as an input. The disparities are measured anyway, and they are
large. Protected attributes have proxies — income, utilisation, address
stability, employment tenure — and a model fitted to those recovers the
protected attribute's information without ever being shown it. "We do not use
gender" is a statement about the feature list, not about the outcome.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.calibration import (
    brier_score,
    calibration_slope_intercept,
    expected_calibration_error,
)
from src.config import DEFAULT_COSTS, DEFAULT_MONITORING, CostParameters, MonitoringThresholds
from src.policy import decision_outcomes, resolve_exposure

logger = logging.getLogger(__name__)

# Minimum group size for a subgroup metric to be reported. Below this the
# estimates are too noisy to act on, and quoting them invites over-reaction to
# sampling noise in a small segment.
MIN_GROUP_SIZE = 100


def group_performance(
    y: np.ndarray,
    p: np.ndarray,
    groups: pd.Series,
    threshold: float,
    min_group_size: int = MIN_GROUP_SIZE,
) -> pd.DataFrame:
    """Per-group approval, error and calibration metrics.

    Args:
        y: Realised default flag.
        p: Calibrated probability of default.
        groups: Protected attribute value per applicant.
        threshold: PD cutoff being applied.
        min_group_size: Groups smaller than this are skipped as too noisy.

    Returns:
        One row per group with size, base rate, approval rate, the rates
        underlying equal opportunity and equalised odds, and within-group
        calibration.
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    approve = p < threshold
    group_values = pd.Series(groups).astype(str).to_numpy()

    rows = []
    for value in sorted(set(group_values)):
        mask = group_values == value
        n = int(mask.sum())
        if n < min_group_size:
            logger.info("Skipping group %r: only %d rows", value, n)
            continue

        y_g, p_g, approve_g = y[mask], p[mask], approve[mask]
        good = y_g == 0
        bad = y_g == 1
        intercept, slope = calibration_slope_intercept(y_g, p_g)

        rows.append(
            {
                "group": value,
                "n": n,
                "share": n / len(y),
                "true_default_rate": float(y_g.mean()),
                "approval_rate": float(approve_g.mean()),
                # P(approve | would repay). The equal opportunity rate: how
                # often a creditworthy applicant actually gets credit.
                "approval_rate_given_good": float(approve_g[good].mean()) if good.any() else np.nan,
                # P(approve | would default). Reported so that "fairness" gains
                # achieved by approving more bads are visible.
                "approval_rate_given_bad": float(approve_g[bad].mean()) if bad.any() else np.nan,
                "mean_predicted_pd": float(p_g.mean()),
                "calibration_bias": float(p_g.mean() - y_g.mean()),
                "ece": expected_calibration_error(y_g, p_g),
                "brier": brier_score(y_g, p_g),
                "calibration_slope": slope,
                "calibration_intercept": intercept,
            }
        )
    return pd.DataFrame(rows)


def fairness_metrics(
    performance: pd.DataFrame,
    attribute: str = "",
    thresholds: MonitoringThresholds = DEFAULT_MONITORING,
) -> dict[str, float | str | bool]:
    """Reduce a per-group table to headline disparity measures.

    Args:
        performance: Output of :func:`group_performance`.
        attribute: Protected attribute name, carried into the output.
        thresholds: Governance thresholds used to flag findings.

    Returns:
        Mapping of disparity measures, including which groups are the extremes
        and whether the finding breaches the escalation thresholds.
    """
    if performance.empty or len(performance) < 2:
        return {"attribute": attribute, "n_groups": float(len(performance))}

    approval = performance["approval_rate"]
    opportunity = performance["approval_rate_given_good"]

    demographic_parity_difference = float(approval.max() - approval.min())
    adverse_impact_ratio = float(approval.min() / approval.max()) if approval.max() > 0 else np.nan
    equal_opportunity_difference = float(opportunity.max() - opportunity.min())
    equalised_odds_difference = max(
        equal_opportunity_difference,
        float(
            performance["approval_rate_given_bad"].max()
            - performance["approval_rate_given_bad"].min()
        ),
    )

    worst_ece = float(performance["ece"].max())
    return {
        "attribute": attribute,
        "n_groups": float(len(performance)),
        "demographic_parity_difference": demographic_parity_difference,
        "adverse_impact_ratio": adverse_impact_ratio,
        "equal_opportunity_difference": equal_opportunity_difference,
        "equalised_odds_difference": equalised_odds_difference,
        "max_group_ece": worst_ece,
        "max_group_calibration_bias": float(performance["calibration_bias"].abs().max()),
        "calibration_bias_spread": float(
            performance["calibration_bias"].max() - performance["calibration_bias"].min()
        ),
        "lowest_approval_group": str(performance.loc[approval.idxmin(), "group"]),
        "highest_approval_group": str(performance.loc[approval.idxmax(), "group"]),
        "worst_calibrated_group": str(performance.loc[performance["ece"].idxmax(), "group"]),
        # Screening flags, not verdicts.
        "flag_adverse_impact": bool(adverse_impact_ratio < 0.80),
        "flag_equal_opportunity": bool(equal_opportunity_difference > thresholds.eod_breach),
        "flag_subgroup_calibration": bool(worst_ece > thresholds.subgroup_ece_breach),
    }


def audit_fairness(
    frame: pd.DataFrame,
    y: np.ndarray,
    p: np.ndarray,
    protected_attributes: list[str],
    threshold: float,
    min_group_size: int = MIN_GROUP_SIZE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full audit across every protected attribute.

    Args:
        frame: Applicant frame containing the protected attribute columns.
        y: Realised default flag.
        p: Calibrated probability of default.
        protected_attributes: Column names to audit.
        threshold: PD cutoff being applied.
        min_group_size: Groups smaller than this are skipped.

    Returns:
        Tuple of ``(per_group, summary)``: the detailed per-group table across
        all attributes, and one summary row per attribute.
    """
    per_group_frames = []
    summaries = []

    for attribute in protected_attributes:
        if attribute not in frame.columns:
            logger.warning("Protected attribute %r not present; skipping", attribute)
            continue
        performance = group_performance(
            y, p, frame[attribute], threshold, min_group_size=min_group_size
        )
        if performance.empty:
            continue
        performance.insert(0, "attribute", attribute)
        per_group_frames.append(performance)
        summaries.append(fairness_metrics(performance, attribute=attribute))

    per_group = (
        pd.concat(per_group_frames, ignore_index=True) if per_group_frames else pd.DataFrame()
    )
    return per_group, pd.DataFrame(summaries)


# --------------------------------------------------------------------------- #
# The trade-off
# --------------------------------------------------------------------------- #


def fairness_cost_frontier(
    y: np.ndarray,
    p: np.ndarray,
    groups: pd.Series,
    base_threshold: float,
    costs: CostParameters = DEFAULT_COSTS,
    exposure: np.ndarray | pd.Series | None = None,
    n_points: int = 21,
) -> pd.DataFrame:
    """Trace the cost of buying approval-rate parity.

    Parameterised by ``lambda`` from 0 to 1. At ``lambda = 0`` every group faces
    the single cost-optimal cutoff. At ``lambda = 1`` every group's cutoff is
    moved to whatever delivers the **same approval rate** for all groups. In
    between, each group's target approval rate is interpolated between the two.

    Group-specific thresholds are used because they make the trade explicit and
    measurable. Be clear about what they are: in most jurisdictions, applying a
    different cutoff to applicants **because of** a protected characteristic is
    disparate treatment and is unlawful, whatever its effect on the disparity
    statistics. This curve is a diagnostic that prices the gap. It is not a
    deployment proposal, and the recommended operating point in ``README.md``
    is not on the parity end of it.

    Args:
        y: Realised default flag.
        p: Calibrated probability of default.
        groups: Protected attribute value per applicant.
        base_threshold: The single cost-optimal cutoff.
        costs: Cost parameters.
        exposure: Per-applicant exposure, or ``None``.
        n_points: Number of lambda values.

    Reading the output honestly — two cautions that the numbers on this book
    make necessary:

    * **The cost curve is not monotone in lambda.** Net cost wanders by about
      3% across the range with no trend, which means the differences between
      individual lambda values are sampling noise, not signal. The correct
      reading is "moving to parity costs approximately nothing here"
      (0.53% of profit end to end), **not** "lambda = 0.7 is the optimum".
      Treating a bumpy flat curve as if it had a peak is how spurious operating
      points get shipped.
    * **Demographic parity and equal opportunity are different objectives with
      different optima.** Equal opportunity difference bottoms out at
      ``lambda = 0.8`` (0.001) and then rises again to 0.032 at full parity.
      Equalising approval rates across groups with different base rates
      necessarily unequalises the rate at which creditworthy applicants are
      approved. You cannot have both, and choosing between them is a value
      judgement rather than a modelling one.

    ``max_group_ece`` is invariant to lambda by construction — calibration is a
    property of the probabilities, not the cutoff — and is carried through only
    as a reminder that **moving thresholds does not repair a miscalibrated
    subgroup**. If a group's probabilities are wrong, every point on this curve
    is built on wrong numbers for them.

    Args:
        y: Realised default flag.
        p: Calibrated probability of default.
        groups: Protected attribute value per applicant.
        base_threshold: The single cost-optimal cutoff.
        costs: Cost parameters.
        exposure: Per-applicant exposure, or ``None``.
        n_points: Number of lambda values.

    Returns:
        One row per lambda with net cost, approval rate, demographic parity
        difference, equal opportunity difference and worst-group ECE.
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    group_values = pd.Series(groups).astype(str).to_numpy()
    exposure_vector = resolve_exposure(len(y), exposure, costs)

    unique_groups = sorted(set(group_values))
    baseline_rates = {
        g: float((p[group_values == g] < base_threshold).mean()) for g in unique_groups
    }
    overall_rate = float((p < base_threshold).mean())

    rows = []
    for lam in np.linspace(0.0, 1.0, n_points):
        approve = np.zeros(len(y), dtype=bool)
        thresholds: dict[str, float] = {}

        for g in unique_groups:
            mask = group_values == g
            target_rate = (1.0 - lam) * baseline_rates[g] + lam * overall_rate
            group_scores = p[mask]
            if target_rate <= 0.0:
                group_threshold = -np.inf
            elif target_rate >= 1.0:
                group_threshold = np.inf
            else:
                group_threshold = float(np.quantile(group_scores, target_rate))
            thresholds[g] = group_threshold
            approve[mask] = group_scores < group_threshold

        outcome = decision_outcomes(y, approve, exposure_vector, costs)
        performance = _performance_from_decisions(y, p, group_values, approve)
        metrics = fairness_metrics(performance)

        rows.append(
            {
                "lambda": float(lam),
                "approval_rate": outcome["approval_rate"],
                "net_cost": outcome["net_cost"],
                "profit_per_application": outcome["profit_per_application"],
                "bad_rate_approved": outcome["bad_rate_approved"],
                "demographic_parity_difference": metrics.get(
                    "demographic_parity_difference", np.nan
                ),
                "adverse_impact_ratio": metrics.get("adverse_impact_ratio", np.nan),
                "equal_opportunity_difference": metrics.get(
                    "equal_opportunity_difference", np.nan
                ),
                "max_group_ece": metrics.get("max_group_ece", np.nan),
                "thresholds": thresholds,
            }
        )
    return pd.DataFrame(rows)


def _performance_from_decisions(
    y: np.ndarray,
    p: np.ndarray,
    group_values: np.ndarray,
    approve: np.ndarray,
) -> pd.DataFrame:
    """Build a per-group table from decisions already made.

    Needed by the frontier, where thresholds vary by group and so
    :func:`group_performance` (which takes a single threshold) does not apply.

    Args:
        y: Realised default flag.
        p: Calibrated probability of default.
        group_values: Protected attribute value per applicant.
        approve: Boolean approve decisions.

    Returns:
        Per-group table in the same shape as :func:`group_performance`.
    """
    rows = []
    for value in sorted(set(group_values)):
        mask = group_values == value
        if mask.sum() < MIN_GROUP_SIZE:
            continue
        y_g, p_g, approve_g = y[mask], p[mask], approve[mask]
        good, bad = y_g == 0, y_g == 1
        rows.append(
            {
                "group": value,
                "n": int(mask.sum()),
                "approval_rate": float(approve_g.mean()),
                "approval_rate_given_good": float(approve_g[good].mean()) if good.any() else np.nan,
                "approval_rate_given_bad": float(approve_g[bad].mean()) if bad.any() else np.nan,
                "true_default_rate": float(y_g.mean()),
                "mean_predicted_pd": float(p_g.mean()),
                "calibration_bias": float(p_g.mean() - y_g.mean()),
                "ece": expected_calibration_error(y_g, p_g),
                "brier": brier_score(y_g, p_g),
            }
        )
    return pd.DataFrame(rows)


def decompose_disparity(
    frame: pd.DataFrame,
    p: np.ndarray,
    group_column: str,
    truth_column: str = "pd_true",
) -> pd.DataFrame:
    """Split a group's predicted-risk gap into a true part and a model part.

    Only possible where the true probability of default is known, so synthetic
    data only. It is included because it isolates the question a real audit
    cannot answer and must instead argue about:

    * ``true_pd_gap`` — the groups really do differ in risk. Not a model defect.
    * ``predicted_pd_gap`` — what the model asserts the difference is.
    * ``model_excess_gap`` — the difference between them. This is the part the
      model invented, and on the synthetic book it is traceable to the injected
      income measurement bias.

    On a real portfolio these three collapse into one observable number and
    cannot be separated, which is why the honest reporting standard is to
    present the disparity together with its uncertainty rather than to claim a
    verdict on causes.

    Args:
        frame: Applicant frame containing the group and truth columns.
        p: Calibrated probability of default.
        group_column: Protected attribute column.
        truth_column: Column holding the true PD.

    Returns:
        One row per group with true PD, predicted PD and the excess.
    """
    if truth_column not in frame.columns:
        raise ValueError(
            f"{truth_column!r} not available; disparity decomposition needs ground truth "
            "and is only possible on synthetic data"
        )

    working = pd.DataFrame(
        {
            "group": frame[group_column].astype(str).to_numpy(),
            "pd_true": frame[truth_column].to_numpy(dtype=float),
            "pd_predicted": np.asarray(p, dtype=float),
        }
    )
    summary = working.groupby("group", observed=True).agg(
        n=("pd_true", "size"),
        mean_pd_true=("pd_true", "mean"),
        mean_pd_predicted=("pd_predicted", "mean"),
    )
    summary["model_excess"] = summary["mean_pd_predicted"] - summary["mean_pd_true"]

    # Reference group is the one with the lowest true risk; gaps are measured
    # against it so the "excess" column reads as bias relative to the safest.
    reference_row = int(summary["mean_pd_true"].to_numpy().argmin())
    reference_true = float(summary["mean_pd_true"].to_numpy()[reference_row])
    reference_predicted = float(summary["mean_pd_predicted"].to_numpy()[reference_row])

    summary["true_pd_gap_vs_lowest"] = summary["mean_pd_true"] - reference_true
    summary["predicted_pd_gap_vs_lowest"] = summary["mean_pd_predicted"] - reference_predicted
    summary["model_excess_gap_vs_lowest"] = (
        summary["predicted_pd_gap_vs_lowest"] - summary["true_pd_gap_vs_lowest"]
    )
    return summary.reset_index()
