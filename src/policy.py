"""Cost-sensitive decisioning: turning a probability into an approve or decline.

The headline result of this system is a **cost**, not an AUC.

Approving an applicant who defaults and declining one who would have repaid are
not the same mistake. On the default assumptions a missed bad costs
``LGD x EAD = 0.75 x 10,000 = 7,500`` and a lost good costs the forgone margin
``0.09 x 10,000 = 900``. One missed bad is worth 8.3 lost goods. Any threshold
chosen by maximising F1 — which weights the two errors equally by
construction — is choosing the wrong operating point on purpose.

Accounting convention
---------------------
Portfolio economics relative to writing no business at all:

* approve a good — ``+ margin_rate x EAD``
* approve a bad  — ``- LGD x EAD``
* decline anyone — ``0``

The opportunity cost of declining a good applicant is implicit in the profit
forgone, so nothing is double counted. "Net cost" throughout is simply negative
profit, so that minimising cost and maximising profit are the same instruction.

The optimal cutoff in closed form
---------------------------------
Approve while the expected profit of doing so is positive::

    (1 - p) x margin x EAD  >  p x LGD x EAD
    p*  =  margin / (margin + LGD)  =  0.09 / 0.84  =  0.1071

:func:`sweep_thresholds` finds this empirically and
:attr:`~src.config.CostParameters.analytic_threshold` derives it in closed
form. They are asserted to agree in the test suite. If they ever disagree,
either the sweep or the cost model is wrong, and it matters which.

Note what the closed form does **not** depend on: the model. The optimal PD
cutoff is a property of the economics alone. What the model determines is how
much profit that cutoff earns — and a miscalibrated model applied at
``p = 0.1071`` is not operating where it thinks it is, which is the entire
argument for :mod:`src.calibration` being upstream of this module.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.config import DEFAULT_COSTS, SCORE_BANDS, CostParameters

logger = logging.getLogger(__name__)


def resolve_exposure(
    n: int, exposure: np.ndarray | pd.Series | None, costs: CostParameters
) -> np.ndarray:
    """Return a per-applicant exposure vector.

    Args:
        n: Number of applicants.
        exposure: Per-applicant exposure at default, or ``None`` to use the
            flat assumption from the cost parameters.
        costs: Cost parameters.

    Returns:
        Array of exposures, length ``n``.
    """
    if exposure is None:
        return np.full(n, costs.exposure_at_default, dtype=float)
    return np.asarray(exposure, dtype=float)


def decision_outcomes(
    y: np.ndarray,
    approve: np.ndarray,
    exposure: np.ndarray,
    costs: CostParameters,
) -> dict[str, float]:
    """Score a set of decisions against realised outcomes.

    Uses **realised** defaults, not predicted probabilities. Evaluating a
    policy with the same probabilities that produced it is circular: a model
    that is confidently wrong would score itself as highly profitable.

    Args:
        y: Realised default flag.
        approve: Boolean approve decisions.
        exposure: Per-applicant exposure at default.
        costs: Cost parameters.

    Returns:
        Mapping of portfolio outcome measures.
    """
    y = np.asarray(y).astype(int)
    approve = np.asarray(approve).astype(bool)

    approved_good = approve & (y == 0)
    approved_bad = approve & (y == 1)

    margin_earned = float(np.sum(exposure[approved_good]) * costs.margin_rate)
    loss_incurred = float(np.sum(exposure[approved_bad]) * costs.lgd)
    profit = margin_earned - loss_incurred
    n_approved = int(approve.sum())

    return {
        "n": float(len(y)),
        "n_approved": float(n_approved),
        "approval_rate": float(approve.mean()),
        "bad_rate_approved": float(y[approve].mean()) if n_approved else float("nan"),
        "bad_rate_declined": float(y[~approve].mean()) if n_approved < len(y) else float("nan"),
        "margin_earned": margin_earned,
        "realised_loss": loss_incurred,
        "total_profit": profit,
        "net_cost": -profit,
        "profit_per_application": profit / len(y) if len(y) else float("nan"),
        "n_missed_bads": float(approved_bad.sum()),
        "n_lost_goods": float(((~approve) & (y == 0)).sum()),
    }


def sweep_thresholds(
    y: np.ndarray,
    p: np.ndarray,
    costs: CostParameters = DEFAULT_COSTS,
    exposure: np.ndarray | pd.Series | None = None,
    n_points: int = 200,
) -> pd.DataFrame:
    """Evaluate every candidate PD cutoff and return the full frontier.

    Args:
        y: Realised default flag.
        p: Calibrated probability of default.
        costs: Cost parameters.
        exposure: Per-applicant exposure, or ``None`` for the flat assumption.
        n_points: Number of cutoffs to evaluate, spread over the observed
            score quantiles so the grid is dense where the applicants are.

    Returns:
        One row per cutoff: approval rate, bad rate among approved, realised
        loss, margin earned, net cost, and the model's own expected loss.
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    exposure_vector = resolve_exposure(len(y), exposure, costs)

    quantiles = np.unique(np.quantile(p, np.linspace(0.0, 1.0, n_points)))
    # Always include the closed-form optimum and the endpoints.
    candidates = np.unique(
        np.concatenate([[0.0], quantiles, [costs.analytic_threshold], [1.0]])
    )

    rows = []
    for threshold in candidates:
        approve = p < threshold
        outcome = decision_outcomes(y, approve, exposure_vector, costs)
        outcome["threshold"] = float(threshold)
        # The model's own view of the loss it is taking on. Compare against
        # realised_loss: the gap is a direct readout of calibration quality.
        outcome["expected_loss_model"] = float(
            np.sum(p[approve] * costs.lgd * exposure_vector[approve])
        )
        outcome["expected_loss_error"] = outcome["expected_loss_model"] - outcome["realised_loss"]
        rows.append(outcome)

    return pd.DataFrame(rows).sort_values("threshold", ignore_index=True)


def optimal_threshold(frontier: pd.DataFrame) -> float:
    """Pick the cutoff minimising realised net cost.

    Args:
        frontier: Output of :func:`sweep_thresholds`.

    Returns:
        The cost-minimising PD cutoff.
    """
    return float(frontier.loc[frontier["net_cost"].idxmin(), "threshold"])


def policy_summary(
    y: np.ndarray,
    p: np.ndarray,
    costs: CostParameters = DEFAULT_COSTS,
    exposure: np.ndarray | pd.Series | None = None,
    threshold: float | None = None,
    label: str = "",
) -> dict[str, float | str]:
    """Summarise one model's economics at a chosen cutoff.

    Args:
        y: Realised default flag.
        p: Calibrated probability of default.
        costs: Cost parameters.
        exposure: Per-applicant exposure, or ``None``.
        threshold: Cutoff to apply. Defaults to the closed-form optimum, which
            is the correct choice for a calibrated model and does not require
            peeking at outcomes.
        label: Identifier carried into the output.

    Returns:
        Mapping of outcome measures, including how far the empirically optimal
        cutoff sits from the closed-form one.
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    exposure_vector = resolve_exposure(len(y), exposure, costs)
    applied = costs.analytic_threshold if threshold is None else threshold

    outcome = decision_outcomes(y, p < applied, exposure_vector, costs)
    frontier = sweep_thresholds(y, p, costs, exposure_vector)
    empirical = optimal_threshold(frontier)

    approved = p < applied
    expected_loss_model = float(np.sum(p[approved] * costs.lgd * exposure_vector[approved]))
    best_net_cost = float(frontier["net_cost"].min())

    summary: dict[str, float | str] = {"label": label, **outcome}
    summary["threshold_applied"] = float(applied)
    summary["threshold_analytic"] = float(costs.analytic_threshold)
    summary["threshold_empirical_best"] = empirical
    summary["net_cost_at_empirical_best"] = best_net_cost
    summary["cost_of_using_analytic"] = outcome["net_cost"] - best_net_cost
    summary["expected_loss_model"] = expected_loss_model
    # Gap between what the model thinks it will lose and what it actually
    # loses. A calibrated model gets this near zero; drift blows it open.
    summary["expected_loss_error"] = expected_loss_model - outcome["realised_loss"]
    return summary


def compare_policies(
    predictions: dict[str, np.ndarray],
    y: np.ndarray,
    costs: CostParameters = DEFAULT_COSTS,
    exposure: np.ndarray | pd.Series | None = None,
    threshold: float | None = None,
) -> pd.DataFrame:
    """Rank models by the only number that matters: total cost.

    This is the headline table. Models are ordered by net cost, not AUC.

    Args:
        predictions: Mapping from model name to calibrated P(default).
        y: Realised default flag.
        costs: Cost parameters.
        exposure: Per-applicant exposure, or ``None``.
        threshold: Cutoff to apply to every model. Defaults to the closed-form
            optimum.

    Returns:
        One row per model, sorted by net cost ascending.
    """
    rows = [
        policy_summary(y, p, costs=costs, exposure=exposure, threshold=threshold, label=name)
        for name, p in predictions.items()
    ]
    return pd.DataFrame(rows).sort_values("net_cost", ignore_index=True)


def approval_frontier(
    predictions: dict[str, np.ndarray],
    y: np.ndarray,
    costs: CostParameters = DEFAULT_COSTS,
    exposure: np.ndarray | pd.Series | None = None,
    n_points: int = 100,
) -> pd.DataFrame:
    """Approval rate against expected loss, for every model.

    The curve a credit committee actually negotiates over: "how much more
    business can we write, and what does the loss rate do if we do?"

    Args:
        predictions: Mapping from model name to calibrated P(default).
        y: Realised default flag.
        costs: Cost parameters.
        exposure: Per-applicant exposure, or ``None``.
        n_points: Cutoff grid resolution.

    Returns:
        Long-format frame of model, threshold, approval rate, loss rate on
        approved balances, and net cost per application.
    """
    frames = []
    for name, p in predictions.items():
        frontier = sweep_thresholds(y, p, costs, exposure, n_points=n_points)
        frontier["model"] = name
        frontier["loss_rate_on_approved"] = np.where(
            frontier["n_approved"] > 0,
            frontier["realised_loss"] / np.maximum(frontier["n_approved"], 1) / costs.exposure_at_default,
            np.nan,
        )
        frontier["net_cost_per_application"] = frontier["net_cost"] / frontier["n"]
        frames.append(
            frontier[
                [
                    "model",
                    "threshold",
                    "approval_rate",
                    "bad_rate_approved",
                    "loss_rate_on_approved",
                    "realised_loss",
                    "margin_earned",
                    "net_cost",
                    "net_cost_per_application",
                    "expected_loss_model",
                    "expected_loss_error",
                ]
            ]
        )
    return pd.concat(frames, ignore_index=True)


def cost_sensitivity(
    y: np.ndarray,
    p: np.ndarray,
    lgd_values: tuple[float, ...] = (0.45, 0.60, 0.75, 0.90),
    margin_values: tuple[float, ...] = (0.05, 0.07, 0.09, 0.12),
    base_costs: CostParameters = DEFAULT_COSTS,
    exposure: np.ndarray | pd.Series | None = None,
) -> pd.DataFrame:
    """Show how the optimal cutoff moves as the cost assumptions move.

    LGD and margin are *assumptions*, not measurements. They come from
    collections performance and pricing, both of which are estimates with their
    own error, and both of which change with the economic cycle. A cutoff
    quoted to four decimal places off a single assumed LGD conveys a precision
    that does not exist.

    This grid is the honest presentation: here is the cutoff, and here is how
    far it moves if the assumptions are wrong.

    Args:
        y: Realised default flag.
        p: Calibrated probability of default.
        lgd_values: LGD values to test.
        margin_values: Margin rates to test.
        base_costs: Cost parameters supplying everything not swept.
        exposure: Per-applicant exposure, or ``None``.

    Returns:
        One row per assumption pair with the resulting cutoff, approval rate
        and net cost.
    """
    rows = []
    for lgd in lgd_values:
        for margin in margin_values:
            costs = CostParameters(
                lgd=lgd,
                exposure_at_default=base_costs.exposure_at_default,
                margin_rate=margin,
                decline_cost_rate=margin,
                fixed_review_cost=base_costs.fixed_review_cost,
            )
            outcome = decision_outcomes(
                y,
                np.asarray(p) < costs.analytic_threshold,
                resolve_exposure(len(y), exposure, costs),
                costs,
            )
            rows.append(
                {
                    "lgd": lgd,
                    "margin_rate": margin,
                    "cost_ratio": costs.cost_ratio,
                    "threshold": costs.analytic_threshold,
                    "approval_rate": outcome["approval_rate"],
                    "bad_rate_approved": outcome["bad_rate_approved"],
                    "net_cost": outcome["net_cost"],
                    "profit_per_application": outcome["profit_per_application"],
                }
            )
    return pd.DataFrame(rows)


def f1_optimal_threshold(y: np.ndarray, p: np.ndarray, n_points: int = 200) -> float:
    """Find the cutoff maximising F1, purely so it can be shown to be wrong.

    F1 weights false positives and false negatives equally. In a book where one
    error costs 8.3 times the other, that is not a neutral default — it is a
    specific and incorrect cost assumption, silently applied.

    Args:
        y: Realised default flag.
        p: Predicted probability of default.
        n_points: Grid resolution.

    Returns:
        The F1-maximising cutoff, expressed as a PD approval threshold.
    """
    from sklearn.metrics import f1_score

    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    grid = np.unique(np.quantile(p, np.linspace(0.01, 0.99, n_points)))
    scores = [f1_score(y, (p >= t).astype(int), zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


def compare_threshold_rules(
    y: np.ndarray,
    p: np.ndarray,
    costs: CostParameters = DEFAULT_COSTS,
    exposure: np.ndarray | pd.Series | None = None,
    label: str = "",
) -> pd.DataFrame:
    """Contrast the cost-optimal cutoff with common alternatives.

    Compares four rules on identical data: the closed-form cost optimum, the
    empirical cost optimum, the F1-maximising cutoff, and a naive 0.5.

    Args:
        y: Realised default flag.
        p: Calibrated probability of default.
        costs: Cost parameters.
        exposure: Per-applicant exposure, or ``None``.
        label: Model identifier carried into the output.

    Returns:
        One row per rule with the cutoff it selects and what that costs.
    """
    exposure_vector = resolve_exposure(len(y), exposure, costs)
    frontier = sweep_thresholds(y, p, costs, exposure_vector)
    best_cost = float(frontier["net_cost"].min())

    rules = {
        "cost_optimal_analytic": costs.analytic_threshold,
        "cost_optimal_empirical": optimal_threshold(frontier),
        "f1_optimal": f1_optimal_threshold(y, p),
        "naive_0.5": 0.5,
    }

    rows = []
    for rule, threshold in rules.items():
        outcome = decision_outcomes(y, np.asarray(p) < threshold, exposure_vector, costs)
        rows.append(
            {
                "model": label,
                "rule": rule,
                "threshold": threshold,
                "approval_rate": outcome["approval_rate"],
                "bad_rate_approved": outcome["bad_rate_approved"],
                "net_cost": outcome["net_cost"],
                "excess_cost_vs_best": outcome["net_cost"] - best_cost,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Applying a policy
# --------------------------------------------------------------------------- #


def assign_score_band(score: float) -> str:
    """Map a scorecard score onto its reporting band.

    Args:
        score: Scorecard points.

    Returns:
        Band label from ``A`` (safest) to ``E`` (riskiest).
    """
    for label, lower, upper in SCORE_BANDS:
        if lower <= score < upper:
            return label
    return SCORE_BANDS[-1][0]


def apply_policy(
    p: np.ndarray,
    threshold: float,
    referral_width: float = 0.0,
) -> np.ndarray:
    """Convert probabilities into decisions.

    Args:
        p: Calibrated probability of default.
        threshold: Approve below this PD.
        referral_width: Half-width of an optional manual review band around the
            cutoff, in PD units. Zero disables referral, which is the default:
            a referral band only pays for itself if the reviewers genuinely add
            information the model lacks, and that claim needs its own evidence.

    Returns:
        Array of ``"approve"``, ``"refer"`` or ``"decline"``.
    """
    p = np.asarray(p, dtype=float)
    decisions = np.where(p < threshold, "approve", "decline").astype(object)
    if referral_width > 0.0:
        near_cutoff = np.abs(p - threshold) <= referral_width
        decisions[near_cutoff] = "refer"
    return decisions
