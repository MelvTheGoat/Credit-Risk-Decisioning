"""Tests for cost-sensitive decisioning.

The central assertion is that the closed-form optimal cutoff and the empirical
sweep agree on well-calibrated predictions. If they ever disagree, either the
sweep or the cost model is wrong, and it matters a great deal which.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.config import CostParameters
from src.policy import (
    apply_policy,
    assign_score_band,
    compare_policies,
    compare_threshold_rules,
    cost_sensitivity,
    decision_outcomes,
    f1_optimal_threshold,
    optimal_threshold,
    policy_summary,
    sweep_thresholds,
)


@pytest.fixture(scope="module")
def calibrated_sample() -> tuple[np.ndarray, np.ndarray]:
    """Large, perfectly calibrated predictions, so the optimum is unambiguous."""
    rng = np.random.default_rng(7)
    p = rng.beta(2, 12, 60_000)
    y = (rng.random(60_000) < p).astype(int)
    return y, p


def test_analytic_threshold_matches_its_definition() -> None:
    costs = CostParameters(lgd=0.75, margin_rate=0.09, decline_cost_rate=0.09)
    assert costs.analytic_threshold == pytest.approx(0.09 / (0.75 + 0.09))
    assert costs.cost_ratio == pytest.approx(0.75 / 0.09)


def test_empirical_optimum_agrees_with_the_closed_form(
    calibrated_sample: tuple[np.ndarray, np.ndarray],
) -> None:
    """The headline consistency check."""
    y, p = calibrated_sample
    costs = CostParameters()
    frontier = sweep_thresholds(y, p, costs, n_points=400)
    empirical = optimal_threshold(frontier)
    assert empirical == pytest.approx(costs.analytic_threshold, abs=0.015)


def test_optimum_moves_correctly_with_the_cost_ratio(
    calibrated_sample: tuple[np.ndarray, np.ndarray],
) -> None:
    """A costlier default must tighten the cutoff."""
    y, p = calibrated_sample
    strict = CostParameters(lgd=0.90, margin_rate=0.05, decline_cost_rate=0.05)
    lenient = CostParameters(lgd=0.45, margin_rate=0.12, decline_cost_rate=0.12)

    assert strict.analytic_threshold < lenient.analytic_threshold
    strict_optimum = optimal_threshold(sweep_thresholds(y, p, strict, n_points=300))
    lenient_optimum = optimal_threshold(sweep_thresholds(y, p, lenient, n_points=300))
    assert strict_optimum < lenient_optimum


def test_cost_accounting_is_arithmetically_correct() -> None:
    y = np.array([0, 0, 1, 1])
    approve = np.array([True, False, True, False])
    exposure = np.full(4, 10_000.0)
    costs = CostParameters(lgd=0.75, margin_rate=0.09, exposure_at_default=10_000.0)

    outcome = decision_outcomes(y, approve, exposure, costs)
    # One good approved earns margin; one bad approved loses LGD x EAD.
    assert outcome["margin_earned"] == pytest.approx(900.0)
    assert outcome["realised_loss"] == pytest.approx(7_500.0)
    assert outcome["total_profit"] == pytest.approx(-6_600.0)
    assert outcome["net_cost"] == pytest.approx(6_600.0)
    assert outcome["n_missed_bads"] == 1
    assert outcome["n_lost_goods"] == 1


def test_declining_everyone_costs_and_earns_nothing() -> None:
    y = np.array([0, 1, 0, 1])
    outcome = decision_outcomes(y, np.zeros(4, dtype=bool), np.full(4, 10_000.0), CostParameters())
    assert outcome["total_profit"] == 0.0
    assert outcome["net_cost"] == 0.0
    assert outcome["approval_rate"] == 0.0


def test_f1_threshold_is_more_expensive_than_the_cost_optimum(
    calibrated_sample: tuple[np.ndarray, np.ndarray],
) -> None:
    """F1 weights both errors equally; the book does not."""
    y, p = calibrated_sample
    costs = CostParameters()
    rules = compare_threshold_rules(y, p, costs=costs).set_index("rule")
    assert rules.loc["f1_optimal", "excess_cost_vs_best"] > 0
    assert rules.loc["naive_0.5", "excess_cost_vs_best"] > rules.loc["f1_optimal", "excess_cost_vs_best"]
    assert rules.loc["cost_optimal_analytic", "excess_cost_vs_best"] < rules.loc[
        "f1_optimal", "excess_cost_vs_best"
    ]


def test_f1_optimum_differs_from_the_cost_optimum(
    calibrated_sample: tuple[np.ndarray, np.ndarray],
) -> None:
    y, p = calibrated_sample
    assert f1_optimal_threshold(y, p) != pytest.approx(CostParameters().analytic_threshold, abs=0.01)


def test_frontier_approval_rate_is_monotone(
    calibrated_sample: tuple[np.ndarray, np.ndarray],
) -> None:
    y, p = calibrated_sample
    frontier = sweep_thresholds(y, p, CostParameters(), n_points=100)
    assert frontier["approval_rate"].is_monotonic_increasing


def test_looser_cutoffs_admit_more_bad_business(
    calibrated_sample: tuple[np.ndarray, np.ndarray],
) -> None:
    """Relaxing the cutoff must raise the default rate of the approved book.

    Asserted as a strong trend rather than strict monotonicity: the running
    bad rate is a sample statistic, so admitting one more applicant can nudge
    it down by chance. Demanding exact monotonicity would be testing the random
    seed.
    """
    y, p = calibrated_sample
    frontier = sweep_thresholds(y, p, CostParameters(), n_points=100)
    populated = frontier[frontier["n_approved"] > 100]

    # Rank correlation, because the relationship is monotone but distinctly
    # concave; a Pearson coefficient would be testing its curvature instead.
    from scipy.stats import spearmanr

    assert spearmanr(populated["threshold"], populated["bad_rate_approved"]).statistic > 0.99
    assert populated["bad_rate_approved"].iloc[-1] > populated["bad_rate_approved"].iloc[0]


def test_expected_loss_tracks_realised_loss_when_calibrated(
    calibrated_sample: tuple[np.ndarray, np.ndarray],
) -> None:
    """A calibrated model can predict its own loss; that is the whole point."""
    y, p = calibrated_sample
    summary = policy_summary(y, p, costs=CostParameters())
    relative_error = abs(float(summary["expected_loss_error"])) / float(summary["realised_loss"])
    assert relative_error < 0.05


def test_miscalibrated_model_cannot_predict_its_own_loss(
    calibrated_sample: tuple[np.ndarray, np.ndarray],
) -> None:
    y, p = calibrated_sample
    summary = policy_summary(y, np.clip(p * 0.5, 1e-6, 1 - 1e-6), costs=CostParameters())
    relative_error = abs(float(summary["expected_loss_error"])) / float(summary["realised_loss"])
    assert relative_error > 0.20


def test_cost_sensitivity_grid_is_complete(
    calibrated_sample: tuple[np.ndarray, np.ndarray],
) -> None:
    y, p = calibrated_sample
    grid = cost_sensitivity(y, p, lgd_values=(0.5, 0.8), margin_values=(0.06, 0.10))
    assert len(grid) == 4
    # A higher LGD at fixed margin must tighten the cutoff.
    tight = grid[(grid["lgd"] == 0.8) & (grid["margin_rate"] == 0.06)]["threshold"].iloc[0]
    loose = grid[(grid["lgd"] == 0.5) & (grid["margin_rate"] == 0.10)]["threshold"].iloc[0]
    assert tight < loose


def test_compare_policies_is_sorted_by_cost(
    calibrated_predictions: dict[str, np.ndarray], test_y: np.ndarray
) -> None:
    comparison = compare_policies(calibrated_predictions, test_y)
    assert comparison["net_cost"].is_monotonic_increasing
    assert set(comparison["label"]) == set(calibrated_predictions)


def test_apply_policy_labels_decisions_correctly() -> None:
    p = np.array([0.01, 0.05, 0.20, 0.50])
    np.testing.assert_array_equal(
        apply_policy(p, 0.10), np.array(["approve", "approve", "decline", "decline"], dtype=object)
    )


def test_referral_band_captures_the_margin() -> None:
    p = np.array([0.01, 0.095, 0.105, 0.50])
    decisions = apply_policy(p, 0.10, referral_width=0.01)
    assert decisions[0] == "approve"
    assert decisions[1] == "refer"
    assert decisions[2] == "refer"
    assert decisions[3] == "decline"


@pytest.mark.parametrize(
    ("score", "expected"),
    [(400.0, "E"), (540.0, "D"), (600.0, "C"), (650.0, "B"), (700.0, "A")],
)
def test_score_bands(score: float, expected: str) -> None:
    assert assign_score_band(score) == expected
