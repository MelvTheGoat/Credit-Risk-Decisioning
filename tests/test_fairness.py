"""Tests for the fairness audit.

The audit must actually detect the bias that was deliberately injected. An
audit that reports "no findings" on a book known to contain a 15% income
measurement bias and an explicit 0.55-logit policy penalty is not reassuring,
it is broken.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.fairness import (
    audit_fairness,
    decompose_disparity,
    fairness_cost_frontier,
    fairness_metrics,
    group_performance,
)


def test_group_performance_reports_every_large_group(
    approved_splits: dict[str, pd.DataFrame], calibrated_predictions: dict[str, np.ndarray],
    test_y: np.ndarray, costs: object,
) -> None:
    test = approved_splits["test"]
    performance = group_performance(
        test_y,
        calibrated_predictions["lightgbm"],
        test["group"],
        costs.analytic_threshold,  # type: ignore[attr-defined]
    )
    assert set(performance["group"]) == {"A", "B"}
    assert performance["n"].sum() <= len(test)
    for column in ("approval_rate", "approval_rate_given_good", "ece", "calibration_bias"):
        assert performance[column].notna().all()


def test_small_groups_are_suppressed(
    approved_splits: dict[str, pd.DataFrame], calibrated_predictions: dict[str, np.ndarray],
    test_y: np.ndarray, costs: object,
) -> None:
    """Noisy subgroup estimates should not be published."""
    test = approved_splits["test"]
    performance = group_performance(
        test_y,
        calibrated_predictions["lightgbm"],
        test["group"],
        costs.analytic_threshold,  # type: ignore[attr-defined]
        min_group_size=10_000,
    )
    assert performance.empty


def test_audit_detects_the_injected_approval_disparity(
    approved_splits: dict[str, pd.DataFrame], calibrated_predictions: dict[str, np.ndarray],
    test_y: np.ndarray, costs: object,
) -> None:
    """A model that never sees `group` still produces a large gap."""
    test = approved_splits["test"]
    per_group, summary = audit_fairness(
        test, test_y, calibrated_predictions["lightgbm"], ["group"], costs.analytic_threshold  # type: ignore[attr-defined]
    )
    row = summary.iloc[0]
    assert row["demographic_parity_difference"] > 0.05
    assert row["adverse_impact_ratio"] < 0.90
    assert row["lowest_approval_group"] == "B"


def test_audit_detects_an_equal_opportunity_gap(
    approved_splits: dict[str, pd.DataFrame], calibrated_predictions: dict[str, np.ndarray],
    test_y: np.ndarray, costs: object,
) -> None:
    """Creditworthy group B applicants are approved less often."""
    test = approved_splits["test"]
    _, summary = audit_fairness(
        test, test_y, calibrated_predictions["lightgbm"], ["group"], costs.analytic_threshold  # type: ignore[attr-defined]
    )
    assert float(summary.iloc[0]["equal_opportunity_difference"]) > 0.03


def test_disparity_decomposition_isolates_the_model_contribution(
    approved_splits: dict[str, pd.DataFrame], calibrated_predictions: dict[str, np.ndarray]
) -> None:
    """The injected measurement bias must show up as manufactured gap.

    The groups genuinely differ in risk, so a gap alone proves nothing. What
    proves something is the model asserting a *larger* gap than the truth, and
    that excess is exactly what the income measurement bias creates.
    """
    test = approved_splits["test"]
    decomposition = decompose_disparity(
        test, calibrated_predictions["lightgbm"], "group"
    ).set_index("group")

    assert float(decomposition.loc["B", "true_pd_gap_vs_lowest"]) > 0
    assert float(decomposition.loc["B", "model_excess_gap_vs_lowest"]) > 0


def test_decomposition_requires_ground_truth(approved_splits: dict[str, pd.DataFrame]) -> None:
    test = approved_splits["test"].drop(columns=["pd_true"])
    with pytest.raises(ValueError, match="synthetic"):
        decompose_disparity(test, np.zeros(len(test)), "group")


def test_audit_covers_multiple_attributes(
    approved_splits: dict[str, pd.DataFrame], calibrated_predictions: dict[str, np.ndarray],
    test_y: np.ndarray, costs: object,
) -> None:
    test = approved_splits["test"]
    per_group, summary = audit_fairness(
        test,
        test_y,
        calibrated_predictions["lightgbm"],
        ["group", "age_band"],
        costs.analytic_threshold,  # type: ignore[attr-defined]
    )
    assert set(summary["attribute"]) == {"group", "age_band"}
    assert set(per_group["attribute"]) == {"group", "age_band"}


def test_missing_attribute_is_skipped_not_fatal(
    approved_splits: dict[str, pd.DataFrame], calibrated_predictions: dict[str, np.ndarray],
    test_y: np.ndarray, costs: object,
) -> None:
    test = approved_splits["test"]
    _, summary = audit_fairness(
        test,
        test_y,
        calibrated_predictions["lightgbm"],
        ["group", "not_a_real_column"],
        costs.analytic_threshold,  # type: ignore[attr-defined]
    )
    assert set(summary["attribute"]) == {"group"}


def test_frontier_starts_at_the_single_global_threshold(
    approved_splits: dict[str, pd.DataFrame], calibrated_predictions: dict[str, np.ndarray],
    test_y: np.ndarray, costs: object,
) -> None:
    """Lambda = 0 must reproduce the ordinary policy exactly."""
    from src.policy import decision_outcomes, resolve_exposure

    test = approved_splits["test"]
    p = calibrated_predictions["lightgbm"]
    threshold = costs.analytic_threshold  # type: ignore[attr-defined]
    exposure = test["loan_amount"].to_numpy()

    frontier = fairness_cost_frontier(
        test_y, p, test["group"], threshold, costs=costs, exposure=exposure, n_points=5  # type: ignore[arg-type]
    )
    direct = decision_outcomes(
        test_y, p < threshold, resolve_exposure(len(test_y), exposure, costs), costs  # type: ignore[arg-type]
    )
    assert float(frontier.iloc[0]["net_cost"]) == pytest.approx(direct["net_cost"], rel=1e-9)
    assert float(frontier.iloc[0]["approval_rate"]) == pytest.approx(direct["approval_rate"], abs=1e-9)


def test_frontier_reaches_approval_parity(
    approved_splits: dict[str, pd.DataFrame], calibrated_predictions: dict[str, np.ndarray],
    test_y: np.ndarray, costs: object,
) -> None:
    """Lambda = 1 must actually equalise approval rates."""
    test = approved_splits["test"]
    frontier = fairness_cost_frontier(
        test_y,
        calibrated_predictions["lightgbm"],
        test["group"],
        costs.analytic_threshold,  # type: ignore[attr-defined]
        costs=costs,  # type: ignore[arg-type]
        n_points=11,
    )
    assert float(frontier.iloc[-1]["demographic_parity_difference"]) < 0.02
    assert float(frontier.iloc[-1]["demographic_parity_difference"]) < float(
        frontier.iloc[0]["demographic_parity_difference"]
    )


def test_frontier_reduces_disparity_monotonically(
    approved_splits: dict[str, pd.DataFrame], calibrated_predictions: dict[str, np.ndarray],
    test_y: np.ndarray, costs: object,
) -> None:
    test = approved_splits["test"]
    frontier = fairness_cost_frontier(
        test_y,
        calibrated_predictions["lightgbm"],
        test["group"],
        costs.analytic_threshold,  # type: ignore[attr-defined]
        costs=costs,  # type: ignore[arg-type]
        n_points=11,
    )
    differences = np.diff(frontier["demographic_parity_difference"].to_numpy())
    assert (differences <= 1e-6).all()


def test_calibration_is_invariant_to_the_threshold(
    approved_splits: dict[str, pd.DataFrame], calibrated_predictions: dict[str, np.ndarray],
    test_y: np.ndarray, costs: object,
) -> None:
    """Moving cutoffs cannot repair a miscalibrated subgroup.

    Encoded as a test because it is the single most misunderstood point in the
    fairness chapter: every point on the trade-off curve is built on the same
    probabilities, so if those are wrong for a group they stay wrong.
    """
    test = approved_splits["test"]
    frontier = fairness_cost_frontier(
        test_y,
        calibrated_predictions["lightgbm"],
        test["group"],
        costs.analytic_threshold,  # type: ignore[attr-defined]
        costs=costs,  # type: ignore[arg-type]
        n_points=7,
    )
    assert frontier["max_group_ece"].nunique() == 1


def test_fairness_metrics_handles_a_single_group() -> None:
    performance = pd.DataFrame(
        [{"group": "A", "n": 100, "approval_rate": 0.7, "approval_rate_given_good": 0.8,
          "approval_rate_given_bad": 0.2, "ece": 0.01, "calibration_bias": 0.0}]
    )
    result = fairness_metrics(performance, attribute="group")
    assert result["n_groups"] == 1.0
    assert "demographic_parity_difference" not in result


def test_flags_fire_on_a_clear_violation() -> None:
    performance = pd.DataFrame(
        [
            {"group": "A", "n": 1000, "approval_rate": 0.90, "approval_rate_given_good": 0.95,
             "approval_rate_given_bad": 0.4, "ece": 0.01, "calibration_bias": 0.00},
            {"group": "B", "n": 1000, "approval_rate": 0.50, "approval_rate_given_good": 0.55,
             "approval_rate_given_bad": 0.2, "ece": 0.09, "calibration_bias": 0.08},
        ]
    )
    result = fairness_metrics(performance, attribute="group")
    assert result["flag_adverse_impact"] is True
    assert result["flag_equal_opportunity"] is True
    assert result["flag_subgroup_calibration"] is True
    assert result["worst_calibrated_group"] == "B"
