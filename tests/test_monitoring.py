"""Tests for drift monitoring, vintage analysis and review triggers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import DEFAULT_SIMULATION, MonitoringThresholds
from src.monitoring import (
    categorical_stability_index,
    classify_psi,
    drift_report,
    evaluate_triggers,
    monitoring_snapshot,
    population_stability_index,
    score_distribution_report,
    vintage_curves,
    vintage_summary,
)


def test_psi_is_zero_for_identical_samples() -> None:
    rng = np.random.default_rng(0)
    sample = rng.normal(size=20_000)
    assert population_stability_index(sample, sample) == pytest.approx(0.0, abs=1e-9)


def test_psi_is_small_for_two_draws_from_one_distribution() -> None:
    rng = np.random.default_rng(1)
    assert population_stability_index(rng.normal(size=20_000), rng.normal(size=20_000)) < 0.02


def test_psi_grows_with_the_size_of_the_shift() -> None:
    rng = np.random.default_rng(2)
    reference = rng.normal(size=20_000)
    values = [
        population_stability_index(reference, rng.normal(loc=shift, size=20_000))
        for shift in (0.1, 0.5, 1.0, 2.0)
    ]
    assert values == sorted(values)
    assert values[-1] > 0.25


def test_psi_is_finite_when_a_bin_empties() -> None:
    rng = np.random.default_rng(3)
    reference = rng.normal(size=10_000)
    current = rng.normal(loc=10.0, size=10_000)  # no overlap at all
    psi = population_stability_index(reference, current)
    assert np.isfinite(psi)
    assert psi > 1.0


def test_categorical_psi_detects_a_mix_shift() -> None:
    reference = pd.Series(["a"] * 700 + ["b"] * 300)
    same = pd.Series(["a"] * 350 + ["b"] * 150)
    shifted = pd.Series(["a"] * 200 + ["b"] * 800)
    assert categorical_stability_index(reference, same) < 0.01
    assert categorical_stability_index(reference, shifted) > 0.25


def test_categorical_psi_handles_an_unseen_level() -> None:
    reference = pd.Series(["a"] * 500 + ["b"] * 500)
    current = pd.Series(["a"] * 400 + ["b"] * 400 + ["c"] * 200)
    assert np.isfinite(categorical_stability_index(reference, current))


@pytest.mark.parametrize(
    ("psi", "expected"),
    [(0.01, "stable"), (0.09, "stable"), (0.10, "watch"), (0.24, "watch"), (0.25, "review"), (0.9, "review")],
)
def test_psi_classification(psi: float, expected: str) -> None:
    assert classify_psi(psi) == expected


def test_psi_classification_handles_nan() -> None:
    assert classify_psi(float("nan")) == "unknown"


def test_drift_report_ranks_the_most_shifted_feature_first(
    approved_splits: dict[str, pd.DataFrame], features: object
) -> None:
    report = drift_report(
        approved_splits["fit"], approved_splits["test"], features.model_features  # type: ignore[attr-defined]
    )
    assert report["psi"].is_monotonic_decreasing
    assert len(report) == len(features.model_features)  # type: ignore[attr-defined]
    # The generator shortens credit history in the drifted vintages, so it
    # should be among the movers.
    assert "credit_history_months" in set(report.head(3)["feature"])


def test_drift_report_skips_missing_features(approved_splits: dict[str, pd.DataFrame]) -> None:
    report = drift_report(
        approved_splits["fit"], approved_splits["test"], ["utilisation", "not_a_column"]
    )
    assert list(report["feature"]) == ["utilisation"]


def test_score_distribution_totals_to_one() -> None:
    rng = np.random.default_rng(4)
    reference = rng.beta(2, 8, 10_000)
    current = rng.beta(2, 6, 10_000)
    table = score_distribution_report(reference, current)
    assert table["reference_share"].sum() == pytest.approx(1.0, abs=0.01)
    assert table["current_share"].sum() == pytest.approx(1.0, abs=0.01)
    assert float(table["psi_total"].iloc[0]) == pytest.approx(
        float(table["psi_contribution"].sum()), abs=1e-9
    )


def test_vintage_curves_are_cumulative(book: pd.DataFrame) -> None:
    curves = vintage_curves(book)
    for _, cohort in curves.groupby("cohort"):
        assert cohort.sort_values("months_on_book")["cumulative_default_rate"].is_monotonic_increasing


def test_vintage_curves_reach_the_final_default_rate(book: pd.DataFrame) -> None:
    curves = vintage_curves(book)
    final = curves[curves["months_on_book"] == 12]
    total_defaults = int(final["cumulative_defaults"].sum())
    assert total_defaults == int(book["default_true"].sum())


def test_vintage_analysis_detects_the_injected_deterioration(book: pd.DataFrame) -> None:
    """The lagging indicator must see what PSI can miss."""
    summary = vintage_summary(vintage_curves(book))
    drift_cohort = DEFAULT_SIMULATION.drift_start_vintage
    early = summary[summary["cohort_start_vintage"] < drift_cohort]
    late = summary[summary["cohort_start_vintage"] >= drift_cohort]
    assert float(late["cumulative_default_rate"].max()) > float(
        early["cumulative_default_rate"].max()
    )
    assert float(summary["relative_change"].max()) > 0.15


def test_triggers_report_an_action_and_a_rationale() -> None:
    triggers = evaluate_triggers(0.3, 0.4, 0.08, 0.06, 0.09)
    assert (triggers["action"].str.len() > 20).all()
    assert (triggers["rationale"].str.len() > 20).all()
    assert triggers["breached"].all()


def test_triggers_stay_quiet_when_everything_is_stable() -> None:
    triggers = evaluate_triggers(0.02, 0.03, 0.01, 0.00, 0.01)
    assert not triggers["breached"].any()


def test_combined_watch_fires_where_single_indicators_do_not() -> None:
    """The rule added because PSI missed a 75% rise in the default rate."""
    triggers = evaluate_triggers(0.103, 0.212, 0.028).set_index("name")
    assert not bool(triggers.loc["score_psi", "breached"])
    assert not bool(triggers.loc["worst_feature_psi", "breached"])
    assert bool(triggers.loc["combined_watch", "breached"])


def test_combined_watch_stays_quiet_when_only_one_indicator_moves() -> None:
    triggers = evaluate_triggers(0.02, 0.212, 0.01).set_index("name")
    assert not bool(triggers.loc["combined_watch", "breached"])


def test_custom_thresholds_are_honoured() -> None:
    strict = MonitoringThresholds(psi_warn=0.01, psi_breach=0.02)
    triggers = evaluate_triggers(0.05, 0.06, thresholds=strict).set_index("name")
    assert bool(triggers.loc["score_psi", "breached"])


def test_snapshot_assembles_the_full_pack(
    approved_splits: dict[str, pd.DataFrame], calibrated_predictions: dict[str, np.ndarray],
    features: object, test_y: np.ndarray,
) -> None:
    rng = np.random.default_rng(0)
    snapshot = monitoring_snapshot(
        approved_splits["fit"],
        approved_splits["test"],
        list(features.model_features),  # type: ignore[attr-defined]
        rng.beta(2, 12, len(approved_splits["fit"])),
        calibrated_predictions["lightgbm"],
        test_y,
    )
    for key in ("feature_drift", "score_distribution", "score_psi", "worst_feature_psi", "triggers"):
        assert key in snapshot
    assert np.isfinite(float(snapshot["score_psi"]))  # type: ignore[arg-type]
    assert not snapshot["triggers"].empty  # type: ignore[union-attr]
