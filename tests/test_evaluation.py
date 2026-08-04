"""Tests for evaluation under class imbalance."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation import (
    discrimination_report,
    evaluate_models,
    gains_table,
    ks_statistic,
    precision_at_capacity,
    resampling_experiment,
)


def test_ks_is_one_for_perfect_separation() -> None:
    y = np.array([0] * 500 + [1] * 500)
    p = np.array([0.1] * 500 + [0.9] * 500)
    assert ks_statistic(y, p) == pytest.approx(1.0, abs=1e-9)


def test_ks_is_near_zero_for_random_scores() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 20_000)
    assert ks_statistic(y, rng.random(20_000)) < 0.05


def test_precision_at_capacity_is_perfect_for_a_perfect_ranker() -> None:
    y = np.array([1] * 100 + [0] * 900)
    p = np.concatenate([np.linspace(1.0, 0.9, 100), np.linspace(0.1, 0.0, 900)])
    table = precision_at_capacity(y, p, capacities=(0.01, 0.05, 0.10)).set_index("capacity")
    assert table.loc[0.01, "precision"] == 1.0
    assert table.loc[0.10, "precision"] == 1.0
    assert table.loc[0.10, "recall"] == 1.0


def test_precision_at_capacity_matches_the_base_rate_for_random_scores() -> None:
    rng = np.random.default_rng(3)
    y = (rng.random(40_000) < 0.2).astype(int)
    table = precision_at_capacity(y, rng.random(40_000), capacities=(0.10,))
    assert float(table["precision"].iloc[0]) == pytest.approx(0.2, abs=0.02)
    assert float(table["lift"].iloc[0]) == pytest.approx(1.0, abs=0.1)


def test_capacity_counts_are_right() -> None:
    y = np.zeros(1_000, dtype=int)
    y[:50] = 1
    table = precision_at_capacity(y, np.linspace(1, 0, 1_000), capacities=(0.01, 0.05))
    assert list(table["n_reviewed"]) == [10, 50]


def test_gains_table_concentrates_bads_in_the_first_band() -> None:
    rng = np.random.default_rng(5)
    p = rng.uniform(0, 1, 20_000)
    y = (rng.random(20_000) < p).astype(int)
    table = gains_table(y, p, n_bands=10)
    assert table["bad_rate"].iloc[0] > table["bad_rate"].iloc[-1]
    assert table["cumulative_bad_capture"].iloc[-1] == pytest.approx(1.0, abs=1e-9)
    assert table["cumulative_bad_capture"].is_monotonic_increasing
    assert table["lift"].iloc[0] > 1.0


def test_accuracy_is_reported_beside_its_baseline(
    calibrated_predictions: dict[str, np.ndarray], test_y: np.ndarray
) -> None:
    """The foil must be present, or accuracy is misleading on its own."""
    report = discrimination_report(test_y, calibrated_predictions["lightgbm"])
    assert "accuracy_at_0p5" in report
    assert "accuracy_majority_baseline" in report
    # On an imbalanced book, accuracy barely beats predicting "never defaults".
    assert float(report["accuracy_at_0p5"]) - float(report["accuracy_majority_baseline"]) < 0.10


def test_pr_auc_is_reported_against_its_baseline(
    calibrated_predictions: dict[str, np.ndarray], test_y: np.ndarray
) -> None:
    report = discrimination_report(test_y, calibrated_predictions["lightgbm"])
    assert float(report["pr_auc_baseline"]) == pytest.approx(float(np.mean(test_y)))
    assert float(report["pr_auc"]) > float(report["pr_auc_baseline"])


def test_gini_is_derived_from_auc(
    calibrated_predictions: dict[str, np.ndarray], test_y: np.ndarray
) -> None:
    report = discrimination_report(test_y, calibrated_predictions["lightgbm"])
    assert float(report["gini"]) == pytest.approx(2 * float(report["roc_auc"]) - 1)


def test_all_models_beat_random(
    calibrated_predictions: dict[str, np.ndarray], test_y: np.ndarray
) -> None:
    table = evaluate_models(calibrated_predictions, test_y)
    assert (table["roc_auc"] > 0.65).all()
    assert (table["ks"] > 0.25).all()


def test_scorecard_is_competitive_with_the_booster(
    calibrated_predictions: dict[str, np.ndarray], test_y: np.ndarray
) -> None:
    """The headline finding, asserted so a regression would be noticed.

    The scorecard is not expected to win. It is expected to be close enough
    that its interpretability and stability are worth more than the gap.
    """
    table = evaluate_models(calibrated_predictions, test_y).set_index("label")
    gap = float(table.loc["lightgbm", "roc_auc"]) - float(table.loc["scorecard_woe_lr", "roc_auc"])
    assert gap < 0.05, f"Booster now beats the scorecard by {gap:.3f} AUC; revisit the write-up"


def test_resampling_damages_calibration_without_helping_ranking(
    approved_splits: dict[str, object], features: object, model_factory: object
) -> None:
    """The documented result: resampling costs calibration and buys nothing."""
    fit = approved_splits["fit"]  # type: ignore[index]
    test = approved_splits["test"]  # type: ignore[index]
    columns = features.model_features  # type: ignore[attr-defined]

    table = resampling_experiment(
        model_factory,
        fit[columns],
        fit["default_observed"].to_numpy(),
        test[columns],
        test["default_observed"].to_numpy(),
    ).set_index("regime")

    baseline_ece = float(table.loc["none", "ece"])
    for regime in ("class_weight", "smote", "random_undersample"):
        if regime not in table.index:
            continue
        assert float(table.loc[regime, "ece"]) > baseline_ece, f"{regime} did not harm calibration"
        # And it does not buy meaningfully better ranking in exchange.
        assert float(table.loc[regime, "pr_auc"]) < float(table.loc["none", "pr_auc"]) + 0.02
