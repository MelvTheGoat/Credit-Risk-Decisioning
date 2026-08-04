"""Calibration regression gate.

Purpose
-------
The failure this guards against is specific and quiet. A change to feature
engineering, a library upgrade, or an "improvement" to the model can leave AUC
untouched — or even nudge it up — while wrecking the calibration that expected
loss, provisioning and the cutoff all depend on. Nothing in an ordinary
accuracy-focused test suite would notice.

So calibration quality is pinned to a committed baseline and CI fails if it
regresses. Discrimination is pinned too, but loosely: the point of the gate is
that a *calibration* regression cannot be traded away for a ranking gain.

How to change the baseline
--------------------------
Run ``python -m tests.test_calibration_gate --update`` and **commit the diff
with a reason**. The baseline moving is a governance event, not housekeeping:
a reviewer should be able to see from the commit history when the model's
calibration changed and why it was accepted.

Tolerances are deliberately loose enough to absorb library-version noise and
tight enough to catch a real regression. They are absolute, not relative,
because a small relative change in an already-small ECE does not matter, while
an absolute jump of two percentage points does.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src.calibration import (
    CalibratedModel,
    brier_score,
    calibration_slope_intercept,
    expected_calibration_error,
    fit_calibrators,
    maximum_calibration_error,
)

BASELINE_PATH = Path(__file__).parent / "calibration_baseline.json"

# Absolute tolerances. A model may drift within these without failing the build.
TOLERANCES: dict[str, float] = {
    "brier": 0.010,
    "ece": 0.015,
    "mce": 0.040,
    "calibration_slope": 0.150,
    "roc_auc": 0.030,
}

# Hard floors, independent of the baseline. Even a "legitimately updated"
# baseline may not fall below these and still be servable.
ABSOLUTE_LIMITS: dict[str, float] = {
    "max_ece": 0.060,
    "max_brier": 0.140,
    "min_roc_auc": 0.740,
    "min_slope": 0.750,
    "max_slope": 1.300,
}


def compute_metrics(
    fitted_zoo: dict[str, Any], approved_splits: dict[str, pd.DataFrame], features: Any
) -> dict[str, dict[str, float]]:
    """Score every model, calibrated, on the out-of-time test set.

    Args:
        fitted_zoo: Fitted models.
        approved_splits: Out-of-time splits, approved applicants only.
        features: Feature specification.

    Returns:
        Nested mapping of model name to metric name to value.
    """
    from sklearn.metrics import roc_auc_score

    calibration = approved_splits["calibration"]
    test = approved_splits["test"]
    columns = features.model_features
    y_test = test["default_observed"].to_numpy().astype(int)

    results: dict[str, dict[str, float]] = {}
    for name, model in fitted_zoo.items():
        calibrator = fit_calibrators(
            model.predict_proba(calibration[columns]),
            calibration["default_observed"].to_numpy(),
        )["platt"]
        p = CalibratedModel(model, calibrator, name).predict_proba(test[columns])
        _, slope = calibration_slope_intercept(y_test, p)
        results[name] = {
            "brier": brier_score(y_test, p),
            "ece": expected_calibration_error(y_test, p),
            "mce": maximum_calibration_error(y_test, p),
            "calibration_slope": slope,
            "roc_auc": float(roc_auc_score(y_test, p)),
        }
    return results


@pytest.fixture(scope="module")
def measured(
    fitted_zoo: dict[str, Any], approved_splits: dict[str, pd.DataFrame], features: Any
) -> dict[str, dict[str, float]]:
    return compute_metrics(fitted_zoo, approved_splits, features)


def test_baseline_file_exists() -> None:
    assert BASELINE_PATH.exists(), (
        f"No calibration baseline at {BASELINE_PATH}. "
        "Create it with: python -m tests.test_calibration_gate --update"
    )


def test_baseline_covers_every_model(measured: dict[str, dict[str, float]]) -> None:
    baseline = json.loads(BASELINE_PATH.read_text())
    missing = set(measured) - set(baseline)
    assert not missing, f"Models absent from the baseline: {sorted(missing)}. Regenerate it."


@pytest.mark.parametrize("metric", ["brier", "ece", "mce"])
def test_calibration_has_not_regressed(measured: dict[str, dict[str, float]], metric: str) -> None:
    """Lower is better for all three; a rise beyond tolerance fails the build."""
    baseline = json.loads(BASELINE_PATH.read_text())
    failures = []
    for model, metrics in measured.items():
        if model not in baseline:
            continue
        expected = float(baseline[model][metric])
        actual = float(metrics[metric])
        if actual > expected + TOLERANCES[metric]:
            failures.append(
                f"{model}.{metric}: {actual:.4f} vs baseline {expected:.4f} "
                f"(tolerance {TOLERANCES[metric]})"
            )
    assert not failures, "Calibration regression:\n  " + "\n  ".join(failures)


def test_calibration_slope_has_not_drifted(measured: dict[str, dict[str, float]]) -> None:
    """Slope must stay near 1 in both directions.

    Unlike ECE this is two-sided: a slope of 1.3 is as wrong as 0.7, in the
    opposite direction, and only this metric distinguishes them.
    """
    baseline = json.loads(BASELINE_PATH.read_text())
    failures = []
    for model, metrics in measured.items():
        if model not in baseline:
            continue
        expected = float(baseline[model]["calibration_slope"])
        actual = float(metrics["calibration_slope"])
        if abs(actual - expected) > TOLERANCES["calibration_slope"]:
            failures.append(f"{model}: slope {actual:.4f} vs baseline {expected:.4f}")
    assert not failures, "Calibration slope drift:\n  " + "\n  ".join(failures)


def test_discrimination_has_not_collapsed(measured: dict[str, dict[str, float]]) -> None:
    """Loosely pinned, so calibration cannot be traded for a ranking gain."""
    baseline = json.loads(BASELINE_PATH.read_text())
    failures = []
    for model, metrics in measured.items():
        if model not in baseline:
            continue
        expected = float(baseline[model]["roc_auc"])
        actual = float(metrics["roc_auc"])
        if actual < expected - TOLERANCES["roc_auc"]:
            failures.append(f"{model}: AUC {actual:.4f} vs baseline {expected:.4f}")
    assert not failures, "Discrimination regression:\n  " + "\n  ".join(failures)


def test_absolute_limits_hold(measured: dict[str, dict[str, float]]) -> None:
    """Floors that apply even to a freshly regenerated baseline.

    Stops the gate being defeated by regenerating the baseline against a model
    that should never have been servable in the first place.
    """
    failures = []
    for model, metrics in measured.items():
        if metrics["ece"] > ABSOLUTE_LIMITS["max_ece"]:
            failures.append(f"{model}: ECE {metrics['ece']:.4f} exceeds {ABSOLUTE_LIMITS['max_ece']}")
        if metrics["brier"] > ABSOLUTE_LIMITS["max_brier"]:
            failures.append(f"{model}: Brier {metrics['brier']:.4f} exceeds {ABSOLUTE_LIMITS['max_brier']}")
        if metrics["roc_auc"] < ABSOLUTE_LIMITS["min_roc_auc"]:
            failures.append(f"{model}: AUC {metrics['roc_auc']:.4f} below {ABSOLUTE_LIMITS['min_roc_auc']}")
        if not (
            ABSOLUTE_LIMITS["min_slope"] <= metrics["calibration_slope"] <= ABSOLUTE_LIMITS["max_slope"]
        ):
            failures.append(
                f"{model}: slope {metrics['calibration_slope']:.4f} outside "
                f"[{ABSOLUTE_LIMITS['min_slope']}, {ABSOLUTE_LIMITS['max_slope']}]"
            )
    assert not failures, "Absolute calibration limits breached:\n  " + "\n  ".join(failures)


def test_calibration_beats_the_uncalibrated_model(
    fitted_zoo: dict[str, Any], approved_splits: dict[str, pd.DataFrame], features: Any
) -> None:
    """Calibration must earn its place, not merely be present."""
    calibration = approved_splits["calibration"]
    test = approved_splits["test"]
    columns = features.model_features
    y_test = test["default_observed"].to_numpy().astype(int)

    model = fitted_zoo["lightgbm"]
    raw = model.predict_proba(test[columns])
    calibrator = fit_calibrators(
        model.predict_proba(calibration[columns]), calibration["default_observed"].to_numpy()
    )["platt"]
    calibrated = calibrator.transform(raw)

    assert expected_calibration_error(y_test, calibrated) <= expected_calibration_error(y_test, raw) + 0.005


def _update_baseline() -> None:  # pragma: no cover - maintenance utility
    """Regenerate the committed baseline. Commit the diff with a reason."""
    from dataclasses import replace

    from src.config import DEFAULT_SIMULATION, SYNTHETIC_FEATURES
    from src.models import build_model_zoo, fit_models
    from src.simulate import observed_training_frame, simulate_loan_book, split_by_vintage
    from tests.conftest import CALIBRATION_END, FIT_END, TEST_N, TEST_START

    book = simulate_loan_book(replace(DEFAULT_SIMULATION, n_applicants=TEST_N))
    splits = {
        name: observed_training_frame(frame)
        for name, frame in split_by_vintage(book, FIT_END, CALIBRATION_END, TEST_START).items()
    }
    fit = splits["fit"]
    zoo = fit_models(
        build_model_zoo(SYNTHETIC_FEATURES),
        fit[SYNTHETIC_FEATURES.model_features],
        fit["default_observed"].to_numpy(),
    )
    metrics = compute_metrics(zoo, splits, SYNTHETIC_FEATURES)
    BASELINE_PATH.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {BASELINE_PATH}")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    import sys

    if "--update" in sys.argv:
        _update_baseline()
    else:
        print(__doc__)
        print("Run with --update to regenerate the baseline.")
