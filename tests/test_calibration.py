"""Tests for calibration metrics and calibrators."""

from __future__ import annotations

import numpy as np
import pytest

from src.calibration import (
    IdentityCalibrator,
    IsotonicCalibrator,
    PlattCalibrator,
    brier_decomposition,
    brier_score,
    calibration_slope_intercept,
    compare_calibrators,
    expected_calibration_error,
    fit_calibrators,
    maximum_calibration_error,
    reliability_table,
    select_calibrator,
)


@pytest.fixture(scope="module")
def perfectly_calibrated() -> tuple[np.ndarray, np.ndarray]:
    """Probabilities that are true by construction."""
    rng = np.random.default_rng(0)
    p = rng.uniform(0.02, 0.6, 20_000)
    y = (rng.random(20_000) < p).astype(int)
    return y, p


def test_brier_is_zero_for_perfect_prediction() -> None:
    y = np.array([0, 1, 0, 1])
    assert brier_score(y, y.astype(float)) == 0.0


def test_brier_is_one_for_perfectly_wrong_prediction() -> None:
    y = np.array([0, 1, 0, 1])
    assert brier_score(y, 1.0 - y.astype(float)) == 1.0


def test_ece_is_near_zero_when_calibrated(
    perfectly_calibrated: tuple[np.ndarray, np.ndarray],
) -> None:
    y, p = perfectly_calibrated
    assert expected_calibration_error(y, p) < 0.015
    assert maximum_calibration_error(y, p) < 0.05


def test_ece_detects_a_scaled_probability(
    perfectly_calibrated: tuple[np.ndarray, np.ndarray],
) -> None:
    """Halving every probability leaves AUC untouched and ruins calibration.

    This is the case the whole calibration module exists for.
    """
    from sklearn.metrics import roc_auc_score

    y, p = perfectly_calibrated
    scaled = p * 0.5
    assert roc_auc_score(y, scaled) == pytest.approx(roc_auc_score(y, p), abs=1e-12)
    assert expected_calibration_error(y, scaled) > 5 * expected_calibration_error(y, p)


def test_calibration_slope_and_intercept_are_ideal_when_calibrated(
    perfectly_calibrated: tuple[np.ndarray, np.ndarray],
) -> None:
    y, p = perfectly_calibrated
    intercept, slope = calibration_slope_intercept(y, p)
    assert slope == pytest.approx(1.0, abs=0.1)
    assert intercept == pytest.approx(0.0, abs=0.1)


def test_calibration_slope_detects_overconfidence(
    perfectly_calibrated: tuple[np.ndarray, np.ndarray],
) -> None:
    """Spreading the log-odds out must push the slope below 1."""
    y, p = perfectly_calibrated
    logits = np.log(p / (1 - p))
    overconfident = 1.0 / (1.0 + np.exp(-logits * 1.8))
    _, slope = calibration_slope_intercept(y, overconfident)
    assert slope < 0.85


def test_brier_decomposition_reconstructs_the_score(
    perfectly_calibrated: tuple[np.ndarray, np.ndarray],
) -> None:
    y, p = perfectly_calibrated
    decomposition = brier_decomposition(y, p)
    assert decomposition.brier == pytest.approx(brier_score(y, p), abs=0.005)
    assert decomposition.reliability >= 0
    assert decomposition.resolution >= 0


def test_reliability_table_covers_every_row(
    perfectly_calibrated: tuple[np.ndarray, np.ndarray],
) -> None:
    y, p = perfectly_calibrated
    table = reliability_table(y, p, n_bins=10)
    assert table["count"].sum() == len(y)
    assert (table["gap"].abs() < 0.05).all()


def test_identity_calibrator_is_a_no_op() -> None:
    p = np.array([0.1, 0.5, 0.9])
    np.testing.assert_allclose(IdentityCalibrator().fit(p, p).transform(p), p, atol=1e-6)


def test_platt_recovers_identity_on_calibrated_input(
    perfectly_calibrated: tuple[np.ndarray, np.ndarray],
) -> None:
    """Platt on already-calibrated scores should be close to a = 0, b = 1."""
    y, p = perfectly_calibrated
    calibrator = PlattCalibrator().fit(p, y)
    intercept, slope = calibrator.parameters
    assert slope == pytest.approx(1.0, abs=0.15)
    assert intercept == pytest.approx(0.0, abs=0.15)


def test_platt_repairs_a_miscalibrated_score(
    perfectly_calibrated: tuple[np.ndarray, np.ndarray],
) -> None:
    y, p = perfectly_calibrated
    broken = p * 0.4
    split = len(y) // 2
    calibrator = PlattCalibrator().fit(broken[:split], y[:split])
    repaired = calibrator.transform(broken[split:])
    assert expected_calibration_error(y[split:], repaired) < expected_calibration_error(
        y[split:], broken[split:]
    )


def test_isotonic_repairs_a_miscalibrated_score(
    perfectly_calibrated: tuple[np.ndarray, np.ndarray],
) -> None:
    y, p = perfectly_calibrated
    broken = p**2
    split = len(y) // 2
    calibrator = IsotonicCalibrator().fit(broken[:split], y[:split])
    repaired = calibrator.transform(broken[split:])
    assert brier_score(y[split:], repaired) < brier_score(y[split:], broken[split:])


def test_platt_preserves_ranking_exactly(
    perfectly_calibrated: tuple[np.ndarray, np.ndarray],
) -> None:
    """Platt is strictly monotone, so AUC is untouched to floating point."""
    from sklearn.metrics import roc_auc_score

    y, p = perfectly_calibrated
    broken = p * 0.4
    calibrated = PlattCalibrator().fit(broken, y).transform(broken)
    assert roc_auc_score(y, calibrated) == pytest.approx(roc_auc_score(y, broken), abs=1e-9)


def test_isotonic_loses_a_little_ranking_to_ties(
    perfectly_calibrated: tuple[np.ndarray, np.ndarray],
) -> None:
    """Isotonic is a step function, so it destroys ranking within each step.

    The documented cost of isotonic. It is small here but it is real, and it is
    a reason to prefer Platt when the ranking inside the high-risk tail is what
    the cutoff acts on.
    """
    from sklearn.metrics import roc_auc_score

    y, p = perfectly_calibrated
    broken = p * 0.4
    calibrated = IsotonicCalibrator().fit(broken, y).transform(broken)

    assert len(np.unique(calibrated)) < len(np.unique(broken))
    assert roc_auc_score(y, calibrated) == pytest.approx(roc_auc_score(y, broken), abs=0.01)


def test_calibrators_improve_on_real_model_output(
    fitted_zoo: dict[str, object], approved_splits: dict[str, object], features: object
) -> None:
    calibration = approved_splits["calibration"]  # type: ignore[index]
    test = approved_splits["test"]  # type: ignore[index]
    columns = features.model_features  # type: ignore[attr-defined]
    model = fitted_zoo["lightgbm"]

    fitted = fit_calibrators(
        model.predict_proba(calibration[columns]),  # type: ignore[attr-defined]
        calibration["default_observed"].to_numpy(),
    )
    comparison = compare_calibrators(
        fitted,
        model.predict_proba(test[columns]),  # type: ignore[attr-defined]
        test["default_observed"].to_numpy(),
        model_name="lightgbm",
    )
    assert set(comparison["calibrator"]) == {"raw", "platt", "isotonic"}
    assert comparison["brier"].notna().all()
    assert select_calibrator(comparison) in {"raw", "platt", "isotonic"}


def test_select_calibrator_prefers_a_slope_near_one() -> None:
    """The documented rule: a good Brier with a bad slope is not acceptable."""
    import pandas as pd

    comparison = pd.DataFrame(
        [
            {"calibrator": "isotonic", "brier": 0.100, "ece_quantile": 0.01, "calibration_slope": 0.60},
            {"calibrator": "platt", "brier": 0.105, "ece_quantile": 0.02, "calibration_slope": 1.02},
        ]
    )
    assert select_calibrator(comparison) == "platt"


def test_select_calibrator_falls_back_when_none_qualifies() -> None:
    import pandas as pd

    comparison = pd.DataFrame(
        [
            {"calibrator": "raw", "brier": 0.20, "ece_quantile": 0.09, "calibration_slope": 0.4},
            {"calibrator": "platt", "brier": 0.15, "ece_quantile": 0.05, "calibration_slope": 0.5},
        ]
    )
    assert select_calibrator(comparison) == "platt"
