"""Shared fixtures.

Everything runs on the synthetic book. That is not a convenience — it is the
only place ground truth for declined applicants and for the true probability of
default exists, so it is the only place the claims this system makes about
reject inference and fairness can actually be checked.

Fixtures are session-scoped and the book is small, so the whole suite fits in
a CI run without anyone being tempted to skip it.
"""

from __future__ import annotations

import warnings
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.calibration import CalibratedModel, fit_calibrators
from src.config import DEFAULT_COSTS, DEFAULT_SIMULATION, SYNTHETIC_FEATURES
from src.models import LightGBMModel, build_model_zoo, fit_models
from src.simulate import (
    mnar_config,
    observed_training_frame,
    simulate_loan_book,
    split_by_vintage,
)

warnings.filterwarnings("ignore")

# Small enough for a fast suite, large enough that metrics are not pure noise.
TEST_N = 6_000
FIT_END, CALIBRATION_END, TEST_START = 20, 24, 24


@pytest.fixture(scope="session")
def features() -> Any:
    """Feature specification for the synthetic book."""
    return SYNTHETIC_FEATURES


@pytest.fixture(scope="session")
def costs() -> Any:
    """Default cost parameters."""
    return DEFAULT_COSTS


@pytest.fixture(scope="session")
def book() -> pd.DataFrame:
    """A synthetic loan book under selection on observables (MAR)."""
    return simulate_loan_book(replace(DEFAULT_SIMULATION, n_applicants=TEST_N))


@pytest.fixture(scope="session")
def mnar_book() -> pd.DataFrame:
    """A synthetic loan book under selection on unobservables (MNAR)."""
    return simulate_loan_book(mnar_config(replace(DEFAULT_SIMULATION, n_applicants=TEST_N)))


@pytest.fixture(scope="session")
def large_book() -> pd.DataFrame:
    """A full-size MAR book for the reject inference conclusions.

    Sized deliberately. Reject inference results depend on how well the scoring
    model is trained: on a small book the model is weak, leaving risk
    heterogeneity inside each score band, which manufactures an apparent
    reject uplift of over 2x and makes the corrections look helpful. That
    effect disappears with enough data (see the table in
    :mod:`src.reject_inference`). Testing the reported MAR conclusion on a
    small book would test the artefact instead of the finding.
    """
    return simulate_loan_book(replace(DEFAULT_SIMULATION, n_applicants=DEFAULT_SIMULATION.n_applicants))


@pytest.fixture(scope="session")
def splits(book: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Out-of-time split of the full book, approved and declined together."""
    return split_by_vintage(book, FIT_END, CALIBRATION_END, TEST_START)


@pytest.fixture(scope="session")
def approved_splits(splits: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Out-of-time split restricted to rows a lender could actually train on."""
    return {name: observed_training_frame(frame) for name, frame in splits.items()}


@pytest.fixture(scope="session")
def fitted_zoo(approved_splits: dict[str, pd.DataFrame], features: Any) -> dict[str, Any]:
    """All three models fitted on the training vintages."""
    fit = approved_splits["fit"]
    return fit_models(
        build_model_zoo(features),
        fit[features.model_features],
        fit["default_observed"].to_numpy(),
    )


@pytest.fixture(scope="session")
def calibrated_predictions(
    fitted_zoo: dict[str, Any], approved_splits: dict[str, pd.DataFrame], features: Any
) -> dict[str, np.ndarray]:
    """Platt-calibrated out-of-time predictions for every model."""
    calibration = approved_splits["calibration"]
    test = approved_splits["test"]
    out = {}
    for name, model in fitted_zoo.items():
        calibrators = fit_calibrators(
            model.predict_proba(calibration[features.model_features]),
            calibration["default_observed"].to_numpy(),
        )
        out[name] = CalibratedModel(model, calibrators["platt"], name).predict_proba(
            test[features.model_features]
        )
    return out


@pytest.fixture(scope="session")
def test_y(approved_splits: dict[str, pd.DataFrame]) -> np.ndarray:
    """Realised default flag on the out-of-time test set."""
    return approved_splits["test"]["default_observed"].to_numpy().astype(int)


@pytest.fixture(scope="session")
def model_factory(features: Any) -> Any:
    """Zero-argument factory producing an unfitted LightGBM model."""

    def factory() -> LightGBMModel:
        return LightGBMModel(list(features.numeric), list(features.categorical))

    return factory


@pytest.fixture
def artifact_dir(tmp_path: Path) -> Path:
    """An isolated artifact directory for tests that write bundles."""
    directory = tmp_path / "artifacts"
    directory.mkdir()
    return directory
