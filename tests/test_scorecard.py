"""Tests for WOE binning and the points scorecard."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.scorecard import (
    MISSING_BIN_LABEL,
    OTHER_BIN_LABEL,
    Scorecard,
    WOEEncoder,
    fit_categorical_binning,
    fit_numeric_binning,
)


@pytest.fixture(scope="module")
def toy() -> tuple[pd.DataFrame, np.ndarray]:
    """A frame where risk falls monotonically with x."""
    rng = np.random.default_rng(0)
    n = 4_000
    x = rng.uniform(0, 1, n)
    y = (rng.random(n) < (0.5 - 0.4 * x)).astype(int)
    frame = pd.DataFrame(
        {"x": x, "noise": rng.normal(size=n), "cat": rng.choice(["a", "b", "c"], n)}
    )
    return frame, y


def test_woe_sign_convention(toy: tuple[pd.DataFrame, np.ndarray]) -> None:
    """Higher WOE must mean lower risk."""
    frame, y = toy
    binning = fit_numeric_binning(frame["x"], y)
    rows = binning.to_frame().dropna(subset=["event_rate"])
    rows = rows[rows["count"] > 0]
    correlation = np.corrcoef(rows["woe"], rows["event_rate"])[0, 1]
    assert correlation < -0.8


def test_numeric_binning_is_monotonic(toy: tuple[pd.DataFrame, np.ndarray]) -> None:
    frame, y = toy
    binning = fit_numeric_binning(frame["x"], y, enforce_monotonic=True)
    labels = binning._numeric_labels()
    rates = [binning.event_rate[label] for label in labels if label in binning.event_rate]
    differences = np.diff(rates)
    assert (differences <= 1e-9).all() or (differences >= -1e-9).all()


def test_bins_respect_the_minimum_size(toy: tuple[pd.DataFrame, np.ndarray]) -> None:
    frame, y = toy
    binning = fit_numeric_binning(frame["x"], y, min_bin_fraction=0.10)
    populated = {k: v for k, v in binning.count.items() if k != MISSING_BIN_LABEL and v > 0}
    assert min(populated.values()) >= 0.05 * len(frame)


def test_information_value_ranks_signal_above_noise(toy: tuple[pd.DataFrame, np.ndarray]) -> None:
    frame, y = toy
    signal = fit_numeric_binning(frame["x"], y)
    noise = fit_numeric_binning(frame["noise"], y)
    assert signal.iv > noise.iv
    assert signal.iv > 0.10


def test_missing_values_get_their_own_bin() -> None:
    rng = np.random.default_rng(1)
    x = pd.Series(rng.uniform(0, 1, 2_000))
    x[:300] = np.nan
    y = (rng.random(2_000) < 0.3).astype(int)
    binning = fit_numeric_binning(x, y)
    assert MISSING_BIN_LABEL in binning.woe
    assert binning.count[MISSING_BIN_LABEL] == 300
    assert np.isfinite(binning.transform(x)).all()


def test_woe_is_finite_when_a_bin_is_all_good() -> None:
    """Smoothing must prevent an infinite WOE in a pure bin."""
    x = pd.Series(np.arange(1_000, dtype=float), name="x")
    y = (np.arange(1_000) > 500).astype(int)  # lower half is entirely good
    binning = fit_numeric_binning(x, y, enforce_monotonic=False)
    assert np.isfinite(list(binning.woe.values())).all()


def test_unseen_category_falls_to_other(toy: tuple[pd.DataFrame, np.ndarray]) -> None:
    frame, y = toy
    binning = fit_categorical_binning(frame["cat"], y)
    unseen = pd.Series(["zzz_never_seen"] * 10, name="cat")
    assert (binning.assign_bins(unseen) == OTHER_BIN_LABEL).all()
    assert np.isfinite(binning.transform(unseen)).all()


def test_encoder_drops_features_below_the_iv_floor(toy: tuple[pd.DataFrame, np.ndarray]) -> None:
    frame, y = toy
    encoder = WOEEncoder(["x", "noise"], ["cat"], min_iv=0.05).fit(frame, y)
    assert "x" in encoder.selected_features
    assert "noise" not in encoder.selected_features


def test_points_table_reproduces_the_score(
    approved_splits: dict[str, pd.DataFrame], features: object
) -> None:
    """The deployable artefact must agree with the model exactly.

    A scorecard whose printed points do not sum to its own score is not a
    scorecard, it is a model with a misleading table attached.
    """
    fit = approved_splits["fit"]
    columns = features.model_features  # type: ignore[attr-defined]
    card = Scorecard(list(features.numeric), list(features.categorical))  # type: ignore[attr-defined]
    card.fit(fit[columns], fit["default_observed"].to_numpy())

    test = approved_splits["test"].head(50)
    scores = card.score(test[columns])
    points = card.points_table()

    for position in range(len(test)):
        row = test.iloc[[position]]
        total = 0.0
        for column in card.feature_names_:
            name = column[: -len("_woe")]
            label = str(card.encoder.binnings[name].assign_bins(row[name]).iloc[0])
            match = points[(points["feature"] == name) & (points["bin"] == label)]
            total += float(match["points"].iloc[0])
        assert total == pytest.approx(float(scores[position]), abs=1e-6)


def test_higher_score_means_lower_predicted_risk(
    approved_splits: dict[str, pd.DataFrame], features: object
) -> None:
    """Score and PD are related by score = offset - factor * logit(p).

    That is strictly decreasing but not linear, so the correct assertion is on
    rank correlation. A Pearson coefficient here would be testing the curvature
    of the logit, not the property that matters.
    """
    from scipy.stats import spearmanr

    fit = approved_splits["fit"]
    columns = features.model_features  # type: ignore[attr-defined]
    card = Scorecard(list(features.numeric), list(features.categorical))  # type: ignore[attr-defined]
    card.fit(fit[columns], fit["default_observed"].to_numpy())

    test = approved_splits["test"]
    correlation = spearmanr(card.score(test[columns]), card.predict_proba(test[columns])).statistic
    assert correlation == pytest.approx(-1.0, abs=1e-9)


def test_sign_flipped_features_are_dropped(
    approved_splits: dict[str, pd.DataFrame], features: object
) -> None:
    """A positive coefficient on WOE would award points for riskier behaviour."""
    fit = approved_splits["fit"]
    columns = features.model_features  # type: ignore[attr-defined]
    card = Scorecard(
        list(features.numeric), list(features.categorical), drop_sign_flips=True  # type: ignore[attr-defined]
    )
    card.fit(fit[columns], fit["default_observed"].to_numpy())
    assert card.coefficient_table()["sign_as_expected"].all()


def test_sample_weights_change_the_fit(
    approved_splits: dict[str, pd.DataFrame], features: object
) -> None:
    """Fuzzy augmentation depends on weights being honoured."""
    fit = approved_splits["fit"]
    columns = features.model_features  # type: ignore[attr-defined]
    y = fit["default_observed"].to_numpy()

    unweighted = Scorecard(list(features.numeric), list(features.categorical))  # type: ignore[attr-defined]
    unweighted.fit(fit[columns], y)

    weights = np.where(y == 1, 3.0, 1.0)
    weighted = Scorecard(list(features.numeric), list(features.categorical))  # type: ignore[attr-defined]
    weighted.fit(fit[columns], y, sample_weight=weights)

    assert weighted.predict_proba(fit[columns]).mean() > unweighted.predict_proba(fit[columns]).mean()


def test_prepare_refuses_before_fit(features: object) -> None:
    """A model asked to prepare a frame before fitting must refuse.

    Silently inferring categories from the frame in front of it is the exact
    corruption the stored ordering exists to prevent: the codes would come from
    the scoring batch rather than the training data.
    """
    from sklearn.exceptions import NotFittedError

    from src.models import LightGBMModel

    model = LightGBMModel(list(features.numeric), list(features.categorical))  # type: ignore[attr-defined]
    frame = pd.DataFrame([dict.fromkeys(features.numeric, 1.0)])  # type: ignore[attr-defined]
    for name in features.categorical:  # type: ignore[attr-defined]
        frame[name] = "own"
    with pytest.raises(NotFittedError, match="call fit"):
        model.prepare(frame)


def test_prepare_uses_training_categories_not_batch_categories(
    approved_splits: dict[str, pd.DataFrame], features: object
) -> None:
    """Category codes must come from training, whatever the scoring batch holds.

    LightGBM encodes categoricals by integer code. If a scoring batch is missing
    a level, re-inferring categories shifts every code and the model scores
    against the wrong encoding — silently, with no error and no warning.
    """
    from src.models import LightGBMModel

    columns = features.model_features  # type: ignore[attr-defined]
    fit = approved_splits["fit"]
    model = LightGBMModel(list(features.numeric), list(features.categorical))  # type: ignore[attr-defined]
    model.fit(fit[columns], fit["default_observed"].to_numpy())

    trained_categories = dict(model.categories_)

    # A batch containing only one housing status, which would infer a
    # single-category ordering if categories were re-derived.
    narrow = fit[fit["housing_status"] == "rent"].head(50)[columns]
    prepared = model.prepare(narrow)

    assert list(prepared["housing_status"].cat.categories) == trained_categories["housing_status"]
    assert model.categories_ == trained_categories, "prepare() must not mutate fitted state"
