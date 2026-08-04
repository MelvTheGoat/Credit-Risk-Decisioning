"""Tests for reject inference.

These assert the *findings*, not just that the code runs, so that a change
which quietly reverses a conclusion fails the build. The headline conclusion —
that reject inference helps under MNAR and hurts under MAR — is encoded
directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.reject_inference import (
    evaluate_on_population,
    fit_approved_only,
    fuzzy_augmentation,
    heckman_correction,
    measure_true_uplift,
    oracle_model,
    parcelling,
    run_reject_inference_study,
    run_uplift_sensitivity,
    summarise_bias_removed,
)
from src.simulate import split_by_vintage


def _train_and_test(book: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    parts = split_by_vintage(book, 20, 24, 24)
    return pd.concat([parts["fit"], parts["calibration"]], ignore_index=True), parts["test"]


def test_baseline_underestimates_risk_for_rejects(
    book: pd.DataFrame, features: object, model_factory: object
) -> None:
    """The problem the module exists to address, demonstrated."""
    train, test = _train_and_test(book)
    columns = features.model_features  # type: ignore[attr-defined]
    approved = train["approved"] == 1

    model = fit_approved_only(
        model_factory, train.loc[approved, columns], train.loc[approved, "default_observed"].to_numpy()
    )
    result = evaluate_on_population(
        model.predict_proba(test[columns]),
        test["default_true"].to_numpy(),
        test["approved"].to_numpy(),
        label="baseline",
    )
    assert float(result["bias_reject"]) < 0, "Approved-only model should under-predict reject risk"


def test_fuzzy_augmentation_shifts_predictions_upward(
    book: pd.DataFrame, features: object, model_factory: object
) -> None:
    train, test = _train_and_test(book)
    columns = features.model_features  # type: ignore[attr-defined]
    approved = train["approved"] == 1

    X_approved = train.loc[approved, columns]
    y_approved = train.loc[approved, "default_observed"].to_numpy()
    X_rejected = train.loc[~approved, columns]

    baseline = fit_approved_only(model_factory, X_approved, y_approved)
    augmented = fuzzy_augmentation(
        model_factory, X_approved, y_approved, X_rejected, uplift=2.5, base_model=baseline
    )
    assert augmented.predict_proba(test[columns]).mean() > baseline.predict_proba(test[columns]).mean()


def test_parcelling_is_reproducible(
    book: pd.DataFrame, features: object, model_factory: object
) -> None:
    train, _ = _train_and_test(book)
    columns = features.model_features  # type: ignore[attr-defined]
    approved = train["approved"] == 1
    X_approved = train.loc[approved, columns]
    y_approved = train.loc[approved, "default_observed"].to_numpy()
    X_rejected = train.loc[~approved, columns]

    first = parcelling(model_factory, X_approved, y_approved, X_rejected, random_state=42)
    second = parcelling(model_factory, X_approved, y_approved, X_rejected, random_state=42)
    np.testing.assert_allclose(
        first.predict_proba(X_approved), second.predict_proba(X_approved), rtol=1e-9
    )


def test_heckman_runs_and_produces_valid_probabilities(
    book: pd.DataFrame, features: object
) -> None:
    train, test = _train_and_test(book)
    columns = features.model_features  # type: ignore[attr-defined]
    approved = train["approved"] == 1

    model = heckman_correction(
        train[[*columns, "branch_capacity_index"]],
        approved.to_numpy().astype(int),
        train.loc[approved, "default_observed"].to_numpy(),
        list(features.numeric),  # type: ignore[attr-defined]
        ["branch_capacity_index"],
    )
    predictions = model.predict_proba(test[[*columns, "branch_capacity_index"]])
    assert predictions.shape == (len(test),)
    assert ((predictions > 0) & (predictions < 1)).all()


def test_oracle_beats_the_baseline_on_the_population(
    mnar_book: pd.DataFrame, features: object, model_factory: object
) -> None:
    """The ceiling must actually be above the floor, or validation is broken."""
    train, test = _train_and_test(mnar_book)
    columns = features.model_features  # type: ignore[attr-defined]
    approved = train["approved"] == 1

    baseline = fit_approved_only(
        model_factory, train.loc[approved, columns], train.loc[approved, "default_observed"].to_numpy()
    )
    oracle = oracle_model(model_factory, train[columns], train["default_true"].to_numpy())

    y_true = test["default_true"].to_numpy()
    approved_test = test["approved"].to_numpy()
    baseline_result = evaluate_on_population(
        baseline.predict_proba(test[columns]), y_true, approved_test, "baseline"
    )
    oracle_result = evaluate_on_population(
        oracle.predict_proba(test[columns]), y_true, approved_test, "oracle"
    )
    assert abs(float(oracle_result["bias_reject"])) < abs(float(baseline_result["bias_reject"]))
    assert float(oracle_result["brier_population"]) < float(baseline_result["brier_population"])


def test_true_uplift_is_measurable_and_near_one_under_mar(
    large_book: pd.DataFrame, features: object, model_factory: object
) -> None:
    """Under MAR with a well-trained model, rejects match approves in-band.

    This is why the assumed 2.5x uplift overcorrects: the assumption is wrong
    by roughly a factor of two, in the direction that inflates risk.

    Uses the full-size book deliberately — on a small book the scoring model is
    weak enough to manufacture an apparent uplift above 2x, which is the trap
    documented in the module.
    """
    train, _ = _train_and_test(large_book)
    columns = features.model_features  # type: ignore[attr-defined]
    approved = train["approved"] == 1

    baseline = fit_approved_only(
        model_factory, train.loc[approved, columns], train.loc[approved, "default_observed"].to_numpy()
    )
    uplift = measure_true_uplift(
        baseline.predict_proba(train[columns]),
        train["approved"].to_numpy(),
        train["default_true"].to_numpy(),
    )
    assert not uplift.empty
    weights = uplift["n_rejected"] / uplift["n_rejected"].sum()
    weighted = float((uplift["uplift"] * weights).sum())
    assert weighted < 1.5, f"Expected in-band uplift near 1 under MAR, measured {weighted:.2f}"


def test_study_runs_all_methods(
    book: pd.DataFrame, features: object, model_factory: object
) -> None:
    train, test = _train_and_test(book)
    study = run_reject_inference_study(
        model_factory,
        train,
        test,
        features.model_features,  # type: ignore[attr-defined]
        list(features.numeric),  # type: ignore[attr-defined]
        ["branch_capacity_index"],
    )
    assert set(study["method"]) == {
        "baseline_approved_only",
        "fuzzy_augmentation",
        "parcelling",
        "heckman_two_step",
        "oracle_full_information",
    }
    assert study["bias_reject"].notna().all()
    assert float(study.loc[study["method"] == "baseline_approved_only", "bias_removed_share"].iloc[0]) == 0.0


def test_reject_inference_helps_under_mnar(
    mnar_book: pd.DataFrame, features: object, model_factory: object
) -> None:
    """When selection is on unobservables, label inference genuinely helps."""
    train, test = _train_and_test(mnar_book)
    study = run_reject_inference_study(
        model_factory,
        train,
        test,
        features.model_features,  # type: ignore[attr-defined]
        list(features.numeric),  # type: ignore[attr-defined]
        ["branch_capacity_index"],
    ).set_index("method")

    baseline_bias = abs(float(study.loc["baseline_approved_only", "bias_reject"]))
    for method in ("fuzzy_augmentation", "parcelling"):
        assert abs(float(study.loc[method, "bias_reject"])) < baseline_bias


def test_reject_inference_hurts_under_mar(
    large_book: pd.DataFrame, features: object, model_factory: object
) -> None:
    """The uncomfortable finding, asserted so it cannot be lost silently.

    Under selection on observables the baseline is already nearly unbiased, and
    applying the conventional 2.5x uplift overcorrects into a worse model.
    """
    train, test = _train_and_test(large_book)
    study = run_reject_inference_study(
        model_factory,
        train,
        test,
        features.model_features,  # type: ignore[attr-defined]
        list(features.numeric),  # type: ignore[attr-defined]
        ["branch_capacity_index"],
    ).set_index("method")

    baseline_bias = abs(float(study.loc["baseline_approved_only", "bias_reject"]))
    overcorrected = [
        method
        for method in ("fuzzy_augmentation", "parcelling")
        if abs(float(study.loc[method, "bias_reject"])) > baseline_bias
    ]
    assert overcorrected, "Expected label-inference methods to overcorrect under MAR"


def test_uplift_sensitivity_covers_the_grid(
    book: pd.DataFrame, features: object, model_factory: object
) -> None:
    train, test = _train_and_test(book)
    sensitivity = run_uplift_sensitivity(
        model_factory,
        train,
        test,
        features.model_features,  # type: ignore[attr-defined]
        uplift_values=(1.0, 2.5),
    )
    assert set(sensitivity["uplift"]) == {1.0, 2.5}
    assert set(sensitivity["method"]) == {"fuzzy_augmentation", "parcelling"}
    # A larger uplift must raise predicted reject risk.
    for method in ("fuzzy_augmentation", "parcelling"):
        subset = sensitivity[sensitivity["method"] == method].set_index("uplift")
        assert float(subset.loc[2.5, "bias_reject"]) > float(subset.loc[1.0, "bias_reject"])


def test_weak_models_manufacture_apparent_uplift(
    features: object, model_factory: object
) -> None:
    """A weak scoring model makes MAR selection look like MNAR.

    Pins the trap documented in the module: measured in-band uplift falls as
    the scoring model gets more data, on the *same* selection mechanism. Anyone
    who measures 2x on a real book and concludes they need reject inference may
    simply have an underpowered model.
    """
    from dataclasses import replace

    from src.config import DEFAULT_SIMULATION
    from src.simulate import simulate_loan_book

    columns = features.model_features  # type: ignore[attr-defined]
    measured = {}

    for n in (6_000, 30_000):
        train, _ = _train_and_test(simulate_loan_book(replace(DEFAULT_SIMULATION, n_applicants=n)))
        approved = train["approved"] == 1
        baseline = fit_approved_only(
            model_factory,
            train.loc[approved, columns],
            train.loc[approved, "default_observed"].to_numpy(),
        )
        uplift = measure_true_uplift(
            baseline.predict_proba(train[columns]),
            train["approved"].to_numpy(),
            train["default_true"].to_numpy(),
        )
        weights = uplift["n_rejected"] / uplift["n_rejected"].sum()
        measured[n] = float((uplift["uplift"] * weights).sum())

    assert measured[6_000] > measured[30_000], (
        f"Expected apparent uplift to fall with more training data, "
        f"got {measured[6_000]:.2f} at n=6k and {measured[30_000]:.2f} at n=30k"
    )


def test_bias_removed_share_handles_a_missing_baseline() -> None:
    frame = pd.DataFrame([{"method": "something", "bias_reject": 0.1, "brier_reject": 0.2}])
    assert summarise_bias_removed(frame)["bias_removed_share"].isna().all()


def test_evaluate_on_population_separates_approved_from_rejects(
    book: pd.DataFrame,
) -> None:
    parts = split_by_vintage(book, 20, 24, 24)
    test = parts["test"]
    result = evaluate_on_population(
        test["pd_true"].to_numpy(), test["default_true"].to_numpy(), test["approved"].to_numpy()
    )
    for key in ("bias_approved", "bias_reject", "true_rate_approved", "true_rate_reject"):
        assert key in result
    # The true PD is by construction almost unbiased for both.
    assert abs(float(result["bias_reject"])) < 0.05
