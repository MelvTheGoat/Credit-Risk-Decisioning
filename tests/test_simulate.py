"""Tests for the synthetic generator.

The generator is the measuring instrument for everything else in this
repository. If the injected bias is not actually there, or the ground truth is
not retained, then every claim the fairness and reject inference modules make
about "detecting" and "correcting" is measuring nothing. These tests calibrate
the instrument.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.config import DEFAULT_SIMULATION
from src.simulate import (
    TRUE_COEFFICIENTS,
    describe_simulation,
    observed_training_frame,
    simulate_loan_book,
    split_by_vintage,
)


def test_book_is_reproducible() -> None:
    config = replace(DEFAULT_SIMULATION, n_applicants=2_000)
    first = simulate_loan_book(config)
    second = simulate_loan_book(config)
    pd.testing.assert_frame_equal(first, second)


def test_different_seeds_give_different_books() -> None:
    base = replace(DEFAULT_SIMULATION, n_applicants=2_000)
    first = simulate_loan_book(base)
    second = simulate_loan_book(replace(base, seed=base.seed + 1))
    assert not first["default_true"].equals(second["default_true"])


def test_ground_truth_exists_for_everyone(book: pd.DataFrame) -> None:
    """The whole point: outcomes for declined applicants are retained."""
    assert book["default_true"].notna().all()
    assert book["pd_true"].notna().all()
    declined = book[book["approved"] == 0]
    assert len(declined) > 0
    assert declined["default_true"].notna().all()


def test_observed_outcome_is_censored_exactly_as_a_real_book(book: pd.DataFrame) -> None:
    declined = book["approved"] == 0
    assert book.loc[declined, "default_observed"].isna().all()
    assert book.loc[~declined, "default_observed"].notna().all()
    # Where observed, it must equal the truth. The censoring hides outcomes,
    # it does not corrupt them.
    approved = book[~declined]
    assert (approved["default_observed"].to_numpy() == approved["default_true"].to_numpy()).all()


def test_approval_rate_hits_its_target(book: pd.DataFrame) -> None:
    assert book["approved"].mean() == pytest.approx(DEFAULT_SIMULATION.target_approval_rate, abs=0.01)


def test_population_default_rate_hits_its_target(book: pd.DataFrame) -> None:
    assert book["default_true"].mean() == pytest.approx(
        DEFAULT_SIMULATION.target_population_default_rate, abs=0.02
    )


def test_selection_makes_the_approved_book_look_safer(book: pd.DataFrame) -> None:
    """The legacy policy works, so approved applicants really are better."""
    summary = describe_simulation(book).iloc[0]
    assert summary["default_rate_approved"] < summary["default_rate_declined"]
    assert summary["selection_gap"] > 0.10


def test_income_measurement_bias_is_injected_exactly(book: pd.DataFrame) -> None:
    """The known fairness violation is present at the configured magnitude."""
    group_a = book[book["group"] == "A"]
    group_b = book[book["group"] == "B"]

    # Group A's recorded income is truthful.
    np.testing.assert_allclose(
        group_a["income_recorded"].to_numpy(), group_a["income_true"].to_numpy(), rtol=1e-6
    )
    # Group B's is understated by exactly the configured factor.
    ratio = group_b["income_recorded"].to_numpy() / group_b["income_true"].to_numpy()
    np.testing.assert_allclose(ratio, DEFAULT_SIMULATION.group_b_income_bias, rtol=1e-6)


def test_policy_discriminates_against_group_b(book: pd.DataFrame) -> None:
    """The injected historical discrimination is present and material."""
    rates = book.groupby("group", observed=True)["approved"].mean()
    assert rates["B"] < rates["A"]
    assert rates["A"] - rates["B"] > 0.10


def test_group_is_not_a_direct_cause_of_default() -> None:
    """No group term in the DGP: every risk gap flows through features."""
    assert not hasattr(TRUE_COEFFICIENTS, "group")
    fields = set(TRUE_COEFFICIENTS.__dataclass_fields__)
    assert not any("group" in name for name in fields)


def test_drift_is_injected_in_later_vintages(book: pd.DataFrame) -> None:
    early = book[book["vintage"] < DEFAULT_SIMULATION.drift_start_vintage]
    late = book[book["vintage"] >= DEFAULT_SIMULATION.drift_start_vintage]
    assert late["default_true"].mean() > early["default_true"].mean()
    assert late["utilisation"].mean() > early["utilisation"].mean()
    assert late["credit_history_months"].mean() < early["credit_history_months"].mean()


def test_default_timing_is_within_the_performance_window(book: pd.DataFrame) -> None:
    defaulters = book[book["default_true"] == 1]
    assert defaulters["months_to_default"].between(1, DEFAULT_SIMULATION.performance_window_months).all()
    # Non-defaulters carry 0, so vintage curves can filter on `> 0`.
    assert (book.loc[book["default_true"] == 0, "months_to_default"] == 0).all()


def test_mar_book_has_no_private_signal_effect(book: pd.DataFrame) -> None:
    """Under MAR the latent officer signal must not touch the outcome."""
    assert DEFAULT_SIMULATION.private_signal_default == 0.0
    correlation = np.corrcoef(book["officer_signal"], book["pd_true"])[0, 1]
    assert abs(correlation) < 0.05


def test_mnar_book_has_a_real_private_signal(mnar_book: pd.DataFrame) -> None:
    """Under MNAR the signal drives both approval and default."""
    assert abs(np.corrcoef(mnar_book["officer_signal"], mnar_book["pd_true"])[0, 1]) > 0.10
    assert abs(np.corrcoef(mnar_book["officer_signal"], mnar_book["approved"])[0, 1]) > 0.10


def test_mnar_selection_gap_exceeds_mar(book: pd.DataFrame, mnar_book: pd.DataFrame) -> None:
    """Selecting on unobservables makes the approved sample less representative."""
    mar_gap = float(describe_simulation(book).iloc[0]["selection_gap"])
    mnar_gap = float(describe_simulation(mnar_book).iloc[0]["selection_gap"])
    assert mnar_gap > mar_gap


def test_exclusion_restriction_is_valid(book: pd.DataFrame) -> None:
    """The instrument must affect approval but not default."""
    assert abs(np.corrcoef(book["branch_capacity_index"], book["approved"])[0, 1]) > 0.05
    assert abs(np.corrcoef(book["branch_capacity_index"], book["pd_true"])[0, 1]) < 0.05


def test_split_is_disjoint_and_ordered_in_time(book: pd.DataFrame) -> None:
    parts = split_by_vintage(book, 20, 24, 24)
    assert parts["fit"]["vintage"].max() < parts["calibration"]["vintage"].min()
    assert parts["calibration"]["vintage"].max() < parts["test"]["vintage"].min()
    assert len(parts["fit"]) + len(parts["calibration"]) + len(parts["test"]) == len(book)
    assert set(parts["fit"]["application_id"]).isdisjoint(parts["test"]["application_id"])


def test_observed_training_frame_keeps_only_approved(book: pd.DataFrame) -> None:
    frame = observed_training_frame(book)
    assert (frame["approved"] == 1).all()
    assert frame["default_observed"].dtype.kind in "iu"


def test_protected_attributes_are_not_model_features(features: object) -> None:
    """Every model is group-blind by construction; the audit still finds gaps."""
    model_features = set(features.model_features)  # type: ignore[attr-defined]
    for attribute in features.protected:  # type: ignore[attr-defined]
        assert attribute not in model_features


def test_oracle_columns_are_not_model_features(features: object) -> None:
    """No model may see the answer key."""
    model_features = set(features.model_features)  # type: ignore[attr-defined]
    for column in ("pd_true", "default_true", "income_true", "officer_signal", "legacy_score"):
        assert column not in model_features
