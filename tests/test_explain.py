"""Tests for explanations and adverse action reason codes.

Reason codes go to an applicant in a legal notice, so these check the
properties that matter legally: the reasons are drawn from the model that
actually decided, they are the strongest adverse factors, they are in plain
language, and anything derived from a protected characteristic is flagged
rather than emitted silently.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.explain import (
    PROTECTED_BASIS_FEATURES,
    REASON_TEMPLATES,
    ShapExplainer,
    explain_decision,
    format_adverse_action_notice,
    scorecard_reason_codes,
    shap_reason_codes,
    worked_example,
)
from src.scorecard import Scorecard


@pytest.fixture(scope="module")
def card(approved_splits: dict[str, pd.DataFrame], features: object) -> Scorecard:
    fit = approved_splits["fit"]
    columns = features.model_features  # type: ignore[attr-defined]
    model = Scorecard(list(features.numeric), list(features.categorical))  # type: ignore[attr-defined]
    return model.fit(fit[columns], fit["default_observed"].to_numpy())


@pytest.fixture(scope="module")
def explainer(
    fitted_zoo: dict[str, object], approved_splits: dict[str, pd.DataFrame], features: object
) -> ShapExplainer:
    return ShapExplainer(
        fitted_zoo["lightgbm"], approved_splits["fit"][features.model_features]  # type: ignore[attr-defined]
    )


def test_scorecard_reason_codes_are_ranked_and_capped(
    card: Scorecard, approved_splits: dict[str, pd.DataFrame], features: object
) -> None:
    row = approved_splits["test"].iloc[[0]][features.model_features]  # type: ignore[attr-defined]
    codes = scorecard_reason_codes(card, row, top_k=4)
    assert len(codes) <= 4
    assert [c.rank for c in codes] == list(range(1, len(codes) + 1))
    contributions = [c.contribution for c in codes]
    assert contributions == sorted(contributions, reverse=True)


def test_reason_code_contributions_are_non_negative(
    card: Scorecard, approved_splits: dict[str, pd.DataFrame], features: object
) -> None:
    """A points shortfall is a shortfall; a negative one is a bug."""
    for position in range(20):
        row = approved_splits["test"].iloc[[position]][features.model_features]  # type: ignore[attr-defined]
        for code in scorecard_reason_codes(card, row):
            assert code.contribution >= -1e-9


def test_worst_applicant_gets_the_expected_reasons(card: Scorecard, features: object) -> None:
    """A maxed-out, delinquent applicant should be told exactly that."""
    applicant = pd.DataFrame(
        [
            {
                "income_recorded": 15_000.0,
                "debt_to_income": 1.2,
                "utilisation": 1.4,
                "n_delinq_24m": 6,
                "credit_history_months": 3.0,
                "n_inquiries_6m": 8,
                "employment_years": 0.2,
                "loan_amount": 20_000.0,
                "age": 24.0,
                "housing_status": "rent",
                "product_type": "card",
            }
        ]
    )[features.model_features]  # type: ignore[attr-defined]
    reasons = {code.feature for code in scorecard_reason_codes(card, applicant, top_k=4)}
    assert reasons & {"utilisation", "debt_to_income", "n_delinq_24m", "income_recorded"}


def test_reason_codes_reject_multiple_rows(
    card: Scorecard, approved_splits: dict[str, pd.DataFrame], features: object
) -> None:
    rows = approved_splits["test"].head(2)[features.model_features]  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="one applicant"):
        scorecard_reason_codes(card, rows)


def test_every_reason_has_plain_language(
    card: Scorecard, approved_splits: dict[str, pd.DataFrame], features: object
) -> None:
    """No applicant should receive a raw feature name."""
    for position in range(20):
        row = approved_splits["test"].iloc[[position]][features.model_features]  # type: ignore[attr-defined]
        for code in scorecard_reason_codes(card, row):
            assert len(code.statement) > 20
            assert "_" not in code.statement
            assert code.statement[0].isupper()


def test_all_model_features_have_templates(features: object) -> None:
    for name in features.model_features:  # type: ignore[attr-defined]
        assert name in REASON_TEMPLATES, f"No plain-language template for {name!r}"


def test_protected_features_are_flagged(
    card: Scorecard, approved_splits: dict[str, pd.DataFrame], features: object
) -> None:
    """Age is a model input and a protected basis; it must be visible as one."""
    assert "age" in PROTECTED_BASIS_FEATURES
    for position in range(30):
        row = approved_splits["test"].iloc[[position]][features.model_features]  # type: ignore[attr-defined]
        for code in scorecard_reason_codes(card, row):
            assert code.protected_basis == (code.feature in PROTECTED_BASIS_FEATURES)


def test_reason_codes_are_json_serialisable(
    card: Scorecard, approved_splits: dict[str, pd.DataFrame], features: object
) -> None:
    """They travel through an HTTP response and into the audit log."""
    row = approved_splits["test"].iloc[[0]][features.model_features]  # type: ignore[attr-defined]
    payload = [code.to_dict() for code in scorecard_reason_codes(card, row)]
    json.dumps(payload)


def test_shap_reasons_are_only_risk_increasing(
    explainer: ShapExplainer, approved_splits: dict[str, pd.DataFrame], features: object
) -> None:
    """A factor that helped the applicant is not a reason they were declined."""
    for position in range(10):
        row = approved_splits["test"].iloc[[position]][features.model_features]  # type: ignore[attr-defined]
        for code in shap_reason_codes(explainer, row):
            assert code.contribution > 0


def test_shap_global_importance_is_sensible(
    explainer: ShapExplainer, approved_splits: dict[str, pd.DataFrame], features: object
) -> None:
    """Importance should recover the strongest terms of the known DGP."""
    importance = explainer.global_importance(
        approved_splits["test"].head(400)[features.model_features]  # type: ignore[attr-defined]
    )
    assert (importance["mean_abs_shap"] >= 0).all()
    assert importance["mean_abs_shap"].is_monotonic_decreasing
    top_three = set(importance.head(3)["feature"])
    assert top_three & {"utilisation", "debt_to_income", "income_recorded"}


def test_approved_applicants_get_no_reason_codes(features: object) -> None:
    applicant = pd.DataFrame([dict.fromkeys(features.model_features, 1.0)])  # type: ignore[attr-defined]
    from src.explain import ReasonCode

    codes = [ReasonCode(1, "utilisation", 0.9, 10.0, "High balances", False)]
    record = explain_decision(applicant, probability=0.01, threshold=0.10, reason_codes=codes)
    assert record["decision"] == "approve"
    assert record["reason_codes"] == []


def test_declined_applicants_get_reason_codes(features: object) -> None:
    applicant = pd.DataFrame([dict.fromkeys(features.model_features, 1.0)])  # type: ignore[attr-defined]
    from src.explain import ReasonCode

    codes = [ReasonCode(1, "utilisation", 0.9, 10.0, "High balances", False)]
    record = explain_decision(applicant, probability=0.5, threshold=0.10, reason_codes=codes)
    assert record["decision"] == "decline"
    assert len(record["reason_codes"]) == 1


def test_expected_value_arithmetic(features: object) -> None:
    from src.config import DEFAULT_COSTS

    applicant = pd.DataFrame([dict.fromkeys(features.model_features, 1.0)])  # type: ignore[attr-defined]
    record = explain_decision(
        applicant, probability=0.2, threshold=0.10, reason_codes=[], exposure=10_000.0
    )
    assert record["expected_loss"] == pytest.approx(0.2 * DEFAULT_COSTS.lgd * 10_000.0)
    assert record["expected_margin"] == pytest.approx(0.8 * DEFAULT_COSTS.margin_rate * 10_000.0)
    assert record["expected_value"] == pytest.approx(
        record["expected_margin"] - record["expected_loss"]
    )


def test_notice_is_empty_for_an_approval(features: object) -> None:
    applicant = pd.DataFrame([dict.fromkeys(features.model_features, 1.0)])  # type: ignore[attr-defined]
    record = explain_decision(applicant, probability=0.01, threshold=0.10, reason_codes=[])
    assert "no adverse action notice" in format_adverse_action_notice(record).lower()


def test_notice_marks_a_protected_basis_reason(features: object) -> None:
    from src.explain import ReasonCode

    applicant = pd.DataFrame([dict.fromkeys(features.model_features, 1.0)])  # type: ignore[attr-defined]
    codes = [ReasonCode(1, "age", 30.0, 5.0, "Your age band is associated with higher risk", True)]
    record = explain_decision(applicant, probability=0.5, threshold=0.10, reason_codes=codes)
    assert "PROTECTED BASIS" in format_adverse_action_notice(record)


def test_worked_example_picks_a_marginal_decline(
    approved_splits: dict[str, pd.DataFrame], calibrated_predictions: dict[str, np.ndarray],
    card: Scorecard, features: object, costs: object,
) -> None:
    """The borderline case, because that is the one that must be defensible."""
    test = approved_splits["test"]
    threshold = costs.analytic_threshold  # type: ignore[attr-defined]
    record = worked_example(
        test,
        calibrated_predictions["lightgbm"],
        threshold,
        scorecard=card,
        feature_columns=features.model_features,  # type: ignore[attr-defined]
        model_version="test-v1",
    )
    assert record["decision"] == "decline"
    assert record["probability_of_default"] >= threshold
    # Marginal: within a hair of the cutoff.
    assert record["probability_of_default"] - threshold < 0.01
    assert record["reason_codes"]
    assert record["score_band"] in {"A", "B", "C", "D", "E"}
    assert "NOTICE OF ADVERSE ACTION" in record["adverse_action_notice"]
    json.dumps(record, default=str)
