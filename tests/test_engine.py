"""Tests for the decision engine.

The most important test in this file is
:func:`test_api_and_batch_produce_identical_decisions`. Real-time and batch
scoring that drift apart give the same applicant different answers depending
which door they came through — indefensible to a regulator, and hard to detect
because each path is individually self-consistent. The guarantee is that they
share one implementation, and this asserts it rather than trusting it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.artifacts import build_manifest, load_bundle, save_bundle
from src.calibration import fit_calibrators
from src.config import DEFAULT_COSTS
from src.engine import DecisionEngine, hash_features


@pytest.fixture(scope="module")
def bundle_dir(
    tmp_path_factory: pytest.TempPathFactory,
    fitted_zoo: dict[str, Any],
    approved_splits: dict[str, pd.DataFrame],
    features: Any,
) -> Path:
    """A real, saved bundle built from the fitted model zoo."""
    directory = tmp_path_factory.mktemp("engine_artifacts")
    calibration = approved_splits["calibration"]
    model = fitted_zoo["lightgbm"]
    scorecard = fitted_zoo["scorecard_woe_lr"].scorecard

    calibrator = fit_calibrators(
        model.predict_proba(calibration[features.model_features]),
        calibration["default_observed"].to_numpy(),
    )["platt"]

    manifest = build_manifest(
        model_name="lightgbm",
        calibrator_name="platt",
        feature_names=features.model_features,
        threshold=DEFAULT_COSTS.analytic_threshold,
        costs=DEFAULT_COSTS,
        version="test-v1",
    )
    save_bundle(model, calibrator, manifest, scorecard=scorecard, directory=directory)
    return directory


@pytest.fixture(scope="module")
def engine(bundle_dir: Path) -> DecisionEngine:
    return DecisionEngine(load_bundle(directory=bundle_dir), costs=DEFAULT_COSTS)


def test_engine_reads_the_threshold_from_the_artifact(engine: DecisionEngine) -> None:
    """The cutoff travels with the model, not with the caller's config."""
    assert engine.threshold == pytest.approx(DEFAULT_COSTS.analytic_threshold)
    assert engine.model_version == "test-v1"


def test_decisions_follow_the_threshold(
    engine: DecisionEngine, approved_splits: dict[str, pd.DataFrame], features: Any
) -> None:
    frame = approved_splits["test"].head(200)
    for decision in engine.decide(frame[features.model_features]):
        expected = "approve" if decision.probability_of_default < engine.threshold else "decline"
        assert decision.decision == expected


def test_expected_loss_uses_the_applicant_exposure(
    engine: DecisionEngine, approved_splits: dict[str, pd.DataFrame], features: Any
) -> None:
    """loan_amount must be picked up as EAD, not the flat assumption."""
    frame = approved_splits["test"].head(50)
    decisions = engine.decide(frame[[*features.model_features]])
    for decision, exposure in zip(decisions, frame["loan_amount"], strict=True):
        assert decision.exposure_at_default == pytest.approx(float(exposure))
        assert decision.expected_loss == pytest.approx(
            decision.probability_of_default * DEFAULT_COSTS.lgd * float(exposure)
        )


def test_reason_codes_only_on_adverse_decisions(
    engine: DecisionEngine, approved_splits: dict[str, pd.DataFrame], features: Any
) -> None:
    decisions = engine.decide(approved_splits["test"].head(200)[features.model_features])
    assert any(d.decision == "decline" for d in decisions)
    for decision in decisions:
        if decision.decision == "approve":
            assert decision.reason_codes == []
        else:
            assert decision.reason_codes


def test_missing_features_fail_loudly(
    engine: DecisionEngine, approved_splits: dict[str, pd.DataFrame], features: Any
) -> None:
    frame = approved_splits["test"].head(5)[features.model_features].drop(columns=["utilisation"])
    with pytest.raises(ValueError, match="Missing required features"):
        engine.decide(frame)


def test_input_hash_is_stable_and_sensitive() -> None:
    base = {"a": 1, "b": "x"}
    assert hash_features(base) == hash_features({"b": "x", "a": 1})  # order-independent
    assert hash_features(base) != hash_features({"a": 2, "b": "x"})


def test_scoring_is_deterministic(
    engine: DecisionEngine, approved_splits: dict[str, pd.DataFrame], features: Any
) -> None:
    """The same applicant must always receive the same decision."""
    frame = approved_splits["test"].head(100)[features.model_features]
    first = engine.decide(frame)
    second = engine.decide(frame)
    for a, b in zip(first, second, strict=True):
        assert a.to_dict() == b.to_dict()


def test_api_and_batch_produce_identical_decisions(
    bundle_dir: Path, approved_splits: dict[str, pd.DataFrame], features: Any, monkeypatch: Any
) -> None:
    """The guarantee: one applicant, one answer, whichever door they use.

    The API path scores a single-row frame built from a request payload; the
    batch path scores the same applicant inside a large frame. Any divergence
    at all is a defect.
    """
    frame = approved_splits["test"].head(120)[features.model_features].reset_index(drop=True)
    engine = DecisionEngine(load_bundle(directory=bundle_dir), costs=DEFAULT_COSTS)

    # Batch: everything at once.
    batch = engine.decide(frame)

    # API: one row at a time, rebuilt from a dict exactly as the endpoint does.
    single = []
    for position in range(len(frame)):
        payload = frame.iloc[position].to_dict()
        single.extend(engine.decide(pd.DataFrame([payload])))

    assert len(batch) == len(single)
    for batched, individual in zip(batch, single, strict=True):
        assert batched.decision == individual.decision
        assert batched.probability_of_default == pytest.approx(
            individual.probability_of_default, abs=1e-12
        )
        assert batched.score == pytest.approx(individual.score, abs=1e-9)
        assert batched.score_band == individual.score_band
        assert batched.input_hash == individual.input_hash
        assert batched.reason_codes == individual.reason_codes


def test_decide_frame_returns_aligned_rows(
    engine: DecisionEngine, approved_splits: dict[str, pd.DataFrame], features: Any
) -> None:
    frame = approved_splits["test"].head(40)[features.model_features]
    scored = engine.decide_frame(frame)
    assert len(scored) == len(frame)
    assert list(scored.index) == list(frame.index)
    for column in ("decision", "probability_of_default", "score_band", "expected_loss"):
        assert column in scored.columns


def test_referral_band_is_applied(
    bundle_dir: Path, approved_splits: dict[str, pd.DataFrame], features: Any
) -> None:
    engine = DecisionEngine(
        load_bundle(directory=bundle_dir), costs=DEFAULT_COSTS, referral_width=0.02
    )
    decisions = engine.decide(approved_splits["test"].head(400)[features.model_features])
    assert any(d.decision == "refer" for d in decisions)
    for decision in decisions:
        if decision.decision == "refer":
            assert abs(decision.probability_of_default - engine.threshold) <= 0.02


def test_threshold_override_changes_the_approval_rate(
    bundle_dir: Path, approved_splits: dict[str, pd.DataFrame], features: Any
) -> None:
    frame = approved_splits["test"].head(300)[features.model_features]
    tight = DecisionEngine(load_bundle(directory=bundle_dir), threshold=0.02)
    loose = DecisionEngine(load_bundle(directory=bundle_dir), threshold=0.40)

    tight_rate = np.mean([d.decision == "approve" for d in tight.decide(frame)])
    loose_rate = np.mean([d.decision == "approve" for d in loose.decide(frame)])
    assert tight_rate < loose_rate


def test_decisions_are_json_serialisable(
    engine: DecisionEngine, approved_splits: dict[str, pd.DataFrame], features: Any
) -> None:
    import json

    for decision in engine.decide(approved_splits["test"].head(30)[features.model_features]):
        json.dumps(decision.to_dict())
