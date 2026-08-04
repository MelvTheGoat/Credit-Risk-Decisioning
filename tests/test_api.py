"""Tests for the decision service.

Covers the endpoint contracts, input validation, the audit trail, and — most
importantly — that the service refuses to start on a mismatched model and
calibrator pair.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.audit import AuditLog
from src.artifacts import ArtifactMismatchError, _sha256, build_manifest, save_bundle
from src.calibration import fit_calibrators
from src.config import DEFAULT_COSTS

RISKY = {
    "income_recorded": 21_000.0,
    "debt_to_income": 0.85,
    "utilisation": 1.05,
    "n_delinq_24m": 4,
    "credit_history_months": 6.0,
    "n_inquiries_6m": 5,
    "employment_years": 0.4,
    "loan_amount": 15_000.0,
    "age": 26.0,
    "housing_status": "rent",
    "product_type": "card",
}

STRONG = {
    "income_recorded": 105_000.0,
    "debt_to_income": 0.10,
    "utilisation": 0.05,
    "n_delinq_24m": 0,
    "credit_history_months": 210.0,
    "n_inquiries_6m": 0,
    "employment_years": 16.0,
    "loan_amount": 8_000.0,
    "age": 49.0,
    "housing_status": "own",
    "product_type": "auto",
}


@pytest.fixture(scope="module")
def service_artifacts(
    tmp_path_factory: pytest.TempPathFactory,
    fitted_zoo: dict[str, Any],
    approved_splits: dict[str, pd.DataFrame],
    features: Any,
) -> Path:
    """A saved bundle plus reference scores, in an isolated directory."""
    directory = tmp_path_factory.mktemp("api_artifacts")
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
        calibration_metrics={"brier": 0.11, "ece_quantile": 0.03},
        performance_metrics={"roc_auc": 0.82},
        training_rows=len(approved_splits["fit"]),
        training_period="vintages 0-19",
        version="api-test-v1",
    )
    version_directory = save_bundle(
        model, calibrator, manifest, scorecard=scorecard, directory=directory
    )
    reference = model.predict_proba(approved_splits["fit"][features.model_features])
    np.save(version_directory / "reference_scores.npy", reference)
    return directory


@pytest.fixture
def client(service_artifacts: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """A running service backed by an isolated artifact dir and audit log."""
    import src.api.main as api_main
    import src.artifacts as artifacts_module

    monkeypatch.setattr(artifacts_module, "ARTIFACT_DIR", service_artifacts)
    api_main.state.audit = AuditLog(tmp_path / "decisions.jsonl")
    api_main.state.reference_scores = None

    with TestClient(api_main.app) as test_client:
        yield test_client


def test_health_reports_the_loaded_model(client: TestClient) -> None:
    payload = client.get("/health").json()
    assert payload["status"] == "ok"
    assert payload["model_loaded"] is True
    assert payload["model_version"] == "api-test-v1"
    assert payload["uptime_seconds"] >= 0


def test_model_info_exposes_the_live_assumptions(client: TestClient) -> None:
    """The endpoint must state the cutoff and why it is what it is."""
    payload = client.get("/model-info").json()
    assert payload["model_version"] == "api-test-v1"
    assert payload["calibrator_name"] == "platt"
    assert payload["threshold"] == pytest.approx(DEFAULT_COSTS.analytic_threshold)
    assert payload["costs"]["lgd"] == DEFAULT_COSTS.lgd
    assert payload["costs"]["cost_ratio"] == pytest.approx(DEFAULT_COSTS.cost_ratio)
    assert payload["calibration_metrics"]["brier"] == pytest.approx(0.11)
    assert payload["training_period"] == "vintages 0-19"


def test_risky_applicant_is_declined_with_reasons(client: TestClient) -> None:
    payload = client.post(
        "/decision", json={"application_id": "APP-RISKY", "features": RISKY}
    ).json()
    assert payload["decision"] == "decline"
    assert payload["probability_of_default"] > payload["threshold"]
    assert payload["expected_value"] < 0
    assert 1 <= len(payload["reason_codes"]) <= 4
    assert payload["reason_codes"][0]["rank"] == 1
    assert len(payload["reason_codes"][0]["statement"]) > 20
    assert payload["score_band"] in {"A", "B", "C", "D", "E"}
    assert payload["model_version"] == "api-test-v1"


def test_strong_applicant_is_approved_without_reasons(client: TestClient) -> None:
    payload = client.post(
        "/decision", json={"application_id": "APP-STRONG", "features": STRONG}
    ).json()
    assert payload["decision"] == "approve"
    assert payload["reason_codes"] == []
    assert payload["expected_value"] > 0


def test_expected_loss_uses_the_requested_exposure(client: TestClient) -> None:
    payload = client.post(
        "/decision",
        json={"application_id": "APP-EAD", "features": STRONG, "exposure_at_default": 50_000.0},
    ).json()
    assert payload["expected_loss"] == pytest.approx(
        payload["probability_of_default"] * DEFAULT_COSTS.lgd * 50_000.0
    )


def test_identical_requests_give_identical_decisions(client: TestClient) -> None:
    request = {"application_id": "APP-SAME", "features": RISKY}
    first = client.post("/decision", json=request).json()
    second = client.post("/decision", json=request).json()
    assert first["probability_of_default"] == second["probability_of_default"]
    assert first["input_hash"] == second["input_hash"]
    assert first["decision"] == second["decision"]


@pytest.mark.parametrize(
    "override",
    [
        {"age": 5},                    # below the minimum
        {"age": 200},                  # above the maximum
        {"utilisation": -0.5},         # negative
        {"income_recorded": 0},        # non-positive
        {"n_delinq_24m": -1},          # negative count
        {"loan_amount": -100},         # negative
    ],
)
def test_out_of_range_inputs_are_rejected(client: TestClient, override: dict[str, Any]) -> None:
    """Coercing a bad value would decline someone for a caller's typo."""
    response = client.post(
        "/decision", json={"application_id": "APP-BAD", "features": {**RISKY, **override}}
    )
    assert response.status_code == 422


def test_unknown_fields_are_rejected(client: TestClient) -> None:
    response = client.post(
        "/decision",
        json={"application_id": "APP-X", "features": {**RISKY, "secret_feature": 1.0}},
    )
    assert response.status_code == 422


def test_missing_features_are_rejected(client: TestClient) -> None:
    partial = {k: v for k, v in RISKY.items() if k != "utilisation"}
    response = client.post("/decision", json={"application_id": "APP-Y", "features": partial})
    assert response.status_code == 422


def test_every_decision_is_written_to_the_audit_log(client: TestClient) -> None:
    """In credit this is not optional."""
    import src.api.main as api_main

    client.post("/decision", json={"application_id": "APP-AUDIT-1", "features": RISKY})
    client.post("/decision", json={"application_id": "APP-AUDIT-2", "features": STRONG})

    entries = api_main.state.audit.find(application_id="APP-AUDIT-1")
    assert len(entries) == 1
    entry = entries[0]
    for field in (
        "timestamp",
        "input_hash",
        "model_version",
        "threshold",
        "probability_of_default",
        "score",
        "decision",
        "reason_codes",
    ):
        assert field in entry, f"audit entry is missing {field!r}"
    assert entry["channel"] == "api"
    assert entry["model_version"] == "api-test-v1"


def test_a_declined_applicant_can_be_reconstructed_months_later(client: TestClient) -> None:
    """The retrieval path a complaint investigation uses."""
    import src.api.main as api_main

    response = client.post(
        "/decision", json={"application_id": "APP-COMPLAINT", "features": RISKY}
    ).json()

    recovered = api_main.state.audit.find(application_id="APP-COMPLAINT", decision="decline")
    assert len(recovered) == 1
    assert recovered[0]["input_hash"] == response["input_hash"]
    assert recovered[0]["threshold"] == response["threshold"]
    assert [r["statement"] for r in recovered[0]["reason_codes"]] == [
        r["statement"] for r in response["reason_codes"]
    ]


def test_audit_log_is_append_only(client: TestClient, tmp_path: Path) -> None:
    import src.api.main as api_main

    for i in range(5):
        client.post("/decision", json={"application_id": f"APP-{i}", "features": RISKY})
    lines = api_main.state.audit.path.read_text().strip().split("\n")
    assert len(lines) >= 5
    for line in lines:
        json.loads(line)  # every line is a complete record


def test_metrics_reports_decision_counts(client: TestClient) -> None:
    client.post("/decision", json={"application_id": "M1", "features": RISKY})
    client.post("/decision", json={"application_id": "M2", "features": STRONG})
    payload = client.get("/metrics").json()
    assert payload["n_decisions"] >= 2
    assert payload["by_decision"]["decline"] >= 1
    assert payload["by_decision"]["approve"] >= 1
    assert 0.0 <= payload["approval_rate"] <= 1.0
    assert payload["by_model_version"]["api-test-v1"] >= 2


def test_metrics_reports_psi_when_a_reference_exists(
    client: TestClient, service_artifacts: Path
) -> None:
    import src.api.main as api_main

    api_main.state.reference_scores = np.load(
        service_artifacts / "api-test-v1" / "reference_scores.npy"
    )
    for i in range(30):
        features = RISKY if i % 2 else STRONG
        client.post("/decision", json={"application_id": f"P{i}", "features": features})

    payload = client.get("/metrics").json()
    assert payload["reference_available"] is True
    assert payload["score_psi"] is not None
    assert payload["psi_status"] in {"stable", "watch", "review"}
    assert len(payload["score_distribution"]) > 0


def test_metrics_says_so_when_there_is_no_reference(
    service_artifacts: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A freshly deployed model has no reference yet; say so, do not guess.

    Reporting a PSI of 0.0 when there is nothing to compare against would read
    as "no drift" and is exactly the wrong thing to tell an operator.
    """
    import shutil

    import src.api.main as api_main
    import src.artifacts as artifacts_module

    fresh_root = tmp_path / "no_reference"
    shutil.copytree(service_artifacts, fresh_root)
    (fresh_root / "api-test-v1" / "reference_scores.npy").unlink()

    monkeypatch.setattr(artifacts_module, "ARTIFACT_DIR", fresh_root)
    api_main.state.audit = AuditLog(tmp_path / "fresh.jsonl")

    with TestClient(api_main.app) as fresh_client:
        fresh_client.post("/decision", json={"application_id": "N1", "features": RISKY})
        payload = fresh_client.get("/metrics").json()

    assert payload["reference_available"] is False
    assert payload["psi_status"] == "no_reference"
    assert payload["score_psi"] is None


def test_service_refuses_to_start_on_a_version_mismatch(
    service_artifacts: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The safety property: down beats quietly wrong.

    A service returning miscalibrated probabilities looks healthy, ranks
    correctly, and misprices every decision. Refusing to start is the cheaper
    failure.
    """
    import shutil

    import src.api.main as api_main
    import src.artifacts as artifacts_module

    broken_root = tmp_path / "broken"
    shutil.copytree(service_artifacts, broken_root)
    version_directory = broken_root / "api-test-v1"

    payload = joblib.load(version_directory / "calibrator.joblib")
    payload["training_run_id"] = "run-from-another-job"
    joblib.dump(payload, version_directory / "calibrator.joblib")

    manifest = json.loads((version_directory / "manifest.json").read_text())
    manifest["file_hashes"] = {
        p.name: _sha256(p) for p in sorted(version_directory.glob("*.joblib"))
    }
    (version_directory / "manifest.json").write_text(json.dumps(manifest))

    monkeypatch.setattr(artifacts_module, "ARTIFACT_DIR", broken_root)
    api_main.state.audit = AuditLog(tmp_path / "broken.jsonl")

    with pytest.raises(ArtifactMismatchError, match="different training runs"), TestClient(
        api_main.app
    ):
        pass
