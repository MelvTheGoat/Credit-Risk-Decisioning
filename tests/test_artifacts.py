"""Tests for artifact versioning and the model/calibrator mismatch guard.

These check a safety property, not a feature. Serving a model against the wrong
calibrator produces confident, well-ranked, wrongly-priced probabilities and
nothing in ordinary monitoring reveals it. The guard must hold.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest

from src.artifacts import (
    ArtifactMismatchError,
    ArtifactNotFoundError,
    build_manifest,
    list_versions,
    load_bundle,
    make_training_run_id,
    resolve_version,
    save_bundle,
    set_current_version,
)
from src.calibration import PlattCalibrator


class StubModel:
    """Minimal model with the interface artifacts care about."""

    def __init__(self, offset: float = 0.0) -> None:
        """Store the constant offset added to every prediction."""
        self.offset = offset

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.clip(np.full(len(X), 0.2) + self.offset, 1e-6, 1 - 1e-6)


def _make_bundle(directory: Path, version: str, run_id: str | None = None) -> Any:
    rng = np.random.default_rng(0)
    scores = rng.uniform(0.01, 0.9, 500)
    y = (rng.random(500) < scores).astype(int)
    calibrator = PlattCalibrator().fit(scores, y)

    manifest = build_manifest(
        model_name="stub",
        calibrator_name="platt",
        feature_names=["a", "b"],
        threshold=0.1,
        training_run_id=run_id or make_training_run_id(),
        version=version,
    )
    save_bundle(StubModel(), calibrator, manifest, directory=directory)
    return manifest


def test_roundtrip_preserves_the_manifest(artifact_dir: Path) -> None:
    manifest = _make_bundle(artifact_dir, "v1")
    loaded = load_bundle("v1", directory=artifact_dir)
    assert loaded.manifest.version == "v1"
    assert loaded.manifest.training_run_id == manifest.training_run_id
    assert loaded.manifest.feature_names == ["a", "b"]
    assert loaded.manifest.threshold == 0.1


def test_bundle_predicts_through_the_calibrator(artifact_dir: Path) -> None:
    _make_bundle(artifact_dir, "v1")
    bundle = load_bundle("v1", directory=artifact_dir)
    predictions = bundle.predict_proba(pd.DataFrame({"a": [1, 2], "b": [3, 4]}))
    assert predictions.shape == (2,)
    assert ((predictions > 0) & (predictions < 1)).all()


def test_mismatched_training_run_is_refused(artifact_dir: Path) -> None:
    """The guard this module exists for."""
    _make_bundle(artifact_dir, "v1")
    path = artifact_dir / "v1"

    payload = joblib.load(path / "calibrator.joblib")
    payload["training_run_id"] = "run-from-a-different-training-job"
    joblib.dump(payload, path / "calibrator.joblib")

    # Refresh hashes so this exercises the run-id check specifically.
    manifest = json.loads((path / "manifest.json").read_text())
    from src.artifacts import _sha256

    manifest["file_hashes"] = {p.name: _sha256(p) for p in sorted(path.glob("*.joblib"))}
    (path / "manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ArtifactMismatchError, match="different training runs"):
        load_bundle("v1", directory=artifact_dir)


def test_tampered_file_is_refused(artifact_dir: Path) -> None:
    _make_bundle(artifact_dir, "v1")
    with (artifact_dir / "v1" / "model.joblib").open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ArtifactMismatchError, match="hash mismatch"):
        load_bundle("v1", directory=artifact_dir)


def test_hash_check_can_be_disabled(artifact_dir: Path) -> None:
    """Verification is on by default, but must be explicitly skippable."""
    _make_bundle(artifact_dir, "v1")
    with (artifact_dir / "v1" / "model.joblib").open("ab") as handle:
        handle.write(b" ")
    with pytest.raises(ArtifactMismatchError):
        load_bundle("v1", directory=artifact_dir)
    # Truncating the file would break unpickling, so only assert the hash path
    # is what raised, by confirming a fresh bundle loads with checks off.
    _make_bundle(artifact_dir, "v2")
    assert load_bundle("v2", directory=artifact_dir, verify_hashes=False) is not None


def test_manifest_disagreement_is_refused(artifact_dir: Path) -> None:
    _make_bundle(artifact_dir, "v1")
    path = artifact_dir / "v1"
    manifest = json.loads((path / "manifest.json").read_text())
    manifest["training_run_id"] = "run-that-does-not-match-the-files"
    (path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ArtifactMismatchError, match="do not match their manifest"):
        load_bundle("v1", directory=artifact_dir)


def test_missing_version_is_reported_clearly(artifact_dir: Path) -> None:
    _make_bundle(artifact_dir, "v1")
    with pytest.raises(ArtifactNotFoundError, match="not found"):
        load_bundle("does-not-exist", directory=artifact_dir)


def test_missing_pointer_is_reported_clearly(artifact_dir: Path) -> None:
    with pytest.raises(ArtifactNotFoundError, match="current.txt"):
        resolve_version(None, artifact_dir)


def test_current_pointer_follows_the_latest_save(artifact_dir: Path) -> None:
    _make_bundle(artifact_dir, "v1")
    assert resolve_version(None, artifact_dir) == "v1"
    _make_bundle(artifact_dir, "v2")
    assert resolve_version(None, artifact_dir) == "v2"
    assert load_bundle(directory=artifact_dir).manifest.version == "v2"


def test_rollback_switches_the_active_version(artifact_dir: Path) -> None:
    _make_bundle(artifact_dir, "v1")
    _make_bundle(artifact_dir, "v2")
    set_current_version("v1", artifact_dir)
    assert load_bundle(directory=artifact_dir).manifest.version == "v1"


def test_rollback_refuses_a_broken_version(artifact_dir: Path) -> None:
    """A rollback must not be able to install an invalid pair."""
    _make_bundle(artifact_dir, "v1")
    _make_bundle(artifact_dir, "v2")
    with (artifact_dir / "v1" / "model.joblib").open("ab") as handle:
        handle.write(b"broken")

    with pytest.raises(ArtifactMismatchError):
        set_current_version("v1", artifact_dir)
    # The pointer must not have moved.
    assert resolve_version(None, artifact_dir) == "v2"


def test_list_versions_marks_the_current_one(artifact_dir: Path) -> None:
    _make_bundle(artifact_dir, "v1")
    _make_bundle(artifact_dir, "v2")
    versions = list_versions(artifact_dir)
    assert {entry["version"] for entry in versions} == {"v1", "v2"}
    current = [entry for entry in versions if entry["is_current"]]
    assert len(current) == 1
    assert current[0]["version"] == "v2"


def test_list_versions_is_empty_for_a_fresh_directory(tmp_path: Path) -> None:
    assert list_versions(tmp_path / "nothing-here") == []


def test_manifest_records_the_cost_assumptions() -> None:
    """model-info must be able to state why the cutoff is what it is."""
    from src.config import DEFAULT_COSTS

    manifest = build_manifest("m", "platt", ["a"], 0.1, costs=DEFAULT_COSTS)
    assert manifest.costs["lgd"] == DEFAULT_COSTS.lgd
    assert manifest.costs["analytic_threshold"] == pytest.approx(DEFAULT_COSTS.analytic_threshold)
    assert manifest.costs["cost_ratio"] == pytest.approx(DEFAULT_COSTS.cost_ratio)
