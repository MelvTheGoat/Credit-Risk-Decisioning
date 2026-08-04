"""Versioned model artifacts, and the checks that stop the wrong pair being served.

The failure this module exists to prevent
-----------------------------------------
A model and its calibrator are two files. They are trained together, and they
are only valid together. Deploy version 4 of the model against version 3 of the
calibrator and nothing crashes: the service starts, scores every applicant,
returns confident probabilities, and every one of them is wrong. Ranking looks
fine. AUC in the monitoring pack looks fine. Expected loss, provisioning and
the cutoff are all silently wrong, and the first hard evidence arrives a year
later in the vintage curves.

So the model and calibrator are stored as **separate files**, because that is
the real deployment topology and therefore the real failure mode, and each
carries the ``training_run_id`` of the run that produced it. Loading a bundle
verifies they match and refuses otherwise. The API calls that check at startup
and will not serve if it fails — a service that is down is a visible incident,
while a service quietly returning miscalibrated probabilities is not.

Content hashes are recorded too, so a file that has been swapped or truncated
on disk is caught rather than assumed good.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib

from src.config import ARTIFACT_DIR, DEFAULT_COSTS, CostParameters

logger = logging.getLogger(__name__)

MODEL_FILENAME = "model.joblib"
CALIBRATOR_FILENAME = "calibrator.joblib"
SCORECARD_FILENAME = "scorecard.joblib"
MANIFEST_FILENAME = "manifest.json"
CURRENT_POINTER = "current.txt"


class ArtifactMismatchError(RuntimeError):
    """Raised when a model and calibrator do not come from the same training run."""


class ArtifactNotFoundError(RuntimeError):
    """Raised when a requested artifact version does not exist on disk."""


@dataclass
class BundleManifest:
    """Everything needed to identify, validate and explain a deployed model.

    Attributes:
        version: Human-readable artifact version, e.g. ``"2026-08-04T1200Z"``.
        training_run_id: Identifier shared by every file from one training run.
            The value the mismatch check compares.
        created_at: UTC ISO timestamp of artifact creation.
        model_name: Which model family is deployed.
        calibrator_name: Which calibration strategy is deployed.
        feature_names: Ordered model input columns.
        threshold: PD cutoff in force.
        costs: Cost assumptions the threshold was derived from.
        calibration_metrics: Calibration metrics measured at build time.
        performance_metrics: Discrimination metrics measured at build time.
        training_rows: Number of rows the model was fitted on.
        training_period: Description of the training window.
        file_hashes: SHA-256 of each artifact file.
        notes: Free-text provenance.
    """

    version: str
    training_run_id: str
    created_at: str
    model_name: str
    calibrator_name: str
    feature_names: list[str]
    threshold: float
    costs: dict[str, float]
    calibration_metrics: dict[str, float] = field(default_factory=dict)
    performance_metrics: dict[str, float] = field(default_factory=dict)
    training_rows: int = 0
    training_period: str = ""
    file_hashes: dict[str, str] = field(default_factory=dict)
    notes: str = ""

    def to_json(self) -> str:
        """Serialise the manifest as indented JSON."""
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> BundleManifest:
        """Rebuild a manifest from a parsed JSON payload.

        Args:
            payload: Parsed manifest dictionary.

        Returns:
            The manifest.
        """
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in payload.items() if k in known})


def _sha256(path: Path) -> str:
    """SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def make_training_run_id() -> str:
    """Generate a training run identifier from the current UTC time."""
    return datetime.now(UTC).strftime("run-%Y%m%dT%H%M%S%fZ")


def make_version() -> str:
    """Generate an artifact version string from the current UTC time."""
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


@dataclass
class ModelBundle:
    """A model, its calibrator, and the manifest binding them together.

    Attributes:
        model: Fitted model exposing ``predict_proba``.
        calibrator: Fitted calibrator exposing ``transform``.
        manifest: Identifying and validating metadata.
        scorecard: Optional fitted scorecard, retained for points-shortfall
            reason codes and a points score even when it is not the deciding
            model.
    """

    model: Any
    calibrator: Any
    manifest: BundleManifest
    scorecard: Any | None = None

    def predict_proba(self, X: Any) -> Any:
        """Return calibrated P(default)."""
        return self.calibrator.transform(self.model.predict_proba(X))


def save_bundle(
    model: Any,
    calibrator: Any,
    manifest: BundleManifest,
    scorecard: Any | None = None,
    directory: Path = ARTIFACT_DIR,
    set_current: bool = True,
) -> Path:
    """Write a bundle to a versioned directory.

    Args:
        model: Fitted model.
        calibrator: Fitted calibrator.
        manifest: Manifest describing the bundle.
        scorecard: Optional fitted scorecard.
        directory: Root artifact directory.
        set_current: Update the ``current.txt`` pointer to this version.

    Returns:
        Path to the version directory.
    """
    version_directory = directory / manifest.version
    version_directory.mkdir(parents=True, exist_ok=True)

    # The training_run_id travels *inside* each file as well as in the
    # manifest, so a file moved between version directories still carries the
    # provenance that the mismatch check reads.
    joblib.dump({"training_run_id": manifest.training_run_id, "object": model}, version_directory / MODEL_FILENAME)
    joblib.dump(
        {"training_run_id": manifest.training_run_id, "object": calibrator},
        version_directory / CALIBRATOR_FILENAME,
    )
    if scorecard is not None:
        joblib.dump(
            {"training_run_id": manifest.training_run_id, "object": scorecard},
            version_directory / SCORECARD_FILENAME,
        )

    manifest.file_hashes = {
        path.name: _sha256(path) for path in sorted(version_directory.glob("*.joblib"))
    }
    (version_directory / MANIFEST_FILENAME).write_text(manifest.to_json())

    if set_current:
        (directory / CURRENT_POINTER).write_text(manifest.version)

    logger.info("Saved bundle %s to %s", manifest.version, version_directory)
    return version_directory


def resolve_version(version: str | None = None, directory: Path = ARTIFACT_DIR) -> str:
    """Resolve a version string, defaulting to the current pointer.

    Args:
        version: Explicit version, or ``None`` to read ``current.txt``.
        directory: Root artifact directory.

    Returns:
        The resolved version string.

    Raises:
        ArtifactNotFoundError: If no version is given and no pointer exists.
    """
    if version:
        return version
    pointer = directory / CURRENT_POINTER
    if not pointer.exists():
        raise ArtifactNotFoundError(
            f"No version requested and no {CURRENT_POINTER} in {directory}. "
            "Train a model first with: python -m scripts.run_experiments"
        )
    return pointer.read_text().strip()


def load_bundle(
    version: str | None = None,
    directory: Path = ARTIFACT_DIR,
    verify_hashes: bool = True,
) -> ModelBundle:
    """Load a bundle, refusing to return a mismatched or corrupted pair.

    Args:
        version: Version to load, or ``None`` for the current pointer.
        directory: Root artifact directory.
        verify_hashes: Check file contents against the recorded hashes.

    Returns:
        The loaded bundle.

    Raises:
        ArtifactNotFoundError: If the version directory or its files are absent.
        ArtifactMismatchError: If the model and calibrator carry different
            training run identifiers, if either disagrees with the manifest, or
            if a file hash does not match.
    """
    resolved = resolve_version(version, directory)
    version_directory = directory / resolved

    if not version_directory.exists():
        available = sorted(p.name for p in directory.glob("*") if p.is_dir())
        raise ArtifactNotFoundError(
            f"Artifact version {resolved!r} not found in {directory}. Available: {available}"
        )

    manifest_path = version_directory / MANIFEST_FILENAME
    if not manifest_path.exists():
        raise ArtifactNotFoundError(f"No manifest at {manifest_path}")
    manifest = BundleManifest.from_dict(json.loads(manifest_path.read_text()))

    for filename in (MODEL_FILENAME, CALIBRATOR_FILENAME):
        if not (version_directory / filename).exists():
            raise ArtifactNotFoundError(f"Missing {filename} in {version_directory}")

    if verify_hashes:
        for filename, expected in manifest.file_hashes.items():
            path = version_directory / filename
            if not path.exists():
                raise ArtifactMismatchError(f"Manifest lists {filename} but it is not present")
            actual = _sha256(path)
            if actual != expected:
                raise ArtifactMismatchError(
                    f"Content hash mismatch for {filename}: manifest {expected}, on disk {actual}. "
                    "The artifact has been modified or replaced since it was built."
                )

    model_payload = joblib.load(version_directory / MODEL_FILENAME)
    calibrator_payload = joblib.load(version_directory / CALIBRATOR_FILENAME)

    model_run = model_payload.get("training_run_id")
    calibrator_run = calibrator_payload.get("training_run_id")

    # The check this module exists for.
    if model_run != calibrator_run:
        raise ArtifactMismatchError(
            f"Model and calibrator are from different training runs: "
            f"model={model_run!r}, calibrator={calibrator_run!r}. "
            "Serving this pair would produce well-ranked, wrongly-priced decisions. "
            "Rebuild the bundle or roll back to a consistent version."
        )
    if model_run != manifest.training_run_id:
        raise ArtifactMismatchError(
            f"Artifacts do not match their manifest: files declare {model_run!r}, "
            f"manifest declares {manifest.training_run_id!r}."
        )

    scorecard = None
    scorecard_path = version_directory / SCORECARD_FILENAME
    if scorecard_path.exists():
        scorecard_payload = joblib.load(scorecard_path)
        if scorecard_payload.get("training_run_id") != manifest.training_run_id:
            raise ArtifactMismatchError(
                f"Scorecard is from a different training run: "
                f"{scorecard_payload.get('training_run_id')!r} vs {manifest.training_run_id!r}"
            )
        scorecard = scorecard_payload["object"]

    logger.info(
        "Loaded bundle %s (model=%s, calibrator=%s, run=%s)",
        manifest.version,
        manifest.model_name,
        manifest.calibrator_name,
        manifest.training_run_id,
    )
    return ModelBundle(
        model=model_payload["object"],
        calibrator=calibrator_payload["object"],
        manifest=manifest,
        scorecard=scorecard,
    )


def list_versions(directory: Path = ARTIFACT_DIR) -> list[dict[str, Any]]:
    """List available artifact versions, newest first.

    Args:
        directory: Root artifact directory.

    Returns:
        One entry per version with its identifying metadata and whether it is
        the current pointer target.
    """
    if not directory.exists():
        return []

    try:
        current = resolve_version(None, directory)
    except ArtifactNotFoundError:
        current = ""

    entries = []
    for path in sorted(directory.glob("*"), reverse=True):
        manifest_path = path / MANIFEST_FILENAME
        if not path.is_dir() or not manifest_path.exists():
            continue
        payload = json.loads(manifest_path.read_text())
        entries.append(
            {
                "version": payload.get("version", path.name),
                "created_at": payload.get("created_at", ""),
                "model_name": payload.get("model_name", ""),
                "calibrator_name": payload.get("calibrator_name", ""),
                "threshold": payload.get("threshold"),
                "training_run_id": payload.get("training_run_id", ""),
                "is_current": payload.get("version", path.name) == current,
            }
        )
    return entries


def set_current_version(version: str, directory: Path = ARTIFACT_DIR) -> None:
    """Point ``current.txt`` at a version, after checking it loads cleanly.

    This is the rollback mechanism. The bundle is loaded and validated before
    the pointer moves, so a rollback cannot install a broken or mismatched pair.

    Args:
        version: Version to activate.
        directory: Root artifact directory.

    Raises:
        ArtifactNotFoundError: If the version does not exist.
        ArtifactMismatchError: If the version fails validation.
    """
    load_bundle(version, directory)
    (directory / CURRENT_POINTER).write_text(version)
    logger.info("Current version set to %s", version)


def build_manifest(
    model_name: str,
    calibrator_name: str,
    feature_names: list[str],
    threshold: float,
    costs: CostParameters = DEFAULT_COSTS,
    calibration_metrics: dict[str, float] | None = None,
    performance_metrics: dict[str, float] | None = None,
    training_rows: int = 0,
    training_period: str = "",
    notes: str = "",
    training_run_id: str | None = None,
    version: str | None = None,
) -> BundleManifest:
    """Construct a manifest for a newly trained bundle.

    Args:
        model_name: Model family identifier.
        calibrator_name: Calibration strategy identifier.
        feature_names: Ordered model input columns.
        threshold: PD cutoff in force.
        costs: Cost assumptions behind the threshold.
        calibration_metrics: Calibration metrics at build time.
        performance_metrics: Discrimination metrics at build time.
        training_rows: Rows the model was fitted on.
        training_period: Description of the training window.
        notes: Free-text provenance.
        training_run_id: Explicit run identifier, generated if omitted.
        version: Explicit version, generated if omitted.

    Returns:
        The manifest.
    """
    return BundleManifest(
        version=version or make_version(),
        training_run_id=training_run_id or make_training_run_id(),
        created_at=datetime.now(UTC).isoformat(),
        model_name=model_name,
        calibrator_name=calibrator_name,
        feature_names=list(feature_names),
        threshold=float(threshold),
        costs={
            "lgd": costs.lgd,
            "exposure_at_default": costs.exposure_at_default,
            "margin_rate": costs.margin_rate,
            "decline_cost_rate": costs.decline_cost_rate,
            "fixed_review_cost": costs.fixed_review_cost,
            "cost_false_negative": costs.cost_false_negative,
            "cost_false_positive": costs.cost_false_positive,
            "cost_ratio": costs.cost_ratio,
            "analytic_threshold": costs.analytic_threshold,
        },
        calibration_metrics=calibration_metrics or {},
        performance_metrics=performance_metrics or {},
        training_rows=training_rows,
        training_period=training_period,
        notes=notes,
    )
