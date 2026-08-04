"""Tests for UCI ingestion.

No network access is used. The archive host is not reachable from CI, and a
test suite that depends on a third-party server is a test suite that fails for
reasons unrelated to the code. Instead a fixture is built to the published
layout and the parsing, validation and feature engineering are exercised
against it.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.ingest import (
    EXPECTED_ROWS,
    PAY_COLUMNS,
    REQUIRED_COLUMNS,
    DownloadError,
    SchemaError,
    data_quality_report,
    download_archive,
    engineer_features,
    extract_archive,
    load_raw,
    sha256_of,
    validate_schema,
)


def _raw_frame(n: int = 200) -> pd.DataFrame:
    """A frame with the published column names and plausible values."""
    rng = np.random.default_rng(0)
    data: dict[str, np.ndarray] = {
        "ID": np.arange(1, n + 1),
        "LIMIT_BAL": rng.choice([10_000, 50_000, 200_000], n).astype(float),
        "SEX": rng.choice([1, 2], n),
        "EDUCATION": rng.choice([0, 1, 2, 3, 4, 5, 6], n),
        "MARRIAGE": rng.choice([0, 1, 2, 3], n),
        "AGE": rng.integers(21, 70, n),
    }
    for column in ("PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"):
        data[column] = rng.choice([-2, -1, 0, 1, 2, 3], n)
    for i in range(1, 7):
        data[f"BILL_AMT{i}"] = rng.integers(-5_000, 200_000, n).astype(float)
        data[f"PAY_AMT{i}"] = rng.integers(0, 40_000, n).astype(float)
    data["default payment next month"] = rng.choice([0, 1], n, p=[0.78, 0.22])
    return pd.DataFrame(data)


def _renamed(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the module's rename mapping, as ``load_raw`` would."""
    from src.ingest import COLUMN_RENAMES

    out = frame.rename(columns=COLUMN_RENAMES)
    out.columns = [str(c).strip().lower().replace(" ", "_") for c in out.columns]
    return out


@pytest.fixture
def archive(tmp_path: Path) -> Path:
    """A zip containing a spreadsheet laid out like the published file."""
    from openpyxl import Workbook

    frame = _raw_frame()
    workbook = Workbook()
    sheet = workbook.active
    # The published file has a title row above the real header.
    sheet.append(["", *[f"X{i}" for i in range(1, 24)], "Y"])
    sheet.append(list(frame.columns))
    for row in frame.itertuples(index=False):
        sheet.append(list(row))

    spreadsheet = tmp_path / "default of credit card clients.xlsx"
    workbook.save(spreadsheet)

    bundle = tmp_path / "archive.zip"
    with zipfile.ZipFile(bundle, "w") as handle:
        handle.write(spreadsheet, spreadsheet.name)
    return bundle


def test_sha256_is_stable_and_sensitive(tmp_path: Path) -> None:
    first = tmp_path / "a.bin"
    first.write_bytes(b"hello")
    second = tmp_path / "b.bin"
    second.write_bytes(b"hello")
    third = tmp_path / "c.bin"
    third.write_bytes(b"hello!")

    assert sha256_of(first) == sha256_of(second)
    assert sha256_of(first) != sha256_of(third)


def test_extract_skips_when_already_present(archive: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Idempotence: a second run must not redo the work."""
    import src.ingest as ingest_module

    monkeypatch.setattr(ingest_module, "RAW_DATA_DIR", tmp_path / "raw")
    (tmp_path / "raw").mkdir()

    first = extract_archive(archive)
    stamp = first.stat().st_mtime_ns
    second = extract_archive(archive)
    assert first == second
    assert second.stat().st_mtime_ns == stamp


def test_extract_rejects_an_archive_without_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.ingest as ingest_module

    monkeypatch.setattr(ingest_module, "RAW_DATA_DIR", tmp_path / "raw")
    (tmp_path / "raw").mkdir()

    empty = tmp_path / "empty.zip"
    with zipfile.ZipFile(empty, "w") as handle:
        handle.writestr("readme.txt", "nothing here")
    with pytest.raises(SchemaError, match="No .xls"):
        extract_archive(empty)


def test_load_raw_finds_the_header_under_the_title_row(
    archive: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.ingest as ingest_module

    monkeypatch.setattr(ingest_module, "RAW_DATA_DIR", tmp_path / "raw")
    (tmp_path / "raw").mkdir()

    frame = load_raw(extract_archive(archive))
    for column in REQUIRED_COLUMNS:
        assert column in frame.columns, f"missing {column!r} after parsing"
    assert len(frame) == 200


def test_validate_accepts_a_well_formed_frame() -> None:
    validate_schema(_renamed(_raw_frame()), expected_rows=200)


def test_validate_rejects_a_missing_column() -> None:
    frame = _renamed(_raw_frame()).drop(columns=["pay_3"])
    with pytest.raises(SchemaError, match="Missing required columns"):
        validate_schema(frame, expected_rows=200)


def test_validate_rejects_a_wrong_row_count() -> None:
    with pytest.raises(SchemaError, match="Expected 30000 rows"):
        validate_schema(_renamed(_raw_frame()), expected_rows=EXPECTED_ROWS)


def test_validate_rejects_a_non_binary_target() -> None:
    frame = _renamed(_raw_frame())
    frame.loc[0, "default_next_month"] = 7
    with pytest.raises(SchemaError, match="binary"):
        validate_schema(frame, expected_rows=200)


def test_validate_rejects_an_implausible_age() -> None:
    frame = _renamed(_raw_frame())
    frame.loc[0, "age"] = 500
    with pytest.raises(SchemaError, match="plausible range"):
        validate_schema(frame, expected_rows=200)


def test_validate_rejects_a_non_positive_credit_limit() -> None:
    frame = _renamed(_raw_frame())
    frame.loc[0, "limit_bal"] = 0
    with pytest.raises(SchemaError, match="Non-positive credit limits"):
        validate_schema(frame, expected_rows=200)


def test_validate_rejects_a_non_numeric_column() -> None:
    """The shifted-column failure: parses fine, wrong in every downstream metric."""
    frame = _renamed(_raw_frame())
    frame["age"] = frame["age"].astype(str)
    with pytest.raises(SchemaError, match="not numeric"):
        validate_schema(frame, expected_rows=200)


def test_quality_report_counts_rows_and_nulls() -> None:
    frame = _renamed(_raw_frame())
    frame.loc[:9, "limit_bal"] = np.nan
    report = data_quality_report(frame)
    assert report.n_rows == 200
    assert report.duplicate_ids == 0
    assert report.null_rates["limit_bal"] == pytest.approx(10 / 200)
    assert 0.0 <= report.default_rate <= 1.0


def test_quality_report_detects_duplicate_ids() -> None:
    frame = _renamed(_raw_frame())
    frame.loc[1, "id"] = frame.loc[0, "id"]
    assert data_quality_report(frame).duplicate_ids == 1


def test_undocumented_education_codes_become_unknown() -> None:
    """Codes 0, 5 and 6 are in the data but not the dictionary."""
    engineered = engineer_features(_renamed(_raw_frame(600)))
    assert set(engineered["education"]) <= {
        "graduate_school",
        "university",
        "high_school",
        "others",
        "unknown",
    }
    assert (engineered["education"] == "unknown").any()


def test_categorical_codes_are_decoded() -> None:
    engineered = engineer_features(_renamed(_raw_frame(600)))
    assert set(engineered["sex"]) <= {"male", "female", "unknown"}
    assert set(engineered["marriage"]) <= {"married", "single", "others", "unknown"}
    assert engineered["age_band"].notna().all()


def test_derived_features_are_computed() -> None:
    engineered = engineer_features(_renamed(_raw_frame(400)))
    for column in ("avg_utilisation", "pay_ratio_1", "max_delinquency", "n_months_delinquent"):
        assert column in engineered.columns
        assert engineered[column].notna().all(), f"{column} has nulls"

    assert (engineered["n_months_delinquent"] <= len(PAY_COLUMNS)).all()
    assert (engineered["n_months_delinquent"] >= 0).all()
    assert (engineered["pay_ratio_1"] <= 5.0).all()


def test_avg_utilisation_matches_its_definition() -> None:
    frame = _renamed(_raw_frame(50))
    engineered = engineer_features(frame)
    expected = frame[[f"bill_amt{i}" for i in range(1, 7)]].mean(axis=1) / frame["limit_bal"]
    np.testing.assert_allclose(engineered["avg_utilisation"], expected, rtol=1e-9)


def test_download_failure_is_actionable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A blocked network must produce an instruction, not a stack trace."""
    import src.ingest as ingest_module

    monkeypatch.setattr(ingest_module, "RAW_DATA_DIR", tmp_path / "raw")
    (tmp_path / "raw").mkdir()

    with pytest.raises(DownloadError) as error:
        download_archive("https://invalid.invalid/nothing.zip")

    message = str(error.value)
    assert "manually" in message
    assert "offline" in message
