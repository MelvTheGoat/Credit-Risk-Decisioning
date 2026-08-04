"""Idempotent ingestion of the UCI Default of Credit Card Clients dataset.

Source: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
30,000 credit card accounts from Taiwan, April to September 2005, with six
months of billing and repayment history and a binary default flag for the
following month. It carries sex, education, marital status and age, which is
why it is used here: the fairness audit needs protected attributes.

The pipeline is safe to re-run. Each stage checks for its own output and skips
if it is already present, so a second invocation costs a checksum verification
and nothing else. ``data/`` is git-ignored; the dataset is never committed.

What this dataset cannot support
--------------------------------
There are **no origination dates**. The file records six months of repayment
history for every account but says nothing about when any account was opened,
so origination cohorts cannot be reconstructed and true vintage analysis is
impossible on it — not difficult, impossible, because the required column does
not exist. Anything claiming to be a vintage curve on this dataset is really a
plot of something else.

The consequences are handled rather than papered over:

* :func:`src.models.lagged_feature_view` builds a pseudo-out-of-time design
  using only the earliest three months of history, which opens a genuine
  forward gap between features and outcome and so catches leakage.
* Population drift robustness cannot be tested here at all. That work lives on
  the synthetic book, which has real origination dates and injected drift.

Two further honest limits: the outcome window is a single month, which is a
much weaker definition of "default" than the 90-days-past-due at 12 months a
lender would use; and the sample is one issuer in one market in one year, so
the base rate and the coefficient magnitudes generalise to nothing in
particular.

Checksum policy
---------------
``EXPECTED_SHA256`` is ``None`` by default. On first download the archive's
hash is computed and written to ``data/raw/checksums.json``, and every later
run verifies against that record — trust on first use, pinned thereafter. If
you have the publisher's hash out of band, set ``CREDIT_RISK_UCI_SHA256`` in
the environment and it is enforced strictly on the first download too.

This is weaker than shipping a pinned hash, and the reason is recorded rather
than hidden: the environment this was developed in cannot reach
``archive.ics.uci.edu``, so no hash could be verified at authoring time and
hard-coding an unverified constant would be worse than admitting the gap.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, ensure_directories

logger = logging.getLogger(__name__)

UCI_URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
ARCHIVE_NAME = "default_of_credit_card_clients.zip"
PROCESSED_NAME = "uci_credit_default.parquet"
CHECKSUM_RECORD = "checksums.json"

EXPECTED_SHA256: str | None = None
CHECKSUM_ENV_VAR = "CREDIT_RISK_UCI_SHA256"

EXPECTED_ROWS = 30_000

# Raw column names as published, mapped to snake_case. PAY_0 really is called
# PAY_0 while its siblings are PAY_2..PAY_6; there is no PAY_1. That is a quirk
# of the source file, not a transcription error here.
COLUMN_RENAMES: dict[str, str] = {
    "ID": "id",
    "LIMIT_BAL": "limit_bal",
    "SEX": "sex_code",
    "EDUCATION": "education_code",
    "MARRIAGE": "marriage_code",
    "AGE": "age",
    "PAY_0": "pay_0",
    "PAY_2": "pay_2",
    "PAY_3": "pay_3",
    "PAY_4": "pay_4",
    "PAY_5": "pay_5",
    "PAY_6": "pay_6",
    "BILL_AMT1": "bill_amt1",
    "BILL_AMT2": "bill_amt2",
    "BILL_AMT3": "bill_amt3",
    "BILL_AMT4": "bill_amt4",
    "BILL_AMT5": "bill_amt5",
    "BILL_AMT6": "bill_amt6",
    "PAY_AMT1": "pay_amt1",
    "PAY_AMT2": "pay_amt2",
    "PAY_AMT3": "pay_amt3",
    "PAY_AMT4": "pay_amt4",
    "PAY_AMT5": "pay_amt5",
    "PAY_AMT6": "pay_amt6",
    "default payment next month": "default_next_month",
    "default.payment.next.month": "default_next_month",
}

SEX_LABELS: dict[int, str] = {1: "male", 2: "female"}

# Codes 0, 5 and 6 appear in the data but are not in the published dictionary.
# They are pooled into "unknown" rather than silently treated as an ordered
# level between "others" and nothing.
EDUCATION_LABELS: dict[int, str] = {
    1: "graduate_school",
    2: "university",
    3: "high_school",
    4: "others",
    0: "unknown",
    5: "unknown",
    6: "unknown",
}

MARRIAGE_LABELS: dict[int, str] = {1: "married", 2: "single", 3: "others", 0: "unknown"}

PAY_COLUMNS: tuple[str, ...] = ("pay_0", "pay_2", "pay_3", "pay_4", "pay_5", "pay_6")
BILL_COLUMNS: tuple[str, ...] = tuple(f"bill_amt{i}" for i in range(1, 7))
PAY_AMOUNT_COLUMNS: tuple[str, ...] = tuple(f"pay_amt{i}" for i in range(1, 7))

REQUIRED_COLUMNS: tuple[str, ...] = (
    "id",
    "limit_bal",
    "sex_code",
    "education_code",
    "marriage_code",
    "age",
    *PAY_COLUMNS,
    *BILL_COLUMNS,
    *PAY_AMOUNT_COLUMNS,
    "default_next_month",
)


class SchemaError(RuntimeError):
    """Raised when the loaded data does not match the expected schema."""


class DownloadError(RuntimeError):
    """Raised when the archive cannot be retrieved."""


@dataclass(frozen=True)
class DataQualityReport:
    """Row counts and null rates, logged on every ingest.

    Attributes:
        n_rows: Number of rows loaded.
        n_columns: Number of columns loaded.
        null_rates: Null rate per column, only for columns with any nulls.
        default_rate: Share of accounts defaulting next month.
        duplicate_ids: Count of duplicated identifiers.
    """

    n_rows: int
    n_columns: int
    null_rates: dict[str, float]
    default_rate: float
    duplicate_ids: int

    def log(self) -> None:
        """Write the report to the logger at INFO level."""
        logger.info("Loaded %d rows x %d columns", self.n_rows, self.n_columns)
        logger.info("Default rate: %.4f", self.default_rate)
        logger.info("Duplicate ids: %d", self.duplicate_ids)
        if self.null_rates:
            for column, rate in sorted(self.null_rates.items(), key=lambda kv: -kv[1]):
                logger.warning("Null rate %.4f in column %r", rate, column)
        else:
            logger.info("No nulls in any column")


def sha256_of(path: Path, chunk_size: int = 1 << 20) -> str:
    """Compute the SHA-256 hash of a file.

    Args:
        path: File to hash.
        chunk_size: Read buffer size.

    Returns:
        Lowercase hexadecimal digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _checksum_record_path() -> Path:
    """Location of the trust-on-first-use checksum record."""
    return RAW_DATA_DIR / CHECKSUM_RECORD


def _read_checksum_record() -> dict[str, str]:
    """Load the recorded checksums, returning an empty mapping if absent."""
    path = _checksum_record_path()
    if not path.exists():
        return {}
    try:
        return dict(json.loads(path.read_text()))
    except (OSError, ValueError) as error:  # pragma: no cover - defensive
        logger.warning("Could not read checksum record: %s", error)
        return {}


def _write_checksum_record(record: dict[str, str]) -> None:
    """Persist the checksum record."""
    _checksum_record_path().write_text(json.dumps(record, indent=2, sort_keys=True))


def verify_checksum(path: Path, name: str = ARCHIVE_NAME) -> str:
    """Verify a file against the pinned or recorded checksum.

    Args:
        path: File to verify.
        name: Key under which the checksum is recorded.

    Returns:
        The file's actual digest.

    Raises:
        SchemaError: If the digest does not match an expected value.
    """
    actual = sha256_of(path)
    expected = EXPECTED_SHA256 or os.environ.get(CHECKSUM_ENV_VAR)

    if expected:
        if actual != expected:
            raise SchemaError(
                f"Checksum mismatch for {path.name}: expected {expected}, got {actual}. "
                "The download is corrupt or the published file has changed."
            )
        logger.info("Checksum verified against pinned value")
        return actual

    record = _read_checksum_record()
    if name in record:
        if record[name] != actual:
            raise SchemaError(
                f"Checksum mismatch for {path.name}: recorded {record[name]}, got {actual}. "
                "Delete data/raw/ to re-establish trust, having satisfied yourself why it changed."
            )
        logger.info("Checksum matches the recorded value")
    else:
        record[name] = actual
        _write_checksum_record(record)
        logger.info("Recorded checksum %s for %s (trust on first use)", actual, name)
    return actual


def download_archive(url: str = UCI_URL, force: bool = False) -> Path:
    """Download the archive, skipping if it is already present.

    Args:
        url: Source URL.
        force: Re-download even if the file exists.

    Returns:
        Path to the archive.

    Raises:
        DownloadError: If the download fails for any reason.
    """
    ensure_directories()
    destination = RAW_DATA_DIR / ARCHIVE_NAME

    if destination.exists() and not force:
        logger.info("Archive already present at %s; skipping download", destination)
        verify_checksum(destination)
        return destination

    logger.info("Downloading %s", url)
    try:
        import requests

        response = requests.get(url, timeout=120, stream=True)
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 16):
                handle.write(chunk)
    except Exception as error:  # noqa: BLE001 - re-raised with actionable context
        if destination.exists():
            destination.unlink()
        raise DownloadError(
            f"Could not download {url}: {error}\n"
            "If this environment restricts outbound network access, fetch the archive "
            f"manually and place it at {destination}, then re-run. The rest of the "
            "pipeline works offline from that point on."
        ) from error

    logger.info("Downloaded %.1f KiB to %s", destination.stat().st_size / 1024, destination)
    verify_checksum(destination)
    return destination


def extract_archive(archive: Path, force: bool = False) -> Path:
    """Extract the data file from the archive, skipping if already extracted.

    Args:
        archive: Path to the zip archive.
        force: Re-extract even if the member is already present.

    Returns:
        Path to the extracted data file.

    Raises:
        SchemaError: If the archive holds no recognisable data file.
    """
    with zipfile.ZipFile(archive) as bundle:
        members = [
            name
            for name in bundle.namelist()
            if name.lower().endswith((".xls", ".xlsx", ".csv")) and not name.startswith("__MACOSX")
        ]
        if not members:
            raise SchemaError(f"No .xls/.xlsx/.csv member found in {archive.name}: {bundle.namelist()}")

        member = members[0]
        target = RAW_DATA_DIR / Path(member).name
        if target.exists() and not force:
            logger.info("Data file already extracted at %s; skipping", target)
            return target

        logger.info("Extracting %s", member)
        with bundle.open(member) as source, target.open("wb") as sink:
            sink.write(source.read())
    return target


def load_raw(path: Path) -> pd.DataFrame:
    """Load the extracted data file and normalise its column names.

    The published spreadsheet has a title row above the real header, so the
    header sits on the second row. CSV variants that have already been
    flattened are handled too.

    Args:
        path: Path to the extracted file.

    Returns:
        Frame with snake_case column names.

    Raises:
        SchemaError: If the file cannot be parsed into the expected columns.
    """
    suffix = path.suffix.lower()

    if suffix == ".csv":
        frame = pd.read_csv(path)
        if "LIMIT_BAL" not in frame.columns and "X1" in frame.columns:
            frame = pd.read_csv(path, header=1)
    else:
        engine: str = "xlrd" if suffix == ".xls" else "openpyxl"
        try:
            frame = pd.read_excel(path, sheet_name=0, header=1, engine=engine)  # type: ignore[call-overload]
        except ImportError as error:
            raise SchemaError(
                f"Reading {suffix} requires the '{engine}' package. "
                f"Install it with: pip install {engine}"
            ) from error

    frame = frame.rename(columns={c: str(c).strip() for c in frame.columns})
    frame = frame.rename(columns=COLUMN_RENAMES)
    frame.columns = [str(c).strip().lower().replace(" ", "_") for c in frame.columns]

    if "default_next_month" not in frame.columns:
        # Some mirrors rename the target; take the last column as a fallback
        # but say loudly that it happened.
        fallback = frame.columns[-1]
        logger.warning("Target column not found by name; assuming last column %r", fallback)
        frame = frame.rename(columns={fallback: "default_next_month"})

    return frame


def validate_schema(frame: pd.DataFrame, expected_rows: int | None = EXPECTED_ROWS) -> None:
    """Check the loaded frame against the expected schema.

    Fails loudly and early. A silently mis-parsed spreadsheet that shifts every
    column by one produces a model that trains, validates and is wrong in ways
    no downstream metric will reveal.

    Args:
        frame: Loaded frame.
        expected_rows: Expected row count, or ``None`` to skip the check.

    Raises:
        SchemaError: On any missing column, wrong row count, non-binary target
            or out-of-range value.
    """
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise SchemaError(f"Missing required columns: {missing}. Found: {list(frame.columns)}")

    if expected_rows is not None and len(frame) != expected_rows:
        raise SchemaError(f"Expected {expected_rows} rows, found {len(frame)}")

    target = frame["default_next_month"]
    unexpected = set(pd.unique(target.dropna())) - {0, 1}
    if unexpected:
        raise SchemaError(f"Target must be binary; found values {sorted(unexpected)}")

    numeric_columns = ["limit_bal", "age", *PAY_COLUMNS, *BILL_COLUMNS, *PAY_AMOUNT_COLUMNS]
    non_numeric = [
        column for column in numeric_columns if not pd.api.types.is_numeric_dtype(frame[column])
    ]
    if non_numeric:
        raise SchemaError(f"Expected numeric columns are not numeric: {non_numeric}")

    if not frame["age"].between(15, 120).all():
        raise SchemaError("Age values outside the plausible range 15-120")

    if (frame["limit_bal"] <= 0).any():
        raise SchemaError("Non-positive credit limits present")

    logger.info("Schema validation passed")


def data_quality_report(frame: pd.DataFrame) -> DataQualityReport:
    """Compute row counts, null rates and the default rate.

    Args:
        frame: Loaded frame.

    Returns:
        The report.
    """
    null_rates = {
        column: float(frame[column].isna().mean())
        for column in frame.columns
        if frame[column].isna().any()
    }
    return DataQualityReport(
        n_rows=len(frame),
        n_columns=frame.shape[1],
        null_rates=null_rates,
        default_rate=float(frame["default_next_month"].mean()),
        duplicate_ids=int(frame["id"].duplicated().sum()) if "id" in frame.columns else 0,
    )


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Decode categorical codes and derive the modelling features.

    Derived features and why each exists:

    * ``avg_utilisation`` — mean billed balance over the credit limit. The
      single most predictive construct in most card books, and not directly
      present in the raw file.
    * ``pay_ratio_1`` — most recent payment over the preceding month's bill.
      Separates an applicant clearing their balance from one paying the
      minimum, which the raw amounts do not distinguish.
    * ``max_delinquency`` and ``n_months_delinquent`` — worst and count of
      delayed months across the six-month history, summarising the ``PAY_*``
      columns without asking a linear model to treat their codes as an
      interval scale, which they are not.

    Args:
        frame: Validated raw frame.

    Returns:
        Frame with decoded labels, derived features and an ``age_band`` column
        for the fairness audit.
    """
    out = frame.copy()

    out["sex"] = out["sex_code"].map(SEX_LABELS).fillna("unknown")
    out["education"] = out["education_code"].map(EDUCATION_LABELS).fillna("unknown")
    out["marriage"] = out["marriage_code"].map(MARRIAGE_LABELS).fillna("unknown")

    from src.simulate import AGE_BAND_EDGES, AGE_BAND_LABELS

    out["age_band"] = pd.Categorical(
        np.asarray(AGE_BAND_LABELS, dtype=object)[
            np.digitize(out["age"].to_numpy(dtype=float), AGE_BAND_EDGES[1:-1], right=False)
        ],
        categories=list(AGE_BAND_LABELS),
        ordered=True,
    )

    limit = out["limit_bal"].replace(0, np.nan)
    out["avg_utilisation"] = out[list(BILL_COLUMNS)].mean(axis=1) / limit

    previous_bill = out["bill_amt2"].where(out["bill_amt2"] > 0, np.nan)
    out["pay_ratio_1"] = (out["pay_amt1"] / previous_bill).clip(upper=5.0)
    # No prior balance means nothing was owed, so a zero payment is not a
    # delinquency signal. Treated as fully paid rather than left missing.
    out["pay_ratio_1"] = out["pay_ratio_1"].fillna(1.0)

    pay_frame = out[list(PAY_COLUMNS)]
    out["max_delinquency"] = pay_frame.max(axis=1)
    out["n_months_delinquent"] = (pay_frame > 0).sum(axis=1)

    out["default"] = out["default_next_month"].astype(int)
    return out


def ingest(
    url: str = UCI_URL,
    force: bool = False,
    save_processed: bool = True,
) -> pd.DataFrame:
    """Run the full ingestion pipeline, idempotently.

    Stages: download, checksum, extract, load, validate, quality report,
    feature engineering, save. Each skips if its output already exists, so
    re-running is cheap and safe.

    Args:
        url: Source URL.
        force: Re-run every stage from scratch.
        save_processed: Write the processed frame to parquet.

    Returns:
        The processed frame.
    """
    ensure_directories()
    processed_path = PROCESSED_DATA_DIR / PROCESSED_NAME

    if processed_path.exists() and not force:
        logger.info("Processed dataset already present at %s; loading", processed_path)
        return pd.read_parquet(processed_path)

    archive = download_archive(url, force=force)
    extracted = extract_archive(archive, force=force)

    raw = load_raw(extracted)
    validate_schema(raw)
    data_quality_report(raw).log()

    processed = engineer_features(raw)
    data_quality_report(processed).log()

    if save_processed:
        processed.to_parquet(processed_path, index=False)
        logger.info("Wrote processed dataset to %s", processed_path)

    return processed


def main() -> int:  # pragma: no cover - CLI entry point
    """Run ingestion from the command line.

    Returns:
        Process exit code: 0 on success, 1 on a handled failure.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        frame = ingest()
    except (DownloadError, SchemaError) as error:
        logger.error("%s", error)
        return 1
    print(f"Ingested {len(frame):,} rows, {frame.shape[1]} columns")
    print(f"Default rate: {frame['default'].mean():.4f}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
