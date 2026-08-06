"""Score a portfolio file through the same code path the API uses.

Real credit teams do both real-time and batch: an application arrives at the
API, and overnight the whole back book is re-scored for limit reviews,
provisioning, and pre-approved offers.

Where those are implemented separately they drift, and the same applicant can
receive different answers depending on which door they came through. So this
script implements no scoring logic. It loads the same artifact, builds the same
:class:`~src.engine.DecisionEngine`, calls the same ``decide`` method, and
writes to the same audit log with ``channel="batch"``.
``tests/test_engine.py`` asserts the two paths produce identical decisions.

Usage::

    python -m scripts.score_batch --input portfolio.csv --output scored.csv
    python -m scripts.score_batch --input portfolio.parquet --version 20260804T120000Z
    python -m scripts.score_batch --demo          # generate and score a sample
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow `python scripts/score_batch.py` as well as `python -m scripts.score_batch`.
# Running a file directly puts the script's own directory on sys.path, not the
# repo root, so the `src` imports below would fail with a bare
# ModuleNotFoundError that gives the caller no hint about the fix.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.api.audit import AuditLog  # noqa: E402
from src.artifacts import ArtifactMismatchError, ArtifactNotFoundError  # noqa: E402
from src.config import DEFAULT_COSTS, SYNTHETIC_FEATURES, ensure_directories  # noqa: E402
from src.engine import DecisionEngine  # noqa: E402

logger = logging.getLogger("score_batch")

CHUNK_SIZE = 20_000


def load_portfolio(path: Path) -> pd.DataFrame:
    """Read a portfolio file, inferring the format from its suffix.

    Args:
        path: Input file, ``.csv`` or ``.parquet``.

    Returns:
        The loaded frame.

    Raises:
        ValueError: On an unsupported suffix.
    """
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported input format {suffix!r}; use .csv or .parquet")


def write_output(frame: pd.DataFrame, path: Path) -> None:
    """Write scored results, inferring the format from the suffix.

    Args:
        frame: Scored portfolio.
        path: Output file.

    Raises:
        ValueError: On an unsupported suffix.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame.to_csv(path, index=False)
    elif suffix in {".parquet", ".pq"}:
        frame.to_parquet(path, index=False)
    else:
        raise ValueError(f"Unsupported output format {suffix!r}; use .csv or .parquet")


def make_demo_portfolio(n: int = 500) -> pd.DataFrame:
    """Generate a small portfolio for demonstration.

    Args:
        n: Number of applicants.

    Returns:
        A frame with the model's input columns plus an identifier.
    """
    from dataclasses import replace

    from src.config import DEFAULT_SIMULATION
    from src.simulate import simulate_loan_book

    book = simulate_loan_book(replace(DEFAULT_SIMULATION, n_applicants=n, seed=999))
    return book[["application_id", *SYNTHETIC_FEATURES.model_features]].copy()


def score_portfolio(
    frame: pd.DataFrame,
    version: str | None = None,
    with_reasons: bool = True,
    write_audit: bool = True,
    id_column: str = "application_id",
) -> pd.DataFrame:
    """Score a portfolio and record every decision in the audit log.

    Args:
        frame: Portfolio to score.
        version: Artifact version, or ``None`` for the current pointer.
        with_reasons: Compute adverse action reason codes for declines.
        write_audit: Append decisions to the audit log. Batch decisions are
            real decisions and are logged by default; a lender must be able to
            evidence a limit reduction as readily as a declined application.
        id_column: Column holding the applicant identifier.

    Returns:
        The input frame with decision columns appended.
    """
    engine = DecisionEngine.from_artifacts(version, costs=DEFAULT_COSTS)
    logger.info(
        "Scoring %d applicants with model %s at threshold %.4f",
        len(frame),
        engine.model_version,
        engine.threshold,
    )

    results = []
    for start in range(0, len(frame), CHUNK_SIZE):
        chunk = frame.iloc[start : start + CHUNK_SIZE]
        decisions = engine.decide(chunk, with_reasons=with_reasons)

        if write_audit:
            identifiers = (
                chunk[id_column].astype(str).tolist() if id_column in chunk.columns else None
            )
            AuditLog().record_many(
                [d.to_dict() for d in decisions], application_ids=identifiers, channel="batch"
            )

        scored = engine.decide_frame(chunk, with_reasons=with_reasons)
        results.append(pd.concat([chunk.reset_index(drop=True), scored.reset_index(drop=True)], axis=1))
        logger.info("Scored %d / %d", min(start + CHUNK_SIZE, len(frame)), len(frame))

    return pd.concat(results, ignore_index=True)


def main(argv: list[str] | None = None) -> int:
    """Command line entry point.

    Args:
        argv: Argument vector, defaulting to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="portfolio file (.csv or .parquet)")
    parser.add_argument("--output", type=Path, default=Path("reports/scored_portfolio.csv"))
    parser.add_argument("--version", default=None, help="artifact version; defaults to current")
    parser.add_argument("--no-reasons", action="store_true", help="skip reason codes (faster)")
    parser.add_argument("--no-audit", action="store_true", help="do not write the audit log")
    parser.add_argument("--demo", action="store_true", help="generate and score a sample portfolio")
    arguments = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ensure_directories()

    if arguments.demo:
        portfolio = make_demo_portfolio()
        logger.info("Generated demo portfolio of %d applicants", len(portfolio))
    elif arguments.input:
        portfolio = load_portfolio(arguments.input)
        logger.info("Loaded %d applicants from %s", len(portfolio), arguments.input)
    else:
        parser.error("provide --input or --demo")

    try:
        scored = score_portfolio(
            portfolio,
            version=arguments.version,
            with_reasons=not arguments.no_reasons,
            write_audit=not arguments.no_audit,
        )
    except (ArtifactMismatchError, ArtifactNotFoundError) as error:
        logger.error("%s", error)
        return 1
    except ValueError as error:
        logger.error("Portfolio does not match the model: %s", error)
        return 1

    write_output(scored, arguments.output)

    counts = scored["decision"].value_counts().to_dict()
    logger.info("Wrote %s", arguments.output)
    logger.info("Decisions: %s", counts)
    logger.info(
        "Approval rate %.4f | mean PD %.4f | total expected loss %.0f",
        float((scored["decision"] == "approve").mean()),
        float(scored["probability_of_default"].mean()),
        float(scored.loc[scored["decision"] == "approve", "expected_loss"].sum()),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
