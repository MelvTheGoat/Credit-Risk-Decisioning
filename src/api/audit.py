"""Append-only audit log for credit decisions.

In consumer credit this is not optional. A declined applicant may complain to
an ombudsman, a regulator may sample decisions from two years ago, and a
supervisor may ask why a specific person was refused. In each case the
institution has to reconstruct **that individual decision** — not a decision
the current model would make today, but the one actually made, by the model
version actually deployed at the time, at the cutoff actually in force.

Every decision is therefore written before it is returned to the caller, with:

* ``timestamp`` — when the decision was made, UTC.
* ``input_hash`` — SHA-256 of the applicant's inputs, so the decision can be
  tied to the exact application data without the log becoming a second copy of
  the applicant database.
* ``probability_of_default``, ``score``, ``score_band``.
* ``decision`` and the ``threshold`` in force at that moment.
* ``reason_codes`` given to the applicant.
* ``model_version``, which resolves to a manifest recording the training data,
  calibration metrics and cost assumptions.

Design choices
--------------
**JSON Lines, append-only.** One self-describing record per line, no
rewriting, no schema migration. The file is greppable with standard tools
during an investigation, which matters at the moment someone actually needs it.
Records are never updated or deleted: a correction is a new record, so the
original decision remains visible.

**Written before the response is returned.** A decision the applicant received
but the institution cannot evidence is the failure mode to avoid, so the log
write is on the critical path rather than fire-and-forget.

**Not a production durability story.** A real deployment would write to
append-only object storage or a WORM store with retention locks, replicate it,
and monitor for gaps. Here it is a local file with an ``fsync``, which
demonstrates the contract without pretending to satisfy a records-retention
policy.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.config import AUDIT_DIR

logger = logging.getLogger(__name__)

DEFAULT_AUDIT_FILE = "decisions.jsonl"


class AuditLog:
    """Append-only JSON Lines log of credit decisions."""

    def __init__(self, path: Path | None = None) -> None:
        """Open (or create) the audit log.

        Args:
            path: Log file path. Defaults to ``audit/decisions.jsonl``.
        """
        self.path = path or (AUDIT_DIR / DEFAULT_AUDIT_FILE)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Serialise writes: FastAPI may handle requests concurrently, and
        # interleaved partial lines would corrupt the record.
        self._lock = threading.Lock()

    def record(
        self,
        decision: dict[str, Any],
        application_id: str | None = None,
        channel: str = "api",
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Append one decision to the log.

        Args:
            decision: Decision record from
                :meth:`src.engine.DecisionEngine.decide`.
            application_id: Caller-supplied identifier, if any.
            channel: Which door the decision came through, ``"api"`` or
                ``"batch"``. Recorded so that a divergence between the two
                would be visible in the log itself.
            extra: Additional fields to store.

        Returns:
            The written entry.
        """
        entry: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "application_id": application_id,
            "channel": channel,
            "input_hash": decision.get("input_hash"),
            "model_version": decision.get("model_version"),
            "threshold": decision.get("threshold"),
            "probability_of_default": decision.get("probability_of_default"),
            "score": decision.get("score"),
            "score_band": decision.get("score_band"),
            "decision": decision.get("decision"),
            "expected_loss": decision.get("expected_loss"),
            "reason_codes": decision.get("reason_codes", []),
        }
        if extra:
            entry.update(extra)

        line = json.dumps(entry, default=str)
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            # Durability on the critical path: a decision the applicant has
            # already received must not be lost to a crash.
            os.fsync(handle.fileno())
        return entry

    def record_many(
        self,
        decisions: list[dict[str, Any]],
        application_ids: list[str] | None = None,
        channel: str = "batch",
    ) -> int:
        """Append many decisions in one pass.

        Args:
            decisions: Decision records.
            application_ids: Identifiers aligned to ``decisions``.
            channel: Origin channel.

        Returns:
            Number of records written.
        """
        identifiers = application_ids or [None] * len(decisions)  # type: ignore[list-item]
        lines = []
        for decision, identifier in zip(decisions, identifiers, strict=True):
            entry = {
                "timestamp": datetime.now(UTC).isoformat(),
                "application_id": identifier,
                "channel": channel,
                "input_hash": decision.get("input_hash"),
                "model_version": decision.get("model_version"),
                "threshold": decision.get("threshold"),
                "probability_of_default": decision.get("probability_of_default"),
                "score": decision.get("score"),
                "score_band": decision.get("score_band"),
                "decision": decision.get("decision"),
                "expected_loss": decision.get("expected_loss"),
                "reason_codes": decision.get("reason_codes", []),
            }
            lines.append(json.dumps(entry, default=str))

        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        logger.info("Wrote %d audit records to %s", len(lines), self.path)
        return len(lines)

    def read_all(self) -> Iterator[dict[str, Any]]:
        """Iterate over every logged decision.

        Yields:
            One parsed entry per line. Malformed lines are skipped with a
            warning rather than aborting the read: during an investigation a
            partially corrupted log is still worth searching.
        """
        if not self.path.exists():
            return
        with self.path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed audit line %d", number)

    def find(
        self,
        application_id: str | None = None,
        input_hash: str | None = None,
        decision: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve decisions matching the given criteria.

        The retrieval path a complaint investigation actually uses. See
        ``RUNBOOK.md`` for the procedure.

        Args:
            application_id: Match this application identifier.
            input_hash: Match this input hash.
            decision: Match this outcome, e.g. ``"decline"``.
            limit: Maximum records to return, newest last.

        Returns:
            Matching entries in log order.
        """
        results = []
        for entry in self.read_all():
            if application_id is not None and entry.get("application_id") != application_id:
                continue
            if input_hash is not None and entry.get("input_hash") != input_hash:
                continue
            if decision is not None and entry.get("decision") != decision:
                continue
            results.append(entry)
        return results[-limit:] if limit else results

    def summary(self) -> dict[str, Any]:
        """Summarise the log for the metrics endpoint.

        Returns:
            Counts by decision and by model version, plus the observed score
            distribution and the time range covered.
        """
        entries = list(self.read_all())
        if not entries:
            return {"n_decisions": 0, "by_decision": {}, "by_model_version": {}}

        by_decision: dict[str, int] = {}
        by_version: dict[str, int] = {}
        for entry in entries:
            by_decision[str(entry.get("decision"))] = by_decision.get(str(entry.get("decision")), 0) + 1
            by_version[str(entry.get("model_version"))] = (
                by_version.get(str(entry.get("model_version")), 0) + 1
            )

        probabilities = [
            float(e["probability_of_default"])
            for e in entries
            if e.get("probability_of_default") is not None
        ]
        return {
            "n_decisions": len(entries),
            "by_decision": by_decision,
            "by_model_version": by_version,
            "first_timestamp": entries[0].get("timestamp"),
            "last_timestamp": entries[-1].get("timestamp"),
            "probabilities": probabilities,
        }
