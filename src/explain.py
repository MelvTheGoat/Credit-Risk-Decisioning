"""Explanation and adverse action reason codes.

Two different jobs, often confused
----------------------------------
**Global explanation** answers "what drives this model?" and is for model risk
management, validation and challenge.

**Adverse action reason codes** answer "why was *this person* declined?" and are
for the applicant. In the United States, ECOA and Regulation B require a
creditor to give the specific principal reasons for adverse action — not a
generic statement, not "you did not meet our criteria", and not a list of every
factor. The UK, EU and most other consumer credit regimes impose comparable
duties. This is a legal obligation attached to an individual decision, and it
has to hold up months later, which is why every decision this system makes is
written to the audit log in :mod:`src.api.audit`.

Two reason code methods are implemented, because they answer the requirement
differently:

* :func:`scorecard_reason_codes` — **points shortfall**. For each
  characteristic, compare the points the applicant earned against the maximum
  points obtainable on that characteristic. The largest shortfalls are the
  principal reasons. This is the long-established scorecard method, it is what
  supervisors expect to see, and it has an unbeatable property: it is exactly
  reproducible from a table, forever, with no model artefact required.
* :func:`shap_reason_codes` — **SHAP contributions**. Works for any model,
  including the booster. The features pushing predicted risk up the most become
  the principal reasons. More flexible, and harder to defend: the explanation
  depends on a background dataset and on the explainer implementation, both of
  which have to be preserved to reproduce it later.

A note on age
-------------
``age`` is a model feature here and it is genuinely predictive. It is also a
protected basis under ECOA. Rather than quietly emitting "your age" as a reason
for declining someone, reason codes derived from protected or
protected-adjacent characteristics are returned with ``protected_basis=True``
so they are visible to legal review instead of going out on a letter
unnoticed. See ``DECISIONS.md`` for why the feature was kept and what a real
deployment would have to settle first.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.config import DEFAULT_COSTS, CostParameters
from src.scorecard import Scorecard

logger = logging.getLogger(__name__)

# Plain-language templates. The applicant reads this sentence, so it must
# describe what they can see about their own file, not the model's internals.
# "WOE of utilisation bin 4 contributed -0.31" is not a reason code.
REASON_TEMPLATES: dict[str, str] = {
    "utilisation": "Your balances are high relative to your available credit limits",
    "debt_to_income": "Your total debt repayments are high relative to your income",
    "income_recorded": "The income recorded on your application is low relative to the amount requested",
    "n_delinq_24m": "There are recent missed or late payments on your credit accounts",
    "credit_history_months": "Your credit history is shorter than we typically require",
    "n_inquiries_6m": "You have applied for credit several times recently",
    "employment_years": "You have been in your current employment for a short time",
    "loan_amount": "The amount requested is high relative to your circumstances",
    "housing_status": "Your residential status is one we associate with higher risk",
    "product_type": "The product applied for carries higher risk in our experience",
    "age": "Your age band is associated with higher risk in our data",
    # UCI dataset characteristics.
    "limit_bal": "Your existing credit limit is low relative to other applicants",
    "pay_0": "Your most recent payment status shows a delay",
    "pay_2": "Your payment status two months ago showed a delay",
    "pay_3": "Your payment status three months ago showed a delay",
    "max_delinquency": "Your worst recent payment delay is longer than we typically accept",
    "n_months_delinquent": "You were behind on payments in several recent months",
    "avg_utilisation": "Your balances are high relative to your credit limit",
    "pay_ratio_1": "Your recent payments are small relative to the balance owed",
}

# Characteristics that are a protected basis, or so closely tied to one that a
# reason code derived from them needs legal sign-off before it reaches an
# applicant.
PROTECTED_BASIS_FEATURES: frozenset[str] = frozenset({"age", "age_band", "sex", "marriage", "education"})


@dataclass(frozen=True)
class ReasonCode:
    """One principal reason contributing to an adverse decision.

    Attributes:
        rank: 1 is the strongest reason.
        feature: Model characteristic the reason derives from.
        value: The applicant's value for that characteristic.
        contribution: Size of the adverse contribution. Points shortfall for
            the scorecard method, SHAP value in log-odds for the SHAP method.
        statement: Plain-language sentence for the applicant.
        protected_basis: Whether this reason derives from a protected or
            protected-adjacent characteristic and needs legal review.
    """

    rank: int
    feature: str
    value: object
    contribution: float
    statement: str
    protected_basis: bool

    def to_dict(self) -> dict[str, object]:
        """Return the reason code as a JSON-serialisable dictionary.

        The feature value arrives as a numpy scalar, which ``json.dumps``
        rejects. Coercing here rather than at each call site means every
        consumer — audit log, API response, batch output — gets a value it can
        serialise.
        """
        payload = asdict(self)
        payload["value"] = _jsonable(payload["value"])
        return payload


def _statement_for(feature: str) -> str:
    """Look up the plain-language sentence for a characteristic."""
    return REASON_TEMPLATES.get(
        feature, f"The information recorded for '{feature}' is associated with higher risk"
    )


def scorecard_reason_codes(
    scorecard: Scorecard,
    applicant: pd.DataFrame,
    top_k: int = 4,
) -> list[ReasonCode]:
    """Derive reason codes from points shortfall against the best achievable.

    For each characteristic in the card, the applicant's awarded points are
    compared with the highest points any bin of that characteristic could have
    awarded. The largest gaps are the principal reasons.

    This is the method a supervisor will recognise. It also has a practical
    virtue no model-based explainer can match: the answer is recoverable from
    the printed points table alone, so a decision can still be explained years
    later without the serialised model.

    Args:
        scorecard: A fitted scorecard.
        applicant: Single-row frame of the applicant's features.
        top_k: Number of reasons to return. Regulation B expects the principal
            reasons, conventionally four.

    Returns:
        Reason codes ordered by contribution, largest first.
    """
    if len(applicant) != 1:
        raise ValueError(f"Expected exactly one applicant row, got {len(applicant)}")

    points = scorecard.points_table()
    rows = []

    for column in scorecard.feature_names_:
        feature = column[: -len("_woe")]
        binning = scorecard.encoder.binnings[feature]
        assigned = str(binning.assign_bins(applicant[feature]).iloc[0])

        feature_points = points[points["feature"] == feature]
        if feature_points.empty:
            continue
        matched = feature_points[feature_points["bin"] == assigned]
        if matched.empty:
            continue

        awarded = float(matched["points"].iloc[0])
        best_possible = float(feature_points["points"].max())
        rows.append(
            {
                "feature": feature,
                "value": applicant[feature].iloc[0],
                "shortfall": best_possible - awarded,
            }
        )

    ordered = sorted(rows, key=lambda r: -float(r["shortfall"]))[:top_k]
    return [
        ReasonCode(
            rank=i,
            feature=str(row["feature"]),
            value=row["value"],
            contribution=float(row["shortfall"]),
            statement=_statement_for(str(row["feature"])),
            protected_basis=str(row["feature"]) in PROTECTED_BASIS_FEATURES,
        )
        for i, row in enumerate(ordered, start=1)
    ]


class ShapExplainer:
    """SHAP wrapper that picks a suitable explainer for the model type.

    Uses ``TreeExplainer`` for LightGBM, which is exact and fast, and falls back
    to the model-agnostic sampling explainer otherwise. The background sample is
    retained: SHAP values are contributions *relative to a reference*, so the
    reference is part of the explanation and has to be versioned with the model
    if the explanation is ever to be reproduced.
    """

    def __init__(self, model: Any, background: pd.DataFrame, max_background: int = 200) -> None:
        """Build an explainer for a fitted model.

        Args:
            model: A fitted model wrapper from :mod:`src.models`.
            background: Representative rows used as the SHAP reference.
            max_background: Rows to sample from ``background``. Kept small
                because the model-agnostic path is quadratic in this number.
        """
        import shap

        self.model = model
        self.feature_names: list[str] = list(model.feature_names)
        self.background = background[self.feature_names].head(max_background).copy()

        # Duck-type on the *public* contract, not on a LightGBM internal. A
        # model that can hand us a prepared frame and expose a fitted booster
        # gets the exact TreeExplainer; anything else falls back to the
        # model-agnostic path. Reaching for `inner.booster_` here would couple
        # this module to LightGBM's attribute names and break silently on a
        # refactor.
        inner = getattr(model, "model", None)
        if callable(getattr(model, "prepare", None)) and inner is not None:
            self.explainer: Any = shap.TreeExplainer(inner)
            self._background_prepared = model.prepare(self.background)
            self.kind = "tree"
        else:
            self.explainer = shap.Explainer(
                lambda data: model.predict_proba(pd.DataFrame(data, columns=self.feature_names)),
                self.background.to_numpy(),
            )
            self._background_prepared = self.background
            self.kind = "agnostic"

    def _prepare(self, X: pd.DataFrame) -> pd.DataFrame:
        """Prepare a frame the same way the wrapped model does.

        A method rather than a stored lambda: a closure over ``model`` makes the
        whole explainer unpicklable, which matters the moment anyone tries to
        cache one alongside a model artifact.
        """
        if self.kind == "tree":
            return self.model.prepare(X)
        return X[self.feature_names]

    def shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """Compute SHAP values.

        Args:
            X: Rows to explain.

        Returns:
            Array of shape ``(n_rows, n_features)`` in log-odds units for the
            tree path.
        """
        prepared = self._prepare(X)
        values = self.explainer.shap_values(prepared) if self.kind == "tree" else self.explainer(
            prepared.to_numpy()
        ).values
        values = np.asarray(values)
        # Some SHAP versions return one matrix per class for binary problems.
        if values.ndim == 3:
            values = values[:, :, -1]
        return values

    def global_importance(self, X: pd.DataFrame) -> pd.DataFrame:
        """Mean absolute SHAP value per feature, descending.

        Args:
            X: Rows to summarise over.

        Returns:
            Frame of feature and mean absolute SHAP contribution.
        """
        values = self.shap_values(X)
        return pd.DataFrame(
            {
                "feature": self.feature_names,
                "mean_abs_shap": np.abs(values).mean(axis=0),
                "mean_shap": values.mean(axis=0),
            }
        ).sort_values("mean_abs_shap", ascending=False, ignore_index=True)


def shap_reason_codes(
    explainer: ShapExplainer,
    applicant: pd.DataFrame,
    top_k: int = 4,
) -> list[ReasonCode]:
    """Derive reason codes from the largest risk-increasing SHAP contributions.

    Args:
        explainer: A fitted :class:`ShapExplainer`.
        applicant: Single-row frame of the applicant's features.
        top_k: Number of reasons to return.

    Returns:
        Reason codes ordered by contribution, largest first. Only features that
        pushed predicted risk *up* are eligible — a factor that helped the
        applicant is not a reason they were declined.
    """
    if len(applicant) != 1:
        raise ValueError(f"Expected exactly one applicant row, got {len(applicant)}")

    values = explainer.shap_values(applicant)[0]
    order = np.argsort(-values)

    codes: list[ReasonCode] = []
    for position in order:
        if values[position] <= 0:
            break
        feature = explainer.feature_names[position]
        codes.append(
            ReasonCode(
                rank=len(codes) + 1,
                feature=feature,
                value=applicant[feature].iloc[0],
                contribution=float(values[position]),
                statement=_statement_for(feature),
                protected_basis=feature in PROTECTED_BASIS_FEATURES,
            )
        )
        if len(codes) >= top_k:
            break
    return codes


# --------------------------------------------------------------------------- #
# Worked example
# --------------------------------------------------------------------------- #


def explain_decision(
    applicant: pd.DataFrame,
    probability: float,
    threshold: float,
    reason_codes: list[ReasonCode],
    score: float | None = None,
    costs: CostParameters = DEFAULT_COSTS,
    exposure: float | None = None,
    model_version: str = "unknown",
) -> dict[str, Any]:
    """Assemble the full record of a single decision.

    This is the object written to the audit log and returned by the API. It
    contains everything needed to reconstruct and defend the decision later:
    the inputs, the probability, the cutoff in force, the economics, and the
    reasons given to the applicant.

    Args:
        applicant: Single-row frame of the applicant's features.
        probability: Calibrated probability of default.
        threshold: PD cutoff in force at decision time.
        reason_codes: Principal reasons, populated for declines.
        score: Scorecard points, if a scorecard produced them.
        costs: Cost parameters in force.
        exposure: Exposure at default for this applicant.
        model_version: Version identifier of the model that decided.

    Returns:
        The decision record.
    """
    from src.policy import assign_score_band

    exposure_value = costs.exposure_at_default if exposure is None else float(exposure)
    approved = bool(probability < threshold)

    return {
        "decision": "approve" if approved else "decline",
        "probability_of_default": float(probability),
        "threshold": float(threshold),
        "score": None if score is None else float(score),
        "score_band": None if score is None else assign_score_band(float(score)),
        "expected_loss": float(probability * costs.lgd * exposure_value),
        "expected_margin": float((1.0 - probability) * costs.margin_rate * exposure_value),
        "expected_value": float(
            (1.0 - probability) * costs.margin_rate * exposure_value
            - probability * costs.lgd * exposure_value
        ),
        "exposure_at_default": exposure_value,
        "model_version": model_version,
        # Reasons are only meaningful for a decline. Sending "principal reasons"
        # to an approved applicant would be nonsense.
        "reason_codes": [] if approved else [code.to_dict() for code in reason_codes],
        "features": {k: _jsonable(v) for k, v in applicant.iloc[0].to_dict().items()},
    }


def _jsonable(value: object) -> object:
    """Coerce numpy and pandas scalars into JSON-serialisable Python types."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, str | int | float | bool):
        return None if isinstance(value, float) and np.isnan(value) else value
    if value is None or value is pd.NaT:
        return None
    return str(value)


def format_adverse_action_notice(record: dict[str, Any], top_k: int = 4) -> str:
    """Render a decision record as the text of an adverse action notice.

    Illustrative only. A production notice carries the creditor's identity, the
    credit bureau used, the applicant's right to a free report and to dispute
    inaccuracies, and the statutory notices — none of which is modelling work.
    The part this system is responsible for is that the **reasons are the real
    reasons**, derived from the same artefact that made the decision.

    Args:
        record: Output of :func:`explain_decision`.
        top_k: Maximum number of reasons to include.

    Returns:
        The notice text.
    """
    if record["decision"] == "approve":
        return "Application approved. No adverse action notice is required."

    lines = [
        "NOTICE OF ADVERSE ACTION",
        "",
        "We are unable to approve your application for credit at this time.",
        "",
        "The principal reasons for our decision are:",
    ]
    for code in record["reason_codes"][:top_k]:
        flag = "  [PROTECTED BASIS - REQUIRES LEGAL REVIEW]" if code["protected_basis"] else ""
        lines.append(f"  {code['rank']}. {code['statement']}{flag}")

    lines += [
        "",
        f"Decision reference: model {record['model_version']}, "
        f"cutoff {record['threshold']:.4f}.",
        "",
        "You have the right to a statement of the specific reasons for this",
        "decision and to obtain a free copy of any credit report used.",
    ]
    return "\n".join(lines)


def worked_example(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
    scorecard: Scorecard | None = None,
    explainer: ShapExplainer | None = None,
    feature_columns: list[str] | None = None,
    model_version: str = "unknown",
    exposure_column: str = "loan_amount",
) -> dict[str, Any]:
    """Take one declined applicant end to end, for the write-up.

    Picks a declined applicant close to the cutoff rather than an extreme one.
    A borderline decline is the case that actually has to be defensible: the
    obvious declines explain themselves, and the marginal ones are where a
    challenge lands.

    Args:
        frame: Applicant frame.
        probabilities: Calibrated P(default) aligned to ``frame``.
        threshold: PD cutoff in force.
        scorecard: Fitted scorecard, for points-shortfall reasons and a score.
        explainer: Fitted SHAP explainer, for SHAP-based reasons.
        feature_columns: Model input columns. Inferred from the scorecard or
            explainer if omitted.
        model_version: Version identifier of the deciding model.
        exposure_column: Column holding exposure at default.

    Returns:
        The decision record, with both reason code methods where available and
        the rendered adverse action notice.
    """
    probabilities = np.asarray(probabilities, dtype=float)
    declined = np.where(probabilities >= threshold)[0]
    if len(declined) == 0:
        raise ValueError("No declined applicants at this threshold")

    # Closest decline to the cutoff: the marginal case.
    chosen = int(declined[np.argmin(probabilities[declined] - threshold)])
    row = frame.iloc[[chosen]]

    if feature_columns is None:
        if scorecard is not None:
            feature_columns = [c[: -len("_woe")] for c in scorecard.feature_names_]
        elif explainer is not None:
            feature_columns = explainer.feature_names
        else:
            raise ValueError("feature_columns is required when no scorecard or explainer is given")

    applicant = row[feature_columns]
    score = float(scorecard.score(row)[0]) if scorecard is not None else None

    scorecard_codes = (
        scorecard_reason_codes(scorecard, row) if scorecard is not None else []
    )
    shap_codes = shap_reason_codes(explainer, row) if explainer is not None else []

    record = explain_decision(
        applicant,
        probability=float(probabilities[chosen]),
        threshold=threshold,
        reason_codes=scorecard_codes or shap_codes,
        score=score,
        exposure=float(row[exposure_column].iloc[0]) if exposure_column in row.columns else None,
        model_version=model_version,
    )
    record["row_index"] = chosen
    record["application_id"] = _jsonable(
        row["application_id"].iloc[0] if "application_id" in row.columns else chosen
    )
    record["reason_codes_scorecard"] = [c.to_dict() for c in scorecard_codes]
    record["reason_codes_shap"] = [c.to_dict() for c in shap_codes]
    record["adverse_action_notice"] = format_adverse_action_notice(record)
    return record
