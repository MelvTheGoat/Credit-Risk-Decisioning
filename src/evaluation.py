"""Evaluation under class imbalance, without the usual self-deception.

Accuracy is meaningless here — state it plainly
-----------------------------------------------
Defaults are the minority class. On the synthetic book roughly 13% of approved
accounts default; on UCI it is 22%. A model that declines nobody and predicts
"will repay" for every applicant scores 87% accuracy on the first and 78% on
the second, and is worth exactly nothing. Any accuracy figure in a credit model
report should be read as a red flag about the report, not a result.

:func:`discrimination_report` therefore prints accuracy **next to the
majority-class baseline** so the comparison is unavoidable, and the metrics
that matter carry the weight:

* **PR-AUC** (average precision) — sensitive to performance on the minority
  class, unlike ROC-AUC which is buoyed by the vast majority of easy negatives.
* **Precision at fixed review capacity** — the operationally real question. A
  review team can work a fixed number of files per day. Precision in the top 1%,
  5% and 10% of risk is what determines whether that team's day is well spent.
  A model can only be deployed at a capacity the business actually has.
* **KS statistic** — the maximum separation between the cumulative good and bad
  distributions. The traditional credit discrimination measure, still the first
  number a credit risk committee asks for.
* **Gini** — ``2 x AUC - 1``, the same information as AUC on the scale the
  industry quotes.

Resampling
----------
:func:`resampling_experiment` tests whether SMOTE and friends actually help.
The received wisdom is that they trade calibration for ranking. On this book
they do not even manage the trade — see ``README.md`` for the table:

* **Class weighting** leaves ranking untouched (AUC 0.8190 against 0.8193
  unweighted) and inflates mean predicted PD from 0.14 to 0.42 against an
  observed 0.17. ECE goes from 0.026 to 0.253.
* **SMOTE** *loses* ranking as well — AUC 0.819 to 0.760, PR-AUC 0.517 to
  0.391 — and is just as badly miscalibrated. Interpolating between minority
  neighbours in a space of binned credit-bureau variables manufactures
  applicants who do not exist.
* **Random undersampling** loses ranking and calibration by throwing away most
  of the majority class.

The mechanism is not subtle: resampling changes the base rate the model is
fitting to, so every probability it emits inherits the distortion. In a system
whose output feeds ``PD x LGD x EAD``, that is fatal, and it is completely
invisible if you only look at AUC.

The imbalance here is real but mild — roughly one bad in eight. Resampling is
a treatment for a disease this dataset does not have.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    roc_auc_score,
)

from src.calibration import (
    brier_score,
    calibration_slope_intercept,
    expected_calibration_error,
)

logger = logging.getLogger(__name__)

DEFAULT_CAPACITIES: tuple[float, ...] = (0.01, 0.05, 0.10)


def ks_statistic(y: np.ndarray, p: np.ndarray) -> float:
    """Kolmogorov-Smirnov separation between good and bad score distributions.

    Args:
        y: Binary default flag, 1 = bad.
        p: Predicted probabilities or any monotone risk score.

    Returns:
        Maximum absolute difference between the cumulative distributions.
        Roughly: below 0.20 is weak, 0.30-0.50 is a healthy application
        scorecard, above 0.70 usually means a leaked feature rather than a
        brilliant model.
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    if len(np.unique(y)) < 2:
        return float("nan")

    order = np.argsort(p)
    y_sorted = y[order]
    cumulative_bad = np.cumsum(y_sorted) / max(int(y_sorted.sum()), 1)
    cumulative_good = np.cumsum(1 - y_sorted) / max(int((1 - y_sorted).sum()), 1)
    return float(np.max(np.abs(cumulative_good - cumulative_bad)))


def precision_at_capacity(
    y: np.ndarray, p: np.ndarray, capacities: tuple[float, ...] = DEFAULT_CAPACITIES
) -> pd.DataFrame:
    """Precision, recall and lift within the riskiest ``k`` fraction of applicants.

    This is the metric that maps onto an actual operating constraint. "Our
    manual review team can look at 5% of applications" is a budget, and this
    table says what that budget buys.

    Args:
        y: Binary default flag.
        p: Predicted probabilities.
        capacities: Fractions of the population to review, riskiest first.

    Returns:
        One row per capacity with the count reviewed, precision (share of
        reviewed files that are genuine bads), recall (share of all bads
        caught), and lift over the base rate.
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    n = len(y)
    base_rate = float(y.mean()) if n else float("nan")
    order = np.argsort(-p)

    rows = []
    for capacity in capacities:
        k = max(int(round(capacity * n)), 1)
        selected = y[order[:k]]
        precision = float(selected.mean())
        rows.append(
            {
                "capacity": capacity,
                "n_reviewed": k,
                "precision": precision,
                "recall": float(selected.sum() / max(int(y.sum()), 1)),
                "lift": precision / base_rate if base_rate > 0 else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def gains_table(y: np.ndarray, p: np.ndarray, n_bands: int = 10) -> pd.DataFrame:
    """Decile gains table, riskiest band first.

    The format a credit risk committee expects: how bads concentrate as you move
    down the score. A healthy card shows a monotone bad rate across bands.

    Args:
        y: Binary default flag.
        p: Predicted probabilities.
        n_bands: Number of equal-mass bands.

    Returns:
        Frame with per-band counts, bad rate, cumulative bad capture and lift.
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    order = np.argsort(-p)
    y_sorted, p_sorted = y[order], p[order]

    bands = np.array_split(np.arange(len(y_sorted)), n_bands)
    total_bad = max(int(y_sorted.sum()), 1)
    base_rate = float(y.mean())

    rows = []
    cumulative_bad = 0
    for i, index in enumerate(bands, start=1):
        if len(index) == 0:
            continue
        band_y = y_sorted[index]
        cumulative_bad += int(band_y.sum())
        rows.append(
            {
                "band": i,
                "n": len(index),
                "mean_predicted": float(p_sorted[index].mean()),
                "bad_rate": float(band_y.mean()),
                "n_bad": int(band_y.sum()),
                "cumulative_bad_capture": cumulative_bad / total_bad,
                "lift": float(band_y.mean()) / base_rate if base_rate > 0 else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def discrimination_report(
    y: np.ndarray,
    p: np.ndarray,
    label: str = "",
    capacities: tuple[float, ...] = DEFAULT_CAPACITIES,
) -> dict[str, float | str]:
    """Headline discrimination and calibration metrics for one model.

    Accuracy is included **only** alongside ``accuracy_majority_baseline``, so
    that the two are always read together. If they are close, accuracy is
    telling you about the base rate rather than the model.

    Args:
        y: Binary default flag.
        p: Predicted probabilities.
        label: Identifier carried into the output.
        capacities: Review capacities for the precision-at-k columns.

    Returns:
        Mapping of metric name to value.
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    base_rate = float(y.mean())

    auc = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
    intercept, slope = calibration_slope_intercept(y, p)

    report: dict[str, float | str] = {
        "label": label,
        "n": float(len(y)),
        "base_rate": base_rate,
        "roc_auc": auc,
        "gini": 2.0 * auc - 1.0,
        "pr_auc": float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
        "pr_auc_baseline": base_rate,
        "ks": ks_statistic(y, p),
        "brier": brier_score(y, p),
        "ece": expected_calibration_error(y, p),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        # Present strictly as a foil. See the module docstring.
        "accuracy_at_0p5": float(accuracy_score(y, (p >= 0.5).astype(int))),
        "accuracy_majority_baseline": max(base_rate, 1.0 - base_rate),
    }

    capacity_table = precision_at_capacity(y, p, capacities)
    for _, row in capacity_table.iterrows():
        tag = f"{row['capacity']:.0%}".replace("%", "pct")
        report[f"precision_at_{tag}"] = float(row["precision"])
        report[f"recall_at_{tag}"] = float(row["recall"])

    return report


def evaluate_models(
    predictions: dict[str, np.ndarray],
    y: np.ndarray,
    capacities: tuple[float, ...] = DEFAULT_CAPACITIES,
) -> pd.DataFrame:
    """Build a comparison table across models.

    Args:
        predictions: Mapping from model name to predicted probabilities.
        y: Binary default flag.
        capacities: Review capacities for precision-at-k.

    Returns:
        One row per model, sorted by PR-AUC descending.
    """
    rows = [
        discrimination_report(y, p, label=name, capacities=capacities)
        for name, p in predictions.items()
    ]
    return pd.DataFrame(rows).sort_values("pr_auc", ascending=False, ignore_index=True)


# --------------------------------------------------------------------------- #
# Resampling
# --------------------------------------------------------------------------- #


def resampling_experiment(
    model_factory: object,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_eval: pd.DataFrame,
    y_eval: np.ndarray,
    random_state: int = 0,
) -> pd.DataFrame:
    """Test whether resampling helps once calibration is accounted for.

    Compares four training regimes on identical evaluation data:

    * ``none`` — train on the data as it is.
    * ``class_weight`` — reweight rather than resample. Changes the loss, not
      the sample, and is the cheapest of the four.
    * ``smote`` — synthesise minority examples by interpolating between
      neighbours.
    * ``random_undersample`` — discard majority rows.

    Read the resulting table across, not down. Compare the ranking columns
    (ROC-AUC, PR-AUC, KS) against the calibration columns (mean predicted PD,
    bias, ECE, Brier) for each regime, and ask what was bought and what was
    paid for it.

    Refitting a calibrator afterwards can undo much of the calibration damage —
    which rather raises the question of what the resampling was for, since the
    ranking is what recalibration cannot restore.

    Args:
        model_factory: Zero-argument callable returning an unfitted model.
        X_train: Training features.
        y_train: Training default flag.
        X_eval: Evaluation features.
        y_eval: Evaluation default flag.
        random_state: Seed for the resamplers.

    Returns:
        One row per regime with ranking and calibration metrics side by side.
    """
    from imblearn.over_sampling import SMOTENC
    from imblearn.under_sampling import RandomUnderSampler

    y_train = np.asarray(y_train).astype(int)
    y_eval = np.asarray(y_eval).astype(int)
    rows = []

    def evaluate(name: str, p: np.ndarray) -> dict[str, float | str]:
        intercept, slope = calibration_slope_intercept(y_eval, p)
        return {
            "regime": name,
            "roc_auc": float(roc_auc_score(y_eval, p)),
            "pr_auc": float(average_precision_score(y_eval, p)),
            "ks": ks_statistic(y_eval, p),
            "observed_rate": float(y_eval.mean()),
            "mean_predicted": float(p.mean()),
            "calibration_bias": float(p.mean() - y_eval.mean()),
            "brier": brier_score(y_eval, p),
            "ece": expected_calibration_error(y_eval, p),
            "calibration_slope": slope,
            "calibration_intercept": intercept,
        }

    # --- baseline -----------------------------------------------------------
    baseline = model_factory()  # type: ignore[operator]
    baseline.fit(X_train, y_train)
    rows.append(evaluate("none", baseline.predict_proba(X_eval)))

    # --- class weighting ----------------------------------------------------
    # Reweight so the two classes carry equal total weight.
    positive_share = float(y_train.mean())
    weights = np.where(y_train == 1, 0.5 / positive_share, 0.5 / (1.0 - positive_share))
    weighted = model_factory()  # type: ignore[operator]
    weighted.fit(X_train, y_train, sample_weight=weights)
    rows.append(evaluate("class_weight", weighted.predict_proba(X_eval)))

    # --- SMOTE --------------------------------------------------------------
    # SMOTENC handles mixed numeric/categorical frames; plain SMOTE cannot.
    categorical_positions = [
        i for i, column in enumerate(X_train.columns) if not pd.api.types.is_numeric_dtype(X_train[column])
    ]
    try:
        if categorical_positions:
            sampler: object = SMOTENC(
                categorical_features=categorical_positions, random_state=random_state
            )
        else:
            from imblearn.over_sampling import SMOTE

            sampler = SMOTE(random_state=random_state)
        X_smote, y_smote = sampler.fit_resample(X_train, y_train)  # type: ignore[attr-defined]
        smote_model = model_factory()  # type: ignore[operator]
        smote_model.fit(X_smote, y_smote)
        rows.append(evaluate("smote", smote_model.predict_proba(X_eval)))
    except (ValueError, TypeError) as error:  # pragma: no cover - defensive
        logger.warning("SMOTE could not be applied: %s", error)

    # --- random undersampling ----------------------------------------------
    undersampler = RandomUnderSampler(random_state=random_state)
    X_under, y_under = undersampler.fit_resample(X_train, y_train)
    under_model = model_factory()  # type: ignore[operator]
    under_model.fit(X_under, y_under)
    rows.append(evaluate("random_undersample", under_model.predict_proba(X_eval)))

    return pd.DataFrame(rows)
