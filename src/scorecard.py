"""Weight-of-evidence binning and a points-based scorecard.

This is what banks actually deploy, and it is the benchmark every other model
in this repository has to beat by enough to justify its cost.

A scorecard is worth defending. It is monotone by construction, so it cannot
tell an applicant that paying down their balance made them riskier. Its points
are additive and auditable, so an adverse action notice writes itself. It
degrades gracefully under drift because coarse bins absorb small distribution
shifts that a tree ensemble will happily overfit. Supervisors understand it.
A 1-2 point AUC gain rarely pays for losing all of that.

Conventions used here
---------------------
* **Good** = did not default (``y == 0``); **bad** = defaulted (``y == 1``).
* ``WOE = ln(P(x in bin | good) / P(x in bin | bad))``, so **higher WOE means
  lower risk**. A logistic regression fitted to predict *bad* on these features
  therefore carries negative coefficients.
* Points rise as risk falls, scaled by points-to-double-the-odds (PDO).

Binning is monotonic by default: adjacent bins that break the dominant
risk trend are merged until the trend is clean. That costs a little
discrimination and buys stability and explicability, which is the trade a
credit risk function almost always wants to make.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.config import DEFAULT_SCALING, ScorecardScaling

logger = logging.getLogger(__name__)

MISSING_BIN_LABEL = "__missing__"
OTHER_BIN_LABEL = "__other__"

# Laplace-style smoothing applied to bin good/bad counts. Prevents infinite WOE
# in a bin that happens to contain no bads, which is common in the safest bin.
WOE_SMOOTHING = 0.5


@dataclass
class FeatureBinning:
    """Binning and WOE table for a single feature.

    Attributes:
        name: Feature name.
        kind: ``"numeric"`` or ``"categorical"``.
        edges: Bin edges for numeric features, ascending, with ``-inf`` and
            ``+inf`` at the ends. Empty for categorical features.
        category_map: Mapping from raw category to bin label, for categorical
            features. Empty for numeric features.
        woe: Mapping from bin label to WOE value.
        event_rate: Mapping from bin label to observed bad rate.
        count: Mapping from bin label to training row count.
        iv: Information value of the feature, summed over bins.
        iv_parts: Per-bin information value contributions.
    """

    name: str
    kind: str
    edges: list[float] = field(default_factory=list)
    category_map: dict[str, str] = field(default_factory=dict)
    woe: dict[str, float] = field(default_factory=dict)
    event_rate: dict[str, float] = field(default_factory=dict)
    count: dict[str, int] = field(default_factory=dict)
    iv: float = 0.0
    iv_parts: dict[str, float] = field(default_factory=dict, repr=False)

    def assign_bins(self, values: pd.Series) -> pd.Series:
        """Map raw feature values onto bin labels.

        Args:
            values: Raw feature column.

        Returns:
            Series of bin labels, aligned to ``values``.
        """
        if self.kind == "numeric":
            numeric = pd.to_numeric(values, errors="coerce")
            labels = pd.Series(
                pd.cut(numeric, bins=self.edges, labels=self._numeric_labels(), right=True),
                index=values.index,
            ).astype(object)
            return labels.where(numeric.notna(), MISSING_BIN_LABEL).fillna(MISSING_BIN_LABEL)

        as_text = values.astype(object).where(values.notna(), MISSING_BIN_LABEL).astype(str)
        # Unseen categories fall to the "other" bucket rather than exploding.
        return as_text.map(lambda v: self.category_map.get(v, OTHER_BIN_LABEL))

    def transform(self, values: pd.Series) -> pd.Series:
        """Convert raw feature values to WOE.

        Args:
            values: Raw feature column.

        Returns:
            WOE values. Bins never seen in training map to 0.0, which is the
            neutral value — it contributes nothing to the score.
        """
        bins = self.assign_bins(values)
        return bins.map(lambda b: self.woe.get(str(b), 0.0)).astype(float)

    def _numeric_labels(self) -> list[str]:
        """Human-readable interval labels for numeric bins."""
        return [f"({self.edges[i]:.4g}, {self.edges[i + 1]:.4g}]" for i in range(len(self.edges) - 1)]

    def to_frame(self) -> pd.DataFrame:
        """Return the WOE table as a dataframe for reporting."""
        rows = [
            {
                "feature": self.name,
                "bin": label,
                "count": self.count.get(label, 0),
                "event_rate": self.event_rate.get(label, float("nan")),
                "woe": woe_value,
                "iv_contribution": self.iv_parts.get(label, float("nan")),
            }
            for label, woe_value in self.woe.items()
        ]
        return pd.DataFrame(rows)


def _compute_woe_table(
    bin_labels: pd.Series,
    y: np.ndarray,
    ordered_labels: list[str],
) -> tuple[dict[str, float], dict[str, float], dict[str, int], dict[str, float], float]:
    """Compute WOE, event rates, counts and IV for a set of bins.

    Args:
        bin_labels: Bin assignment per row.
        y: Binary outcome, 1 = bad.
        ordered_labels: Bin labels in the order they should be reported.

    Returns:
        Tuple of ``(woe, event_rate, count, iv_parts, total_iv)``.
    """
    total_bad = float(y.sum())
    total_good = float(len(y) - y.sum())

    woe: dict[str, float] = {}
    event_rate: dict[str, float] = {}
    count: dict[str, int] = {}
    iv_parts: dict[str, float] = {}

    labels_array = bin_labels.to_numpy()
    for label in ordered_labels:
        mask = labels_array == label
        n = int(mask.sum())
        if n == 0:
            continue
        bad = float(y[mask].sum())
        good = float(n - bad)

        # Smoothing keeps WOE finite when a bin is all-good or all-bad.
        pct_good = (good + WOE_SMOOTHING) / (total_good + WOE_SMOOTHING * len(ordered_labels))
        pct_bad = (bad + WOE_SMOOTHING) / (total_bad + WOE_SMOOTHING * len(ordered_labels))

        woe[label] = float(np.log(pct_good / pct_bad))
        event_rate[label] = bad / n
        count[label] = n
        iv_parts[label] = float((pct_good - pct_bad) * woe[label])

    return woe, event_rate, count, iv_parts, float(sum(iv_parts.values()))


def _initial_numeric_edges(x: np.ndarray, max_bins: int) -> list[float]:
    """Quantile-based starting edges, de-duplicated for spiky distributions."""
    quantiles = np.linspace(0.0, 1.0, max_bins + 1)[1:-1]
    inner = np.unique(np.quantile(x, quantiles))
    return [-np.inf, *[float(e) for e in inner], np.inf]


def _merge_small_and_monotonic(
    x: np.ndarray,
    y: np.ndarray,
    edges: list[float],
    min_bin_fraction: float,
    enforce_monotonic: bool,
) -> list[float]:
    """Merge bins until every bin is large enough and the WOE trend is monotone.

    Two passes, in this order:

    1. **Size.** Any bin holding fewer than ``min_bin_fraction`` of rows, or
       containing only goods or only bads, is merged into its smaller
       neighbour. Tiny bins produce WOE values that are pure noise and will not
       reproduce next quarter.
    2. **Monotonicity.** The dominant direction of the bad rate is taken from
       the sign of the correlation between bin index and bad rate. Adjacent
       pairs that violate that direction are merged worst-first until the trend
       is clean.

    Args:
        x: Non-missing numeric values.
        y: Binary outcome aligned to ``x``.
        edges: Starting bin edges.
        min_bin_fraction: Minimum share of rows per bin.
        enforce_monotonic: Whether to run the monotonicity pass.

    Returns:
        Final edges.
    """
    min_count = max(int(min_bin_fraction * len(x)), 25)

    def bin_stats(current: list[float]) -> tuple[np.ndarray, np.ndarray]:
        idx = np.digitize(x, current[1:-1], right=True)
        n_bins = len(current) - 1
        counts = np.array([int((idx == b).sum()) for b in range(n_bins)])
        rates = np.array(
            [float(y[idx == b].mean()) if counts[b] > 0 else np.nan for b in range(n_bins)]
        )
        return counts, rates

    # ---- pass 1: size and purity ----
    changed = True
    while changed and len(edges) > 3:
        changed = False
        counts, rates = bin_stats(edges)
        for b in range(len(counts)):
            too_small = counts[b] < min_count
            degenerate = counts[b] > 0 and (rates[b] in (0.0, 1.0))
            if too_small or degenerate:
                # Drop the edge that fuses this bin with its smaller neighbour.
                if b == 0:
                    drop = 1
                elif b == len(counts) - 1:
                    drop = len(edges) - 2
                else:
                    drop = b if counts[b - 1] <= counts[b + 1] else b + 1
                edges = edges[:drop] + edges[drop + 1 :]
                changed = True
                break

    if not enforce_monotonic:
        return edges

    # ---- pass 2: monotonicity ----
    counts, rates = bin_stats(edges)
    valid = ~np.isnan(rates)
    if valid.sum() < 3:
        return edges
    direction = float(np.sign(np.corrcoef(np.arange(len(rates))[valid], rates[valid])[0, 1]))
    if direction == 0.0 or np.isnan(direction):
        direction = 1.0

    while len(edges) > 3:
        _, rates = bin_stats(edges)
        diffs = np.diff(rates) * direction
        violations = np.where(diffs < 0)[0]
        if len(violations) == 0:
            break
        # Merge the worst violation first: the adjacent pair most out of order.
        worst = int(violations[np.argmin(diffs[violations])])
        edges = edges[: worst + 1] + edges[worst + 2 :]

    return edges


def fit_numeric_binning(
    x: pd.Series,
    y: np.ndarray,
    max_bins: int = 8,
    min_bin_fraction: float = 0.05,
    enforce_monotonic: bool = True,
) -> FeatureBinning:
    """Fit a monotonic WOE binning for one numeric feature.

    Args:
        x: Raw numeric column.
        y: Binary outcome, 1 = bad.
        max_bins: Maximum number of bins before merging.
        min_bin_fraction: Minimum share of rows a bin must hold.
        enforce_monotonic: Enforce a monotone bad-rate trend across bins.

    Returns:
        Fitted :class:`FeatureBinning`.
    """
    numeric = pd.to_numeric(x, errors="coerce")
    observed = numeric.notna().to_numpy()
    x_obs = numeric.to_numpy(dtype=float)[observed]
    y_obs = y[observed]

    if len(np.unique(x_obs)) < 2:
        edges = [-np.inf, np.inf]
    else:
        edges = _initial_numeric_edges(x_obs, max_bins)
        edges = _merge_small_and_monotonic(
            x_obs, y_obs, edges, min_bin_fraction, enforce_monotonic
        )

    binning = FeatureBinning(name=str(x.name), kind="numeric", edges=edges)
    labels = binning.assign_bins(x)
    ordered = [*binning._numeric_labels(), MISSING_BIN_LABEL]
    woe, rate, count, iv_parts, iv = _compute_woe_table(labels, y, ordered)
    binning.woe, binning.event_rate, binning.count = woe, rate, count
    binning.iv_parts, binning.iv = iv_parts, iv
    return binning


def fit_categorical_binning(
    x: pd.Series,
    y: np.ndarray,
    min_bin_fraction: float = 0.05,
) -> FeatureBinning:
    """Fit a WOE binning for one categorical feature.

    Levels holding less than ``min_bin_fraction`` of rows are pooled into a
    single ``__other__`` bucket, which also receives any category not seen
    during training.

    Args:
        x: Raw categorical column.
        y: Binary outcome, 1 = bad.
        min_bin_fraction: Minimum share of rows a level must hold to stand alone.

    Returns:
        Fitted :class:`FeatureBinning`.
    """
    as_text = x.astype(object).where(x.notna(), MISSING_BIN_LABEL).astype(str)
    counts = as_text.value_counts()
    min_count = max(int(min_bin_fraction * len(as_text)), 25)

    category_map = {
        str(level): (str(level) if n >= min_count else OTHER_BIN_LABEL)
        for level, n in counts.items()
    }
    binning = FeatureBinning(name=str(x.name), kind="categorical", category_map=category_map)

    labels = binning.assign_bins(x)
    ordered = sorted(set(category_map.values()) | {OTHER_BIN_LABEL})
    woe, rate, count, iv_parts, iv = _compute_woe_table(labels, y, ordered)
    binning.woe, binning.event_rate, binning.count = woe, rate, count
    binning.iv_parts, binning.iv = iv_parts, iv
    return binning


class WOEEncoder:
    """Fit and apply WOE transformations across a feature set."""

    def __init__(
        self,
        numeric_features: list[str],
        categorical_features: list[str],
        max_bins: int = 8,
        min_bin_fraction: float = 0.05,
        enforce_monotonic: bool = True,
        min_iv: float = 0.02,
    ) -> None:
        """Initialise the encoder.

        Args:
            numeric_features: Numeric column names.
            categorical_features: Categorical column names.
            max_bins: Maximum starting bins for numeric features.
            min_bin_fraction: Minimum share of rows per bin.
            enforce_monotonic: Enforce monotone bad-rate trends on numerics.
            min_iv: Features with information value below this are dropped.
                0.02 is the conventional "no useful signal" floor. Keeping
                noise features in a scorecard destabilises it without adding
                discrimination.
        """
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.max_bins = max_bins
        self.min_bin_fraction = min_bin_fraction
        self.enforce_monotonic = enforce_monotonic
        self.min_iv = min_iv
        self.binnings: dict[str, FeatureBinning] = {}
        self.selected_features: list[str] = []

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> WOEEncoder:
        """Fit binnings for every feature and drop those below the IV floor.

        Args:
            X: Feature frame.
            y: Binary outcome, 1 = bad.

        Returns:
            Self.
        """
        y = np.asarray(y).astype(int)
        for name in self.numeric_features:
            self.binnings[name] = fit_numeric_binning(
                X[name], y, self.max_bins, self.min_bin_fraction, self.enforce_monotonic
            )
        for name in self.categorical_features:
            self.binnings[name] = fit_categorical_binning(X[name], y, self.min_bin_fraction)

        self.selected_features = [
            name for name, binning in self.binnings.items() if binning.iv >= self.min_iv
        ]
        dropped = sorted(set(self.binnings) - set(self.selected_features))
        if dropped:
            logger.info("Dropped %d features below IV floor %.3f: %s", len(dropped), self.min_iv, dropped)
        if not self.selected_features:
            # Never return an empty design matrix; fall back to the best feature.
            self.selected_features = [max(self.binnings, key=lambda k: self.binnings[k].iv)]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features to WOE space.

        Args:
            X: Feature frame.

        Returns:
            Frame of WOE values, columns suffixed ``_woe``.
        """
        return pd.DataFrame(
            {f"{name}_woe": self.binnings[name].transform(X[name]) for name in self.selected_features},
            index=X.index,
        )

    def fit_transform(self, X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
        """Fit then transform. See :meth:`fit` and :meth:`transform`."""
        return self.fit(X, y).transform(X)

    def information_values(self) -> pd.DataFrame:
        """Information value per feature, descending, with the usual guidance.

        Returns:
            Frame of feature, IV, a strength label, and whether it was kept.
        """
        rows = []
        for name, binning in sorted(self.binnings.items(), key=lambda kv: -kv[1].iv):
            rows.append(
                {
                    "feature": name,
                    "iv": binning.iv,
                    "strength": _iv_strength(binning.iv),
                    "n_bins": len(binning.woe),
                    "selected": name in self.selected_features,
                }
            )
        return pd.DataFrame(rows)

    def woe_tables(self) -> pd.DataFrame:
        """Concatenated WOE tables for every fitted feature."""
        return pd.concat([b.to_frame() for b in self.binnings.values()], ignore_index=True)


def _iv_strength(iv: float) -> str:
    """Label an information value using the conventional bands."""
    if iv < 0.02:
        return "unpredictive"
    if iv < 0.10:
        return "weak"
    if iv < 0.30:
        return "medium"
    if iv < 0.50:
        return "strong"
    return "suspiciously strong"


class Scorecard:
    """WOE binning plus logistic regression, presented as a points scorecard."""

    def __init__(
        self,
        numeric_features: list[str],
        categorical_features: list[str],
        scaling: ScorecardScaling = DEFAULT_SCALING,
        c: float = 1.0,
        max_bins: int = 8,
        min_bin_fraction: float = 0.05,
        enforce_monotonic: bool = True,
        min_iv: float = 0.02,
        drop_sign_flips: bool = True,
    ) -> None:
        """Initialise the scorecard.

        Args:
            numeric_features: Numeric column names.
            categorical_features: Categorical column names.
            scaling: Points scaling convention.
            c: Inverse L2 regularisation strength for the logistic regression.
            max_bins: Maximum starting bins for numeric features.
            min_bin_fraction: Minimum share of rows per bin.
            enforce_monotonic: Enforce monotone bad-rate trends on numerics.
            min_iv: Information value floor for feature selection.
            drop_sign_flips: Iteratively remove features whose fitted
                coefficient is positive. Since higher WOE means lower risk,
                a positive coefficient says the multivariate fit reversed the
                feature's own univariate trend — almost always an artefact of
                correlation with a stronger feature. Such a feature makes the
                scorecard indefensible: it would award *more* points for
                behaviour the bank calls riskier, and no adverse action notice
                derived from it would survive challenge. Standard scorecard
                practice is to drop it, which is what this does.
        """
        self.scaling = scaling
        self.drop_sign_flips = drop_sign_flips
        self.dropped_for_sign_flip: list[str] = []
        self.encoder = WOEEncoder(
            numeric_features,
            categorical_features,
            max_bins=max_bins,
            min_bin_fraction=min_bin_fraction,
            enforce_monotonic=enforce_monotonic,
            min_iv=min_iv,
        )
        self.model = LogisticRegression(C=c, max_iter=2000, solver="lbfgs")
        self.feature_names_: list[str] = []

    @property
    def factor(self) -> float:
        """Points per unit log-odds."""
        return self.scaling.pdo / np.log(2.0)

    @property
    def offset(self) -> float:
        """Points offset anchoring the base score to the base odds."""
        return self.scaling.base_score - self.factor * np.log(self.scaling.base_odds)

    def fit(self, X: pd.DataFrame, y: np.ndarray, sample_weight: np.ndarray | None = None) -> Scorecard:
        """Fit binnings and the logistic regression.

        Args:
            X: Feature frame.
            y: Binary outcome, 1 = default.
            sample_weight: Optional per-row weights. Used by the fuzzy
                augmentation reject inference method.

        Returns:
            Self.
        """
        y = np.asarray(y).astype(int)
        woe_frame = self.encoder.fit_transform(X, y)
        self.feature_names_ = list(woe_frame.columns)
        self.model.fit(woe_frame.to_numpy(), y, sample_weight=sample_weight)

        if self.drop_sign_flips:
            # Refit after each removal: dropping one feature can resolve or
            # create a flip elsewhere, so a single pass is not enough. Always
            # keep at least two features so the scorecard stays a scorecard.
            while len(self.feature_names_) > 2:
                coefficients = self.model.coef_[0]
                positive = np.where(coefficients > 0)[0]
                if len(positive) == 0:
                    break
                worst = int(positive[np.argmax(coefficients[positive])])
                dropped = self.feature_names_[worst][: -len("_woe")]
                self.dropped_for_sign_flip.append(dropped)
                logger.info(
                    "Dropped '%s' from scorecard: coefficient %+.3f reverses its univariate trend",
                    dropped,
                    float(coefficients[worst]),
                )
                self.feature_names_ = [n for i, n in enumerate(self.feature_names_) if i != worst]
                self.model.fit(woe_frame[self.feature_names_].to_numpy(), y, sample_weight=sample_weight)

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probability of default.

        Args:
            X: Feature frame.

        Returns:
            Array of P(default), one per row.
        """
        woe_frame = self.encoder.transform(X)[self.feature_names_]
        return self.model.predict_proba(woe_frame.to_numpy())[:, 1]

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """Compute scorecard points. Higher points mean lower risk.

        Args:
            X: Feature frame.

        Returns:
            Array of points.
        """
        woe_frame = self.encoder.transform(X)[self.feature_names_]
        log_odds_bad = self.model.decision_function(woe_frame.to_numpy())
        # score = offset + factor * ln(good odds) = offset - factor * ln(bad odds)
        return self.offset - self.factor * log_odds_bad

    def points_table(self) -> pd.DataFrame:
        """The deployable scorecard: points awarded per feature bin.

        This table *is* the model. It can be handed to an operations team, put
        in a policy document, and applied with a spreadsheet. Reproducing the
        score by summing these points is a test in the suite.

        Returns:
            Frame of feature, bin, count, event rate, WOE and points.
        """
        n_features = len(self.feature_names_)
        intercept = float(self.model.intercept_[0])
        rows: list[dict[str, Any]] = []

        for column, coefficient in zip(self.feature_names_, self.model.coef_[0], strict=True):
            feature = column[: -len("_woe")]
            binning = self.encoder.binnings[feature]
            for label, woe_value in binning.woe.items():
                points = -self.factor * (coefficient * woe_value + intercept / n_features)
                points += self.offset / n_features
                rows.append(
                    {
                        "feature": feature,
                        "bin": label,
                        "count": binning.count.get(label, 0),
                        "event_rate": binning.event_rate.get(label, float("nan")),
                        "woe": woe_value,
                        "coefficient": float(coefficient),
                        "points": float(points),
                    }
                )
        return pd.DataFrame(rows)

    def coefficient_table(self) -> pd.DataFrame:
        """Fitted coefficients alongside each feature's information value.

        A well-behaved scorecard has all-negative coefficients on WOE, because
        higher WOE means safer. A positive coefficient is a red flag: it means
        the multivariate fit reversed the univariate trend, usually through
        correlation with another feature, and the bin should be revisited.

        Returns:
            Frame of feature, coefficient, IV and a sign-sanity flag.
        """
        rows = []
        for column, coefficient in zip(self.feature_names_, self.model.coef_[0], strict=True):
            feature = column[: -len("_woe")]
            rows.append(
                {
                    "feature": feature,
                    "coefficient": float(coefficient),
                    "iv": self.encoder.binnings[feature].iv,
                    "sign_as_expected": bool(coefficient <= 0),
                }
            )
        return pd.DataFrame(rows).sort_values("iv", ascending=False, ignore_index=True)
