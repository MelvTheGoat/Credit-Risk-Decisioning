"""Reject inference: modelling the applicants the previous policy never let in.

The problem
-----------
A lender only observes repayment for applicants it **approved**. Everyone else
has features but no outcome. Train on approved-only data and the model learns
``P(default | features, approved by the old policy)`` while the business
deploys it to answer ``P(default | features)``. Those are different quantities
whenever the old policy used information the new model cannot see — judgemental
overrides, unrecorded conversations, branch discretion — which is always.

The failure is not random. The old policy declined the applicants it thought
were worst, so the training sample is *positively selected*, and the model
inherits a systematically optimistic view of exactly the region near the
cutoff where its decisions actually matter. It then approves into that region
with unwarranted confidence.

On this synthetic book the size of the problem is known exactly: approved
accounts default at 12.1% and declined applicants would have defaulted at
38.3%. A model trained on the approved population and applied to everyone is
predicting into a population three times riskier than the one it learned from.

Why this module can be validated at all
---------------------------------------
On a real book, any claim that a correction "removes bias" is unfalsifiable —
the counterfactual outcome for a declined applicant does not exist. The
synthetic generator retains ``default_true`` for **every** applicant including
the declined, so each method can be scored against the truth it is trying to
recover, and against an **oracle** model trained on labels no real lender could
ever have.

Methods implemented
-------------------
* :func:`fit_approved_only` — the uncorrected baseline.
* :func:`fuzzy_augmentation` — each reject enters training twice, as a good and
  as a bad, weighted by its inferred default probability.
* :func:`parcelling` — rejects are binned by score and assigned labels at the
  approved bad rate in their band, scaled up by an assumed uplift factor.
* :func:`heckman_correction` — a two-step selection model using an exclusion
  restriction, correcting the outcome equation with the inverse Mills ratio.
* :func:`oracle_model` — trained on true labels for everyone. Impossible in
  practice; included as the upper bound that says how much of the gap any
  method could conceivably have closed.

The uplift assumption
---------------------
Fuzzy augmentation and parcelling both need an assumption about how much worse
a reject is than an approved applicant with the same score. Industry guidance
is somewhere between 2x and 4x, and on a real book **the assumption cannot be
checked** — it is the same missing counterfactual all over again. Here it can
be: :func:`measure_true_uplift` reports the real value, so the sensitivity of
each method to getting the assumption wrong is measurable rather than asserted.

What the experiment actually found
----------------------------------
The study is run under **both selection regimes** (see
:func:`src.simulate.mnar_config`), and they give opposite answers:

* **Selection on observables (MAR).** The baseline was already nearly
  unbiased — the oracle only moved reject bias from -0.061 to -0.043, so there
  was almost nothing to win. Fuzzy augmentation and parcelling, applying an
  assumed 2.5x uplift against a true 1.28x, **overcorrected and made things
  worse**, ending at +0.091 and +0.098. They added more bias than they removed.

* **Selection on unobservables (MNAR).** The baseline was badly biased on
  rejects (-0.171). Here the same methods, with a true uplift of 2.12 against
  the assumed 2.5, **worked**: bias fell to -0.027 and -0.006 and Brier score
  on rejects improved by about 15%.

Two conclusions follow, and neither is comfortable:

1. **Reject inference is not a free improvement.** It helps when selection
   depends on things the model cannot see, and it actively damages the model
   when selection does not. Applying it reflexively is a coin flip.
2. **You cannot tell which regime you are in from a real book**, because the
   evidence that would settle it is the missing counterfactual itself.

The Heckman correction removed essentially none of the bias in either regime
(5% at best) **despite being handed a clean, valid exclusion restriction that
was constructed for it**. A real deployment would not have one. That is a
strong argument against the method in this setting, not against this
implementation of it.

A trap worth knowing about: weak models manufacture apparent uplift
---------------------------------------------------------------------
Measured on the *same* MAR book, the true reject-to-approve uplift depends on
how much training data the scoring model had:

=================  ==============  ====================
Training rows      Measured uplift Baseline reject bias
=================  ==============  ====================
4,000              2.25            -0.120
10,000             1.21            -0.059
20,000             1.09            -0.051
=================  ==============  ====================

Nothing about the selection mechanism changed. What changed is the model. A
weak model leaves risk heterogeneity *inside* each score band, and because the
legacy policy sorted on that heterogeneity, rejects within a band genuinely
are worse — so the uplift is real, measurable, and entirely an artefact of the
scoring model rather than of selection on unobservables. As the model improves
it absorbs what the policy knew, and the uplift collapses toward 1.

The practical consequence is uncomfortable: **measuring an uplift of 2x on
your own book is not evidence that you need reject inference.** It is equally
consistent with your model being underpowered, in which case the fix is a
better model and more data, and applying reject inference on top will
overcorrect. The two explanations are not distinguishable from the uplift
figure alone.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from src.calibration import brier_score, expected_calibration_error
from src.config import RANDOM_SEED

logger = logging.getLogger(__name__)

# Assumed ratio of reject bad rate to approved bad rate within a score band.
# 2.5 sits in the middle of the usual 2x-4x industry range.
DEFAULT_UPLIFT = 2.5

ModelFactory = Callable[[], Any]


def fit_approved_only(
    model_factory: ModelFactory,
    X_approved: pd.DataFrame,
    y_approved: np.ndarray,
) -> Any:
    """Train on approved applicants only — the uncorrected baseline.

    Args:
        model_factory: Zero-argument callable returning an unfitted model.
        X_approved: Features of approved applicants.
        y_approved: Observed default flag for approved applicants.

    Returns:
        The fitted model.
    """
    model = model_factory()
    model.fit(X_approved, np.asarray(y_approved).astype(int))
    return model


def oracle_model(
    model_factory: ModelFactory,
    X_all: pd.DataFrame,
    y_true_all: np.ndarray,
) -> Any:
    """Train on true outcomes for the whole population.

    Only possible on synthetic data. This is the ceiling: it says how much of
    the selection bias *any* correction could remove, so that a method closing
    30% of the gap can be recognised as closing 30% rather than reported as a
    success in the abstract.

    Args:
        model_factory: Zero-argument callable returning an unfitted model.
        X_all: Features of every applicant.
        y_true_all: True default flag for every applicant.

    Returns:
        The fitted model.
    """
    model = model_factory()
    model.fit(X_all, np.asarray(y_true_all).astype(int))
    return model


def fuzzy_augmentation(
    model_factory: ModelFactory,
    X_approved: pd.DataFrame,
    y_approved: np.ndarray,
    X_rejected: pd.DataFrame,
    uplift: float = DEFAULT_UPLIFT,
    base_model: Any | None = None,
) -> Any:
    """Fuzzy augmentation: each reject enters training as both a good and a bad.

    A baseline model scores the rejects. Each reject is then duplicated: one
    copy labelled bad with weight ``p_uplifted``, one labelled good with weight
    ``1 - p_uplifted``. The model is refitted on approved rows plus the weighted
    duplicates.

    The appeal over parcelling is that no labels are invented — the uncertainty
    is carried explicitly as weight rather than resolved by a coin flip. The
    weakness is circularity: the weights come from the very model whose bias is
    the problem, so the correction can only redistribute information the biased
    model already had. The uplift factor is the only genuinely new information
    entering the procedure, which is why the method is so sensitive to it.

    Args:
        model_factory: Zero-argument callable returning an unfitted model.
        X_approved: Features of approved applicants.
        y_approved: Observed default flag for approved applicants.
        X_rejected: Features of declined applicants.
        uplift: Multiplier on the inferred reject default probability.
        base_model: Pre-fitted scoring model. Fitted from the approved data if
            not supplied.

    Returns:
        The refitted model.
    """
    y_approved = np.asarray(y_approved).astype(int)
    if base_model is None:
        base_model = fit_approved_only(model_factory, X_approved, y_approved)

    p_reject = np.clip(base_model.predict_proba(X_rejected) * uplift, 1e-4, 1.0 - 1e-4)

    X_augmented = pd.concat([X_approved, X_rejected, X_rejected], ignore_index=True)
    y_augmented = np.concatenate(
        [y_approved, np.ones(len(X_rejected), dtype=int), np.zeros(len(X_rejected), dtype=int)]
    )
    weights = np.concatenate([np.ones(len(X_approved)), p_reject, 1.0 - p_reject])

    model = model_factory()
    model.fit(X_augmented, y_augmented, sample_weight=weights)
    return model


def parcelling(
    model_factory: ModelFactory,
    X_approved: pd.DataFrame,
    y_approved: np.ndarray,
    X_rejected: pd.DataFrame,
    uplift: float = DEFAULT_UPLIFT,
    n_bands: int = 10,
    base_model: Any | None = None,
    random_state: int = RANDOM_SEED,
) -> Any:
    """Parcelling: assign rejects hard labels at an uplifted band bad rate.

    Rejects are scored and placed into score bands. Within each band the
    approved bad rate is measured, multiplied by ``uplift``, and used as the
    probability of stamping each reject in that band as a bad.

    More common in practice than fuzzy augmentation because the output is an
    ordinary labelled dataset that any downstream tool can consume. The cost is
    that it fabricates hard labels from a rate, adding sampling noise that
    fuzzy augmentation avoids, and the result depends on the random seed.

    Args:
        model_factory: Zero-argument callable returning an unfitted model.
        X_approved: Features of approved applicants.
        y_approved: Observed default flag for approved applicants.
        X_rejected: Features of declined applicants.
        uplift: Multiplier on each band's approved bad rate.
        n_bands: Number of score bands.
        base_model: Pre-fitted scoring model. Fitted from approved data if not
            supplied.
        random_state: Seed for the label draw.

    Returns:
        The refitted model.
    """
    rng = np.random.default_rng(random_state)
    y_approved = np.asarray(y_approved).astype(int)
    if base_model is None:
        base_model = fit_approved_only(model_factory, X_approved, y_approved)

    p_approved = base_model.predict_proba(X_approved)
    p_rejected = base_model.predict_proba(X_rejected)

    edges = np.unique(np.quantile(p_approved, np.linspace(0.0, 1.0, n_bands + 1)))
    approved_band = np.clip(np.digitize(p_approved, edges[1:-1], right=True), 0, len(edges) - 2)
    rejected_band = np.clip(np.digitize(p_rejected, edges[1:-1], right=True), 0, len(edges) - 2)

    inferred = np.zeros(len(X_rejected), dtype=int)
    for band in np.unique(rejected_band):
        in_band_approved = approved_band == band
        if not in_band_approved.any():
            continue
        band_rate = float(np.clip(y_approved[in_band_approved].mean() * uplift, 0.0, 1.0))
        mask = rejected_band == band
        inferred[mask] = (rng.random(int(mask.sum())) < band_rate).astype(int)

    X_combined = pd.concat([X_approved, X_rejected], ignore_index=True)
    y_combined = np.concatenate([y_approved, inferred])

    logger.info(
        "Parcelling inferred a %.1f%% bad rate across %d rejects (approved base %.1f%%)",
        100.0 * inferred.mean(),
        len(inferred),
        100.0 * y_approved.mean(),
    )

    model = model_factory()
    model.fit(X_combined, y_combined)
    return model


class HeckmanCorrectedModel:
    """Two-step selection correction with an inverse Mills ratio control.

    Step one fits a probit for *approval* on all applicants, using the model
    features plus an **exclusion restriction** — a variable that drives approval
    but has no causal effect on default. Here that is
    ``branch_capacity_index``: a busy branch approves less, but branch workload
    does not make a borrower repay.

    Step two fits the outcome model on approved applicants only, with the
    inverse Mills ratio ``phi(z'g) / Phi(z'g)`` added as a regressor. That term
    absorbs the correlation between selection and outcome, so the remaining
    coefficients estimate the unconditional relationship. At prediction time the
    Mills term is set to zero, which is what makes the output a population
    estimate rather than a selected-sample one.

    Two honest caveats:

    * Heckman assumes **jointly normal errors**. The DGP here is logistic, so
      the correction is already approximate before any estimation error.
    * The exclusion restriction is the whole game, and a clean one **rarely
      exists on a real book**. Without it, identification rests entirely on the
      non-linearity of the Mills ratio, which is weak and unstable. This
      generator was built to hand Heckman its best case; a real deployment
      would not get one, and results here should be read as an upper bound on
      how well the method can do.
    """

    def __init__(
        self,
        feature_columns: list[str],
        selection_instruments: list[str],
    ) -> None:
        """Initialise the corrected model.

        Args:
            feature_columns: Numeric outcome-equation regressors.
            selection_instruments: Columns entering the selection equation but
                excluded from the outcome equation.
        """
        self.feature_columns = feature_columns
        self.selection_instruments = selection_instruments
        self.selection_model: Any = None
        self.outcome_model: Any = None
        self.means_: pd.Series | None = None
        self.stds_: pd.Series | None = None

    def _design(self, X: pd.DataFrame, columns: list[str]) -> np.ndarray:
        """Standardise and add an intercept column."""
        assert self.means_ is not None and self.stds_ is not None
        standardised = (X[columns] - self.means_[columns]) / self.stds_[columns]
        return np.column_stack([np.ones(len(X)), standardised.to_numpy(dtype=float)])

    def fit(
        self,
        X_all: pd.DataFrame,
        approved: np.ndarray,
        y_approved: np.ndarray,
    ) -> HeckmanCorrectedModel:
        """Fit the selection and outcome equations.

        Args:
            X_all: Features for every applicant, including the instruments.
            approved: Binary approval flag for every applicant.
            y_approved: Observed default flag for approved applicants only,
                ordered to match ``X_all[approved == 1]``.

        Returns:
            Self.
        """
        import statsmodels.api as sm
        from scipy.stats import norm

        selection_columns = [*self.feature_columns, *self.selection_instruments]
        numeric = X_all[selection_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        self.means_ = numeric.mean()
        self.stds_ = numeric.std().replace(0.0, 1.0)

        approved = np.asarray(approved).astype(int)

        # --- step 1: probit for approval, on everyone -----------------------
        selection_design = self._design(numeric, selection_columns)
        self.selection_model = sm.Probit(approved, selection_design).fit(disp=0)
        linear_index = np.asarray(self.selection_model.predict(selection_design, linear=True))

        # Inverse Mills ratio for the selected (approved) sample.
        mills = norm.pdf(linear_index) / np.clip(norm.cdf(linear_index), 1e-8, None)

        # --- step 2: outcome equation on approved, with the Mills control ---
        outcome_design = np.column_stack(
            [self._design(numeric.loc[approved == 1], self.feature_columns), mills[approved == 1]]
        )
        self.outcome_model = sm.Logit(
            np.asarray(y_approved).astype(int), outcome_design
        ).fit(disp=0, maxiter=200)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict unconditional P(default) with the selection term switched off.

        Args:
            X: Features for the applicants to score.

        Returns:
            Array of P(default).
        """
        assert self.outcome_model is not None
        selection_columns = [*self.feature_columns, *self.selection_instruments]
        numeric = X[selection_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        # Mills term set to zero: we want the population relationship, not the
        # one conditional on having been approved.
        design = np.column_stack(
            [self._design(numeric, self.feature_columns), np.zeros(len(X))]
        )
        return np.asarray(self.outcome_model.predict(design), dtype=float)


def heckman_correction(
    X_all: pd.DataFrame,
    approved: np.ndarray,
    y_approved: np.ndarray,
    feature_columns: list[str],
    selection_instruments: list[str],
) -> HeckmanCorrectedModel:
    """Fit a Heckman-corrected outcome model. See :class:`HeckmanCorrectedModel`.

    Args:
        X_all: Features for every applicant, including instruments.
        approved: Binary approval flag for every applicant.
        y_approved: Observed default flag for approved applicants.
        feature_columns: Numeric outcome-equation regressors.
        selection_instruments: Selection-only columns.

    Returns:
        The fitted corrected model.
    """
    return HeckmanCorrectedModel(feature_columns, selection_instruments).fit(
        X_all, approved, y_approved
    )


# --------------------------------------------------------------------------- #
# Validation against ground truth
# --------------------------------------------------------------------------- #


def measure_true_uplift(
    scores: np.ndarray,
    approved: np.ndarray,
    y_true: np.ndarray,
    n_bands: int = 10,
) -> pd.DataFrame:
    """Measure the real reject-versus-approve bad rate ratio, by score band.

    This is the assumption that fuzzy augmentation and parcelling have to make
    blind on a real book. Measuring it here turns "we assumed 2.5x, as the
    literature suggests" into "we assumed 2.5x and the truth was X", which is
    the difference between a defensible method and a hopeful one.

    Args:
        scores: Model scores for every applicant.
        approved: Binary approval flag.
        y_true: True default flag for every applicant, including rejects.
        n_bands: Number of score bands.

    Returns:
        One row per band with approved and rejected bad rates and their ratio.
    """
    approved = np.asarray(approved).astype(bool)
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)

    edges = np.unique(np.quantile(scores, np.linspace(0.0, 1.0, n_bands + 1)))
    band = np.clip(np.digitize(scores, edges[1:-1], right=True), 0, len(edges) - 2)

    rows = []
    for b in np.unique(band):
        in_band = band == b
        approved_mask = in_band & approved
        rejected_mask = in_band & ~approved
        if approved_mask.sum() < 20 or rejected_mask.sum() < 20:
            continue
        approved_rate = float(y_true[approved_mask].mean())
        rejected_rate = float(y_true[rejected_mask].mean())
        rows.append(
            {
                "band": int(b),
                "n_approved": int(approved_mask.sum()),
                "n_rejected": int(rejected_mask.sum()),
                "bad_rate_approved": approved_rate,
                "bad_rate_rejected": rejected_rate,
                "uplift": rejected_rate / approved_rate if approved_rate > 0 else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def evaluate_on_population(
    p: np.ndarray,
    y_true: np.ndarray,
    approved: np.ndarray,
    label: str = "",
) -> dict[str, float | str]:
    """Score predictions against true outcomes for the whole population.

    The columns that matter are the ones restricted to **rejects**. A method
    can look fine overall simply because rejects are a minority, while still
    being badly wrong about exactly the population it was built to fix.

    Args:
        p: Predicted P(default) for every applicant.
        y_true: True default flag for every applicant.
        approved: Binary approval flag.
        label: Method identifier carried into the output.

    Returns:
        Mapping of metrics on the full population, on approved, and on rejects.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    p = np.asarray(p, dtype=float)
    y_true = np.asarray(y_true).astype(int)
    is_reject = ~np.asarray(approved).astype(bool)

    result: dict[str, float | str] = {
        "method": label,
        "auc_population": float(roc_auc_score(y_true, p)),
        "pr_auc_population": float(average_precision_score(y_true, p)),
        "brier_population": brier_score(y_true, p),
        "ece_population": expected_calibration_error(y_true, p),
        "mean_pd_population": float(p.mean()),
        "true_rate_population": float(y_true.mean()),
        "bias_population": float(p.mean() - y_true.mean()),
    }

    for name, mask in (("approved", ~is_reject), ("reject", is_reject)):
        if mask.sum() == 0:
            continue
        result[f"mean_pd_{name}"] = float(p[mask].mean())
        result[f"true_rate_{name}"] = float(y_true[mask].mean())
        result[f"bias_{name}"] = float(p[mask].mean() - y_true[mask].mean())
        result[f"brier_{name}"] = brier_score(y_true[mask], p[mask])
        if len(np.unique(y_true[mask])) > 1:
            result[f"auc_{name}"] = float(roc_auc_score(y_true[mask], p[mask]))

    return result


def summarise_bias_removed(comparison: pd.DataFrame) -> pd.DataFrame:
    """Express each method's improvement as a share of the achievable gap.

    Absolute bias numbers are hard to read. This restates them as: of the bias
    the uncorrected baseline carried on rejects, what fraction did this method
    remove, where the oracle removes 100% by construction?

    A negative value means the method made things **worse** than doing nothing,
    which is a result worth reporting rather than hiding. A value above 1.0
    means the method landed closer to zero bias than the oracle did; that is
    possible because the oracle is limited by model capacity and irreducible
    noise rather than by selection, so it is not a bound on bias alone.

    Args:
        comparison: Frame from :func:`evaluate_on_population` rows, containing
            ``baseline_approved_only`` and ``oracle_full_information`` methods.

    Returns:
        The comparison with ``bias_removed_share`` and ``brier_improvement``
        columns appended.
    """
    frame = comparison.copy()
    methods = frame["method"].tolist()

    if "baseline_approved_only" not in methods:
        frame["bias_removed_share"] = float("nan")
        return frame

    def value(method: str, column: str) -> float:
        return float(frame.loc[frame["method"] == method, column].to_numpy()[0])

    baseline_bias = abs(value("baseline_approved_only", "bias_reject"))
    baseline_brier = value("baseline_approved_only", "brier_reject")

    oracle_bias = (
        abs(value("oracle_full_information", "bias_reject"))
        if "oracle_full_information" in methods
        else 0.0
    )
    achievable = baseline_bias - oracle_bias

    frame["bias_removed_share"] = [
        (baseline_bias - abs(bias)) / achievable if achievable > 1e-9 else float("nan")
        for bias in frame["bias_reject"]
    ]
    frame["brier_improvement_reject"] = baseline_brier - frame["brier_reject"]
    return frame


def run_uplift_sensitivity(
    model_factory: ModelFactory,
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    feature_columns: list[str],
    uplift_values: tuple[float, ...] = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0),
    outcome_column: str = "default_observed",
    truth_column: str = "default_true",
    approved_column: str = "approved",
) -> pd.DataFrame:
    """Measure how sensitive the label-inference methods are to the uplift guess.

    The uplift factor is the only genuinely new information fuzzy augmentation
    and parcelling inject, so it determines whether they help or hurt. On a real
    book it is chosen from published guidance and cannot be validated. This
    function shows the size of the bet being placed.

    Args:
        model_factory: Zero-argument callable returning an unfitted model.
        train_frame: Training applicants, approved and declined together.
        test_frame: Evaluation applicants, approved and declined together.
        feature_columns: Model input columns.
        uplift_values: Uplift factors to test.
        outcome_column: Censored observed outcome column.
        truth_column: True outcome column, synthetic data only.
        approved_column: Legacy approval flag column.

    Returns:
        One row per method and uplift value with reject bias and Brier score.
    """
    approved_train = train_frame[approved_column].to_numpy().astype(bool)
    X_approved = train_frame.loc[approved_train, feature_columns]
    y_approved = train_frame.loc[approved_train, outcome_column].to_numpy().astype(int)
    X_rejected = train_frame.loc[~approved_train, feature_columns]

    X_test = test_frame[feature_columns]
    y_test_true = test_frame[truth_column].to_numpy().astype(int)
    approved_test = test_frame[approved_column].to_numpy().astype(bool)

    baseline = fit_approved_only(model_factory, X_approved, y_approved)

    rows = []
    for uplift in uplift_values:
        for name, builder in (
            ("fuzzy_augmentation", fuzzy_augmentation),
            ("parcelling", parcelling),
        ):
            model = builder(
                model_factory,
                X_approved,
                y_approved,
                X_rejected,
                uplift=uplift,
                base_model=baseline,
            )
            result = evaluate_on_population(
                model.predict_proba(X_test), y_test_true, approved_test, label=name
            )
            rows.append(
                {
                    "method": name,
                    "uplift": uplift,
                    "bias_reject": result["bias_reject"],
                    "abs_bias_reject": abs(float(result["bias_reject"])),
                    "brier_reject": result["brier_reject"],
                    "auc_population": result["auc_population"],
                }
            )
    return pd.DataFrame(rows)


def run_reject_inference_study(
    model_factory: ModelFactory,
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    feature_columns: list[str],
    numeric_columns: list[str],
    selection_instruments: list[str],
    uplift: float = DEFAULT_UPLIFT,
    outcome_column: str = "default_observed",
    truth_column: str = "default_true",
    approved_column: str = "approved",
) -> pd.DataFrame:
    """Fit every reject inference method and score them against ground truth.

    Args:
        model_factory: Zero-argument callable returning an unfitted model.
        train_frame: Training applicants, approved and declined together.
        test_frame: Evaluation applicants, approved and declined together.
        feature_columns: Model input columns.
        numeric_columns: Numeric subset, used by the Heckman equations, which
            cannot take raw categoricals.
        selection_instruments: Exclusion-restriction columns.
        uplift: Assumed reject bad-rate multiplier.
        outcome_column: Column holding the censored observed outcome.
        truth_column: Column holding the true outcome for everyone. Synthetic
            data only.
        approved_column: Column holding the legacy approval flag.

    Returns:
        One row per method with population, approved and reject metrics, plus
        the share of baseline reject bias each method removed.
    """
    approved_train = train_frame[approved_column].to_numpy().astype(bool)
    X_train_all = train_frame[[*feature_columns, *selection_instruments]]
    X_approved = train_frame.loc[approved_train, feature_columns]
    y_approved = train_frame.loc[approved_train, outcome_column].to_numpy().astype(int)
    X_rejected = train_frame.loc[~approved_train, feature_columns]

    X_test = test_frame[[*feature_columns, *selection_instruments]]
    y_test_true = test_frame[truth_column].to_numpy().astype(int)
    approved_test = test_frame[approved_column].to_numpy().astype(bool)

    baseline = fit_approved_only(model_factory, X_approved, y_approved)

    fitted: dict[str, Any] = {
        "baseline_approved_only": baseline,
        "fuzzy_augmentation": fuzzy_augmentation(
            model_factory, X_approved, y_approved, X_rejected, uplift=uplift, base_model=baseline
        ),
        "parcelling": parcelling(
            model_factory, X_approved, y_approved, X_rejected, uplift=uplift, base_model=baseline
        ),
        "heckman_two_step": heckman_correction(
            X_train_all,
            approved_train.astype(int),
            y_approved,
            numeric_columns,
            selection_instruments,
        ),
        "oracle_full_information": oracle_model(
            model_factory, train_frame[feature_columns], train_frame[truth_column].to_numpy()
        ),
    }

    rows = []
    for name, model in fitted.items():
        # The Heckman model consumes the instrument columns; the rest do not.
        features = X_test if name == "heckman_two_step" else X_test[feature_columns]
        rows.append(
            evaluate_on_population(
                model.predict_proba(features), y_test_true, approved_test, label=name
            )
        )

    return summarise_bias_removed(pd.DataFrame(rows))
