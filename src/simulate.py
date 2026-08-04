"""Synthetic loan book with a known data-generating process.

Why this module exists
----------------------
Two of the hardest claims this system makes are not verifiable on real data:

1. **Reject inference works (or does not).** On a real book you never observe
   what would have happened to a declined applicant. Any claim that a
   correction "removes bias" is unfalsifiable.
2. **The fairness audit detects the bias that is actually there.** On real data
   you cannot separate a measurement artefact from a genuine risk difference,
   because you never see the true risk.

So the synthetic generator is built first and deliberately contains ground
truth that a real book hides:

* ``pd_true`` — the true probability of default for **every** applicant.
* ``default_true`` — the realised outcome for **every** applicant, including
  those the legacy policy declined.
* ``approved`` — the legacy policy's decision, which censors the observed
  outcome exactly the way a real book is censored.

Three things are injected with known magnitude:

* **Population drift** from :attr:`~src.config.SimulationConfig.drift_start_vintage`
  onward — later vintages are pushed into thinner-file, higher-utilisation
  segments and a mild macro deterioration is applied. Monitoring must detect it.
* **A fairness violation.** Recorded income for group B is understated by
  ``group_b_income_bias`` while true default risk depends on *true* income. The
  model therefore over-predicts risk for group B through no fault of its own
  and through no feature it can see. This is measurement bias, the kind that
  survives dropping the protected attribute.
* **Historical policy discrimination.** The legacy approval policy applies an
  extra penalty of ``group_b_policy_penalty`` logits to group B, so group B's
  approved population is more positively selected than group A's. Reject
  inference must contend with this.

There is also a deliberate **exclusion restriction**: ``branch_capacity_index``
affects the legacy approval decision but has no causal effect on default. It
gives the Heckman correction in :mod:`src.reject_inference` an instrument.
Honest caveat: a clean instrument like this rarely exists on a real book, which
is a large part of why Heckman corrections disappoint in practice.

Group B is also given genuinely different feature distributions (somewhat lower
income, somewhat higher utilisation). That produces a *real* base-rate
difference on top of the *unfair* measurement bias. The two are entangled by
construction, because that is the situation a real fairness audit faces: some
of the approval gap is justified by risk and some is an artefact, and no metric
separates them on its own.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from src.config import DEFAULT_SIMULATION, SimulationConfig

logger = logging.getLogger(__name__)

HOUSING_STATUSES: tuple[str, ...] = ("own", "mortgage", "rent", "other")
PRODUCT_TYPES: tuple[str, ...] = ("card", "personal_loan", "auto")

# Effect of housing status on the default logit. Renters carry more risk than
# owners in essentially every consumer book; "other" is a thin, noisy segment.
HOUSING_EFFECT: dict[str, float] = {
    "own": -0.30,
    "mortgage": -0.15,
    "rent": 0.22,
    "other": 0.35,
}

PRODUCT_EFFECT: dict[str, float] = {
    "card": 0.12,
    "personal_loan": 0.05,
    "auto": -0.20,
}

AGE_BAND_EDGES: tuple[float, ...] = (0.0, 25.0, 35.0, 45.0, 55.0, 200.0)
AGE_BAND_LABELS: tuple[str, ...] = ("<25", "25-34", "35-44", "45-54", "55+")


@dataclass(frozen=True)
class TrueCoefficients:
    """The known default-generating process, in logit units.

    These are the answer key. No model is given them; several tests check that
    fitted models recover their sign and rough ordering.

    Note that there is **no group term**. Group membership has no direct causal
    effect on default in this world. Every risk difference between groups flows
    through legitimate features, and every *unfair* difference flows through the
    recorded-income measurement bias.

    The last three terms are deliberately **non-linear**. Without them the DGP
    would be linear in the logit, every linear model would be correctly
    specified, and gradient boosting would have nothing whatsoever to find —
    which would rig the scorecard-versus-booster comparison in the scorecard's
    favour and make the headline finding worthless. All three are ordinary
    credit phenomena:

    * ``utilisation_x_delinquency`` — high utilisation and recent delinquency
      compound rather than add. Being maxed out matters much more if you have
      already missed payments.
    * ``high_dti_cliff`` — affordability rules bite at a threshold, not
      smoothly.
    * ``thin_file_penalty`` — a step change in risk for applicants with almost
      no credit history, which no monotone transform of history length
      captures.

    Attributes:
        intercept: Base logit, later solved to hit the target default rate.
        log_income_true: Effect of true (not recorded) log income.
        debt_to_income: Effect of standardised debt-to-income.
        utilisation: Effect of standardised utilisation.
        n_delinq_24m: Effect per delinquency in the last 24 months, capped at 6.
        credit_history_months: Effect of standardised log history length.
        n_inquiries_6m: Effect of standardised recent inquiry count.
        employment_years: Effect of standardised log employment tenure.
        loan_to_income: Effect of standardised loan-to-income.
        age: Effect of standardised age.
        utilisation_x_delinquency: Interaction between utilisation and capped
            delinquency count.
        high_dti_cliff: Step penalty above ``high_dti_threshold``.
        high_dti_threshold: Debt-to-income level at which the cliff applies.
        thin_file_penalty: Step penalty below ``thin_file_months``.
        thin_file_months: Credit history length defining a thin file.
    """

    intercept: float = -1.55
    log_income_true: float = -0.75
    debt_to_income: float = 0.85
    utilisation: float = 0.90
    n_delinq_24m: float = 0.55
    credit_history_months: float = -0.40
    n_inquiries_6m: float = 0.30
    employment_years: float = -0.25
    loan_to_income: float = 0.35
    age: float = -0.20
    utilisation_x_delinquency: float = 0.42
    high_dti_cliff: float = 0.38
    high_dti_threshold: float = 0.45
    thin_file_penalty: float = 0.30
    thin_file_months: float = 12.0


TRUE_COEFFICIENTS = TrueCoefficients()


def _standardise(values: np.ndarray) -> np.ndarray:
    """Centre and scale to unit variance, guarding against a zero denominator."""
    std = float(values.std())
    if std <= 0.0:
        return np.zeros_like(values, dtype=float)
    return (values - float(values.mean())) / std


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function."""
    out = np.empty_like(z, dtype=float)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


def _solve_intercept_shift(
    base_logit: np.ndarray,
    target_rate: float,
    tolerance: float = 1e-6,
    max_iterations: int = 200,
) -> float:
    """Bisect for the intercept shift that makes the mean event rate hit a target.

    Args:
        base_logit: Logit values before the shift.
        target_rate: Desired mean of ``sigmoid(base_logit + shift)``.
        tolerance: Absolute tolerance on the achieved rate.
        max_iterations: Bisection iteration cap.

    Returns:
        The additive shift in logit units.
    """
    low, high = -20.0, 20.0
    shift = 0.0
    for _ in range(max_iterations):
        shift = (low + high) / 2.0
        rate = float(_sigmoid(base_logit + shift).mean())
        if abs(rate - target_rate) < tolerance:
            break
        if rate < target_rate:
            low = shift
        else:
            high = shift
    return shift


def _assign_age_band(age: np.ndarray) -> np.ndarray:
    """Map continuous age onto the reporting bands used by the fairness audit."""
    indices = np.digitize(age, AGE_BAND_EDGES[1:-1], right=False)
    return np.asarray(AGE_BAND_LABELS, dtype=object)[indices]


def simulate_loan_book(config: SimulationConfig = DEFAULT_SIMULATION) -> pd.DataFrame:
    """Generate a synthetic loan book with full ground truth.

    Args:
        config: Simulation parameters. See
            :class:`~src.config.SimulationConfig`.

    Returns:
        One row per application with these column families:

        * **Identity / time** — ``application_id``, ``vintage``,
          ``origination_date``, ``observation_date``.
        * **Protected** — ``group`` (A/B), ``age_band``. Never model inputs.
        * **Model features** — ``income_recorded``, ``debt_to_income``,
          ``utilisation``, ``n_delinq_24m``, ``credit_history_months``,
          ``n_inquiries_6m``, ``employment_years``, ``loan_amount``, ``age``,
          ``housing_status``, ``product_type``.
        * **Selection-only** — ``branch_capacity_index``, the exclusion
          restriction for the Heckman correction.
        * **Oracle, never a model input** — ``income_true``, ``pd_true``,
          ``default_true``. These do not exist on a real book.
        * **Legacy policy** — ``legacy_score``, ``approved``.
        * **Observed target** — ``default_observed``, which is ``default_true``
          for approved applicants and ``NaN`` for declined ones. This is the
          only label a real model would ever get to train on.

    Notes:
        The observation snapshot is taken ``performance_window_months`` after
        the final origination, so every vintage is fully seasoned. A real book
        has an immature tail; excluding it here keeps the vintage analysis
        interpretable and is recorded as a simplification in ``DECISIONS.md``.
    """
    rng = np.random.default_rng(config.seed)
    n = config.n_applicants

    # ----------------------------------------------------------------- time --
    vintage = rng.integers(0, config.n_vintages, size=n)
    vintage_starts = pd.date_range(config.start_date, periods=config.n_vintages, freq="MS")
    origination_date = vintage_starts.to_numpy()[vintage]
    observation_date = (
        pd.DatetimeIndex(origination_date) + pd.DateOffset(months=config.performance_window_months)
    ).to_numpy()

    # Drift indicator: later vintages are pushed into thinner-file segments.
    is_drifted = vintage >= config.drift_start_vintage
    drift_intensity = np.where(
        is_drifted,
        (vintage - config.drift_start_vintage + 1) / max(config.n_vintages - config.drift_start_vintage, 1),
        0.0,
    )

    # ------------------------------------------------------------ protected --
    is_group_b = rng.random(n) < config.group_b_share
    group = np.where(is_group_b, "B", "A")

    age = np.clip(rng.gamma(shape=9.0, scale=4.2, size=n) + 19.0, 19.0, 78.0)
    age_band = _assign_age_band(age)

    # ------------------------------------------------------------- features --
    # Group B earns genuinely less on average. This is a real economic
    # difference, not the injected unfairness, and it produces a real base-rate
    # gap that no threshold choice can wish away.
    income_log_mean = 10.62 - 0.16 * is_group_b - 0.12 * drift_intensity
    income_true = np.exp(rng.normal(income_log_mean, 0.46, size=n))
    income_true = np.clip(income_true, 9_000.0, 900_000.0)

    # THE INJECTED FAIRNESS VIOLATION.
    # Recorded income understates true income for group B — informal or
    # cash-based earnings that the application process fails to capture. True
    # default risk depends on income_true; the model only ever sees
    # income_recorded. The model is therefore biased against group B through a
    # channel it cannot observe and dropping `group` does nothing about it.
    income_recorded = np.where(is_group_b, income_true * config.group_b_income_bias, income_true)
    income_recorded = np.round(income_recorded, 2)

    utilisation = np.clip(
        rng.beta(2.0, 3.4, size=n) + 0.10 * is_group_b + 0.07 * drift_intensity,
        0.0,
        1.6,
    )
    debt_to_income = np.clip(rng.gamma(3.0, 0.11, size=n) + 0.03 * drift_intensity, 0.0, 2.5)
    n_delinq_24m = rng.poisson(0.35 + 0.55 * utilisation + 0.12 * drift_intensity, size=n)
    credit_history_months = np.clip(
        rng.gamma(4.0, 22.0, size=n) * (0.55 + 0.45 * (age - 19.0) / 59.0) - 25.0 * drift_intensity,
        1.0,
        480.0,
    )
    n_inquiries_6m = rng.poisson(0.9 + 0.35 * drift_intensity, size=n)
    employment_years = np.clip(
        rng.gamma(2.0, 3.0, size=n) * np.clip((age - 18.0) / 30.0, 0.1, 2.0),
        0.0,
        45.0,
    )
    loan_amount = np.round(
        np.clip(income_true * rng.uniform(0.08, 0.55, size=n), 1_000.0, 200_000.0), -2
    )
    loan_to_income = loan_amount / income_true

    housing_probabilities = np.array([0.24, 0.30, 0.38, 0.08])
    housing_status = rng.choice(HOUSING_STATUSES, size=n, p=housing_probabilities)
    product_type = rng.choice(PRODUCT_TYPES, size=n, p=np.array([0.55, 0.30, 0.15]))

    # Exclusion restriction: branch workload affects whether an application is
    # approved but has no causal path to default.
    branch_capacity_index = rng.normal(0.0, 1.0, size=n)

    # Latent loan-officer judgement. Never a model feature. When the
    # private_signal_* parameters are non-zero this drives both the true
    # default outcome and the legacy approval decision, which makes selection
    # depend on something the model can never observe. See SimulationConfig.
    officer_signal = rng.normal(0.0, 1.0, size=n)

    # ------------------------------------------------------- true PD (DGP) ---
    c = TRUE_COEFFICIENTS
    logit = (
        c.intercept
        + c.log_income_true * _standardise(np.log(income_true))
        + c.debt_to_income * _standardise(debt_to_income)
        + c.utilisation * _standardise(utilisation)
        + c.n_delinq_24m * np.minimum(n_delinq_24m, 6)
        + c.credit_history_months * _standardise(np.log1p(credit_history_months))
        + c.n_inquiries_6m * _standardise(n_inquiries_6m.astype(float))
        + c.employment_years * _standardise(np.log1p(employment_years))
        + c.loan_to_income * _standardise(loan_to_income)
        + c.age * _standardise(age)
    )
    logit += np.array([HOUSING_EFFECT[h] for h in housing_status])
    logit += np.array([PRODUCT_EFFECT[p] for p in product_type])

    # Non-linear structure: an interaction and two threshold effects. These are
    # what a gradient booster can find and an additive scorecard cannot.
    logit += c.utilisation_x_delinquency * (
        _standardise(utilisation) * np.minimum(n_delinq_24m, 3)
    )
    logit += c.high_dti_cliff * (debt_to_income > c.high_dti_threshold)
    logit += c.thin_file_penalty * (credit_history_months < c.thin_file_months)

    # Unobserved-by-the-model risk that the loan officer could see. Zero in the
    # default (MAR) book.
    logit += config.private_signal_default * officer_signal

    # Macro deterioration in the drifted vintages, plus mild seasonality so the
    # vintage curves are not perfectly flat.
    logit += 0.25 * drift_intensity
    logit += 0.06 * np.sin(2.0 * np.pi * (vintage % 12) / 12.0)

    # Irreducible noise: the part of default no feature set can explain. Without
    # it the problem is too easy and every model looks like an oracle.
    logit += rng.normal(0.0, 0.55, size=n)

    # Solve the intercept so the population default rate lands on a realistic
    # target rather than whatever the coefficients happen to produce.
    logit += _solve_intercept_shift(logit, config.target_population_default_rate)

    pd_true = _sigmoid(logit)
    default_true = (rng.random(n) < pd_true).astype(int)

    # --------------------------------------------- legacy approval policy ----
    # A crude legacy scorecard on a subset of the *recorded* features, plus
    # judgemental noise, plus an explicit penalty on group B. It sees
    # income_recorded, so it inherits the measurement bias as well.
    legacy_logit = (
        -0.62 * _standardise(np.log(income_recorded))
        + 0.70 * _standardise(debt_to_income)
        + 0.80 * _standardise(utilisation)
        + 0.60 * np.minimum(n_delinq_24m, 6)
        + 0.25 * _standardise(n_inquiries_6m.astype(float))
        + config.group_b_policy_penalty * is_group_b
        + 0.45 * branch_capacity_index
        + config.private_signal_policy * officer_signal
        + rng.normal(0.0, config.legacy_policy_noise, size=n)
    )
    # Cut at the quantile that delivers the target approval rate exactly. A
    # fixed 0.5 cut on a probability would leave the realised approval rate at
    # the mercy of the score distribution.
    legacy_cutoff = float(np.quantile(legacy_logit, config.target_approval_rate))
    legacy_score = _sigmoid(legacy_logit)
    approved = (legacy_logit < legacy_cutoff).astype(int)

    default_observed = np.where(approved == 1, default_true.astype(float), np.nan)

    frame = pd.DataFrame(
        {
            "application_id": np.arange(1, n + 1),
            "vintage": vintage.astype(int),
            "origination_date": origination_date,
            "observation_date": observation_date,
            "group": group,
            "age_band": pd.Categorical(age_band, categories=list(AGE_BAND_LABELS), ordered=True),
            "age": np.round(age, 1),
            "income_recorded": income_recorded,
            "debt_to_income": np.round(debt_to_income, 4),
            "utilisation": np.round(utilisation, 4),
            "n_delinq_24m": n_delinq_24m.astype(int),
            "credit_history_months": np.round(credit_history_months, 1),
            "n_inquiries_6m": n_inquiries_6m.astype(int),
            "employment_years": np.round(employment_years, 2),
            "loan_amount": loan_amount,
            "housing_status": housing_status,
            "product_type": product_type,
            "branch_capacity_index": np.round(branch_capacity_index, 4),
            # Oracle columns. A real book has none of these.
            "officer_signal": np.round(officer_signal, 4),
            "income_true": np.round(income_true, 2),
            "pd_true": pd_true,
            "default_true": default_true,
            # Legacy policy and the censored label.
            "legacy_score": legacy_score,
            "approved": approved,
            "default_observed": default_observed,
        }
    )

    logger.info(
        "Simulated %d applications | approval rate %.3f | population default rate %.3f | "
        "approved default rate %.3f | declined default rate %.3f",
        n,
        float(frame["approved"].mean()),
        float(frame["default_true"].mean()),
        float(frame.loc[frame["approved"] == 1, "default_true"].mean()),
        float(frame.loc[frame["approved"] == 0, "default_true"].mean()),
    )
    return frame


def mnar_config(
    config: SimulationConfig = DEFAULT_SIMULATION,
    default_strength: float = 0.85,
    policy_strength: float = 1.30,
) -> SimulationConfig:
    """Build a variant where selection depends on something the model cannot see.

    In the default book the legacy policy acted only on features the model also
    receives, so a model trained on approved applicants generalises to declined
    ones. That is **selection on observables**, and reject inference has nothing
    to repair.

    This variant switches on a latent loan-officer judgement that drives both
    the true default outcome and the approval decision, and is never a model
    feature. Now the approved sample really is unrepresentative in a way no
    amount of feature engineering can reach, and a correction has genuine work
    to do.

    Running both regimes is the point. The two produce opposite conclusions
    about whether reject inference is worth doing, and **on a real book there is
    no way to tell which regime you are in** — the evidence that would settle it
    is precisely the missing counterfactual.

    Args:
        config: Base configuration to modify.
        default_strength: Effect of the latent signal on the default logit.
        policy_strength: Effect of the latent signal on the approval decision.

    Returns:
        A configuration with the private signal switched on.
    """
    return replace(
        config,
        private_signal_default=default_strength,
        private_signal_policy=policy_strength,
    )


def split_by_vintage(
    frame: pd.DataFrame,
    fit_end_vintage: int,
    calibration_end_vintage: int,
    test_start_vintage: int,
) -> dict[str, pd.DataFrame]:
    """Split a loan book out of time by origination vintage.

    A random split leaks the future into the past. Credit portfolios drift, and
    a model validated on a random split will look better than it will ever
    perform. Splitting by origination cohort is the only honest option when
    origination dates exist.

    Args:
        frame: Loan book with a ``vintage`` column.
        fit_end_vintage: Model fitting uses vintages below this.
        calibration_end_vintage: Calibrator fitting uses vintages in
            ``[fit_end_vintage, calibration_end_vintage)``.
        test_start_vintage: Out-of-time test uses vintages at or above this.

    Returns:
        Mapping with keys ``fit``, ``calibration`` and ``test``.
    """
    return {
        "fit": frame[frame["vintage"] < fit_end_vintage].copy(),
        "calibration": frame[
            (frame["vintage"] >= fit_end_vintage) & (frame["vintage"] < calibration_end_vintage)
        ].copy(),
        "test": frame[frame["vintage"] >= test_start_vintage].copy(),
    }


def observed_training_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Restrict to the rows a real lender would actually be able to train on.

    Only approved applicants have an observed outcome. Everything downstream of
    this function is working with a sample selected by the previous policy —
    which is the whole problem :mod:`src.reject_inference` exists to address.

    Args:
        frame: Loan book.

    Returns:
        Approved rows only, with ``default_observed`` cast to ``int``.
    """
    approved = frame[frame["approved"] == 1].copy()
    approved["default_observed"] = approved["default_observed"].astype(int)
    return approved


def describe_simulation(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarise the injected structure so it can be asserted on and reported.

    Args:
        frame: Loan book from :func:`simulate_loan_book`.

    Returns:
        Single-row dataframe of headline quantities: approval rates and true
        default rates overall and by group, and the selection gap that reject
        inference is trying to close.
    """
    approved = frame["approved"] == 1
    group_b = frame["group"] == "B"
    rows = {
        "n_applications": float(len(frame)),
        "approval_rate": float(frame["approved"].mean()),
        "approval_rate_group_a": float(frame.loc[~group_b, "approved"].mean()),
        "approval_rate_group_b": float(frame.loc[group_b, "approved"].mean()),
        "default_rate_population": float(frame["default_true"].mean()),
        "default_rate_approved": float(frame.loc[approved, "default_true"].mean()),
        "default_rate_declined": float(frame.loc[~approved, "default_true"].mean()),
        "default_rate_group_a": float(frame.loc[~group_b, "default_true"].mean()),
        "default_rate_group_b": float(frame.loc[group_b, "default_true"].mean()),
        "selection_gap": float(
            frame.loc[~approved, "default_true"].mean() - frame.loc[approved, "default_true"].mean()
        ),
        "mean_pd_true_approved": float(frame.loc[approved, "pd_true"].mean()),
        "mean_pd_true_declined": float(frame.loc[~approved, "pd_true"].mean()),
    }
    return pd.DataFrame([rows])


if __name__ == "__main__":  # pragma: no cover - manual inspection helper
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    book = simulate_loan_book()
    with pd.option_context("display.width", 160, "display.max_columns", 40):
        print(describe_simulation(book).T)
