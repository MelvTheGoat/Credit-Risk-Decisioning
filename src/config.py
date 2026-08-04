"""Central configuration: paths, seeds, cost assumptions and governance thresholds.

Every number that encodes a business judgement lives here rather than being
buried in a module, so that a reviewer can audit the assumptions in one place
and a risk officer can change them without touching modelling code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
RAW_DATA_DIR: Final[Path] = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Final[Path] = DATA_DIR / "processed"
ARTIFACT_DIR: Final[Path] = PROJECT_ROOT / "artifacts"
REPORT_DIR: Final[Path] = PROJECT_ROOT / "reports"
FIGURE_DIR: Final[Path] = REPORT_DIR / "figures"
TABLE_DIR: Final[Path] = REPORT_DIR / "tables"
AUDIT_DIR: Final[Path] = PROJECT_ROOT / "audit"

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #

RANDOM_SEED: Final[int] = 20240517


# --------------------------------------------------------------------------- #
# Cost model
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CostParameters:
    """Economics of a single credit decision.

    The system optimises expected loss, not F1. Expected credit loss on an
    approved account is the regulatory identity ``EL = PD x LGD x EAD``. Against
    that sits the margin forgone when a good applicant is declined, which is a
    real cost to the lender even though it never appears in a loss provision.

    Attributes:
        lgd: Loss given default, the fraction of exposure not recovered after
            default and collections. 0.75 is a common unsecured-revolving
            assumption; secured products would be far lower.
        exposure_at_default: Default EAD in currency units when an applicant's
            own limit is unknown. Used as the unit of account throughout.
        margin_rate: Net lifetime margin earned on a non-defaulting approved
            account, as a fraction of exposure. Interest and fee income net of
            funding and servicing cost.
        decline_cost_rate: Cost of declining an applicant who would not have
            defaulted, as a fraction of exposure. Set equal to ``margin_rate``
            by default: the loss is exactly the margin forgone.
        fixed_review_cost: Per-application cost of manual review, applied to
            applications routed to the referral band.
    """

    lgd: float = 0.75
    exposure_at_default: float = 10_000.0
    margin_rate: float = 0.09
    decline_cost_rate: float = 0.09
    fixed_review_cost: float = 25.0

    @property
    def cost_false_negative(self) -> float:
        """Cost of approving an applicant who defaults (a missed bad)."""
        return self.lgd * self.exposure_at_default

    @property
    def cost_false_positive(self) -> float:
        """Cost of declining an applicant who would have repaid (a lost good)."""
        return self.decline_cost_rate * self.exposure_at_default

    @property
    def cost_ratio(self) -> float:
        """How many lost goods one missed bad is worth."""
        return self.cost_false_negative / self.cost_false_positive

    @property
    def analytic_threshold(self) -> float:
        """PD cutoff that minimises expected cost, derived in closed form.

        Approving applicant ``i`` costs ``p * C_fn``; declining costs
        ``(1 - p) * C_fp``. Approve while the former is smaller, so the
        indifference point is ``p* = C_fp / (C_fn + C_fp)``.

        The empirical threshold sweep in :mod:`src.policy` should land on this
        value; if it does not, either the sweep or the cost model is wrong.
        """
        return self.cost_false_positive / (self.cost_false_negative + self.cost_false_positive)


DEFAULT_COSTS: Final[CostParameters] = CostParameters()


# --------------------------------------------------------------------------- #
# Scorecard scaling
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ScorecardScaling:
    """Points-scaling convention for the WOE scorecard.

    Attributes:
        base_score: Score at which the base odds hold.
        base_odds: Good:bad odds at ``base_score``. 19.0 means 19 goods per bad.
        pdo: Points to double the odds. The industry convention is 20.
    """

    base_score: float = 600.0
    base_odds: float = 19.0
    pdo: float = 20.0


DEFAULT_SCALING: Final[ScorecardScaling] = ScorecardScaling()


# --------------------------------------------------------------------------- #
# Score bands
# --------------------------------------------------------------------------- #

# Band edges on the points scale, low score = high risk.
SCORE_BANDS: Final[tuple[tuple[str, float, float], ...]] = (
    ("E", -1e9, 520.0),
    ("D", 520.0, 570.0),
    ("C", 570.0, 620.0),
    ("B", 620.0, 670.0),
    ("A", 670.0, 1e9),
)


# --------------------------------------------------------------------------- #
# Monitoring / governance thresholds
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class MonitoringThresholds:
    """Trigger levels for model review.

    Attributes:
        psi_warn: PSI above which a feature or score distribution is watched.
            0.10 is the long-standing industry convention for "some shift".
        psi_breach: PSI above which a formal model review is opened. 0.25 is
            the conventional "significant shift" level.
        ece_breach: Expected calibration error above which the calibrator is
            considered stale and must be refit.
        auc_drop_breach: Absolute fall in out-of-time AUC versus the validation
            baseline that triggers review.
        subgroup_ece_breach: Within-group ECE above which a subgroup is
            reported as materially miscalibrated, regardless of overall ECE.
        eod_breach: Equal opportunity difference above which the fairness
            finding is escalated to the model risk committee.
    """

    psi_warn: float = 0.10
    psi_breach: float = 0.25
    ece_breach: float = 0.05
    auc_drop_breach: float = 0.05
    subgroup_ece_breach: float = 0.05
    eod_breach: float = 0.05


DEFAULT_MONITORING: Final[MonitoringThresholds] = MonitoringThresholds()


# --------------------------------------------------------------------------- #
# Synthetic simulation
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SimulationConfig:
    """Parameters of the synthetic loan book.

    The synthetic generator exists so that reject inference and fairness work
    can be validated against ground truth that is unobservable in real data.

    Attributes:
        n_applicants: Number of applications generated.
        n_vintages: Number of monthly origination cohorts (36 = three years).
        start_date: Origination date of the first vintage, ISO format.
        performance_window_months: Months of observation before an account's
            default flag is final.
        drift_start_vintage: Vintage from which population drift is injected.
        target_population_default_rate: Population-wide true default rate the
            generator solves its intercept to hit. Set to a realistic unsecured
            consumer level rather than something that makes modelling easy.
        target_approval_rate: Approval rate of the simulated legacy policy.
        legacy_policy_noise: Standard deviation of the judgemental / unobserved
            component of the legacy policy, in logit units. Larger values make
            the legacy policy less sharp and the selection problem more
            realistic; a near-deterministic policy would make reject inference
            trivially impossible through complete separation.
        group_b_share: Population share of the protected group B.
        group_b_income_bias: Multiplicative measurement bias applied to
            *recorded* income for group B. 0.85 means recorded income is 15%
            below true income, while true default risk depends on true income.
            This is the injected fairness violation.
        group_b_policy_penalty: Extra legacy-policy score penalty applied to
            group B, in logit units. This is the injected historical
            discrimination that reject inference must contend with.
        seed: Base random seed.
    """

    n_applicants: int = 30_000
    n_vintages: int = 36
    start_date: str = "2019-01-01"
    performance_window_months: int = 12
    drift_start_vintage: int = 24
    target_population_default_rate: float = 0.20
    target_approval_rate: float = 0.70
    legacy_policy_noise: float = 1.60
    group_b_share: float = 0.42
    group_b_income_bias: float = 0.85
    group_b_policy_penalty: float = 0.55
    seed: int = RANDOM_SEED


DEFAULT_SIMULATION: Final[SimulationConfig] = SimulationConfig()


# --------------------------------------------------------------------------- #
# Temporal split
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SplitConfig:
    """Out-of-time split boundaries expressed in vintage indices.

    Attributes:
        fit_end_vintage: Vintages ``[0, fit_end_vintage)`` train the model.
        calibration_end_vintage: Vintages ``[fit_end_vintage,
            calibration_end_vintage)`` fit the calibrator. Held out from model
            fitting so the calibrator never sees in-sample scores.
        test_start_vintage: Vintages ``>= test_start_vintage`` are the
            out-of-time test set. Deliberately covers the drift period.
    """

    fit_end_vintage: int = 20
    calibration_end_vintage: int = 24
    test_start_vintage: int = 24


DEFAULT_SPLIT: Final[SplitConfig] = SplitConfig()


# --------------------------------------------------------------------------- #
# Feature groupings
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FeatureSpec:
    """Which columns are model inputs and which are protected attributes.

    Protected attributes are deliberately excluded from ``model_features``.
    They are retained in the dataframe for auditing only. Excluding them does
    not make the model fair — proxies survive — which is precisely what the
    fairness audit is for.
    """

    numeric: tuple[str, ...] = field(default_factory=tuple)
    categorical: tuple[str, ...] = field(default_factory=tuple)
    protected: tuple[str, ...] = field(default_factory=tuple)

    @property
    def model_features(self) -> list[str]:
        """All columns used as model inputs."""
        return [*self.numeric, *self.categorical]


SYNTHETIC_FEATURES: Final[FeatureSpec] = FeatureSpec(
    numeric=(
        "income_recorded",
        "debt_to_income",
        "utilisation",
        "n_delinq_24m",
        "credit_history_months",
        "n_inquiries_6m",
        "employment_years",
        "loan_amount",
        "age",
    ),
    categorical=("housing_status", "product_type"),
    protected=("group", "age_band"),
)

UCI_FEATURES: Final[FeatureSpec] = FeatureSpec(
    numeric=(
        "limit_bal",
        "age",
        "pay_0",
        "pay_2",
        "pay_3",
        "pay_4",
        "pay_5",
        "pay_6",
        "bill_amt1",
        "bill_amt2",
        "bill_amt3",
        "bill_amt4",
        "bill_amt5",
        "bill_amt6",
        "pay_amt1",
        "pay_amt2",
        "pay_amt3",
        "pay_amt4",
        "pay_amt5",
        "pay_amt6",
        "avg_utilisation",
        "pay_ratio_1",
        "max_delinquency",
        "n_months_delinquent",
    ),
    categorical=(),
    protected=("sex", "education", "marriage", "age_band"),
)


def ensure_directories() -> None:
    """Create the output directory tree if it does not already exist."""
    for directory in (
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        ARTIFACT_DIR,
        REPORT_DIR,
        FIGURE_DIR,
        TABLE_DIR,
        AUDIT_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
