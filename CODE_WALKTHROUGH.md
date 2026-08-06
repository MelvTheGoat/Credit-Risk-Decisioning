# Building a Credit Risk Decisioning System

## A walkthrough, in build order

We're sitting in front of an empty folder. Before we type anything, I want to talk you through what we're about to build and — more importantly — *why each piece has to exist*, because the order matters enormously here and getting it wrong will cost you a rewrite.

Here's the one idea that organises everything else:

> **The two hardest claims a credit model makes are unfalsifiable on real data.**

Claim one: "our reject inference removes selection bias." You cannot check that, because you never observe what would have happened to an applicant you declined. Claim two: "our fairness audit detects real bias." You cannot check that either, because you never observe an applicant's *true* risk — only whether they happened to default.

Every credit modelling project that starts by downloading a dataset ends up unable to validate its most important work. So we're going to start somewhere else: we're going to build a **simulator** with known ground truth, and we're going to build it first, before we touch real data. Everything downstream gets validated against a world where we know the answer.

That single decision reorders the entire project. Here's the sequence:

1. **`pyproject.toml`, `.gitignore`** — the shape of the thing
2. **`src/config.py`** — every business assumption, in one auditable place
3. **`src/simulate.py`** — the measuring instrument
4. **`src/scorecard.py`** — the benchmark that everything else has to beat
5. **`src/models.py`** — the challengers
6. **`src/calibration.py`** — turning scores into probabilities you can price with
7. **`src/evaluation.py`** — metrics that don't lie about imbalance
8. **`src/policy.py`** — the actual answer: a cost
9. **`src/reject_inference.py`** — the intellectual core
10. **`src/fairness.py`** — the audit
11. **`src/explain.py`** — reason codes, because a decline is a legal act
12. **`src/monitoring.py`** — what breaks in production
13. **`src/ingest.py`** — real data, deliberately late
14. **`src/artifacts.py`** — versioning, and the mismatch guard
15. **`src/engine.py`** — the single decision path
16. **`src/api/`** — the service
17. **`scripts/`** — drivers
18. **`tests/`** — what each one is actually protecting
19. **Docker, CI, docs**

If you build these out of order you will get stuck in specific, predictable places, and I'll flag each one as we pass it.

---

## 1. `pyproject.toml` and `.gitignore`

### Why this exists

Boring, but it constrains everything. Two things in here are load-bearing.

First, the package layout. We're using a `src/` package imported as `src.*`, not a flat module dump. That means `from src.config import DEFAULT_COSTS` works from anywhere in the repo, and there's exactly one canonical import path for every symbol. Flat layouts produce the situation where `config.py` means different things depending on your working directory, and you find out in production.

Second, lint and type configuration. We turn on `D` (pydocstyle) in ruff:

```toml
[tool.ruff.lint]
select = ["E", "F", "W", "I", "N", "UP", "B", "C4", "SIM", "ARG", "D"]
ignore = [
    "D203",  # conflicts with D211
    "D213",  # conflicts with D212
    "E501",  # line length handled by formatter
    "N803",  # allow X / y sklearn-style argument names
    "N806",  # allow X_train style locals
]
```

`N803` and `N806` are the interesting ones. Ruff's naming rules want `x_train`, not `X_train`. But in a codebase that hands frames to scikit-learn, capital-`X`-for-matrix is universal convention and fighting it makes the code read *worse*. That's a deliberate override, not laziness — we're saying the ecosystem convention beats the linter's opinion.

And mypy:

```toml
[tool.mypy]
disallow_untyped_defs = true
disallow_incomplete_defs = true
```

This one has teeth. It means every function you write needs annotations, including the little private helpers. It will annoy you for about a day and then start catching things.

### `.gitignore` — one decision worth explaining

```gitignore
# Serialised models. Rebuilt by `python -m scripts.run_experiments` and by the
# Docker build; binary, versioned, and not useful in review.
artifacts/

# Append-only decision log. Contains decisions about (synthetic) applicants and
# never belongs in source control.
audit/

# Report outputs. The tables and figures ARE committed: they are the evidence
# behind every number quoted in README.md, they are small (~590K), and the
# README's figures have to resolve for it to render. Bulk scoring output is not.
reports/scored_portfolio.csv
reports/*.parquet
```

The split here is the point. `artifacts/` is ignored because it's binary build output. `reports/` is **not** ignored, because those CSVs are the evidence for every number in the README. If someone challenges a claim in your write-up, the answer should be a file in the repo, not "re-run it and trust me."

`data/` is ignored too. Never commit the dataset.

---

## 2. `src/config.py` — assumptions first

### Why this file exists, and why it's first

Because a credit model is mostly assumptions wearing a model's clothes. LGD, margin, the PSI review threshold, the IV floor — these are business judgements, and if they're scattered across ten modules as inline literals then nobody can audit them and a risk officer can't change one without a code review.

So: every number that encodes a judgement lives here. Everything else imports it.

Build this *before* the simulator, because the simulator needs `SimulationConfig` and the policy module needs `CostParameters`, and you don't want to be threading magic numbers through function signatures and then refactoring.

### The cost model

This is the heart of the whole system, so let's go slowly.

`src/config.py:65-69`:

```python
    lgd: float = 0.75
    exposure_at_default: float = 10_000.0
    margin_rate: float = 0.09
    decline_cost_rate: float = 0.09
    fixed_review_cost: float = 25.0
```

`lgd = 0.75` means when someone defaults, we recover 25% after collections. That's a standard unsecured-revolving assumption. If you're modelling mortgages this is catastrophically wrong — secured LGD is more like 0.2 — and every threshold downstream would move.

`margin_rate = 0.09` is the net lifetime margin on a good account, as a fraction of exposure. Interest and fees minus funding and servicing.

Now the derived properties, `src/config.py:71-97`:

```python
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
        return self.cost_false_positive / (self.cost_false_negative + self.cost_false_positive)
```

**Slow down on `analytic_threshold`.** This one line is the most important in the repository.

The derivation: approving applicant *i* has expected cost `p × C_fn`. Declining has expected cost `(1 − p) × C_fp`. Approve while the first is smaller. Set them equal and solve:

```
p × C_fn = (1 − p) × C_fp
p × C_fn + p × C_fp = C_fp
p* = C_fp / (C_fn + C_fp)
```

With our defaults that's `900 / (7500 + 900) = 0.1071`.

**The trap.** The tempting thing to write is `0.5`. Or, if you've been reading about imbalanced classification, the base rate — `0.20`. Or you tune a threshold to maximise F1 and get `0.211`. All three are wrong, and here's the thing that should worry you: **all three produce results that look fine.** Confusion matrices, precision, recall — everything reports plausible numbers. You only find out you were wrong when you total up the money, and by then you've written a year of business.

We'll measure exactly how wrong later: the F1 threshold costs 729,735 more than the optimum on our test set, and a naive 0.5 cutoff *loses money outright*.

Notice what `p*` does **not** depend on: the model. It's a property of the economics alone. That's a genuinely surprising fact the first time you see it, and it's why this lives in `config.py` next to the cost parameters rather than in `policy.py` next to the model code.

### Score bands

`src/config.py:131-137`:

```python
SCORE_BANDS: Final[tuple[tuple[str, float, float], ...]] = (
    ("E", -1e9, 520.0),
    ("D", 520.0, 570.0),
    ("C", 570.0, 620.0),
    ("B", 620.0, 670.0),
    ("A", 670.0, 1e9),
)
```

`-1e9` and `1e9` instead of `-inf`/`inf` because these get serialised to JSON in the API response and manifest, and `float('inf')` is not valid JSON. `json.dumps` will happily emit `Infinity`, which is a JSON extension that many parsers reject. Using a big finite number avoids a class of interop bug that only shows up when someone consumes your API from a strict parser.

### Monitoring thresholds

`src/config.py:164-169`:

```python
    psi_warn: float = 0.10
    psi_breach: float = 0.25
    ece_breach: float = 0.05
    auc_drop_breach: float = 0.05
    subgroup_ece_breach: float = 0.05
    eod_breach: float = 0.05
```

`0.10` and `0.25` for PSI are the industry convention, and I want to be honest about *why* we keep them: not because they're statistically principled — they aren't, PSI has no sampling distribution attached to these cut-offs and scales with sample size — but because they're the numbers a credit risk committee already recognises and acts on. A monitoring framework nobody acts on is worse than none.

We'll come back to this. On our own data, these thresholds **miss** a 75% rise in the default rate. That's a finding, and we handle it without simply lowering the number until it fires.

### The simulation config, and the parameter that changes everything

`src/config.py:237-250`:

```python
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
    private_signal_default: float = 0.0
    private_signal_policy: float = 0.0
    seed: int = RANDOM_SEED
```

Most of these are self-explanatory. Three are not.

**`legacy_policy_noise = 1.60`.** This is the standard deviation of the judgemental component of the old approval policy. It sounds like a nuisance parameter. It is not. Set it too low and the old policy becomes a near-deterministic function of the features, which means approval is perfectly predictable from the model's own inputs — complete separation — and reject inference becomes trivially impossible for uninteresting numerical reasons. Set it too high and the policy is random, selection bias vanishes, and there's nothing to study. 1.6 gives a policy with realistic discriminating power: approved accounts default at 12.4%, declined at 37.6%, roughly a 3× gap. That's what a decent real scorecard achieves.

**`group_b_income_bias = 0.85`.** The injected fairness violation. Recorded income for group B is 15% below their true income, while true default risk depends on *true* income. We'll come back to why this specific form matters.

**`private_signal_default` / `private_signal_policy`, both `0.0` by default.** These two select the *selection regime*, and they're the single most consequential switch in the project. Both zero means the old policy used only information the model can also see — selection on observables, MAR. Non-zero means the old policy acted on a latent signal the model never receives — MNAR. Under MAR reject inference has nothing to fix; under MNAR it has real work. The default book is MAR and `mnar_config()` builds the variant.

I did not have these parameters when I first wrote this file. They were added later, after the reject inference results came out looking odd. I'll tell that story properly when we get there.

### Feature spec — the line that defines "fair"

`src/config.py:307-321`:

```python
SYNTHETIC_FEATURES: Final[FeatureSpec] = FeatureSpec(
    numeric=(
        "income_recorded",
        ...
        "age",
    ),
    categorical=("housing_status", "product_type"),
    protected=("group", "age_band"),
)
```

and `src/config.py:301-304`:

```python
    @property
    def model_features(self) -> list[str]:
        """All columns used as model inputs."""
        return [*self.numeric, *self.categorical]
```

**This is the signature decision that constrains everything downstream.** `model_features` excludes `protected`. Every model in the system is fitted on `model_features`, so no model ever sees `group` or `age_band`. The protected attributes stay in the dataframe purely so the audit can slice by them.

The tempting thing — and I've seen it in production code — is to have one feature list and drop protected columns at fit time inside each model. That version leaks: someone adds a model, forgets the drop, and now you have a protected attribute in a live scorecard. Making `model_features` a *property that cannot include them* means the exclusion is structural.

Note `age` **is** in `numeric` while `age_band` is in `protected`. That's deliberate and uncomfortable, and I'll come back to it in `explain.py`.

---

## 3. `src/simulate.py` — the measuring instrument

### Why this file exists

Because without it you cannot validate reject inference or the fairness audit, full stop. Everything else in this repo is testable against real data. These two are not.

It's a separate file from `config.py` because config holds *parameters* and this holds *process*. It's separate from everything else because it's the one module that gets to know the answers.

### What "ground truth" actually means here

Three columns exist that no real book has:

- `pd_true` — the true probability of default for every applicant
- `default_true` — the realised outcome for **everyone**, including the declined
- `officer_signal` — the latent signal, when the MNAR variant is on

And one column that a real book *does* have, which is where the whole problem lives:

- `default_observed` — `default_true` for approved applicants, `NaN` for the declined

### The DGP, and why it looks the way it does

`src/simulate.py:133-147`:

```python
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
```

Notice what is **not** in this list: any term involving `group`. Group membership has no direct causal effect on default in this world. Every risk difference between groups flows through legitimate features. That's what makes the fairness decomposition possible later — we can separate "genuinely different risk" from "the model made this up."

The last five parameters are non-linear terms, and there's a story attached.

**They weren't there originally.** The first version of this DGP was purely linear in the logit. I ran the model comparison and LightGBM *lost* to plain logistic regression. Which makes complete sense: if the true model is linear-in-logit, a linear model is correctly specified and a booster has nothing to find but noise. But it would have meant publishing "gradient boosting doesn't beat a scorecard" from a setup that guaranteed that result. That's rigging the answer.

So the interaction and two threshold effects went in — and all three are ordinary credit phenomena, not synthetic contrivances. High utilisation and recent delinquency genuinely compound. Affordability rules genuinely bite at a threshold. Thin files genuinely carry a step change in risk.

`src/simulate.py:340-346`:

```python
    # Non-linear structure: an interaction and two threshold effects. These are
    # what a gradient booster can find and an additive scorecard cannot.
    logit += c.utilisation_x_delinquency * (
        _standardise(utilisation) * np.minimum(n_delinq_24m, 3)
    )
    logit += c.high_dti_cliff * (debt_to_income > c.high_dti_threshold)
    logit += c.thin_file_penalty * (credit_history_months < c.thin_file_months)
```

`np.minimum(n_delinq_24m, 3)` caps the interaction. Without the cap, an applicant with 14 delinquencies gets an enormous interaction term and the logit runs away. Capping is what a real scorecard does too — beyond three missed payments, the marginal information from a fourth is small.

### The injected fairness violation — read this one carefully

`src/simulate.py:277-284`:

```python
    # THE INJECTED FAIRNESS VIOLATION.
    # Recorded income understates true income for group B — informal or
    # cash-based earnings that the application process fails to capture. True
    # default risk depends on income_true; the model only ever sees
    # income_recorded. The model is therefore biased against group B through a
    # channel it cannot observe and dropping `group` does nothing about it.
    income_recorded = np.where(is_group_b, income_true * config.group_b_income_bias, income_true)
    income_recorded = np.round(income_recorded, 2)
```

And critically, at `src/simulate.py:327`:

```python
        + c.log_income_true * _standardise(np.log(income_true))
```

**`income_true`, not `income_recorded`.** That one word is the entire fairness violation. The outcome depends on true income; the model sees recorded income; for group B those differ by 15%.

**The trap.** The natural thing to write is `np.log(income_recorded)` — you're generating features, why would you use a column the model can't see? And it would run fine. Every distribution would look right. The fairness audit would find a disparity (because group B genuinely earns less) but the *decomposition* would find zero manufactured bias, because the model would be seeing exactly what generated the outcome. You'd conclude "the gap is entirely explained by risk" and you'd have proved nothing except that your simulator has no bias in it.

This is the archetypal case of a wrong version that looks *better*: cleaner code, sensible-seeming, and it silently deletes the phenomenon you built the simulator to study.

Why measurement bias specifically, rather than a discriminatory coefficient? Because measurement bias is the failure mode that **survives dropping the protected attribute**. A coefficient on `group` is a bug any reviewer catches. This one is invisible to the model, invisible to the feature list, and only detectable if you have ground truth — which is precisely the point.

### Solving for the target rate instead of hand-tuning

`src/simulate.py:361-363`:

```python
    # Solve the intercept so the population default rate lands on a realistic
    # target rather than whatever the coefficients happen to produce.
    logit += _solve_intercept_shift(logit, config.target_population_default_rate)
```

The first version of this file just set `intercept = -1.55` and hoped. It produced a 33% population default rate — wildly unrealistic. Bisection (`_solve_intercept_shift`, `src/simulate.py:171-199`) makes the target explicit and adjustable: change `target_population_default_rate` and the generator hits it.

### The approval cutoff — a genuine off-by-conceptual-error

`src/simulate.py:398-403`:

```python
    # Cut at the quantile that delivers the target approval rate exactly. A
    # fixed 0.5 cut on a probability would leave the realised approval rate at
    # the mercy of the score distribution.
    legacy_cutoff = float(np.quantile(legacy_logit, config.target_approval_rate))
    legacy_score = _sigmoid(legacy_logit)
    approved = (legacy_logit < legacy_cutoff).astype(int)
```

**The trap, and I walked straight into it.** The obvious version is:

```python
# WRONG — this is what I wrote first
legacy_logit += _solve_intercept_shift(legacy_logit, 1.0 - config.target_approval_rate)
legacy_score = _sigmoid(legacy_logit)
approved = (legacy_score < 0.5).astype(int)
```

That solves for the *mean predicted probability* to equal 0.30, then thresholds at 0.5. But the mean of a sigmoid is not its median, so the realised approval rate came out at **0.768** against a target of 0.70. Nearly seven percentage points off, from code that looks entirely reasonable.

And here's why it's dangerous rather than merely wrong: nothing downstream errors. You get a loan book. It has an approval rate. Every subsequent analysis runs. You just quietly aren't studying the population you think you're studying, and every reject-inference number is computed on a different selection intensity than you documented.

The quantile version is exact by construction. Note it thresholds on `legacy_logit`, not `legacy_score` — thresholding the sigmoid would work too since sigmoid is monotone, but going through the logit avoids any question about tie handling at the boundary after a floating-point transform.

### Default timing — added late, for the vintage curves

`src/simulate.py:368-381`:

```python
    months = np.arange(1, config.performance_window_months + 1)
    hazard = np.exp(-0.5 * ((months - 6.5) / 3.0) ** 2)
    hazard = hazard / hazard.sum()
    months_to_default = np.where(
        default_true == 1,
        rng.choice(months, size=n, p=hazard),
        0,
    )
```

This was **not** in the first version. I added it when I got to `monitoring.py` and realised vintage analysis would produce a single point per cohort rather than a curve, which is useless. A Gaussian hazard peaking at month 6.5 is the right shape for unsecured consumer lending: very early defaults are rare (and usually fraud), the bulk emerge once the account has been carried a while.

Note `0` for non-defaulters, not `NaN`. That lets the vintage curve filter with `(timing <= month) & (timing > 0)` without null handling. If you use `NaN`, the comparison silently returns `False` and you get the right answer by accident — until someone changes the comparison direction.

### The oracle columns

`src/simulate.py:427-434`:

```python
            # Oracle columns. A real book has none of these.
            "officer_signal": np.round(officer_signal, 4),
            "income_true": np.round(income_true, 2),
            "pd_true": pd_true,
            "default_true": default_true,
            # Months on book at default; 0 for non-defaulters. Drives the
            # vintage curves in src.monitoring.
            "months_to_default": months_to_default.astype(int),
```

They're in the frame, and there's exactly one thing stopping them leaking into a model: `SYNTHETIC_FEATURES.model_features` doesn't list them. That's thin protection resting on a config constant, so there's a test that asserts it (`tests/test_simulate.py::test_oracle_columns_are_not_model_features`). That test is doing real work — it's the only thing between you and a model that trivially achieves AUC 0.99 because someone typed `pd_true` into a feature list.

### The censoring

`src/simulate.py:405`:

```python
    default_observed = np.where(approved == 1, default_true.astype(float), np.nan)
```

`.astype(float)` before the `where` because `np.where` with an int array and `np.nan` upcasts anyway, but doing it explicitly means the dtype is predictable rather than depending on numpy's promotion rules. The column has to be float — it holds NaN — and `observed_training_frame` casts back to int after filtering to approved rows.

---

## 4. `src/scorecard.py` — the benchmark

### Why this comes before the fancy models

Because it's the incumbent. In most credit functions there is already a WOE scorecard in production, and the burden of proof sits with anything proposing to replace it. If you build LightGBM first, you'll unconsciously treat the scorecard as the thing you compare against *afterwards*, and your write-up will read that way.

Also, practically: the scorecard produces a `points_table()` that the reason-code system depends on. Build it first and `explain.py` has something to lean on.

### The WOE sign convention — pick one and never waver

Two conventions exist. `WOE = ln(%good / %bad)` or `ln(%bad / %good)`. Both are used in industry. We pick the first:

`src/scorecard.py:163-170`:

```python
        # Smoothing keeps WOE finite when a bin is all-good or all-bad.
        pct_good = (good + WOE_SMOOTHING) / (total_good + WOE_SMOOTHING * len(ordered_labels))
        pct_bad = (bad + WOE_SMOOTHING) / (total_bad + WOE_SMOOTHING * len(ordered_labels))

        woe[label] = float(np.log(pct_good / pct_bad))
        event_rate[label] = bad / n
        count[label] = n
        iv_parts[label] = float((pct_good - pct_bad) * woe[label])
```

**Higher WOE means lower risk.** That single choice propagates: the logistic regression predicting *bad* will have **negative** coefficients on WOE, points must be computed with a negative sign, and the sign-flip detector looks for *positive* coefficients. Get the convention backwards and every one of those flips, and the code will still run and produce a plausible-looking scorecard with inverted points.

**The smoothing.** `WOE_SMOOTHING = 0.5` added to counts, with the denominator adjusted by `0.5 × n_bins` so the shares still sum to 1. Without it, a bin containing zero bads gives `ln(x/0) = inf`, which propagates to an infinite coefficient and a `LinAlgError` or a silently useless model.

**The trap:** the tempting fix is to clip the WOE afterwards — `np.clip(woe, -5, 5)`. That gets you a finite number, but it's a *lie about the data*: it says "this bin is 5 units safer than average" when what the data actually says is "we have no bads here, which might be signal or might be four observations." Laplace smoothing shrinks toward the prior in proportion to how little data you have, which is the honest behaviour. The clipped version is worse *and* looks fine.

### Monotonic binning — the merge loop

This is the most intricate function in the file. `src/scorecard.py:212-265`.

Two passes, and the order matters.

Pass one, size and purity, `src/scorecard.py:223-241`:

```python
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
```

The `break` after each merge, then re-entering the `while`, is deliberate. You must recompute `counts` and `rates` after every merge, because merging changes bin membership. The tempting version — iterate over all bins collecting merges, then apply them — is wrong, because your indices refer to the pre-merge edge list and after the first merge they're all off by one.

**The edge-index arithmetic is the off-by-one hazard.** `edges` has `n_bins + 1` entries. Bin `b` sits between `edges[b]` and `edges[b+1]`. To merge bin `b` with bin `b+1` you drop `edges[b+1]`. To merge bin `b` with bin `b−1` you drop `edges[b]`. So:

- `b == 0` has no left neighbour → merge right → drop `edges[1]`
- `b == last` has no right neighbour → merge left → drop `edges[len(edges) - 2]`
- otherwise merge into whichever neighbour is *smaller*, because merging into the larger one buries the small bin's signal

Get any of these wrong by one and you merge the wrong pair. The result is a scorecard with slightly different bin boundaries. It will still fit. It will still have plausible IV. You will never notice.

The degeneracy check is exact float equality, which normally deserves suspicion. Here it's correct — these are means of 0/1 arrays, so a pure bin is exactly `0.0` or `1.0` with no rounding involved. It was originally written as `rates[b] in (0.0, 1.0)`, which is the same test but reads like a mistake, so it's now spelled out with a comment saying why exact comparison is fine. Cheap change; saves every future reviewer the same double-take.

Pass two, monotonicity, `src/scorecard.py:246-265`:

```python
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
```

We *infer* the direction rather than assuming it. Utilisation rises with risk; income falls with risk. Hard-coding "increasing" would mangle every protective feature.

`diffs = np.diff(rates) * direction` normalises so a violation is always `< 0` regardless of direction. Neat, and it means one branch instead of two.

`np.argmin(diffs[violations])` — worst violation first, not first violation. Merging the worst first tends to resolve several violations at once and produces coarser, more stable bins. Merging first-found produces more merges and a card that moves around between refits.

The `edges[: worst + 1] + edges[worst + 2 :]` drops `edges[worst + 1]`, merging bins `worst` and `worst + 1`. `np.diff` index `i` compares bins `i` and `i+1`, so a violation at index `worst` means those two bins are out of order. Off by one here and you merge an innocent adjacent pair while leaving the violation in place — and the loop still terminates, because it terminates on merge count, so you get a *non-monotone* card from a function that claims to enforce monotonicity.

### The points formula

`src/scorecard.py:514-522`:

```python
    @property
    def factor(self) -> float:
        """Points per unit log-odds."""
        return self.scaling.pdo / np.log(2.0)

    @property
    def offset(self) -> float:
        """Points offset anchoring the base score to the base odds."""
        return self.scaling.base_score - self.factor * np.log(self.scaling.base_odds)
```

`pdo / ln(2)` because "points to double the odds" means: adding `pdo` points multiplies the odds by 2, so `pdo = factor × ln(2)`.

`src/scorecard.py:584-587`:

```python
        woe_frame = self.encoder.transform(X)[self.feature_names_]
        log_odds_bad = self.model.decision_function(woe_frame.to_numpy())
        # score = offset + factor * ln(good odds) = offset - factor * ln(bad odds)
        return self.offset - self.factor * log_odds_bad
```

**The minus sign is the whole thing.** `decision_function` returns log-odds of *bad*. Score is defined on *good* odds. `ln(good odds) = −ln(bad odds)`.

**The trap:** write `self.offset + self.factor * log_odds_bad` and you get a scorecard where high scores mean high risk. It runs. It produces numbers in the 400–700 range. Its AUC is identical (rank correlation is just negated, and AUC is symmetric under that if you flip the label convention). You'd catch it eventually when the band assignment put your safest applicants in band E — but only if you looked.

And the per-bin points, `src/scorecard.py:606-608`:

```python
            for label, woe_value in binning.woe.items():
                points = -self.factor * (coefficient * woe_value + intercept / n_features)
                points += self.offset / n_features
```

The intercept and offset are divided by `n_features` and distributed across characteristics, so that summing the points across features reconstructs the total score exactly. That reconstruction is asserted in `tests/test_scorecard.py::test_points_table_reproduces_the_score` — which matters, because the points table *is* the deployable artefact. A scorecard whose printed table doesn't sum to its own score is not a scorecard, it's a model with a misleading table attached.

### Sign-flip removal

`src/scorecard.py:541-559`:

```python
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
                ...
                self.model.fit(woe_frame[self.feature_names_].to_numpy(), y, sample_weight=sample_weight)
```

Given our sign convention, a *positive* coefficient means the multivariate fit reversed the feature's own univariate trend. On our data this correctly catches `loan_amount`: bigger loans look protective univariately (rich people borrow more) but are risky conditional on income. A card containing it would award more points for a larger loan, and no adverse action notice derived from that survives challenge.

Refitting inside the loop rather than dropping all positives at once is the correct-but-slower choice: removing one feature redistributes its correlated signal and can resolve — or create — a flip elsewhere.

**Honest note on this file:** `WOEEncoder` and `Scorecard` are two classes doing one job, and `Scorecard` mostly delegates. If I were writing it again I'd probably collapse them, or make `WOEEncoder` a proper scikit-learn transformer with `get_feature_names_out` so it composes into a `Pipeline`. As it stands the split exists mainly because I built the encoder first and the scorecard on top, and the seam is still visible.

---

## 5. `src/models.py` — the challengers

### Why a separate file, and why a Protocol

Because calibration, policy, explanation and serving all need to hold "a thing that predicts probabilities" without caring which. `src/models.py` defines that contract:

```python
@runtime_checkable
class ProbabilityModel(Protocol):
    """Minimal interface every model in this system satisfies."""

    def fit(
        self, X: pd.DataFrame, y: np.ndarray, sample_weight: np.ndarray | None = ...
    ) -> ProbabilityModel:
        """Fit the model to features and a binary default flag."""
        ...

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return P(default) as a one-dimensional array."""
        ...
```

**Two signature decisions here constrain the entire rest of the project.**

First: `predict_proba` returns a **one-dimensional array of P(default)**, not scikit-learn's `(n, 2)` matrix. Every consumer downstream — calibrators, `sweep_thresholds`, the fairness audit, the engine — takes a flat vector. If you leave the sklearn shape, every single call site needs `[:, 1]` and one of them will eventually be `[:, 0]`, which gives you probability-of-*good* and a model that appears to have AUC 0.16. That's an inverted model that looks like a catastrophically bad one rather than a bug, and people waste days on it.

Second: `sample_weight` is in the `fit` signature **from the start**, even though nothing uses it yet at this point in the build. It's there because `reject_inference.fuzzy_augmentation` will need it three files from now. If you leave it out, you get to reject inference and discover the entire model zoo needs re-signing.

### LightGBM settings — restrained, and not on principle

```python
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        num_leaves: int = 8,
        max_depth: int = 3,
        min_child_samples: int = 200,
```

These look timid. They were **selected by a sweep on the held-out validation block**, never the test set, and the restrained settings won outright: raising capacity to 1500 trees and 63 leaves *lowers* out-of-time AUC from 0.819 to 0.794.

This matters for the honesty of the headline comparison. If the scorecard beats a deliberately crippled booster, the finding is worthless. Document the sweep.

### Stable categories across splits

`src/models.py:141-152`:

```python
    def _prepare(self, X: pd.DataFrame, fitting: bool) -> pd.DataFrame:
        """Coerce categoricals to a stable category dtype shared across splits."""
        prepared = X[self.feature_names].copy()
        for column in self.categorical_features:
            if fitting:
                self.categories_[column] = sorted(prepared[column].astype(str).unique())
            prepared[column] = pd.Categorical(
                prepared[column].astype(str), categories=self.categories_[column]
            )
        for column in self.numeric_features:
            prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
        return prepared
```

**This is a silent-corruption trap and it is worth understanding properly.**

LightGBM encodes `pd.Categorical` by its integer codes. Those codes come from the *category ordering*. If you let pandas infer categories independently on train and test, and the test split happens to contain a different set of levels — say `product_type` has no `auto` in one month — then the codes shift. `card` might be `0` at train time and `0` at test time by luck, or `1` if a level dropped out.

The model doesn't error. It doesn't warn. It scores every applicant against the wrong category. Your AUC drops by a few points and you assume it's drift.

Storing `categories_` at fit time and reusing it at transform time is what makes this correct. `sorted()` makes it deterministic rather than dependent on row order.

`X[self.feature_names]` also enforces column *order*, which matters for the same reason.

The first version of this folded both jobs into one `_prepare(X, fitting=True)`, which mutated `self.categories_` as a side effect of a method whose name suggests it's pure — and gave you a bare `KeyError` if you scored before fitting. It's now split, and `prepare` refuses rather than guessing:

```python
        if self.categorical_features and not self.categories_:
            raise NotFittedError(
                "Category orderings are not available; call fit() before prepare(). "
                "Preparing with inferred categories would encode this frame "
                "differently from the training data."
            )
```

That error message earns its length. The failure it prevents isn't "you forgot to fit" — it's "you scored a batch against categories derived from that batch", which produces numbers rather than an exception.

### The UCI validation strategy — being honest about what you cannot do

```python
def lagged_feature_view(features: FeatureSpec) -> FeatureSpec:
    """Restrict a UCI feature spec to the earliest repayment window.
    ...
    This tests leakage-resistance. It does **not** test drift robustness,
    because the dataset contains no time axis to drift along.
    """
```

UCI has six months of repayment history and **no origination dates**. Temporal validation isn't hard on it, it's *impossible* — the column doesn't exist. Your three options are: random split (dishonest), pretend row order is temporal (worse), or build a genuine forward gap between the feature window and the outcome window. We do the third and state plainly what it does and doesn't buy.

---

## 6. `src/calibration.py` — the probability layer

### Why this is a first-class module and not three helper functions

Because of one fact that people nod along to and then ignore:

**AUC is invariant to any monotone transformation of the score.**

Multiply every predicted probability by 0.5. AUC doesn't move by a thousandth. Every number the business acts on — expected loss, provisioning, risk-based pricing, the cost-optimal cutoff — is now wrong by a factor of two.

There is a test that pins exactly this, and it's one of my favourites in the suite:

```python
def test_ece_detects_a_scaled_probability(...)
    y, p = perfectly_calibrated
    scaled = p * 0.5
    assert roc_auc_score(y, scaled) == pytest.approx(roc_auc_score(y, p), abs=1e-12)
    assert expected_calibration_error(y, scaled) > 5 * expected_calibration_error(y, p)
```

AUC identical to twelve decimal places; ECE five times worse. That's the whole argument in four lines.

### Platt on the logit, not the probability

`src/calibration.py:57-60`:

```python
def _logit(p: np.ndarray) -> np.ndarray:
    """Log-odds of a probability, clipped away from 0 and 1."""
    clipped = np.clip(np.asarray(p, dtype=float), PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON)
    return np.log(clipped / (1.0 - clipped))
```

`PROBABILITY_EPSILON = 1e-6` keeps logits inside ±13.8. A model that returns a hard 0 or 1 — LightGBM will, on a pure leaf — otherwise produces `-inf` and the calibrator's `fit` dies with a cryptic sklearn error.

Fitting Platt on `logit(p)` rather than raw `p` is what makes the fitted parameters *mean* something: `a = 0, b = 1` recovers the identity, so the two numbers are literally the calibration intercept and slope. Fit on raw probability and you get two coefficients that are correct but uninterpretable.

### Calibration slope — the diagnostic ECE can't give you

`src/calibration.py:397-402`:

```python
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    model = LogisticRegression(C=1e6, max_iter=2000, solver="lbfgs")
    model.fit(_logit(p).reshape(-1, 1), y)
    return float(model.intercept_[0]), float(model.coef_[0][0])
```

`C=1e6` is effectively no regularisation. **This is important and easy to get wrong.** sklearn's `LogisticRegression` defaults to `C=1.0`, which is meaningful L2 penalty. Leave the default and the fitted slope is shrunk toward zero, so a perfectly calibrated model reports a slope of maybe 0.93 instead of 1.00. You'd conclude your model is slightly overconfident when it's fine. Every subsequent calibrator comparison inherits the distortion — and, again, nothing errors.

The `len(np.unique(y)) < 2` guard returns NaN rather than raising. That's the right call for a *metric*: a subgroup with no defaults is a legitimate thing to encounter in a fairness audit, and it should produce "not measurable" rather than crash the whole audit.

The interpretation is what earns this its place next to ECE:

- **slope < 1** — predictions too spread out, overconfident. Classic over-fitted booster.
- **slope > 1** — too compressed, underconfident. Classic over-regularised model.
- **intercept ≠ 0** — the overall level is off, usually a base-rate shift.

ECE tells you *how much* you're wrong. Slope and intercept tell you *how*, and those have different remedies.

### ECE on quantile bins, and reporting both

`src/calibration.py:346-350`:

```python
    table = reliability_table(y, p, n_bins=n_bins, strategy=strategy)
    if table.empty:
        return float("nan")
    weights = table["count"].to_numpy(dtype=float)
    return float(np.sum(weights * table["gap"].abs().to_numpy()) / weights.sum())
```

Population-weighted, not a plain mean over bins. An unweighted mean gives a bin with 4 applicants the same influence as one with 4,000.

Default strategy is `"quantile"` (equal mass). On an imbalanced credit book with most PDs under 0.2, uniform-width bins put nearly everyone in the first two bins and leave eight nearly empty — which flatters the result. We compute both and report both, so nobody has to argue about which was chosen.

### Isotonic — and the trap of evaluating a calibrator on its own data

Isotonic fits an arbitrary monotone step function. It will usually win on in-sample Brier. It has two real costs, and our own numbers show both:

| Evaluated | Isotonic ECE | Isotonic slope |
|---|---|---|
| In-period (the data it was fitted on) | **0.0000** | 1.000 |
| Out-of-time (the data it will face) | 0.0251 | **0.726** |

A perfect zero in-period. A slope of 0.726 out of time, meaning it made the model *more* overconfident. If you only ever evaluated in-period — which is exactly what you do if you fit and score on the same frame — you'd ship isotonic and think you'd nailed it.

There's a test for the other isotonic cost too:

```python
def test_isotonic_loses_a_little_ranking_to_ties(...):
    assert len(np.unique(calibrated)) < len(np.unique(broken))
    assert roc_auc_score(y, calibrated) == pytest.approx(roc_auc_score(y, broken), abs=0.01)
```

Isotonic is a *step* function, so it creates ties, so it does **not** preserve AUC exactly. Platt does, to floating point. I originally wrote one test asserting both preserve ranking to `1e-3` and it failed — correctly. That failure was the code telling me something true, and the fix was to split the test and assert the real property of each, not to loosen the tolerance until it passed.

### The selection rule, written down

```python
def select_calibrator(comparison: pd.DataFrame, slope_tolerance: float = 0.15) -> str:
    eligible = comparison[(comparison["calibration_slope"] - 1.0).abs() <= slope_tolerance]
    if not eligible.empty:
        return str(eligible.sort_values("ece_quantile").iloc[0]["calibrator"])
    logger.warning(...)
    return str(comparison.sort_values("brier").iloc[0]["calibrator"])
```

Slope gate first, then ECE. A good Brier with a slope of 0.73 means the model is still systematically overconfident, and no amount of average-case accuracy makes that acceptable for pricing.

Writing the rule down rather than taking an argmin means a reviewer can disagree with the *policy* explicitly, instead of reverse-engineering it.

**And it produced an uncomfortable result, which we report rather than hide:** on the current build this rule selects **`raw`** — the identity. Recalibration did not improve on LightGBM's own output out of time. That's a real finding. It also means the deployed configuration doesn't exercise the Platt or isotonic code paths at all, which is worth knowing when you read the coverage numbers.

---

## 7. `src/evaluation.py` — metrics that don't lie

### The accuracy foil

```python
        # Present strictly as a foil. See the module docstring.
        "accuracy_at_0p5": float(accuracy_score(y, (p >= 0.5).astype(int))),
        "accuracy_majority_baseline": max(base_rate, 1.0 - base_rate),
```

These two always appear together, and that's the entire design. On our test set: accuracy 0.857, majority baseline 0.829. The model's entire apparent advantage over predicting "nobody ever defaults" is 2.7 percentage points.

I considered removing accuracy entirely. I kept it because it's the number a non-specialist stakeholder will ask for, and refusing to show it is less useful than showing it next to the thing that makes it meaningless. There's a test that asserts they stay close together, which is a slightly odd-looking test until you realise it's protecting the *rhetorical* structure.

### KS statistic

`src/evaluation.py:93-97`:

```python
    order = np.argsort(p)
    y_sorted = y[order]
    cumulative_bad = np.cumsum(y_sorted) / max(int(y_sorted.sum()), 1)
    cumulative_good = np.cumsum(1 - y_sorted) / max(int((1 - y_sorted).sum()), 1)
    return float(np.max(np.abs(cumulative_good - cumulative_bad)))
```

The `max(..., 1)` guards prevent division by zero when a slice has no bads or no goods. Returns a meaningless-but-finite number rather than a NaN that poisons a whole comparison table.

`np.argsort(p)` ascending — the direction doesn't matter because we take the max of the absolute difference, but it does matter that both cumulative curves use the *same* ordering. Sorting one and not the other is the kind of bug that gives you KS ≈ 1.0 and a very exciting afternoon.

### Precision at review capacity

```python
    for capacity in capacities:
        k = max(int(round(capacity * n)), 1)
        selected = y[order[:k]]
```

`order = np.argsort(-p)` — descending, riskiest first. `max(..., 1)` so a 1% capacity on a 50-row frame reviews one file rather than zero.

This is the metric that maps onto an actual constraint. "Our review team can work 5% of applications" is a budget, and this says what the budget buys. A model you can only deploy at a capacity the business doesn't have is not deployable.

### The resampling experiment

The received wisdom is that SMOTE trades calibration for ranking. Our numbers say it doesn't even manage the trade:

| Regime | ROC-AUC | PR-AUC | Mean predicted | Observed | ECE |
|---|---|---|---|---|---|
| none | **0.8379** | **0.5736** | 0.145 | 0.171 | **0.0255** |
| class_weight | 0.8355 | 0.5716 | 0.424 | 0.171 | 0.2531 |
| SMOTE | 0.7829 | 0.4406 | 0.390 | 0.171 | 0.2197 |
| random undersample | 0.8238 | 0.5279 | 0.450 | 0.171 | 0.2790 |

Class weighting leaves ranking untouched and inflates mean predicted PD from 0.145 to 0.424. SMOTE *loses* 0.13 PR-AUC as well. The mechanism isn't subtle — resampling changes the base rate the model is fitting to, and every emitted probability inherits the distortion.

I wrote the module docstring before running this, describing the conventional expectation. Then the numbers came in and I rewrote it to match what actually happened. Worth saying out loud: if your docstring predicts your results, one of them is fiction.

The experiment uses `SMOTENC` rather than `SMOTE` when categoricals are present:

```python
    categorical_positions = [
        i for i, column in enumerate(X_train.columns) if not pd.api.types.is_numeric_dtype(X_train[column])
    ]
```

Plain `SMOTE` on a frame with string columns raises. `SMOTENC` needs *positional* indices, not names — a detail that costs you twenty minutes the first time.

---

## 8. `src/policy.py` — the actual answer

### Why this is a separate file and why it's the headline

Every module before this produces a *score*. This one produces a *decision*, and decisions have costs. The headline number of the whole system comes out of here, and it's measured in currency.

### The accounting convention

`src/policy.py:98-119`:

```python
    approved_good = approve & (y == 0)
    approved_bad = approve & (y == 1)

    margin_earned = float(np.sum(exposure[approved_good]) * costs.margin_rate)
    loss_incurred = float(np.sum(exposure[approved_bad]) * costs.lgd)
    profit = margin_earned - loss_incurred
```

Relative to writing no business: approve a good `+margin × EAD`, approve a bad `−LGD × EAD`, decline `0`.

**The trap here is double-counting.** The natural instinct is: "declining a good applicant has an opportunity cost, so I should charge `C_fp` for every declined good." But if you *also* credit `+margin` for every approved good, you've counted the same quantity twice, and the resulting "cost" scales with your approval rate in a way that has no economic meaning. The frontier comes out visibly wrong — but only if you know what shape it should be.

Under our convention the opportunity cost is implicit: declining a good means forgoing `+margin`, which shows up as profit you didn't earn. `net_cost = -profit` so that "minimise cost" and "maximise profit" are the same instruction.

### Evaluate on realised outcomes, not predictions

The docstring at `src/policy.py:80-84` says it:

```
    Uses **realised** defaults, not predicted probabilities. Evaluating a
    policy with the same probabilities that produced it is circular: a model
    that is confidently wrong would score itself as highly profitable.
```

This is worth dwelling on because the circular version is *very* tempting — you have `p` right there, and using it means you don't need labels. A model that assigns PD 0.01 to everyone would, evaluated against its own predictions, report near-zero expected loss and enormous profit. It would top your league table.

### The sweep, and forcing the analytic point into the grid

`src/policy.py:147-151`:

```python
    quantiles = np.unique(np.quantile(p, np.linspace(0.0, 1.0, n_points)))
    # Always include the closed-form optimum and the endpoints.
    candidates = np.unique(
        np.concatenate([[0.0], quantiles, [costs.analytic_threshold], [1.0]])
    )
```

The grid is on score *quantiles*, not evenly spaced probabilities. With most PDs bunched below 0.2, a linear grid on [0, 1] spends 80% of its points in a region containing almost no applicants. Quantile spacing puts resolution where the people are.

Explicitly injecting `analytic_threshold` matters because the test asserts the empirical optimum lands on it. If the closed-form point isn't in the grid, the test compares against the nearest quantile and passes or fails on grid resolution rather than on correctness.

### The strict inequality

`src/policy.py:155`:

```python
        approve = p < threshold
```

Strict `<`, consistently, everywhere — here, in `policy_summary`, in `apply_policy`, in `DecisionEngine.decide`. At `threshold = 0.0` this approves nobody, which is the correct boundary. Use `<=` and threshold zero approves everyone whose PD is exactly zero, which is nobody in practice but makes the frontier endpoints inconsistent.

The real reason to care: if `sweep_thresholds` uses `<` and the engine uses `<=`, an applicant sitting exactly on the cutoff gets a different answer from the offline analysis than from the live service. Our worked example has PD `0.10715` against a threshold of `0.10714` — that's how close to the boundary real marginal cases sit.

### Expected loss versus realised loss

`src/policy.py:158-163`:

```python
        # The model's own view of the loss it is taking on. Compare against
        # realised_loss: the gap is a direct readout of calibration quality.
        outcome["expected_loss_model"] = float(
            np.sum(p[approve] * costs.lgd * exposure_vector[approve])
        )
        outcome["expected_loss_error"] = outcome["expected_loss_model"] - outcome["realised_loss"]
```

This is the calibration argument restated in money. A well-calibrated model can predict its own loss; a miscalibrated one can't. On our out-of-time test every model under-predicts its own loss by 490k–785k.

There are two tests pinning this in both directions — one asserting the error is under 5% for a calibrated model, one asserting it exceeds 20% when you multiply the probabilities by 0.5. That second test is checking that the *metric* has teeth, not that the model is good.

### Threshold rules, computed to be shown wrong

```python
def f1_optimal_threshold(y: np.ndarray, p: np.ndarray, n_points: int = 200) -> float:
    """Find the cutoff maximising F1, purely so it can be shown to be wrong.
```

Results on our test set:

| Rule | Cutoff | Net cost | Excess vs best |
|---|---|---|---|
| Cost-optimal (empirical) | 0.120 | −2,584,041 | 0 |
| Cost-optimal (closed form) | 0.107 | −2,506,563 | 77,478 |
| F1-optimal | 0.211 | −1,854,306 | **729,735** |
| Naive 0.5 | 0.500 | **+1,227,921** | **3,811,962** |

The F1 threshold costs more than twice the entire gap between the best and worst *model* in the comparison. The naive 0.5 loses money outright — positive net cost.

That 77,478 gap between the closed-form and empirical optima isn't noise, and it isn't a bug in the closed form. It's the price of the model being miscalibrated on a drifted book. The closed form is exactly right *for a calibrated model*.

**We deploy the analytic threshold, not the empirical one.** The empirical optimum is fitted to the test set's realised outcomes; using it would be selecting a hyperparameter on test data.

### `optimal_threshold` and a mypy-driven rewrite

`src/policy.py:178-179`:

```python
    best_row = int(frontier["net_cost"].to_numpy().argmin())
    return float(frontier["threshold"].to_numpy()[best_row])
```

The natural version is `frontier.loc[frontier["net_cost"].idxmin(), "threshold"]`. That's more idiomatic pandas and it's what I wrote first. It fails mypy under `pandas-stubs`, because `.loc[...]` on a DataFrame returns a union type that includes `str`, `datetime` and a dozen others, and `float()` won't accept it.

Going via `.to_numpy()` sidesteps the typing problem and is also faster. Worth noting it's not *purely* cosmetic: `idxmin` returns an index *label*, and if the frontier ever carried a non-default index, `.loc` with that label would still work while positional `.to_numpy()[i]` would not. Here `sweep_thresholds` always returns `ignore_index=True` so they coincide — but that's a coupling worth being aware of.

---

## 9. `src/reject_inference.py` — the intellectual core

### The problem

Your training data has outcomes only for applicants the previous policy **approved**. The model learns `P(default | features, approved by old policy)` and you deploy it to answer `P(default | features)`. On our book approved accounts default at 12.4% and declined applicants would have defaulted at 37.6%.

### Why the oracle is the most important function here

```python
def oracle_model(model_factory, X_all, y_true_all) -> Any:
    """Train on true outcomes for the whole population.

    Only possible on synthetic data. This is the ceiling: it says how much of
    the selection bias *any* correction could remove, so that a method closing
    30% of the gap can be recognised as closing 30% rather than reported as a
    success in the abstract.
    """
```

Without a ceiling, "this method reduced reject bias by 0.02" is uninterpretable. With it, you discover that under MAR the *entire achievable* improvement was 0.020 — the oracle only moves reject bias from −0.051 to −0.031. Suddenly every method's result reframes.

### Fuzzy augmentation

`src/reject_inference.py:212-226`:

```python
    y_approved = np.asarray(y_approved).astype(int)
    if base_model is None:
        base_model = fit_approved_only(model_factory, X_approved, y_approved)

    p_reject = np.clip(base_model.predict_proba(X_rejected) * uplift, 1e-4, 1.0 - 1e-4)

    X_augmented = pd.concat([X_approved, X_rejected, X_rejected], ignore_index=True)
    y_augmented = np.concatenate(
        [y_approved, np.ones(len(X_rejected), dtype=int), np.zeros(len(X_rejected), dtype=int)]
    )
    weights = np.concatenate([np.ones(len(X_approved)), p_reject, 1.0 - p_reject])
```

Each reject appears **twice** — once labelled bad with weight `p`, once labelled good with weight `1 − p`. The concatenation order of `X`, `y` and `weights` must match exactly across all three: approved block, then rejects-as-bad, then rejects-as-good.

**The trap.** Get the weight order backwards — `[ones, 1 - p_reject, p_reject]` — and you've labelled every reject's *bad* copy with the probability of being *good*. The model trains fine. The output is a systematically inverted correction for the reject population. And because rejects are only 30% of the augmented data and the approved block dominates, the resulting model still has a perfectly respectable AUC. You would not notice from any headline metric.

`ignore_index=True` on the concat matters: without it you get duplicate index values, and any downstream `.loc` or `groupby` on that frame behaves unpredictably.

The clip to `[1e-4, 1-1e-4]` is because `p × uplift` with uplift 2.5 can exceed 1.0, and a negative weight (`1 − 1.3`) makes most estimators either error or silently do something indefensible.

**The honest weakness**, stated in the docstring: the weights come from the very model whose bias is the problem. The only genuinely new information entering the procedure is the uplift factor — which is why the method is so sensitive to it.

### The uplift assumption, and measuring it

Industry guidance says 2× to 4×. On a real book **you cannot check it** — it's the missing counterfactual again. Here we can:

```python
def measure_true_uplift(scores, approved, y_true, n_bands: int = 10) -> pd.DataFrame:
```

with a guard:

```python
        if approved_mask.sum() < 20 or rejected_mask.sum() < 20:
            continue
```

Bands with almost no rejects produce uplift ratios that are pure noise, and they're at the safe end of the score where almost everyone was approved. Including them drags the volume-weighted average around meaninglessly.

The measured answer on our MAR book: **1.09**, against an assumed 2.5. That mismatch is why the corrections overcorrect.

### The finding, and the second regime

Under MAR (true uplift 1.09):

| Method | Reject bias | Bias removed |
|---|---|---|
| Baseline (approved only) | −0.051 | 0% |
| Fuzzy augmentation | **+0.102** | **−268%** |
| Parcelling | **+0.102** | **−269%** |
| Heckman two-step | −0.082 | −162% |
| *Oracle* | *−0.031* | *100%* |

Every correction made it worse.

That result is what caused me to go back and add `private_signal_default` / `private_signal_policy` to `SimulationConfig`. Because the honest question was: is reject inference useless, or is my simulator in the easy regime? The answer was the second. My legacy policy selected only on features the model could also see — selection on observables — so the approved-only model already generalised and there was nothing to fix.

Under MNAR (true uplift 1.90):

| Method | Reject bias | Reject Brier | Bias removed |
|---|---|---|---|
| Baseline | −0.154 | 0.2010 | 0% |
| Fuzzy augmentation | **−0.007** | **0.1756** | 211% |
| Parcelling | +0.009 | 0.1764 | 208% |
| Heckman two-step | −0.146 | 0.2033 | **11%** |
| *Oracle* | *−0.084* | *0.1795* | *100%* |

Here it genuinely works. Same code, opposite conclusion, and **on a real book you cannot tell which regime you're in**, because the evidence that would settle it is the missing counterfactual.

`bias_removed_share` can exceed 1.0 — the oracle is limited by model capacity and irreducible noise, not by selection, so it isn't a bound on bias alone. That's documented at the function rather than left as a puzzle.

### The trap I only found by writing tests

The measured uplift depends on how good your *model* is, not just on the selection mechanism. Same MAR book:

| Training rows | Measured uplift | Baseline reject bias |
|---|---|---|
| 4,000 | 2.25 | −0.120 |
| 10,000 | 1.21 | −0.059 |
| 20,000 | 1.09 | −0.051 |

Nothing about selection changed. A weak model leaves risk heterogeneity *inside* each score band, and the old policy sorted on that heterogeneity — so rejects within a band really are worse. The uplift is real, measurable, and entirely an artefact of the scoring model.

**Practical consequence: measuring a 2× uplift on your own book is not evidence that you need reject inference.** It's equally consistent with an underpowered model, in which case reject inference will overcorrect. The two explanations are not distinguishable from the uplift figure alone.

I found this because a test failed. I'd written the MAR conclusions against a 6,000-row book (fast tests) and asserted them; on the small book the conclusions *reversed*. The temptation was to loosen the assertion. The right move was to work out why, discover a real phenomenon, and give the test a full-size fixture with a comment explaining exactly why it can't be small.

### Heckman

`src/reject_inference.py:380-394`:

```python
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
```

`predict(..., linear=True)` returns the **linear index** `z'γ`, not the probability. The Mills ratio is `φ(z'γ)/Φ(z'γ)` and needs the index. Pass the probability and you get a finite, plausible-looking number that is not the Mills ratio, and the correction silently does nothing useful.

`np.clip(norm.cdf(...), 1e-8, None)` because for a very negative index the CDF underflows to exactly 0 and you divide by zero.

Then prediction:

```python
        # Mills term set to zero: we want the population relationship, not the
        # one conditional on having been approved.
        design = np.column_stack(
            [self._design(numeric, self.feature_columns), np.zeros(len(X))]
        )
```

**Zeroing the Mills column at prediction time is the entire point of the method** and it's the step people miss. If you compute the Mills ratio at prediction time and include it, you're predicting `E[y | X, approved]` all over again — exactly the conditional quantity you were trying to escape. Setting it to zero gives you the unconditional relationship.

The wrong version produces predictions that are *closer to the approved-only baseline*, which is to say it looks like the correction "didn't do much" rather than like a bug.

Two honest caveats in the docstring: Heckman assumes jointly normal errors and our DGP is logistic, so it's approximate before any estimation error; and the exclusion restriction is a gift we manufactured. A real book rarely has one. The 11% figure should be read as an *upper bound*.

---

## 10. `src/fairness.py` — the audit

### The framing decision

This module does not de-bias anything. It measures, and it presents the choice. Which point on the accuracy/fairness frontier a lender operates at is a policy decision with legal content, and a modeller silently picking one has made that decision on someone else's behalf without telling them.

### Equal opportunity, defined for credit

```python
                # P(approve | would repay). The equal opportunity rate: how
                # often a creditworthy applicant actually gets credit.
                "approval_rate_given_good": float(approve_g[good].mean()) if good.any() else np.nan,
                # P(approve | would default). Reported so that "fairness" gains
                # achieved by approving more bads are visible.
                "approval_rate_given_bad": float(approve_g[bad].mean()) if bad.any() else np.nan,
```

The favourable outcome is approval; the "qualified" group is those who wouldn't default (`y == 0`). So equal opportunity is parity of `P(approve | y=0)`.

Both rates get reported. If you only report the first, a method that equalises it by approving more bad applicants from the disadvantaged group looks like an improvement. It isn't — you've made worse loans to the people you were trying to help.

### Small-group suppression

```python
MIN_GROUP_SIZE = 100
```

with

```python
        if n < min_group_size:
            logger.info("Skipping group %r: only %d rows", value, n)
            continue
```

Below about a hundred, subgroup metrics are noise. Publishing them invites over-reaction to sampling variation, and in a fairness context an over-reaction can be as damaging as a miss. It logs rather than silently dropping, so the omission is visible.

### The decomposition — the payoff for building the simulator first

```python
    summary["model_excess"] = summary["mean_pd_predicted"] - summary["mean_pd_true"]

    # Reference group is the one with the lowest true risk; gaps are measured
    # against it so the "excess" column reads as bias relative to the safest.
    reference_row = int(summary["mean_pd_true"].to_numpy().argmin())
    reference_true = float(summary["mean_pd_true"].to_numpy()[reference_row])
    reference_predicted = float(summary["mean_pd_predicted"].to_numpy()[reference_row])

    summary["true_pd_gap_vs_lowest"] = summary["mean_pd_true"] - reference_true
    summary["predicted_pd_gap_vs_lowest"] = summary["mean_pd_predicted"] - reference_predicted
```

Result:

| Group | True mean PD | Predicted mean PD | True gap | Predicted gap | **Manufactured** |
|---|---|---|---|---|---|
| A | 0.1535 | 0.1221 | — | — | — |
| B | 0.2013 | 0.1901 | 0.0478 | 0.0679 | **+0.0202** |

The groups differ in real risk by 4.8 points. The model claims 6.8. **The extra 2.0 points is manufactured** — that's the injected income measurement bias, detected. The simulator paid for itself right here.

Note we measure *gaps relative to a reference group*, not raw `model_excess`. Both groups show negative raw excess (−0.031, −0.011) because the whole model under-predicts on the drifted test set. Comparing raw excess would conflate a global calibration problem with a fairness problem. The *relative* gap isolates the fairness component.

**The trap**: reporting `model_excess` directly and concluding "the model under-predicts for both groups, so there's no bias." That reads as reassuring and is exactly backwards.

The function raises rather than degrading:

```python
    if truth_column not in frame.columns:
        raise ValueError(
            f"{truth_column!r} not available; disparity decomposition needs ground truth "
            "and is only possible on synthetic data"
        )
```

That's a deliberate error-handling split. Missing *protected attributes* → log and skip (a real audit runs on whatever attributes it has). Missing *ground truth* → raise, because silently returning a decomposition computed without truth would be actively misleading.

### The trade-off frontier

```python
        for g in unique_groups:
            mask = group_values == g
            target_rate = (1.0 - lam) * baseline_rates[g] + lam * overall_rate
            group_scores = p[mask]
            if target_rate <= 0.0:
                group_threshold = -np.inf
            elif target_rate >= 1.0:
                group_threshold = np.inf
            else:
                group_threshold = float(np.quantile(group_scores, target_rate))
            thresholds[g] = group_threshold
            approve[mask] = group_scores < group_threshold
```

λ=0 must reproduce the single global threshold exactly, and there's a test asserting it does to `rel=1e-9`. That anchoring matters because the whole chart is read relative to that point — if λ=0 drifted, every "cost of parity" number would be measured from the wrong baseline.

The `≤0` / `≥1` guards handle degenerate targets, where `np.quantile` would either error or return a boundary value that approves/declines an off-by-one number of people.

Results:

| λ | Net cost | DPD | AIR | EOD |
|---|---|---|---|---|
| 0.0 | −2,506,563 | 0.159 | 0.755 | 0.136 |
| 0.4 | −2,502,660 | 0.095 | 0.849 | 0.063 |
| **0.8** | −2,484,132 | 0.032 | 0.947 | **0.004** |
| 1.0 | −2,465,535 | **0.000** | **1.000** | 0.037 |

Full parity costs 41,028 — **1.64% of profit**. And the curve is non-monotone, which the docstring insists you read correctly: "moving to parity costs approximately nothing here", *not* "λ=0.7 is the optimum." A bumpy flat curve has no peak, and treating one as if it did is how spurious operating points get shipped.

Also note EOD bottoms at λ=0.8 and *rises* at full parity. Equalising approval rates across groups with different base rates necessarily unequalises the rate at which creditworthy applicants get approved. You cannot have both.

`max_group_ece` is constant across every λ, by construction, and there's a test asserting `nunique() == 1`. That test looks pointless until you understand what it's protecting: the idea that you can fix a miscalibrated subgroup by moving its cutoff. You can't. Every point on that curve is built on the same probabilities.

**A note on how this file got messy and was cleaned up:** `fairness_cost_frontier` carried a duplicated `Args:` block for a while — I inserted the "Reading the output honestly" section into the middle of the docstring and left the original `Args:` stranded above it. Harmless at runtime, renders twice in generated docs, and exactly the kind of thing that survives indefinitely because no test looks at docstrings. Worth a periodic `inspect.getdoc(fn).count("Args:")` sweep if you write docstrings this long.

---

## 11. `src/explain.py` — reason codes

### Two different jobs, routinely confused

**Global explanation** answers "what drives this model?" — that's for validation and challenge. **Adverse action reason codes** answer "why was *this person* declined?" — that's a legal obligation attached to an individual decision, and it has to still hold up months later.

### Points shortfall — the deployed method

`src/explain.py:162-195`:

```python
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
```

For each characteristic, compare points awarded against the **best achievable on that characteristic**. Largest shortfalls are the principal reasons.

**The trap.** The tempting baseline is the *population mean* points rather than the maximum. That version produces *negative* shortfalls for applicants who are above average on a characteristic — and then your top-4 sort can rank a feature the applicant did *well* on as a reason for declining them. It looks fine for a poor applicant (everything's below average) and produces nonsense for a marginal one. Marginal applicants are exactly who receive adverse action notices.

Using the max guarantees shortfall ≥ 0. There's a test asserting exactly that across twenty applicants.

The virtue of this method over SHAP: it's recoverable from the printed points table alone. In three years, with the pickle format changed and the SHAP version long gone, you can still reconstruct why someone was declined from a CSV.

### The `age` problem

```python
PROTECTED_BASIS_FEATURES: frozenset[str] = frozenset({"age", "age_band", "sex", "marriage", "education"})
```

and

```python
            protected_basis=str(row["feature"]) in PROTECTED_BASIS_FEATURES,
```

`age` is a model feature and it's a protected basis under ECOA. Three options: drop it (understates the model's real behaviour and loses genuine signal), emit "your age band" onto a letter unnoticed (bad), or flag it for legal review.

We flag. It's a **deferral, not a resolution**, and it's listed as a pre-launch blocker in the model card. On the worked example SHAP flagged `age` at rank 4; points shortfall didn't. The two methods disagreeing on a protected-basis reason is exactly the situation you want surfaced rather than averaged away.

### Reason codes only for adverse decisions

```python
        # Reasons are only meaningful for a decline. Sending "principal reasons"
        # to an approved applicant would be nonsense.
        "reason_codes": [] if approved else [code.to_dict() for code in reason_codes],
```

Note the caller still *computes* the codes and passes them in; the filter is at the record level. Slightly wasteful, and the engine avoids it by not computing them at all for approvals.

### `ShapExplainer` — the messiest thing in the repo

```python
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
```

This was the worst code in the project until recently, and the history is worth knowing because the smell is common.

The original reached into `model._prepare(X, fitting=False)` — a private method on another class — because `TreeExplainer` needs the *prepared* frame with the stable categorical dtypes and there was no public accessor. It duck-typed on `hasattr(inner, "booster_")`, coupling this module to a LightGBM attribute name. And it stashed a lambda closing over `model`, which quietly made the whole explainer unpicklable.

The fix was to give `LightGBMModel` a public `prepare()` and duck-type on *that*. Note the second-order benefit: `_prepare` is now a method rather than a stored lambda, so an explainer can be pickled alongside a model artifact:

```python
    def _prepare(self, X: pd.DataFrame) -> pd.DataFrame:
        """Prepare a frame the same way the wrapped model does.

        A method rather than a stored lambda: a closure over ``model`` makes the
        whole explainer unpicklable, which matters the moment anyone tries to
        cache one alongside a model artifact.
        """
        if self.kind == "tree":
            return self.model.prepare(X)
        return X[self.feature_names]
```

The general lesson: when you find yourself reaching for another object's underscore-prefixed method, the missing thing is usually a public method on *that* object, not a workaround in yours.

Also note `values.ndim == 3` handling:

```python
        # Some SHAP versions return one matrix per class for binary problems.
        if values.ndim == 3:
            values = values[:, :, -1]
```

SHAP changed this return shape between versions. `[:, :, -1]` takes the positive class. The defensive branch stays, but `pyproject.toml` now upper-bounds the dependency (`shap>=0.44,<0.50`) so a third shape nobody has tested against can't arrive through a routine `pip install -U`. Handling two known shapes is defensible; leaving the range open and hoping is not.

---

## 12. `src/monitoring.py` — production reality

### PSI, and where the bins come from

`src/monitoring.py:125-141`:

```python
    if edges is None:
        edges = np.unique(np.quantile(reference, np.linspace(0.0, 1.0, n_bins + 1)))
    if len(edges) < 3:
        return 0.0

    inner = edges[1:-1]
    reference_counts = np.bincount(
        np.digitize(reference, inner, right=True), minlength=len(edges) - 1
    ).astype(float)
    current_counts = np.bincount(
        np.digitize(current, inner, right=True), minlength=len(edges) - 1
    ).astype(float)

    reference_share = np.maximum(reference_counts / reference_counts.sum(), PSI_EPSILON)
    current_share = np.maximum(current_counts / current_counts.sum(), PSI_EPSILON)

    return float(np.sum((current_share - reference_share) * np.log(current_share / reference_share)))
```

**The single most important line is `np.quantile(reference, ...)`.**

**The trap.** The natural thing is to bin on the combined sample, or on the current sample — it feels more balanced, and it guarantees non-empty bins. It's wrong. Re-binning on current data partly absorbs the very shift you're measuring. Your PSI comes out lower than reality. **The metric fails in the direction of reassurance**, which is the worst possible direction for a monitoring metric.

`minlength=len(edges) - 1` on the bincount matters: without it, if the current sample has nothing in the top bins, `np.bincount` returns a *shorter* array and the subtraction against reference either broadcasts wrongly or raises.

`PSI_EPSILON = 1e-4` floors the shares so an empty bin gives a large-but-finite contribution rather than `inf` or `nan`.

### The finding: PSI missed it

Vintage analysis sees the deterioration immediately:

| Cohort | 12-month default rate | vs first |
|---|---|---|
| v00–05 | 17.51% | — |
| v18–23 | 16.36% | −6.6% |
| v24–29 | 21.76% | +24.3% |
| v30–35 | **30.65%** | **+75.0%** |

Conventional input-drift monitoring did not fire:

| Indicator | Value | Threshold | Status |
|---|---|---|---|
| Score PSI | 0.103 | 0.25 | watch, not breached |
| Worst feature PSI | 0.212 | 0.25 | watch, not breached |
| Calibration ECE | 0.026 | 0.05 | fine |

A framework operated strictly on 0.25 would have raised **nothing** while losses rose three quarters.

The response is *not* to lower the threshold until it fires on this incident — that's fitting the governance to the accident, and it produces false positives on every other book. Instead:

```python
    # Combined-watch rule. Added because on this book the single-indicator
    # thresholds did not fire on a deterioration that raised realised losses by
    # 75%. Two indicators sitting in the watch band at once is itself a signal,
    # and waiting for either to reach "review" independently costs a year of
    # business.
    both_watching = bool(
        score_psi >= thresholds.psi_warn and worst_feature_psi >= thresholds.psi_warn
    )
```

And state plainly that input-drift metrics cannot see a change in the feature-to-outcome *relationship*, which is where the damage came from.

### Vintage curves

```python
        for month in range(1, max_months + 1):
            defaulted = int(((outcome == 1) & (timing <= month) & (timing > 0)).sum())
```

`timing > 0` excludes non-defaulters, who carry `0`. Drop that clause and every non-defaulter counts as having defaulted at month 0, giving you a 100% default rate at month 1 — which is at least obviously wrong. The subtler error is `timing < month` instead of `<=`, which shifts every curve one month right and understates early-life defaults.

`group_size: int = 6` pools vintages into six-month cohorts. Monthly cohorts on a 30,000-row book are too thin to read.

### Triggers carry actions

```python
@dataclass(frozen=True)
class TriggerResult:
    name: str
    value: float
    threshold: float
    breached: bool
    action: str
    rationale: str
```

`action` and `rationale` are **required fields**, not optional. That's a design decision expressed in a type: a monitoring pack that stops at a chart answers "is something wrong?" but not "and then what?". Making them required means you cannot add a trigger without answering both.

---

## 13. `src/ingest.py` — real data, deliberately late

### Why this is file thirteen and not file one

Because building it first would have meant building the whole system around a dataset that can't validate the two things that matter most. By the time we get here, the synthetic path is complete and this is an *additional* data source rather than the foundation.

### Idempotence

Every stage checks for its own output and skips:

```python
    if destination.exists() and not force:
        logger.info("Archive already present at %s; skipping download", destination)
        verify_checksum(destination)
        return destination
```

Note it still verifies the checksum on the skip path. Skipping the *download* is fine; skipping the *verification* would mean a corrupted local file is trusted forever.

### The checksum policy, and being honest about a gap

```python
EXPECTED_SHA256: str | None = None
CHECKSUM_ENV_VAR = "CREDIT_RISK_UCI_SHA256"
```

Trust-on-first-use: hash on first download, record to `data/raw/checksums.json`, enforce thereafter. Strict pinning available via environment variable.

**Why not a hard-coded hash?** Because the environment this was developed in cannot reach `archive.ics.uci.edu` — it's blocked by egress policy. No hash could be verified at authoring time. Hard-coding an unverified constant would *look* like a pinned checksum while providing none of the assurance, which is worse than admitting the gap. That admission is in the module docstring, not buried in a commit message.

**Be clear about what this means: the real-data path has never been run against the actual file.** It's implemented, schema-validated, and tested against a fixture built to the published layout. It is not verified end to end.

### Schema validation — fail loudly

```python
    numeric_columns = ["limit_bal", "age", *PAY_COLUMNS, *BILL_COLUMNS, *PAY_AMOUNT_COLUMNS]
    non_numeric = [
        column for column in numeric_columns if not pd.api.types.is_numeric_dtype(frame[column])
    ]
    if non_numeric:
        raise SchemaError(f"Expected numeric columns are not numeric: {non_numeric}")

    if not frame["age"].between(15, 120).all():
        raise SchemaError("Age values outside the plausible range 15-120")
```

The failure this is guarding against is a **column shift**. The published file has a title row above the real header. Parse with `header=0` and every column is offset by one row, or worse, the header row becomes data. The frame still loads. It has 30,000 rows and 25 columns. It trains a model. Every result is wrong.

The `age between 15 and 120` check catches this cheaply, because a shifted column puts credit limits in the age field and 200000 isn't a plausible age. Six negative tests cover the six failure modes.

### The one place we deliberately swallow an error

```python
    previous_bill = out["bill_amt2"].where(out["bill_amt2"] > 0, np.nan)
    out["pay_ratio_1"] = (out["pay_amt1"] / previous_bill).clip(upper=5.0)
    # No prior balance means nothing was owed, so a zero payment is not a
    # delinquency signal. Treated as fully paid rather than left missing.
    out["pay_ratio_1"] = out["pay_ratio_1"].fillna(1.0)
```

Zero or negative prior bill → NaN → filled with 1.0, meaning "paid in full." That's a *modelling* judgement disguised as null handling, and it's flagged as such in the comment. If you filled with 0.0 you'd be telling the model these applicants paid nothing, which reads as maximally delinquent when in fact they owed nothing.

`clip(upper=5.0)` because someone paying 50× their previous bill is a data artefact, and an unbounded ratio gives a few rows enormous leverage in a linear model.

---

## 14. `src/artifacts.py` — the mismatch guard

### The failure this exists to prevent

A model and its calibrator are two files, trained together, valid only together. Deploy model v4 against calibrator v3 and **nothing crashes**. The service starts. It scores everyone. It returns confident probabilities, all of them wrong. Ranking is fine, so AUC in your monitoring pack is fine. Expected loss, provisioning and the cutoff are all silently wrong, and the first hard evidence arrives a year later in the vintage curves.

### Why separate files

Storing them in one pickle would make the failure impossible — and would also not model reality. In a real deployment they *are* separate artefacts and they *can* be mismatched. Making the failure possible and then catching it is the useful design.

```python
    joblib.dump({"training_run_id": manifest.training_run_id, "object": model}, version_directory / MODEL_FILENAME)
```

The run ID travels **inside** each file, not just in the manifest. A file moved between version directories still carries its provenance. If the ID only lived in the manifest, copying `calibrator.joblib` from an old version into a new directory would pass every check.

Then `src/artifacts.py:301-313`:

```python
    # The check this module exists for.
    if model_run != calibrator_run:
        raise ArtifactMismatchError(
            f"Model and calibrator are from different training runs: "
            f"model={model_run!r}, calibrator={calibrator_run!r}. "
            "Serving this pair would produce well-ranked, wrongly-priced decisions. "
            "Rebuild the bundle or roll back to a consistent version."
        )
    if model_run != manifest.training_run_id:
        raise ArtifactMismatchError(...)
```

Two checks, not one. The files must agree with *each other* **and** with the manifest. Only the first and you can have a consistent pair whose manifest describes something else — and the manifest is what `/model-info` reports and what the audit log resolves against.

### Rollback that can't install a broken version

```python
def set_current_version(version: str, directory: Path | None = None) -> None:
    root = resolve_artifact_root(directory)
    load_bundle(version, root)
    (root / CURRENT_POINTER).write_text(version)
```

`load_bundle` **before** moving the pointer. It validates and raises. So a rollback under incident pressure cannot install a mismatched pair — which is precisely when you'd be least careful.

### `resolve_artifact_root` — a refactor forced by tests

```python
def resolve_artifact_root(directory: Path | None = None) -> Path:
    """Resolve the artifact root, defaulting to the configured directory.

    Resolved at call time rather than bound as a default argument, so tests and
    alternative deployments can redirect it without reimporting the module.
    """
    return ARTIFACT_DIR if directory is None else directory
```

Originally every function had `directory: Path = ARTIFACT_DIR`. Python binds default arguments **at function definition time**, so monkeypatching `src.artifacts.ARTIFACT_DIR` in a test does nothing — the default was captured at import. The API tests needed an isolated artifact directory, so this had to change to `None` + call-time resolution.

That's a general lesson: a module-level constant as a default argument is a testability trap, and you find out late.

---

## 15. `src/engine.py` — the single decision path

### Why this file exists

Because real-time and batch scoring that drift apart give the same applicant different answers depending which door they came through. That's indefensible to a regulator and genuinely hard to detect, because each path is individually self-consistent.

Both the API and the batch scorer call `DecisionEngine.decide` and neither implements any scoring, thresholding, banding or reason-code logic.

And it's asserted rather than trusted:

```python
def test_api_and_batch_produce_identical_decisions(...):
    # Batch: everything at once.
    batch = engine.decide(frame)

    # API: one row at a time, rebuilt from a dict exactly as the endpoint does.
    single = []
    for position in range(len(frame)):
        payload = frame.iloc[position].to_dict()
        single.extend(engine.decide(pd.DataFrame([payload])))
```

Scoring 120 applicants as a block and one at a time, requiring byte-identical results. This catches the whole class of bug where a preprocessing step depends on batch statistics — a scaler fitted on the incoming frame, a category inferred from the batch — which works perfectly in bulk and produces different answers for a single-row request.

### The exposure fallback chain

`src/engine.py:249-254`:

```python
        if exposure is not None:
            exposures = np.asarray(exposure, dtype=float)
        elif "loan_amount" in frame.columns:
            exposures = frame["loan_amount"].to_numpy(dtype=float)
        else:
            exposures = np.full(len(frame), self.costs.exposure_at_default, dtype=float)
```

Explicit argument → `loan_amount` → flat assumption. Getting this order backwards — flat default before the column — means every batch run uses 10,000 for everyone and your expected loss totals are wrong by whatever the actual loan size distribution is. Nothing errors; the numbers are just uniformly wrong in a way that looks plausible.

### Referral checked first

`src/engine.py:261-266`:

```python
            if self.referral_width > 0.0 and abs(probability - self.threshold) <= self.referral_width:
                outcome = "refer"
            elif probability < self.threshold:
                outcome = "approve"
            else:
                outcome = "decline"
```

Referral band before approve/decline, because the band straddles the cutoff. Check approve first and everything below threshold is already approved, so the band only ever catches the decline side — a silently asymmetric referral policy.

`referral_width` defaults to `0.0`, disabling it. A referral band only pays for itself if reviewers add information the model lacks, and that claim needs its own evidence. Defaulting it on would assume the conclusion.

### The input hash

```python
def hash_features(features: dict[str, Any]) -> str:
    canonical = json.dumps(features, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()
```

`sort_keys=True` is what makes it order-independent, and there's a test for it. Without it, two identical applications submitted with differently-ordered JSON produce different hashes, and your audit lookup by hash silently misses.

We hash rather than store the features so the audit log doesn't become a second copy of the applicant database, while anyone holding the original application can recompute and prove the tie.

The rounding is the second thing that matters:

```python
    rounded = {
        key: round(value, HASH_FLOAT_PRECISION) if isinstance(value, float) else value
        for key, value in features.items()
    }
```

Without it, `default=str` serialises floats by `repr`, so the digest depends on a float's exact trailing digits. An application whose utilisation arrives as `0.1 + 0.2` through one caller and `0.3` through another hashes differently while describing the same person — and your audit lookup by hash silently misses. For a record whose entire purpose is reconstructing a decision years later, tying it to floating-point noise is a needless fragility. Six decimals is finer than any feature this system accepts and coarse enough to be representation-independent; two tests pin both halves of that.

---

## 16. `src/api/` — the service

### `schemas.py` — strict on purpose

```python
    model_config = ConfigDict(extra="forbid")

    income_recorded: float = Field(gt=0, le=10_000_000, description="Annual income as recorded")
    debt_to_income: float = Field(ge=0, le=10, description="Debt repayments over income")
    utilisation: float = Field(ge=0, le=5, description="Balances over available limits")
```

Every field bounded, unknown fields rejected. A service that silently coerces a malformed value will decline someone for a reason that traces back to a typo in a caller's payload — and the audit log will faithfully record the *wrong reason*. Rejecting the request is the safer failure.

`extra="forbid"` catches the renamed-field bug: a caller sends `income` instead of `income_recorded`, and without `forbid` you'd get a validation error on the missing field but with `forbid` you get both halves of the story.

One wart: `ModelInfoResponse` needs

```python
    model_config = ConfigDict(protected_namespaces=())
```

because pydantic v2 reserves the `model_` prefix and we have `model_version`, `model_name`. The alternative is renaming the fields, which would make the API less clear to serve pydantic's namespace convention. Correct call, ugly line.

### `audit.py` — append-only, written before responding

```python
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            # Durability on the critical path: a decision the applicant has
            # already received must not be lost to a crash.
            os.fsync(handle.fileno())
```

Three things:

`self._lock` — FastAPI can handle requests concurrently, and interleaved partial lines corrupt the record irrecoverably.

`os.fsync` — `flush()` only pushes to the OS buffer. Without fsync, a crash loses decisions that applicants have already been told about.

**Write before responding.** The log write is on the critical path, deliberately. The failure mode to avoid is a decision the applicant received that the institution cannot evidence. That costs latency, and it's the right trade in this domain.

Malformed lines are skipped rather than fatal:

```python
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed audit line %d", number)
```

Deliberate split: during an investigation, a partially corrupted log is still worth searching. Refusing to read the whole file because line 40,000 is truncated helps nobody.

**Honest limits:** this is a local file. A real deployment needs WORM storage with retention locks, replication, and gap monitoring. The module says so rather than implying this is production-grade.

### `main.py` — refusing to start

```python
    try:
        state.load()
    except ArtifactMismatchError as error:
        # Deliberately fatal. See the module docstring.
        state.startup_error = str(error)
        logger.critical("REFUSING TO START: %s", error)
        raise
```

Re-raise, don't degrade. A service that's down is an incident someone notices in minutes. A service quietly serving a model against the wrong calibrator returns confident, well-ranked, wrongly-priced probabilities and nothing in ordinary monitoring reveals it.

### The stale-reference bug the tests found

```python
        # Always reassign, including to None. Leaving a previously loaded
        # reference in place would mean that reloading or rolling back to a
        # version without reference scores kept comparing against a different
        # model's distribution, and reporting the result as this model's PSI.
        reference_path = resolve_artifact_root(None) / manifest.version / REFERENCE_SCORES_FILENAME
        if reference_path.exists():
            self.reference_scores = np.load(reference_path)
            ...
        else:
            self.reference_scores = None
```

The original only *assigned* in the `if` branch. It passed every test I'd written until a new test loaded a second model without reference scores and got the *first* model's distribution back, reported as the second model's PSI.

The test failed only when run as part of the file, not alone — classic shared-state pollution. The lazy fix is to reset state in the fixture. The right fix was to notice that the pollution was revealing a real production bug: roll back to a version without reference scores and you'd compare against a different model's distribution forever.

---

## 17. `scripts/` — the drivers

### `run_experiments.py`

One script produces every table, every figure, and the deployable artifact. The single most valuable line is at the end:

```python
    # Prove the served artifact reproduces the experiment's own numbers.
    engine = DecisionEngine.from_artifacts()
    served = engine.predict_proba(X_test)
    max_difference = float(np.max(np.abs(served - primary_predictions)))
    logger.info("Served vs experiment max |difference|: %.3e", max_difference)
    if max_difference > 1e-9:
        logger.error("Served artifact does not reproduce experiment predictions")
        return 1
```

It scores the test set through the *serialised, reloaded* artifact and compares against the in-memory predictions the report was built from. Non-zero exit if they differ.

This catches the entire class of serialisation bug: a fitted attribute that doesn't pickle, a category ordering lost in the round-trip, a calibrator saved before its final fit. All of which produce a service that runs and disagrees with your report.

Current value: `0.000e+00`.

`matplotlib.use("Agg")` before importing pyplot — headless backend, or it tries to open a display in CI. The `# noqa: E402` comments on the subsequent imports are the price of that ordering constraint.

### `score_batch.py`

```python
        if write_audit:
            identifiers = (
                chunk[id_column].astype(str).tolist() if id_column in chunk.columns else None
            )
            AuditLog().record_many(
                [d.to_dict() for d in decisions], application_ids=identifiers, channel="batch"
            )
```

`channel="batch"` so a divergence between the two paths would be visible in the log itself. Batch decisions are logged **by default** — a lender must be able to evidence a limit reduction as readily as a declined application.

`CHUNK_SIZE = 20_000` bounds memory on a large portfolio.

Running `python scripts/score_batch.py` directly used to fail with `ModuleNotFoundError: No module named 'src'`, because Python puts the *script's own directory* on `sys.path`, not the repo root. Only `python -m scripts.score_batch` worked. The docstring said so, but the direct invocation is the natural thing to try and the error gave no hint. Now there's a bootstrap:

```python
# Allow `python scripts/score_batch.py` as well as `python -m scripts.score_batch`.
# Running a file directly puts the script's own directory on sys.path, not the
# repo root, so the `src` imports below would fail with a bare
# ModuleNotFoundError that gives the caller no hint about the fix.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
```

`__package__ in (None, "")` is the reliable "am I being run as a script?" test — it's `None` for a directly-executed file and the package name under `-m`. The cost is `# noqa: E402` on the imports below it, because they now follow a statement.

---

## 18. `tests/` — what each one actually protects

223 tests. Rather than list them, here's what the load-bearing ones are *for*.

### `conftest.py` — the sizing decision

```python
TEST_N = 6_000
```

Small enough that the suite runs in 40 seconds. Session-scoped fixtures so models are fitted once.

But then:

```python
@pytest.fixture(scope="session")
def large_book() -> pd.DataFrame:
    """A full-size MAR book for the reject inference conclusions.

    Sized deliberately. Reject inference results depend on how well the scoring
    model is trained: on a small book the model is weak, leaving risk
    heterogeneity inside each score band, which manufactures an apparent
    reject uplift of over 2x and makes the corrections look helpful.
    """
```

A second fixture at full size, existing solely because the MAR conclusions *reverse* on a small book. That comment is doing real work — without it, a future maintainer sees a slow fixture and "optimises" it, and the tests silently start asserting the opposite of the documented finding.

### Tests that protect specific bugs

**`test_income_measurement_bias_is_injected_exactly`** — asserts group A's recorded income equals true income to `rtol=1e-6` and group B's ratio equals 0.85 exactly. Catches the `income_recorded` vs `income_true` swap in the DGP, which would silently delete the phenomenon the whole simulator exists to create.

**`test_oracle_columns_are_not_model_features`** — the only thing between you and `pd_true` in a feature list.

**`test_points_table_reproduces_the_score`** — sums points from the table for 50 real applicants and compares to `score()` at `abs=1e-6`. Catches any drift between the deployable artefact and the model.

**`test_empirical_optimum_agrees_with_the_closed_form`** — the consistency check between `analytic_threshold` and the sweep. If these disagree, either the closed form or the cost accounting is wrong.

**`test_cost_accounting_is_arithmetically_correct`** — four hand-computed applicants:

```python
    assert outcome["margin_earned"] == pytest.approx(900.0)
    assert outcome["realised_loss"] == pytest.approx(7_500.0)
    assert outcome["total_profit"] == pytest.approx(-6_600.0)
```

Unglamorous and the most valuable test in `test_policy.py`. Every headline number depends on this arithmetic.

**`test_ece_detects_a_scaled_probability`** — AUC unchanged to 1e-12, ECE 5× worse. The one-test argument for the calibration module's existence.

**`test_reject_inference_hurts_under_mar`** — asserts the uncomfortable finding so it can't be silently lost:

```python
    assert overcorrected, "Expected label-inference methods to overcorrect under MAR"
```

**`test_calibration_is_invariant_to_the_threshold`** — `frontier["max_group_ece"].nunique() == 1`. Protects the *idea* that you can't fix miscalibration by moving cutoffs.

**`test_api_and_batch_produce_identical_decisions`** — described above; the most important test in the repo.

**`test_service_refuses_to_start_on_a_version_mismatch`** — asserts a *failure* is reachable. Tests that assert something breaks are as important as tests that assert something works, and they're rarer.

### The calibration gate

```python
TOLERANCES: dict[str, float] = {
    "brier": 0.010,
    "ece": 0.015,
    "mce": 0.040,
    "calibration_slope": 0.150,
    "roc_auc": 0.030,
}

ABSOLUTE_LIMITS: dict[str, float] = {
    "max_ece": 0.060,
    "max_brier": 0.140,
    "min_roc_auc": 0.740,
    "min_slope": 0.750,
    "max_slope": 1.300,
}
```

Two layers. Tolerances are relative to a committed baseline — they catch *regression*. Absolute limits are independent of it — they stop the gate being defeated by regenerating the baseline against a model that should never have shipped. Without the second layer the gate is circular: any regression can be laundered by running `--update`.

The slope test is two-sided (`abs(actual - expected)`) while ECE and Brier are one-sided (`actual > expected + tol`). Correct: lower ECE is always better, but a slope of 1.3 is as wrong as 0.7.

I verified the gate actually fails by injecting a fake pristine baseline. A regression gate you've never seen fail is a gate you don't know works.

### Three test bugs I hit, and what they taught

**Pearson where I needed Spearman, twice.** `score` vs `predict_proba` are related by `score = offset − factor × logit(p)` — strictly decreasing, badly non-linear. Pearson gave −0.87 and I'd asserted < −0.95. The relationship is *perfectly* monotone; Spearman gives exactly −1.0. Same mistake on threshold vs bad-rate. The lesson: if you're asserting "these move together," rank correlation is usually what you mean, and Pearson will make you loosen a tolerance to hide a shape you didn't think about.

**Isotonic doesn't preserve AUC.** I'd asserted both calibrators preserve ranking to `1e-3`. Isotonic failed at 0.0015. The failure was right — isotonic creates ties. The fix was to split the test and assert each method's actual property.

---

## 19. Docker and CI

`Dockerfile` trains in the builder stage, then:

```dockerfile
# Fail the build if the artifact cannot be loaded. Catching a model/calibrator
# mismatch here costs a red build; catching it in production costs a year of
# wrongly priced lending.
RUN python -c "from src.artifacts import load_bundle; b = load_bundle(); print('artifact ok:', b.manifest.version)"
```

The image that starts is the image that was validated. No training at container start.

**Caveat: this has not been executed.** There's no Docker daemon in the development environment. The Dockerfile and compose file are written and CI builds them, but that job has not run.

CI runs lint, mypy, tests on 3.11 and 3.12, the calibration gate, an end-to-end train/score/serve smoke test, and the Docker build. Nothing reaches the network for data — the UCI host is unreliable from CI, and a build that fails because a third-party server is down teaches nobody anything.

---

## Rebuild checklist

Starting from an empty directory, in this order:

1. `pyproject.toml` — package layout, ruff with `D`, mypy `disallow_untyped_defs`, pytest paths
2. `.gitignore` — ignore `data/`, `artifacts/`, `audit/`; keep `reports/tables` and `reports/figures`
3. `src/__init__.py`, `scripts/__init__.py`, `tests/__init__.py` — one-line docstrings each
4. `src/config.py` — paths, seed, `CostParameters` with `analytic_threshold`, `ScorecardScaling`, `SCORE_BANDS`, `MonitoringThresholds`, `SimulationConfig`, `SplitConfig`, `FeatureSpec` with protected excluded from `model_features`
5. `src/simulate.py` — `TrueCoefficients` (with the non-linear terms), `_sigmoid`, `_standardise`, `_solve_intercept_shift`, `simulate_loan_book`, `split_by_vintage`, `observed_training_frame`, `describe_simulation`
6. `tests/conftest.py` — session-scoped `book`, `splits`, `approved_splits` fixtures
7. `tests/test_simulate.py` — assert the injections exist at exact magnitude before trusting anything downstream
8. `src/scorecard.py` — `FeatureBinning`, `_compute_woe_table`, `_merge_small_and_monotonic`, `WOEEncoder`, `Scorecard` with `points_table` and sign-flip removal
9. `tests/test_scorecard.py` — points-table reconstruction first; it's the contract
10. `src/models.py` — `ProbabilityModel` Protocol (1-D `predict_proba`, `sample_weight` in `fit`), `LightGBMModel` with stable categories, `PenalisedLogisticModel`, `ScorecardModel`, `build_model_zoo`
11. `src/calibration.py` — `_logit`, the three calibrators, Brier + decomposition, `reliability_table`, ECE/MCE, `calibration_slope_intercept` with `C=1e6`, `CalibratedModel`, `select_calibrator`
12. `tests/test_calibration.py` — start with the scaled-probability test
13. `src/evaluation.py` — `ks_statistic`, `precision_at_capacity`, `gains_table`, `discrimination_report` with the accuracy foil, `resampling_experiment`
14. `src/policy.py` — `decision_outcomes`, `sweep_thresholds`, `optimal_threshold`, `policy_summary`, `compare_policies`, `approval_frontier`, `cost_sensitivity`, `f1_optimal_threshold`, `compare_threshold_rules`, `assign_score_band`, `apply_policy`
15. `tests/test_policy.py` — hand-computed cost arithmetic, then analytic-vs-empirical agreement
16. `src/reject_inference.py` — `fit_approved_only`, `oracle_model`, `fuzzy_augmentation`, `parcelling`, `HeckmanCorrectedModel`, `measure_true_uplift`, `evaluate_on_population`, `summarise_bias_removed`, `run_uplift_sensitivity`, `run_reject_inference_study`
17. Add `private_signal_*` to `SimulationConfig` and `mnar_config()` to `simulate.py` — you'll discover you need these when the MAR results come out flat
18. `src/fairness.py` — `group_performance`, `fairness_metrics`, `audit_fairness`, `fairness_cost_frontier`, `decompose_disparity`
19. `src/explain.py` — `REASON_TEMPLATES`, `PROTECTED_BASIS_FEATURES`, `ReasonCode`, `scorecard_reason_codes`, `ShapExplainer`, `explain_decision`, `format_adverse_action_notice`, `worked_example`
20. Add `months_to_default` to `simulate.py` — you'll need it in the next step
21. `src/monitoring.py` — `population_stability_index`, `categorical_stability_index`, `drift_report`, `score_distribution_report`, `vintage_curves`, `vintage_summary`, `evaluate_triggers`, `monitoring_snapshot`
22. `src/ingest.py` — download, checksum, extract, `load_raw`, `validate_schema`, `data_quality_report`, `engineer_features`, `ingest`
23. `src/artifacts.py` — `BundleManifest`, `resolve_artifact_root`, `save_bundle`, `load_bundle` with both mismatch checks, `set_current_version`, `list_versions`, `build_manifest`
24. `src/engine.py` — `hash_features`, `Decision`, `DecisionEngine.decide`
25. `src/api/schemas.py` — bounded pydantic models, `extra="forbid"`
26. `src/api/audit.py` — `AuditLog` with lock + fsync
27. `src/api/main.py` — lifespan that refuses to start, four endpoints
28. `scripts/run_experiments.py` — every table and figure, then the artifact, then the reproduction check
29. `scripts/score_batch.py` — same engine, `channel="batch"`
30. `tests/test_artifacts.py`, `tests/test_engine.py`, `tests/test_api.py` — mismatch guards and the API/batch identity test
31. `tests/test_calibration_gate.py` + `tests/calibration_baseline.json` — generate with `--update`, then verify it fails on an injected regression
32. `Dockerfile`, `docker-compose.yml` — train in builder, validate, run unprivileged
33. `.github/workflows/ci.yml` — lint, tests, gate, end-to-end, docker
34. `README.md`, `MODEL_CARD.md`, `DECISIONS.md`, `RUNBOOK.md` — written last, from the actual numbers

The two places you'll get stuck if you deviate: putting `sample_weight` into the model protocol *after* writing reject inference (you'll re-sign every model), and writing `artifacts.py` with `directory: Path = ARTIFACT_DIR` defaults (you'll rewrite six signatures when the API tests need isolation).
