# Decisions

Every judgement call in this repository and the reasoning behind it. Where I
chose the simpler option, that is recorded as a decision rather than left
implicit. Where a decision is contestable, the counter-argument is stated so a
reviewer can disagree with the reasoning rather than guess at it.

---

## Contents

- [A. Data and simulation](#a-data-and-simulation)
- [B. Modelling](#b-modelling)
- [C. Calibration](#c-calibration)
- [D. Evaluation](#d-evaluation)
- [E. Cost and policy](#e-cost-and-policy)
- [F. Reject inference](#f-reject-inference)
- [G. Fairness](#g-fairness)
- [H. Explainability](#h-explainability)
- [I. Monitoring](#i-monitoring)
- [J. Engineering](#j-engineering)
- [K. Simplifications](#k-simplifications)

---

## A. Data and simulation

### A1. Synthetic data was built first, before touching real data

**Decision.** `src/simulate.py` was written and validated before any modelling.

**Why.** Two of the claims this system makes are unfalsifiable on real data.
"Reject inference removes bias" cannot be checked without the outcome for
declined applicants, which no lender has. "The fairness audit detects real bias"
cannot be checked without knowing the true risk, which nobody observes. Building
a generator with retained ground truth is the only way to test either.

**Consequence.** Every headline number in the README is from synthetic data and
transfers to no real portfolio. That trade is worth it: a validated method on
fictional data is more useful than an unvalidated method on real data.

### A2. The DGP has no direct protected-attribute term

**Decision.** Group membership has no causal effect on default. All risk
differences flow through legitimate features; all *unfair* differences flow
through recorded income.

**Why.** It makes "some of the gap is real risk and some is manufactured"
precisely decomposable, which is exactly the question a real audit must argue
about without evidence. A direct group term would have made the answer trivial
and the exercise pointless.

### A3. The fairness violation is measurement bias, not a coefficient

**Decision.** Recorded income understates group B's true income by 15%, while
true risk depends on true income.

**Why.** Measurement bias is the failure mode that **survives dropping the
protected attribute**, which is the standard and inadequate mitigation. A model
that never sees `group` still marks group B as riskier than they are, through a
channel it cannot observe. That is the interesting case; a coefficient on a
protected attribute is a bug anyone would catch.

**Contestable.** A reviewer might argue that omitted-variable bias or label bias
(discriminatory historical defaults) are more common in practice. Both would be
reasonable additions; measurement bias was chosen because it is the cleanest to
inject at known magnitude and the hardest to detect.

### A4. Non-linearity was added to the DGP after the first results

**Decision.** After the first run, a utilisation × delinquency interaction, a
debt-to-income cliff at 0.45, and a thin-file step below 12 months were added.

**Why.** The original DGP was linear in the logit. Every linear model was
correctly specified, LightGBM had nothing to find, and it *lost* to plain
logistic regression. Reporting "gradient boosting does not beat a scorecard"
from that setup would have been rigging the headline finding. All three
additions are ordinary credit phenomena.

### A5. Two selection regimes, not one

**Decision.** `mnar_config()` produces a variant where a latent loan-officer
signal drives both approval and default.

**Why.** The first reject inference run showed the corrections making things
worse. Investigation showed why: the legacy policy selected only on features the
model could also see, so selection was ignorable and there was nothing to fix.
That is a real and important scenario but it is the *easy* one. Reporting only
it would have been a partial answer. Running both regimes turned "reject
inference doesn't help" into the far more useful "it depends on which regime you
are in, and you cannot tell from a real book."

### A6. An exclusion restriction was provided deliberately

**Decision.** `branch_capacity_index` affects approval and not default.

**Why.** Heckman needs one. Providing it gives the method its **best case**, so
that a poor result cannot be dismissed as a rigged setup. It removed 11% of the
bias at best, and that is with a gift a real book would not receive.

### A7. All accounts are fully seasoned

**Decision.** The snapshot is taken 12 months after the last origination, so
every vintage is mature. *Simplification.*

**Why.** A real book has an immature tail, which complicates vintage analysis
and the definition of the training population. Excluding it keeps the vintage
curves interpretable. Recorded here because it makes the monitoring section
easier than reality.

### A8. Realistic parameters, solved rather than hand-tuned

**Decision.** The intercept is solved by bisection to hit a 20% population
default rate; the approval cutoff is a quantile that delivers exactly 70%.

**Why.** The first version produced a 33% default rate and a 65% default rate
among declines — a policy far more predictive than any real scorecard. Solving
for the targets makes the parameters explicit and adjustable rather than the
accidental output of hand-picked coefficients.

---

## B. Modelling

### B1. The scorecard is the benchmark, not a footnote

**Decision.** `build_model_zoo` lists the scorecard first and every comparison
table includes it.

**Why.** It is the incumbent in most credit functions. The burden of proof sits
with anything proposing to replace it, and reporting it last invites the reader
to treat it as an afterthought.

### B2. LightGBM hyperparameters were selected on validation, and are restrained

**Decision.** 300 trees, 8 leaves, depth 3, `min_child_samples=200`, chosen by a
sweep on the held-out calibration block.

**Why.** Restraint here is a finding, not a preference: raising capacity to 1500
trees and 63 leaves *lowers* out-of-time AUC from 0.819 to 0.794. The extra
capacity is spent memorising a training period the test period no longer
resembles. Documenting the sweep matters because otherwise the scorecard
comparison looks rigged by a deliberately crippled opponent.

### B3. Sign-flipped features are dropped from the scorecard

**Decision.** Features whose fitted WOE coefficient is positive are removed
iteratively, refitting each time.

**Why.** Higher WOE means lower risk, so a positive coefficient means the
multivariate fit reversed the feature's own univariate trend. Such a card awards
*more* points for behaviour the bank calls riskier and no adverse action notice
derived from it would survive challenge. On this data it correctly catches
`loan_amount`, whose correlation with income makes it look protective. Dropping
it left test AUC unchanged — it was contributing nothing but indefensibility.

### B4. Monotonic binning is on by default

**Decision.** Adjacent bins violating the dominant risk trend are merged
worst-first.

**Why.** It stops the card telling an applicant that paying down their balance
made them riskier. The measured cost is **0.0004 AUC** — nothing. This was
worth measuring rather than assuming; the result is that the most defensible
property of a scorecard is free.

### B5. Information value floor of 0.02

**Decision.** Features below IV 0.02 are dropped.

**Why.** The conventional "no useful signal" level. Keeping noise features
destabilises a card across refits without adding discrimination. On this data it
drops `housing_status` and `product_type`.

### B6. UCI gets a pseudo-out-of-time design, not a random split

**Decision.** `lagged_feature_view` restricts UCI to the earliest three months
of repayment history.

**Why.** UCI has **no origination dates**. True temporal validation is
impossible — not difficult, impossible, because the column does not exist. The
options were: a random split (dishonest), pretend an ID ordering is temporal
(worse), or open a genuine forward gap between the feature window and the
outcome. The third catches leakage, which is half of what temporal validation is
for. It cannot test drift robustness, and that is stated wherever it appears.

### B7. Repeated stratified folds with dispersion, where no time axis exists

**Decision.** `repeated_stratified_evaluation` reports per-fold scores.

**Why.** A single random split hides instability behind a lucky seed. On an
imbalanced target, fold-to-fold variance is frequently larger than the
difference between the models being compared.

---

## C. Calibration

### C1. Calibration is a first-class module with its own CI gate

**Decision.** Not a post-processing step.

**Why.** AUC is invariant to any monotone transform of the score. Multiply every
probability by 0.5 and AUC does not move by a thousandth, while every number the
business acts on becomes wrong. Expected loss, provisioning, risk-based pricing
and the cost-optimal cutoff all consume a probability.

### C2. Platt is fitted on the logit, not the raw probability

**Decision.** `PlattCalibrator` regresses on `logit(p)`.

**Why.** It makes the identity recoverable at `a=0, b=1`, so the fitted
parameters are directly interpretable as the calibration intercept and slope.
Fitting on raw probability gives parameters that mean nothing on their own.

### C3. ECE is reported on quantile bins by default, and both are shown

**Decision.** Quantile (equal-mass) bins are the default; uniform bins are also
reported.

**Why.** On an imbalanced credit book, uniform-width bins put nearly every
applicant in the first two bins and leave the rest almost empty, which flatters
the result. Reporting both prevents an argument about which was chosen.

### C4. Calibration slope and intercept are reported alongside ECE

**Decision.** Both, always.

**Why.** ECE says *how much* a model is wrong; slope and intercept say *how*. A
slope below 1 means predictions are too spread out; an intercept away from 0
means the level is off. Those have different remedies and ECE cannot distinguish
them. This is also the diagnostic a supervisor will ask for.

### C5. The calibrator selection rule is written down, not searched

**Decision.** Prefer a slope within 0.15 of 1.0; among those, lowest ECE; fall
back to lowest Brier with a logged warning.

**Why.** A good Brier score with a slope of 0.73 means the model is still
systematically overconfident, which no amount of average-case accuracy makes
acceptable for pricing. Writing the rule down lets a reviewer disagree with the
policy explicitly rather than reverse-engineer it from an argmax.

**Consequence.** On the current build the rule selects **`raw`** for LightGBM —
recalibration did not improve on the model's own output out of time. That is
reported rather than quietly replaced with a calibrator that "should" win.

### C6. Calibration is reported both in-period and out-of-time

**Decision.** Both views, in the same table.

**Why.** Out-of-time is the deployment condition, but it confounds calibrator
quality with drift. In-period isolates the calibrator and is optimistic by
construction. The contrast is itself the finding: isotonic achieves an ECE of
**exactly 0.0000** in-period and a slope of **0.726** out of time. That is what
evaluating a calibrator on its own fitting data buys you.

---

## D. Evaluation

### D1. Accuracy is reported only beside the majority-class baseline

**Decision.** `accuracy_at_0p5` never appears without
`accuracy_majority_baseline`.

**Why.** 0.857 sounds respectable until you see that predicting "nobody
defaults" scores 0.830. Removing accuracy entirely would be cleaner, but it is
the number a non-specialist stakeholder will ask for; showing it next to its
foil is more useful than refusing to show it.

### D2. Precision at fixed review capacity, not at an arbitrary threshold

**Decision.** Top 1%, 5% and 10%.

**Why.** It maps onto an actual operating constraint. "Our review team can work
5% of applications" is a budget, and the metric says what that budget buys. A
model can only be deployed at a capacity the business has.

### D3. The resampling experiment tests calibration, not just ranking

**Decision.** Every regime is scored on ranking *and* calibration side by side.

**Why.** The usual claim is that resampling trades calibration for ranking. On
this book it does not even manage the trade: class weighting leaves ranking
untouched and inflates mean predicted PD from 0.145 to 0.424; SMOTE *loses*
0.13 PR-AUC as well. The experiment is designed so that result is visible rather
than assumed either way.

---

## E. Cost and policy

### E1. Accounting is relative to writing no business

**Decision.** Approve a good `+margin × EAD`; approve a bad `−LGD × EAD`;
decline `0`.

**Why.** The opportunity cost of declining a good applicant is implicit in the
margin forgone, so nothing is double-counted. The obvious alternative — charging
an explicit cost for each declined good *and* crediting margin for each approved
good — counts the same quantity twice.

### E2. The closed-form optimum is derived and cross-checked empirically

**Decision.** `p* = margin / (margin + LGD)`, asserted in tests to agree with
the sweep.

**Why.** If they disagree, either the sweep or the cost model is wrong and it
matters which. The closed form also makes explicit that **the optimal cutoff
does not depend on the model** — it is a property of the economics alone.

### E3. The applied cutoff is the analytic one, not the empirical best

**Decision.** Deploy at 0.1071, not the 0.120 that minimised realised cost on
the test set.

**Why.** The empirical optimum is fitted to the test set's realised outcomes and
would not generalise; using it would be selecting a hyperparameter on the test
data. The 77,478 gap between them is reported as **the cost of the model being
miscalibrated on a drifted book**, which is the honest interpretation.

### E4. F1 and naive-0.5 thresholds are computed purely to be shown wrong

**Decision.** `f1_optimal_threshold` exists so it can be beaten.

**Why.** F1 weights both error types equally. That is not a neutral default; it
is a specific and incorrect cost assumption applied silently. Showing that it
costs 729,735 — more than twice the entire gap between the best and worst model
— makes the point better than an argument would.

### E5. Cost sensitivity is swept and reported

**Decision.** LGD × margin grid.

**Why.** Both are assumptions with their own error, and both move with the
cycle. The cutoff ranges from 0.053 to 0.211 across plausible values. Quoting
one cutoff to four decimals implies a precision that does not exist.

### E6. Referral bands exist but are off by default

**Decision.** `referral_width=0.0`. *Simplification.*

**Why.** A manual review band only pays for itself if reviewers add information
the model lacks, and that claim needs its own evidence. Defaulting it on would
assume the conclusion.

---

## F. Reject inference

### F1. An oracle model is included

**Decision.** A model trained on true labels for everyone.

**Why.** Without a ceiling, "this method reduced bias by 0.02" is
uninterpretable. The oracle shows that under MAR the *entire achievable*
improvement was 0.020, which reframes every method's result. It is impossible in
practice and labelled as such.

### F2. Results are reported separately for rejects

**Decision.** Every metric is broken out for approved and rejected applicants.

**Why.** Rejects are 30% of the population, so a method can look fine overall
while being badly wrong about exactly the group it was built to fix.

### F3. The uplift assumption is measured, not just assumed

**Decision.** `measure_true_uplift` reports the real value.

**Why.** Fuzzy augmentation and parcelling both need an assumption that cannot
be validated on a real book. Measuring it turns "we assumed 2.5×, as the
literature suggests" into "we assumed 2.5× and the truth was 1.09×", which
explains the overcorrection instead of leaving it mysterious.

### F4. The negative result is reported prominently

**Decision.** "Reject inference tripled the bias under MAR" is in the README
headline section.

**Why.** It is the finding. A system that reports its methods worked when they
did not is worth less than one that reports honestly.

### F5. The sample-size trap is documented and tested

**Decision.** A test pins that measured uplift falls as training data grows.

**Why.** Discovered while writing the tests: the same MAR book yields a measured
uplift of 2.25 at 4,000 training rows and 1.09 at 20,000. Nothing about the
selection mechanism changed — a weak model leaves risk heterogeneity inside each
score band, and the old policy sorted on it. **Measuring a 2× uplift on a real
book is not evidence you need reject inference**; it is equally consistent with
an underpowered model. This seemed important enough to encode as a test so it
cannot be lost.

---

## G. Fairness

### G1. Protected attributes are excluded from every model, and the audit runs anyway

**Decision.** Excluded as inputs, retained for auditing.

**Why.** Exclusion is the standard mitigation and it is insufficient. The
adverse impact ratio is 0.755 by group and 0.451 by age band with no protected
attribute in any model. Demonstrating that is more useful than asserting it.

### G2. Both demographic parity and equal opportunity are reported

**Decision.** Both, plus equalised odds.

**Why.** They conflict, and the conflict is visible in the results: equal
opportunity difference is minimised at λ=0.8 and *rises* at full parity. A
report showing only one hides the choice being made.

### G3. Within-group calibration is treated as the most serious finding

**Decision.** Reported per group and given its own escalation trigger.

**Why.** It is the failure that harms people and the one aggregate metrics
conceal. A model can be well calibrated overall and systematically wrong for a
subgroup, whose members are then priced, provisioned and declined against
probabilities that are simply wrong for them. On this book the under-25 band is
both the most declined and the worst calibrated.

### G4. Group-specific thresholds are a diagnostic, labelled as such in code

**Decision.** The frontier is computed; deployment at λ=0 is recommended.

**Why.** In most jurisdictions applying a different cutoff *because of* a
protected characteristic is disparate treatment and unlawful, whatever it does
to the statistics. The curve prices the gap, which is useful; shipping a point
on it is a different act. Saying so in the docstring stops the function being
mistaken for a remedy.

### G5. The recommendation is stated, and stated to be a policy choice

**Decision.** Recommend λ=0 plus an upstream fix to income capture; state
explicitly that a different institution could reasonably choose otherwise.

**Why.** The brief asked for a recommendation, and refusing to give one would be
hiding behind neutrality. But the choice has legal and ethical content that a
modeller is not positioned to settle alone. The reasoning is given in full so it
can be disagreed with. What is not acceptable is making the choice silently
inside a model.

### G6. Small groups are suppressed

**Decision.** Groups under 100 rows are skipped, logged.

**Why.** Subgroup estimates on tiny samples are noise, and publishing them
invites over-reaction to sampling variation.

---

## H. Explainability

### H1. Two reason-code methods, with points shortfall deployed

**Decision.** Points shortfall is the deployed method; SHAP is available.

**Why.** Points shortfall is recoverable from the printed points table forever,
with no model artefact, no background dataset and no library version. When
someone asks in three years why an applicant was declined, that property is
worth more than flexibility. SHAP is retained because it is the only option for
a non-additive model.

**Finding.** They agree on the top two reasons and diverge after — which matters
when the output is a legal notice.

### H2. Age is kept as a feature and its reason codes are flagged

**Decision.** `protected_basis: true` rather than removal or silent emission.

**Why.** Age is genuinely predictive and present in both datasets. Removing it
would understate the model's real behaviour; emitting "your age band" onto a
letter without review would be worse. Flagging surfaces it for legal review,
which is where the decision belongs. **This is a deferral, not a resolution**,
and it is listed as a pre-launch blocker in the model card.

**Contestable.** A reviewer might reasonably say age should simply be dropped.
The counter-argument is that a demonstrably sound empirically-derived scoring
system may use age in some jurisdictions, and pretending otherwise avoids the
question rather than answering it.

### H3. Reason codes are only produced for adverse decisions

**Decision.** Approved applicants get an empty list.

**Why.** "Principal reasons for adverse action" is meaningless for an approval.

### H4. The worked example is the decline closest to the cutoff

**Decision.** Not a random or extreme decline.

**Why.** Obvious declines explain themselves. The marginal case — this one has an
expected value of −0.18 on a 17,600 exposure — is where a challenge lands and
where the explanation has to hold.

---

## I. Monitoring

### I1. Conventional PSI thresholds were kept even though they missed the drift

**Decision.** 0.10 watch / 0.25 review retained, and a combined-watch rule added.

**Why.** This was the hardest call here. Score PSI reached 0.103 and worst
feature PSI 0.212 while realised losses rose 75% — neither breached 0.25.

Lowering the threshold until it fires on this incident would be fitting the
governance to the accident, and would produce false positives on every other
book. Instead: keep the conventional levels, which have the real virtue that a
credit committee already acts on them; add a rule that **two indicators in the
watch band simultaneously** is itself a finding; and state plainly that input
drift metrics cannot see a change in the feature-to-outcome relationship, which
is where the damage came from.

### I2. PSI bins come from the reference sample

**Decision.** Not re-binned on the current sample.

**Why.** Re-binning each period would partly absorb the very shift the metric
exists to detect.

### I3. Vintage cohorts are pooled six-monthly

**Decision.** Not monthly. *Simplification.*

**Why.** Monthly cohorts are too thin to read on a 30,000-row book. Six-month
pools are the common compromise.

### I4. Every trigger carries an action and a rationale

**Decision.** `TriggerResult` includes both as required fields.

**Why.** A monitoring pack that stops at a chart answers "is something wrong?"
but not "and then what?". Forcing an action into the data structure means the
question cannot be skipped.

---

## J. Engineering

### J1. One decision path for API and batch, asserted by test

**Decision.** Both call `DecisionEngine.decide`; a test requires byte-identical
output.

**Why.** Separate implementations drift, usually in small things — a feature
clipped in one, a threshold updated in one config. The same applicant then gets
different answers depending which door they came through, which is indefensible
to a regulator and hard to detect because each path is individually
self-consistent. A convention would not hold; a test does.

### J2. Model and calibrator are separate files with a shared run ID

**Decision.** Separate files, verified on load, service refuses to start on a
mismatch.

**Why.** Storing them together would make the failure impossible but would not
model reality — in a real deployment they are separate artefacts and *can* be
mismatched. Making the failure possible and then catching it is the useful
design. A service that is down is an incident someone notices in minutes; one
quietly serving a model against the wrong calibrator returns confident,
well-ranked, wrongly-priced probabilities that nothing in ordinary monitoring
reveals, because **AUC is untouched by the mismatch**.

### J3. The threshold travels inside the artifact

**Decision.** Not in an environment variable or caller config.

**Why.** A decision must be reconstructible months later, including the cutoff
in force at the time. If the threshold lives in deployment config, that history
is gone. `/model-info` reads it live from the manifest, so documentation cannot
drift from what is running.

### J4. The audit log is append-only JSONL, written before responding

**Decision.** One record per line, `fsync`'d, never rewritten.

**Why.** JSONL is greppable with standard tools during an investigation, which
is when it matters. Writing before responding means a decision the applicant has
received can always be evidenced. Corrections are new records, so the original
decision stays visible. **This is not a production durability story** — a real
deployment needs WORM storage with retention locks — and the module says so.

### J5. The audit log stores an input hash, not the inputs

**Decision.** SHA-256 of canonicalised features.

**Why.** It ties a decision to its exact inputs without the log becoming a
second copy of the applicant database. Anyone holding the original application
can recompute the hash and prove which inputs produced which decision.

### J6. Strict input validation, rejecting rather than coercing

**Decision.** Pydantic bounds on every field, `extra="forbid"`.

**Why.** A service that coerces a malformed value will decline someone for a
reason traceable to a caller's typo, and the audit log will faithfully record
the wrong reason. Rejecting the request is the safer failure.

### J7. Docker trains in the build and fails if the artifact will not load

**Decision.** Training happens in the builder stage; a load check gates the
runtime stage.

**Why.** The image that starts is the image that was validated. No training at
container start, and a mismatch costs a red build rather than a year of wrongly
priced lending.

### J8. The UCI checksum is trust-on-first-use, and the reason is documented

**Decision.** `EXPECTED_SHA256 = None`, recorded on first download and enforced
thereafter; strict pinning available via environment variable.

**Why.** The development environment cannot reach `archive.ics.uci.edu`, so no
publisher hash could be verified. Hard-coding an unverified constant would look
like a pinned checksum while providing none of the assurance. Admitting the gap
in the module docstring is weaker but honest.

---

## K. Simplifications

Chosen deliberately, per the brief's instruction to prefer the simpler option
and record it.

| Simplification | Instead of | Why |
|---|---|---|
| All vintages fully seasoned | Immature tail with partial performance | Keeps vintage curves interpretable; a real book is messier |
| Flat LGD and margin | Modelled from collections and pricing | Both are assumptions here; the sensitivity grid shows what that costs |
| Referral band off by default | Three-way approve/refer/decline | Referral only pays if reviewers add information; that needs its own evidence |
| Two protected attributes on synthetic data | Full intersectional analysis | Intersections are where the worst disparities hide; noted as a limitation, not done |
| 12-month fixed performance window | Survival modelling of time to default | Timing is simulated for vintage curves; the model predicts a binary 12-month outcome |
| Local JSONL audit log | WORM object storage with retention locks | Demonstrates the contract without pretending to satisfy a retention policy |
| Single train/calibrate/test split | Nested CV or walk-forward backtesting | One out-of-time split is the honest minimum; walk-forward would be better |
| No hyperparameter search beyond a small sweep | Bayesian optimisation | The sweep showed restrained settings win; more search would overfit the validation block |
| No missing-value injection | Realistic missingness patterns | Binning handles missing as its own bin and is tested, but the synthetic book has none |
| Reason codes fixed at four | Jurisdiction-specific counts | Four is the common convention; a real deployment would parameterise per market |

---

## Things I would do next

1. **A random-approval holdout.** The only way to learn the true reject
   counterfactual on a real book, and the only way to settle which selection
   regime you are in. Expensive, and worth it.
2. **Walk-forward backtesting** across all 36 vintages rather than one split, to
   see how fast the model decays and how often it genuinely needs refitting.
3. **Intersectional fairness analysis.** Group × age band is where the worst
   disparities usually hide, and it is not done here.
4. **Model LGD and EAD** rather than assuming them, with their own uncertainty
   propagated into the cutoff.
5. **Resolve the age question** properly, with legal input, rather than flagging
   it and moving on.
6. **A challenger/champion framework** so a new model can run in shadow against
   live traffic before it decides anything.
