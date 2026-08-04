# Model Card: Credit Risk Decisioning System

**Version:** 0.1.0 · **Date:** 2026-08-04 · **Status:** demonstration, not for
production lending

---

## 1. Model details

| | |
|---|---|
| **Deployed model** | LightGBM binary classifier (300 trees, 8 leaves, depth 3, `min_child_samples=200`) |
| **Calibrator** | Selected per run by a written rule; `raw` (identity) selected on the current build |
| **Benchmark** | WOE binning + L2 logistic regression, output as a points scorecard |
| **Output** | Calibrated probability of default over a 12-month horizon |
| **Decision** | Approve if `PD < 0.1071`, the closed-form cost optimum |
| **Score scale** | Points, base 600 at 19:1 good:bad odds, 20 points to double the odds |
| **Artifact versioning** | Model and calibrator carry a shared `training_run_id`, verified on load |
| **Live metadata** | `GET /model-info` returns the deployed version, cutoff, cost assumptions and calibration metrics |

Hyperparameters were selected on a **held-out validation block, never the test
set**. Restrained settings won: raising capacity to 1500 trees and 63 leaves
*lowers* out-of-time AUC from 0.838 to 0.794.

---

## 2. Intended use

### Primary intended use

A **reference implementation** of end-to-end credit decisioning, demonstrating:

- calibrated probability estimation with explicit calibrator comparison
- cost-sensitive threshold selection under asymmetric misclassification cost
- adverse action reason code generation
- fairness auditing that reports trade-offs rather than silently resolving them
- reject inference validated against ground truth
- production concerns: versioning, audit trails, drift monitoring

### Intended users

Credit risk modellers, model validators, model risk management functions, and
engineers building decisioning infrastructure. It is a worked example of the
*shape* of the problem and the honest reporting standard, not a model to deploy.

### Out of scope — do not use this for

- **Any real lending decision about any real person.** It is trained on
  synthetic data from a generator written for this repository. Its coefficients
  encode a fictional world.
- **Any regulated credit decision** without independent validation, legal
  review of the reason codes, fair-lending review of the disparities in §5, and
  sign-off from a model risk function.
- **Any jurisdiction's compliance evidence.** The adverse action notice here is
  illustrative and omits the statutory content a real notice requires.
- **Transfer of any number in this repository to a real portfolio.** Base
  rates, coefficients, the optimal cutoff and every effect size are properties
  of the simulator.
- **Non-credit decisions** — employment, insurance, housing, benefits. The cost
  model, the fairness framing and the reason codes are all specific to consumer
  credit.
- **Populations unlike the training distribution.** No thin-file/no-file
  handling, no commercial lending, no secured products (LGD of 0.75 is an
  unsecured assumption and is badly wrong for a mortgage).

---

## 3. Training data

### Primary: synthetic loan book (`src/simulate.py`)

| | |
|---|---|
| Applications | 30,000 across 36 monthly origination vintages |
| Training rows | 12,095 (vintages 0–19, **approved applicants only**) |
| Calibration rows | 2,389 (vintages 20–23) |
| Test rows | 6,516 (vintages 24–35, out of time, contains injected drift) |
| Population default rate | 19.96% |
| Default rate, approved | 12.41% |
| Default rate, declined | 37.57% *(unobservable on a real book)* |

**Features (11):** recorded income, debt-to-income, utilisation, delinquencies
in 24 months, credit history months, inquiries in 6 months, employment years,
loan amount, age, housing status, product type.

**Deliberately injected, with known magnitude:**

- A **known default-generating process** with no direct protected-attribute
  term, plus a utilisation × delinquency interaction, a debt-to-income cliff at
  0.45 and a thin-file step below 12 months.
- **Population drift** from vintage 24 — thinner files, higher utilisation,
  mild macro deterioration.
- **A fairness violation**: recorded income understates group B's true income by
  15% while true risk depends on true income. Measurement bias, of the kind that
  survives dropping the protected attribute.
- **Historical policy discrimination**: the legacy approval policy penalises
  group B by 0.55 logits.
- **An exclusion restriction** (`branch_capacity_index`) affecting approval but
  not default, to give the Heckman correction its best case.

Group B also has genuinely lower income and higher utilisation, producing a
*real* base-rate difference on top of the *unfair* measurement bias. The two are
entangled by construction, because that is the situation a real audit faces.

### Secondary: UCI Default of Credit Card Clients

30,000 credit card accounts, Taiwan, April–September 2005, with sex, education,
marital status and age. Ingestion is implemented, schema-validated and tested,
but **could not be executed in the development environment** because the archive
host is blocked by egress policy. No results in this repository come from it.

### What the training data excludes

Models never receive `group`, `age_band`, `sex`, `education`, `marriage`, or any
oracle column (`pd_true`, `default_true`, `income_true`, `officer_signal`,
`legacy_score`). Asserted by test.

**Training data is selected by the previous approval policy.** Only approved
applicants have observed outcomes. This is the central limitation of every
credit model and is addressed — with mixed results — in the reject inference
work.

---

## 4. Performance

Out-of-time test set (vintages 24–35, 6,516 approved applicants, 17.05% default
rate).

### Discrimination

| Model | ROC-AUC | Gini | PR-AUC | KS | Brier | ECE |
|---|---|---|---|---|---|---|
| **LightGBM (deployed)** | **0.8379** | 0.6758 | 0.5736 | 0.5238 | 0.1042 | 0.0255 |
| Logistic L2 | 0.8321 | 0.6642 | 0.5646 | 0.5039 | 0.1059 | 0.0289 |
| WOE scorecard | 0.8148 | 0.6297 | 0.5156 | 0.4847 | 0.1112 | 0.0303 |
| *PR-AUC baseline* | | | *0.1705* | | | |

**Accuracy is not reported as a headline.** At a 0.5 cutoff the deployed model
scores 0.857; predicting "nobody defaults" scores 0.829.

### Operational precision

| Review capacity | Precision | Recall | Lift |
|---|---|---|---|
| Top 1% | 0.908 | 0.053 | 5.32× |
| Top 5% | 0.764 | 0.224 | 4.48× |
| Top 10% | 0.656 | 0.385 | 3.85× |

### Economics at the deployed cutoff

| | LightGBM | Scorecard |
|---|---|---|
| Approval rate | 59.8% | 59.9% |
| Default rate among approved | 5.3% | 6.4% |
| Profit per application | 384.68 | 332.78 |

The scorecard forgoes **13.5% of profit**. See
[README §3](README.md#3-scorecard-vs-booster-decomposed) — most of that gap is
coarse binning, which is a tunable dial, not additivity.

### Calibration

Mean predicted PD 0.145 against an observed 0.171 — the model **under-predicts
out of time by about 2.6 points** because the book drifted after the calibrator
was fitted. Calibration intercept +0.30, slope 1.035.

**This is a known, material limitation of the current build.** Expected loss
computed from these probabilities will be understated by roughly 15% relative
to realised. The remedy is operational, not architectural: refresh the
calibrator on recent matured outcomes (see [RUNBOOK.md](RUNBOOK.md)).

---

## 5. Performance by subgroup

**This section is the reason the model card exists.** All figures at the
deployed cutoff.

### By group (the synthetic protected attribute)

| Group | n | True default rate | Approval rate | Approval \| would repay | Mean predicted PD | ECE |
|---|---|---|---|---|---|---|
| A | 4,325 | 14.71% | **65.13%** | 72.59% | 0.1221 | 0.0255 |
| B | 2,191 | 21.68% | **49.20%** | 58.97% | 0.1901 | 0.0271 |

| Metric | Value | Assessment |
|---|---|---|
| Demographic parity difference | 0.159 | |
| **Adverse impact ratio** | **0.755** | ⚠ **below the four-fifths screening level** |
| **Equal opportunity difference** | **0.136** | ⚠ **flagged** |
| Within-group calibration (max ECE) | 0.027 | acceptable |

### By age band

| Band | n | True default rate | Approval rate | Approval \| would repay | ECE |
|---|---|---|---|---|---|
| <25 | 258 | 26.36% | **35.66%** | 46.84% | **0.0694** ⚠ |
| 25–34 | 2,348 | 20.87% | 49.83% | 59.04% | 0.0267 |
| 35–44 | 2,111 | 15.77% | 60.82% | 68.73% | 0.0255 |
| 45–54 | 1,105 | 14.48% | 72.40% | 79.47% | 0.0452 |
| 55+ | 694 | 8.65% | **79.11%** | 83.75% | 0.0152 |

| Metric | Value | Assessment |
|---|---|---|
| Demographic parity difference | 0.434 | |
| **Adverse impact ratio** | **0.451** | ⚠ **materially below 0.80** |
| **Equal opportunity difference** | **0.369** | ⚠ **flagged** |
| **Worst within-group ECE (<25)** | **0.069** | ⚠ **above the 0.05 escalation level** |

**The under-25 band is both the most declined and the worst calibrated.** Its
probabilities are the least reliable of any group, and it receives the harshest
treatment. That combination is the one that harms people, and no threshold
adjustment repairs it — calibration is a property of the probabilities, not the
cutoff.

### How much of the disparity is the model's fault

Possible only because ground truth exists:

| Group | True mean PD | Predicted mean PD | True gap | Predicted gap | **Manufactured** |
|---|---|---|---|---|---|
| A | 0.1535 | 0.1221 | — | — | — |
| B | 0.2013 | 0.1901 | 0.0478 | 0.0679 | **+0.0202** |

The groups differ in real risk by 4.8 points; the model claims 6.8. **2.0 points
of the gap are manufactured by the model** — traceable to the injected income
measurement bias, invisible to the model, and unaffected by dropping the
protected attribute.

---

## 6. Ethical considerations

### Dropping protected attributes does not make a model fair

No model here sees `group`, `sex`, `age_band`, `education` or `marriage`. The
adverse impact ratio is still 0.755 by group and 0.451 by age band. Income,
utilisation, employment tenure and credit history carry the protected
attribute's information. **Fairness is a property of outcomes, not of the
feature list.**

### Some of the disparity is real risk and some is not

The groups genuinely differ in default rate. Demographic parity can therefore
only be reached by approving worse applicants from one group or declining better
ones from another. **This does not make the disparity acceptable** — 2.0 points
of it are manufactured by the model, and the equal opportunity difference of
0.136 counts real people who would have repaid and were refused.

On a real book these components cannot be separated. The honest standard is to
report the disparity *with* its uncertainty and argue about causes explicitly,
never to assert that a gap is "explained by risk" because a model produced it.

### Demographic parity and equal opportunity conflict

Equal opportunity difference is minimised at λ=0.8 (0.004) and *rises* to 0.037
at full approval-rate parity. With different base rates you cannot have both.
Choosing between them is a value judgement.

### The recommendation, and that it is a policy choice

The system recommends **a single cutoff for everyone (λ=0)**, escalation of the
adverse impact and equal opportunity findings, and an **upstream fix to income
capture** — because group-specific cutoffs are likely disparate treatment, and
because they cannot repair the miscalibration that is the actual defect.

A different institution could look at the same table and reasonably choose
differently. What is not reasonable is choosing silently inside a model. The
full reasoning is in [README §8](README.md#8-fairness-audit).

### Age is a protected basis and is used as a feature

`age` is retained because it is genuinely predictive and present in both
datasets. Reason codes derived from it are returned flagged
`protected_basis: true` for legal review rather than emitted onto a letter. This
would have to be settled before any real deployment, not after.

### Reject inference can make a model less fair

The legacy policy discriminated against group B, so group B's approved
population is more positively selected. Corrections that infer reject outcomes
from a model trained on that sample can propagate the historical discrimination
into the new model with a statistical veneer. Under MAR, the corrections here
*tripled* the reject bias.

### Explanations must be true

Reason codes are derived from the same artefact that made the decision, and the
worked example shows that points-shortfall and SHAP **disagree beyond the top
two reasons**. An explanation generated by a different mechanism than the
decision is not an explanation.

---

## 7. Caveats and recommendations

### Before any real use

1. Retrain on real data with a real outcome definition (90+ days past due at 12
   months, not "late next month").
2. Have the reason codes reviewed by counsel in every jurisdiction of operation.
3. Resolve the `age` question.
4. Estimate LGD and EAD rather than assuming them; the cutoff moves from 0.053
   to 0.211 across plausible values.
5. Re-run the fairness audit on the real population and escalate the findings
   before launch, not after.
6. Replace the local audit log with WORM storage under the applicable retention
   policy.
7. Establish the reject inference regime empirically if possible — via a
   deliberate random-approval holdout, which is the only real way to learn the
   counterfactual — and do not apply corrections without it.

### Monitoring in production

| Trigger | Threshold | Action |
|---|---|---|
| Score PSI | ≥ 0.25 | Open a model review; refit the calibrator first |
| Worst feature PSI | ≥ 0.25 | Investigate the feature: pipeline break or real shift |
| Combined watch | both ≥ 0.10 | Bring the review forward; pull vintage curves now |
| Calibration ECE | ≥ 0.05 | Refit the calibrator on recent matured outcomes |
| AUC drop | ≥ 0.05 | Refit or redevelop; recalibration cannot restore ranking |
| Subgroup ECE | ≥ 0.05 | Escalate to fairness review; **do not adjust cutoffs** |

**The single most important monitoring finding in this repository:** PSI at
0.103 and 0.212 did **not** breach the conventional 0.25 level while realised
losses rose 75%. Input-drift metrics cannot see a change in the
feature-to-outcome relationship. Do not rely on them alone.

---

## 8. Quantitative reproduction

```bash
pip install -e ".[dev]"
python -m scripts.run_experiments     # every table in this card
pytest tests/ -q                      # 223 tests
```

Fixed seed (`RANDOM_SEED = 20240517`). Tables land in `reports/tables/`,
figures in `reports/figures/`.

Calibration is pinned by a regression gate
(`tests/test_calibration_gate.py`): Brier, ECE, MCE and calibration slope are
compared against a committed baseline, with absolute servability floors so the
gate cannot be defeated by regenerating the baseline. Moving that baseline is a
governance event and CI says so on failure.
