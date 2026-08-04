# Runbook

Operational procedures for the credit risk decisioning service. Each one is a
sequence you can follow under time pressure without reading the source.

**Conventions.** Commands assume the repository root and an activated
virtualenv. Anything that changes what applicants experience is marked
**⚠ PRODUCTION CHANGE** and requires the approval named in the procedure.

---

## Contents

- [Health check](#health-check)
- [1. Change the cutoff](#1-change-the-cutoff) ⚠
- [2. Roll back a model](#2-roll-back-a-model) ⚠
- [3. Re-run the fairness audit](#3-re-run-the-fairness-audit)
- [4. Retrieve the audit trail for a declined applicant](#4-retrieve-the-audit-trail-for-a-declined-applicant)
- [5. Refresh the calibrator](#5-refresh-the-calibrator) ⚠
- [6. Deploy a new model](#6-deploy-a-new-model) ⚠
- [7. Run the monitoring pack](#7-run-the-monitoring-pack)
- [Incident: service will not start](#incident-service-will-not-start)
- [Incident: PSI breach](#incident-psi-breach)
- [Incident: subgroup calibration breach](#incident-subgroup-calibration-breach)

---

## Health check

Always start here.

```bash
curl -s localhost:8000/health     | python -m json.tool
curl -s localhost:8000/model-info | python -m json.tool
curl -s localhost:8000/metrics    | python -m json.tool
```

`/model-info` is the authoritative statement of what is running: version,
training run, cutoff in force, cost assumptions behind it, and calibration
metrics at build time. It reads live from the artifact manifest, so it cannot
drift from reality the way a wiki page can.

```bash
# What is available, and what is live
python -c "
from src.artifacts import list_versions
for v in list_versions():
    print(('* ' if v['is_current'] else '  ') + f\"{v['version']}  {v['model_name']:16s} \"
          f\"cut={v['threshold']:.4f}  {v['created_at']}\")"
```

---

## 1. Change the cutoff

**⚠ PRODUCTION CHANGE.** Approval: Head of Credit Risk. The cutoff determines
who gets credit; changing it is a policy act, not a configuration tweak.

### First: is the cutoff actually the problem?

The cutoff is a property of **the economics, not the model**:

```
p* = margin / (margin + LGD)
```

If someone wants a different approval rate, the honest question is whether the
cost assumptions have changed. If LGD has genuinely risen, change LGD. If
nothing has changed and the request is "approve more people", that is a decision
to accept a worse loss rate, and it should be recorded as such.

### Step 1 — see what the change costs

```bash
python -c "
from src.policy import cost_sensitivity, sweep_thresholds, optimal_threshold
from src.config import DEFAULT_COSTS
import pandas as pd, numpy as np
pd.set_option('display.width', 200)

# What the current assumptions imply
print('current cutoff:', round(DEFAULT_COSTS.analytic_threshold, 4))
print('cost ratio (missed bad : lost good):', round(DEFAULT_COSTS.cost_ratio, 2))
"

# Full frontier and sensitivity from the last experiment run
head -30 reports/tables/05_approval_frontier.csv
cat reports/tables/05_cost_sensitivity.csv
```

Read `reports/tables/05_cost_sensitivity.csv`: across plausible LGD (0.45–0.90)
and margin (5–12%), the cutoff moves from **0.053 to 0.211** and the approval
rate from **42% to 77%**. Locate the proposed cutoff on that grid and state
which cost assumption it corresponds to.

### Step 2 — change the assumption, not the number

Edit `src/config.py`:

```python
@dataclass(frozen=True)
class CostParameters:
    lgd: float = 0.75          # <- from collections performance
    margin_rate: float = 0.09  # <- from pricing
```

Changing `lgd` or `margin_rate` moves `analytic_threshold` automatically.
**Prefer this to overriding the threshold directly**, because it keeps the
cutoff traceable to a stated business assumption.

### Step 3 — rebuild and verify

```bash
python -m scripts.run_experiments
curl -s localhost:8000/model-info | python -c "
import json,sys; d=json.load(sys.stdin)
print('cutoff  ', d['threshold'])
print('lgd     ', d['costs']['lgd'])
print('margin  ', d['costs']['margin_rate'])
print('ratio   ', round(d['costs']['cost_ratio'],2))"
```

### Emergency override (temporary only)

```bash
CREDIT_RISK_THRESHOLD_OVERRIDE=0.085 uvicorn src.api.main:app --port 8000
```

This is not wired in by default and deliberately so — a threshold that lives in
deployment config is a threshold you cannot reconstruct months later. If you need
it, add it explicitly and open a ticket to fold it back into the artifact.

### Record

Log in the change record: old cutoff, new cutoff, the cost assumption that
changed, expected approval-rate impact from the frontier, approver, date.

---

## 2. Roll back a model

**⚠ PRODUCTION CHANGE.** Approval: Head of Credit Risk, or on-call during an
incident with retrospective sign-off.

```bash
# 1. What can we roll back to?
python -c "
from src.artifacts import list_versions
for v in list_versions():
    print(('* ' if v['is_current'] else '  ') + f\"{v['version']}  {v['created_at']}  cut={v['threshold']}\")"

# 2. Validate the target BEFORE switching. This loads the bundle, checks the
#    model and calibrator share a training run, and verifies content hashes.
python -c "
from src.artifacts import load_bundle
b = load_bundle('20260804T105115Z')
print('OK:', b.manifest.version, '| run:', b.manifest.training_run_id)
print('cutoff:', b.manifest.threshold)
print('build-time calibration:', b.manifest.calibration_metrics)"

# 3. Switch. set_current_version re-validates and REFUSES to point at a
#    broken or mismatched bundle, so a rollback cannot install one.
python -c "
from src.artifacts import set_current_version
set_current_version('20260804T105115Z')"

# 4. Restart and confirm
docker compose restart api      # or: systemctl restart credit-risk-api
sleep 5
curl -s localhost:8000/model-info | python -c "
import json,sys; d=json.load(sys.stdin); print('now serving:', d['model_version'], 'cutoff', d['threshold'])"
```

### After a rollback

- Decisions made under the previous version remain valid and are tagged with
  that `model_version` in the audit log. **Do not backfill or restate them** —
  the record must show what actually happened.
- Check whether the rolled-back version has `reference_scores.npy`. If not,
  `/metrics` will correctly report `psi_status: no_reference` rather than
  comparing against the wrong model's distribution.
- Applicants declined under the withdrawn version may need re-decisioning.
  That is a commercial and legal call, not a technical one.

---

## 3. Re-run the fairness audit

Run this at least quarterly, after any model change, and whenever a subgroup
calibration trigger fires.

```bash
python -m scripts.run_experiments        # regenerates the full audit
```

Then read, in this order:

```bash
# 1. Headline disparities per protected attribute
cat reports/tables/07_fairness_summary.csv

# 2. Per-group detail: approval rates, equal-opportunity rates, within-group ECE
cat reports/tables/07_fairness_by_group.csv

# 3. Synthetic data only: how much of the gap the model manufactured
cat reports/tables/07_disparity_decomposition.csv

# 4. What parity would cost
cat reports/tables/07_fairness_tradeoff.csv
```

### Ad-hoc audit against live decisions

```bash
python -c "
import pandas as pd, numpy as np
from src.fairness import audit_fairness
from src.engine import DecisionEngine
from src.config import DEFAULT_COSTS

# Replace with your own scored population carrying protected attributes and
# matured outcomes.
frame = pd.read_parquet('data/processed/scored_with_outcomes.parquet')
engine = DecisionEngine.from_artifacts()
p = engine.predict_proba(frame)

per_group, summary = audit_fairness(
    frame, frame['default'].to_numpy(), p,
    ['sex', 'age_band', 'education', 'marriage'],
    engine.threshold,
)
pd.set_option('display.width', 220)
print(summary.T)
print(per_group)"
```

### Reading the result

| Finding | Threshold | Action |
|---|---|---|
| Adverse impact ratio < 0.80 | screening level | Document; investigate proxies; report to fair-lending review |
| Equal opportunity difference > 0.05 | escalation | **Escalate to model risk committee.** Counts people who would have repaid and were refused |
| Within-group ECE > 0.05 | escalation | **Escalate.** See [incident procedure](#incident-subgroup-calibration-breach) |

**Do not respond by adjusting cutoffs per group.** In most jurisdictions that
is disparate treatment, and it cannot repair miscalibration — `max_group_ece` is
invariant to the threshold by construction. Fix the measurement or the features.

---

## 4. Retrieve the audit trail for a declined applicant

The procedure for a complaint, an ombudsman referral, or a supervisory sample.
You must be able to reconstruct the decision **as it was made**, not as the
current model would make it.

### By application ID

```bash
python -c "
from src.api.audit import AuditLog
import json
for entry in AuditLog().find(application_id='APP-12345'):
    print(json.dumps(entry, indent=2))"
```

### By input hash

When you have the original application data but not the reference:

```bash
python -c "
from src.api.audit import AuditLog
from src.engine import hash_features
import json

features = {
    'income_recorded': 21000.0, 'debt_to_income': 0.85, 'utilisation': 1.05,
    'n_delinq_24m': 4, 'credit_history_months': 6.0, 'n_inquiries_6m': 5,
    'employment_years': 0.4, 'loan_amount': 15000.0, 'age': 26.0,
    'housing_status': 'rent', 'product_type': 'card',
}
digest = hash_features(features)
print('input hash:', digest)
for entry in AuditLog().find(input_hash=digest):
    print(json.dumps(entry, indent=2))"
```

### All declines in a period

```bash
python -c "
from src.api.audit import AuditLog
import pandas as pd
frame = pd.DataFrame(AuditLog().find(decision='decline'))
frame['timestamp'] = pd.to_datetime(frame['timestamp'])
window = frame[(frame.timestamp >= '2026-07-01') & (frame.timestamp < '2026-08-01')]
print(len(window), 'declines')
print(window[['timestamp','application_id','probability_of_default','score_band','model_version']].head(20))"
```

### What the record contains, and what it proves

| Field | Answers |
|---|---|
| `timestamp` | when the decision was made |
| `input_hash` | which inputs produced it — recompute from the original application to prove the tie |
| `probability_of_default`, `score`, `score_band` | what the model said |
| `decision`, `threshold` | the outcome and the cutoff **in force at that moment** |
| `reason_codes` | exactly what the applicant was told |
| `model_version` | resolves to a manifest with training data, calibration metrics and cost assumptions |
| `channel` | `api` or `batch` |

### Reconstructing the full context

```bash
python -c "
from src.artifacts import load_bundle
import json
b = load_bundle('20260804T105115Z')      # the model_version from the record
m = b.manifest
print('trained on   :', m.training_period, f'({m.training_rows} rows)')
print('calibrator   :', m.calibrator_name)
print('cutoff       :', m.threshold)
print('cost basis   :', json.dumps(m.costs, indent=2))
print('calibration  :', json.dumps(m.calibration_metrics, indent=2))
print('notes        :', m.notes)"
```

### If the applicant disputes the reasons

The reason codes in the log are the ones sent. To re-derive them from the
deployed artifact and confirm they reproduce:

```bash
python -c "
import pandas as pd
from src.engine import DecisionEngine
engine = DecisionEngine.from_artifacts('20260804T105115Z')
features = {...}   # the original application
d = engine.decide(pd.DataFrame([features]))[0]
print('hash matches log:', d.input_hash)
for c in d.reason_codes:
    print(c['rank'], c['statement'], '[PROTECTED BASIS]' if c['protected_basis'] else '')"
```

If the hash matches the logged hash, the inputs are confirmed identical and the
reasons are reproducible. **If it does not match, the application data you have
is not what was scored** — investigate the upstream capture before anything else.

---

## 5. Refresh the calibrator

**⚠ PRODUCTION CHANGE.** Approval: model owner.

### When

- `/metrics` or the monitoring pack shows ECE ≥ 0.05
- Mean predicted PD diverges materially from realised
- After a population shift, **before** considering a full refit

Recalibration is the cheap fix and should always be tried first: it repairs the
*level* of the probabilities without touching the ranking. If AUC has dropped,
recalibration cannot help — see [step 6](#6-deploy-a-new-model).

### Why this matters here

The current build **under-predicts out of time**: mean predicted PD 0.145
against an observed 0.171, calibration intercept +0.30. Expected loss computed
from these probabilities is understated by roughly 15%. That is drift after the
calibrator was fitted, and **no calibrator fitted before the drift can fix it**.
The remedy is to refit on recent matured outcomes.

```bash
# 1. Assemble a calibration set of RECENT, MATURED outcomes that the model did
#    NOT train on. Both properties matter: in-sample scores produce a
#    calibrator that corrects in-sample optimism and nothing else.

# 2. Compare calibrators on it
python -c "
import pandas as pd
from src.calibration import fit_calibrators, compare_calibrators, select_calibrator
from src.engine import DecisionEngine

frame = pd.read_parquet('data/processed/recent_matured.parquet')
engine = DecisionEngine.from_artifacts()
raw = engine.bundle.model.predict_proba(frame[engine.feature_names])
y = frame['default'].to_numpy()

split = len(frame) // 2
fitted = fit_calibrators(raw[:split], y[:split])
comparison = compare_calibrators(fitted, raw[split:], y[split:], model_name='refresh')
pd.set_option('display.width', 220)
print(comparison[['calibrator','brier','ece_quantile','calibration_slope','mean_predicted','observed_rate']])
print('selected:', select_calibrator(comparison))"

# 3. Rebuild the bundle so model and calibrator share a training run
python -m scripts.run_experiments

# 4. The gate must pass before deployment
pytest tests/test_calibration_gate.py -v
```

**Never hand-edit a calibrator file.** The model and calibrator carry a shared
`training_run_id` and the service will refuse to start if they disagree — which
is the intended behaviour, not an obstacle to work around.

---

## 6. Deploy a new model

**⚠ PRODUCTION CHANGE.** Approval: model owner + model risk.

```bash
# 1. Train, evaluate, and build the artifact. The script asserts that the
#    saved artifact reproduces its own predictions exactly and exits non-zero
#    if it does not.
python -m scripts.run_experiments

# 2. Full quality gate
ruff check src/ tests/ scripts/
mypy src/
pytest tests/ -q
pytest tests/test_calibration_gate.py -v       # calibration regression gate

# 3. Review before switching
cat reports/tables/05_policy_cost_comparison.csv     # the headline: cost
cat reports/tables/03_calibration_comparison.csv     # calibration
cat reports/tables/07_fairness_summary.csv           # fairness -- MANDATORY
cat reports/tables/04_discrimination_calibrated.csv  # discrimination

# 4. Confirm the bundle loads and versions match
python -c "
from src.artifacts import load_bundle
b = load_bundle(); print('OK', b.manifest.version, b.manifest.training_run_id)"

# 5. Deploy
docker compose up --build -d
curl -s localhost:8000/health | python -m json.tool
```

### Pre-deployment checklist

- [ ] Calibration gate passes, or the baseline change is justified **in the commit message**
- [ ] Fairness audit reviewed; no new escalation-level finding
- [ ] Cost comparison shows the new model is not worse on **net cost**, not just AUC
- [ ] Cutoff unchanged, or a separate cutoff change is approved
- [ ] Previous version retained for rollback
- [ ] Reason codes spot-checked on a sample of declines

---

## 7. Run the monitoring pack

Monthly, or after any alert.

```bash
python -m scripts.run_experiments

cat reports/tables/09_triggers.csv          # what fired and what to do
cat reports/tables/09_feature_drift.csv     # PSI per feature
cat reports/tables/09_vintage_summary.csv   # the lagging indicator
open reports/figures/vintage_curves.png
```

### Read the vintage curves first

**This is the most important instruction in this runbook.** On this book, score
PSI reached 0.103 and worst-feature PSI 0.212 — neither breaching the
conventional 0.25 — while the 12-month default rate rose from 17.5% to 30.7%,
a **75% increase**.

Input-drift metrics answer "did the applicants change?" They cannot see a change
in the *relationship* between features and default, which is where the damage
came from. Do not treat a green PSI as reassurance.

| Trigger | Level | Action |
|---|---|---|
| `score_psi` | ≥ 0.25 | Model review. Refit the calibrator before the model |
| `worst_feature_psi` | ≥ 0.25 | Investigate that feature: pipeline break or genuine shift |
| `combined_watch` | both ≥ 0.10 | **Bring the review forward. Pull vintage curves now.** |
| `calibration_ece` | ≥ 0.05 | [Refresh the calibrator](#5-refresh-the-calibrator) |
| `auc_drop` | ≥ 0.05 | Refit or redevelop. Recalibration cannot restore ranking |
| `subgroup_calibration` | ≥ 0.05 | [Escalate](#incident-subgroup-calibration-breach) |

---

## Incident: service will not start

### Symptom

```
CRITICAL REFUSING TO START: Model and calibrator are from different training
runs: model='run-2026...', calibrator='run-2025...'
```

### This is working as designed

The service refuses rather than serving a mismatched pair. A mismatch produces
confident, correctly-ranked, **wrongly-priced** probabilities. AUC is unaffected,
so nothing in ordinary monitoring would reveal it, and the first hard evidence
would arrive a year later in the vintage curves. A service that is down is an
incident someone notices in minutes.

### Resolve

```bash
# 1. Confirm the diagnosis
python -c "
from src.artifacts import load_bundle
try:
    load_bundle()
except Exception as e:
    print(type(e).__name__); print(e)"

# 2. Roll back to a known-good version (fastest path to service)
python -c "
from src.artifacts import list_versions
for v in list_versions(): print(v['version'], v['created_at'], v['training_run_id'])"

python -c "from src.artifacts import set_current_version; set_current_version('<known-good>')"
docker compose restart api

# 3. Then fix the root cause: rebuild the broken version properly
python -m scripts.run_experiments
```

### Other startup failures

| Message | Cause | Fix |
|---|---|---|
| `Content hash mismatch for model.joblib` | File modified or truncated since build | Rebuild; investigate how it changed |
| `No version requested and no current.txt` | Never trained, or artifacts wiped | `python -m scripts.run_experiments` |
| `Artifacts do not match their manifest` | Manifest edited by hand | Rebuild. Do not hand-edit manifests |

---

## Incident: PSI breach

```bash
cat reports/tables/09_feature_drift.csv     # which feature moved
cat reports/tables/09_score_distribution.csv
```

**Triage in this order** — the cheap causes are the common ones:

1. **Is it a data pipeline break?** A single feature at high PSI while others
   are stable usually means an upstream definition changed, a join dropped rows,
   or a bureau attribute was renamed. Check the feature's `reference_mean`
   against `current_mean` in the drift report. Far cheaper to fix than a model.
2. **Is it a deliberate business change?** A new marketing channel, a new
   product, an expansion into a different segment. Expected drift, but the model
   may not cover the new population.
3. **Is it genuine population shift?** Then check the vintage curves before
   deciding anything.

Then:

```bash
# Has calibration actually degraded, or just the input distribution?
curl -s localhost:8000/metrics | python -m json.tool
```

- **Calibration degraded, ranking intact** → [refresh the calibrator](#5-refresh-the-calibrator)
- **Ranking degraded (AUC drop ≥ 0.05)** → refit or redevelop; recalibration will not help
- **Neither degraded** → document and continue monitoring. A population can
  shift substantially with no loss of performance.

---

## Incident: subgroup calibration breach

**A group's probabilities are wrong for them.** Every member is priced,
provisioned and decided against numbers that do not apply to them. This is the
fairness failure that harms people and aggregate metrics conceal.

```bash
cat reports/tables/07_fairness_by_group.csv   # find the group, check its ECE and n
```

### Do not

- **Do not adjust that group's cutoff.** `max_group_ece` is invariant to the
  threshold by construction. Moving the cutoff redistributes decisions without
  repairing the probabilities, and in most jurisdictions a group-specific cutoff
  is disparate treatment.
- **Do not drop the protected attribute** as a remedy. It is not an input to any
  model here and the disparity exists anyway.

### Do

1. **Check the sample size.** Below a few hundred, the estimate may be noise.
   `group_performance` suppresses groups under 100; sizes just above that are
   still shaky.
2. **Look for a measurement problem.** On this book the group B excess traces to
   income being recorded 15% low. That is an upstream data-capture defect, and
   fixing it removes the error rather than offsetting it — the only remedy with
   no fairness/accuracy trade-off.
3. **Check for a coverage problem.** If the group is underrepresented in
   training, the model has simply not learned it. More data for that segment is
   the fix.
4. **Escalate to the model risk committee** with the disparity decomposition,
   the trade-off curve, and a recommendation. The decision about what to do is
   theirs, and it is a policy decision.
5. **Record it**, including the decision to accept a residual disparity if that
   is the outcome. An undocumented acceptance is indistinguishable from having
   missed it.
