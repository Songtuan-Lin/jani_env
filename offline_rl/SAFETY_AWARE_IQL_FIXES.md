# Fixing the Safe-Action Lower Bound on Failure-Dominated Benchmarks

This document records the diagnosis of *why* the safe-action Q-value lower bound
(`DiscreteIQLLossActionSafeLB`, see [SAFETY_AWARE_IQL.md](SAFETY_AWARE_IQL.md))
underperforms on some benchmarks, and a catalogue of candidate fixes with their
trade-offs. It is a design/brainstorming record, not yet an implementation plan.

---

## 1. The problem, restated

The safe-action lower bound clamps the Q-value target of *safe* actions up to 0:

```python
target = r + γ * V(s')
if is_action_safe:
    target = max(target, 0)
```

This is theoretically sound — a safe action guarantees `Q*(s, a) ≥ 0` — but it
performs **worse than vanilla IQL** on benchmarks where:

- almost all actions are safe (unsafe actions occur only in the last 1–3 steps), and
- the offline data is **failure-dominated** (most trajectories end in failure).

### Empirical confirmation (beluga_5_2, 100k steps, controlled A/B)

| | Vanilla IQL | action_safe LB |
|---|---|---|
| Success rate | **16.6%** | **0.9%** |
| Failure rate | 83.4% | 99.1% |
| Avg reward | −0.668 | −0.982 |

The lower bound is **~18× worse** on success rate.

### Mechanism (verified by the advantage diagnostic)

See [advantage_diagnostic.py](advantage_diagnostic.py). It measures the exact
IQL advantage the actor uses, `A = min_a' Q(s, a_taken) − V(s)`, and splits
safe-action transitions by trajectory outcome (goal-path vs failure-path).

Final-state safe-action advantage spread (`std`) and fraction flattened near
zero (`~0%`):

| group | vanilla std / ~0% | action_safe std / ~0% |
|---|---|---|
| safe / **failure-path** | 0.043 / **7.2%** | 0.033 / **20.0%** |
| safe / **goal-path** | 0.092 / 6.1% | 0.083 / 13.7% |
| safe / **all** | 0.051 / 7.0% | 0.042 / **19.3%** |

Over training, action_safe's failure-path `~0%` climbs monotonically
(0.8 → 7.9 → 10.5 → 20.8%) while vanilla stays flat at ~5–7%. Histograms show
action_safe's failure-path advantages as a single narrow spike vs vanilla's
broader spread.

**Diagnosis — information destruction, not unsoundness.** Because the data is
failure-dominated, the natural target `r + γV(s')` for safe actions is negative
almost everywhere, so `max(·, 0)` lifts a large fraction of safe-action targets
to a *common* 0. This:

1. flattens the safe-action advantage distribution (near-zero mass ~3× vanilla),
2. so the AWR actor weight `exp(β·A)` becomes near-uniform over safe actions,
3. so the actor stops discriminating and **imitates the behavior policy**, which
   on this data fails ~99% of the time.

The bound injects no false fact (`Q* ≥ 0` is true) but overwrites the data's only
shaped signal with a flat constant — replacing "weak but informative" targets
with "correct but useless" ones.

**Why benchmark-dependent:** damage scales with how often the clamp is active =
how negative the natural targets are = how failure-dominated the data is. On
benchmarks with more successful trajectories, more safe-action targets are
naturally positive, the clamp is a no-op there, and the bound reverts to a
harmless light correction.

### Requirements any fix must satisfy

- **(P1)** Preserve relative ordering among safe actions — never map distinct
  values to a common constant.
- **(P2)** Keep using the oracle's correct knowledge (safe ⇒ `Q* ≥ 0`) without
  overriding the data's shape.

### Available information

The **only** extra signal is the safety oracle (Tarjan). Crucially it exposes
**two** capabilities, one of which the current approach wastes:

- `is_state_action_safe(action)` → **per-action** safety: is *any* given action
  safe at the current state? (current code only records the *taken* action's bit)
- `current_state_safety_with_action()` → is the state safe + one *witness* safe
  action.

---

## 2. The search-vs-learning axis (key framing)

A central distinction governs which fixes are acceptable:

- **Search-side use:** consult the oracle *at decision/inference time* to
  constrain actions. Powerful, but the oracle must ship with the controller
  forever, and the "policy" is really executing the oracle — it benchmarks
  search, not learning, and cannot generalize past the oracle's reach.
- **Learning-side use:** use the oracle *only at training time* to shape what the
  network internalizes, then **discard the oracle at deployment**. The network
  must stand alone.

The goal is a *learned* policy, so we favor learning-side fixes. A clean test of
which side a method is on: **evaluate with the oracle disabled.** If success
holds, the policy internalized the knowledge (learning). If it collapses, the
method was doing search.

---

## 3. Candidate fixes

### Idea A — Safety as an action mask / shield (SEARCH-LEANING)

Use `is_state_action_safe(a)` for *every* action to build a per-state
`safe_mask`, then constrain the policy to safe actions.

**Variants for "pushing probability mass off unsafe actions":**

1. **Hard logit masking (structural).** `logits.masked_fill(~safe_mask, -inf)`
   before forming the Categorical. Unsafe probability is *exactly* 0, no tuning.
   Requires gating the AWR term on `is_action_safe` (skip regressing toward
   unsafe dataset actions, else `exp(β·A)·(−∞)` blows up).
2. **Soft NLL penalty.** `L_safety = −log( Σ_{a safe} π(a|s) )`, added with
   weight λ. Drives unsafe mass → 0 asymptotically; differentiable; annealable.
3. **Cross-entropy toward the safe set.** KL(target ‖ π) with target zero on
   unsafe actions. Avoid a *uniform-over-safe* target — it competes with the
   discrimination we want to preserve.
4. **Logit barrier on unsafe actions.** `relu(logits[~safe])².mean()`. Direct but
   less principled than reasoning about probability mass (variant 2).

**Pros:** attacks the *dominant* failure (eventually picking an unsafe action) in
the **action channel**, which the value-collapse cannot touch. Robust even if Q
is garbage. Uses the oracle's strongest (per-action) capability.

**Cons / why it's search-leaning:**
- **Hard masking (variant 1) needs the oracle at inference** — the policy can't
  stand alone; you ship the oracle as part of the controller, defeating
  distillation.
- It benchmarks the *oracle*, not the *learner*: high success can't be
  attributed to better learning.
- It can't generalize to states the oracle never labeled.

**Verdict:** As a *runtime safety wrapper* it's strong; as a *learning* fix it
sidesteps the research question. **Hard masking at inference is rejected.**
Variant 2, applied as a **training-only** auxiliary loss (and NOT re-masked at
eval), survives — it becomes distillation: the oracle teaches the net which
actions to avoid, then leaves.

### Idea B — Give the lower bound *shape* instead of a flat 0 (LEARNING-SIDE)

Keep the bound's intent but stop clamping to a constant (the flatness is the
direct cause of the collapse, violating P1).

- **Soft / penalty form:** don't rewrite the target; add a one-sided penalty to
  the Q-loss: `λ · mean( relu(−Q(s, a_safe))² )`. Nudges safe-action Q toward
  ≥ 0 *without erasing differences* — a safe action at −0.05 and one at −0.20 are
  both pushed up but their gap survives, because the rest of the TD target still
  flows.
- **Per-action floor over all safe actions:** using oracle per-action labels,
  apply the ≥ 0 floor to *every* safe action's Q at a state, not just the taken
  one. Corrects more values and lets the actor's ranking come from the unclamped
  TD signal where it is positive.

**Pros:** minimal, surgical, train-only, no inference-time oracle.
**Cons:** still fights the same headwind — where data is failure-dominated, V is
depressed and the floor still tends to dominate. Expect a *modest* gain.

### Idea C — Fix the *baseline*, not the Q targets (LEARNING-SIDE)

The actor advantage is `A = Q − V`. With 99% failure, `V(s)` is dragged negative,
so clamped-to-0 safe actions get a *spurious uniform* positive advantage
`A = 0 − V > 0`, upweighting mediocre safe actions equally.

Fix: compute the AWR advantage **relative to the best safe action**, using oracle
per-action labels:

```
A(s, a) = Q(s, a) − max_{a' safe} Q(s, a')
```

This makes the baseline data-driven, restricts the comparison to safe actions,
restores ranking even when V is depressed, and never references the flat floor.
A clean, ~one-line change to `actor_loss`. The oracle labels are used **only at
training time**; the resulting actor needs no oracle at test.

**Pros:** directly repairs the *discrimination* problem we diagnosed; learning-side.
**Cons:** changes IQL's actor objective (departs from the published recipe);
needs per-action safe labels.

### Idea D — Oracle-guided data reweighting / augmentation (LEARNING-SIDE)

The root cause is that the dataset never demonstrates safe long-horizon play.
Use `current_state_safety_with_action()` to roll out the oracle's *safe* policy
from dataset states, generating non-failing trajectories, and add them to the
buffer to rebalance the 9:1 failure imbalance the diagnostic implicated.

**Pros:** attacks the data imbalance at its source; oracle used once at
data-prep, absent from training and deployment; "only the oracle" respected.
**Cons:** closer to imitation than learning; the witness safe action is arbitrary
among safe actions (not necessarily optimal), so use it to *balance* the data,
not replace it.

---

## 4. Recommendation

Stay on the **learning side** (oracle at train time, gone at test time), and
combine the fix for *discrimination* with a fix for *unsafe-action avoidance*:

> **Idea C (safe-relative advantage baseline) + Idea A variant 2 as a
> training-only soft penalty.**

- **C** repairs the collapse we diagnosed: advantages no longer flatten to a
  constant floor, because the baseline is the best *safe* action rather than a
  depressed V or a constant 0 — with no inference-time oracle.
- **A-variant-2 (soft, annealed, train-only)** teaches the actor to *internalize*
  avoidance of unsafe actions instead of being *forbidden* them. At deployment
  the actor is a plain softmax that has absorbed the safety knowledge.

**Honest caveat:** a soft train-only penalty will not drive unsafe probability to
exactly zero, so it won't match a hard shield's safety guarantee. That is the
correct price of staying learning-side — matching the shield would just be the
oracle again, and the genuine research question becomes: *how much of the
oracle's safety can the policy internalize from training signal alone?*

**Evaluation discipline (built-in honesty check):** always also evaluate with the
**oracle disabled**. If success holds up without inference-time masking, the
policy genuinely learned; if it collapses, the method was leaning on search.

All candidates are testable with the existing harness
([advantage_diagnostic.py](advantage_diagnostic.py) + the vanilla/action_safe
A/B in [iql.py](iql.py)): a successful fix should show failure-path safe
`frac_near_zero` returning toward vanilla levels (or becoming irrelevant) **and**
success rate recovering above vanilla's 16.6%.

---

## 5. Summary table

| Idea | Mechanism | Channel | Oracle at inference? | Side | Expected effect |
|---|---|---|---|---|---|
| A.1 hard mask | forbid unsafe actions | action | **yes** | search | strong safety, but benchmarks the oracle (rejected) |
| A.2 soft NLL penalty | push π mass off unsafe | action | no (train-only) | learning | internalized avoidance; no hard guarantee |
| B soft Q-floor | `relu(−Q)²` penalty | value | no | learning | modest; preserves ordering |
| B per-action floor | floor all safe Q | value | no | learning | corrects more values |
| C safe-relative adv | `Q − max_{a' safe} Q` | actor | no | learning | repairs discrimination directly |
| D oracle rollouts | rebalance dataset | data | no | learning | fixes imbalance; imitation-flavored |

**Primary recommendation: C + A.2 (train-only), evaluated with the oracle off.**
