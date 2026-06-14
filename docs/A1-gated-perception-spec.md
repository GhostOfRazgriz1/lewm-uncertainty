# A (stage 1) — trust-gated perception under intermittent observation corruption (spec)

## Question
Direction A: does a JEPA world model's *free* uncertainty improve **deployment under distribution shift**?
Stage 1 is the cheap mechanism check (state estimation, no planner): under intermittent observation
corruption, can the **shell signal** `|‖encode(obs)‖ − √d|` decide *which observations to trust* well enough
to keep a near-oracle latent state estimate — with no corruption labels and no retraining?

This is the regime our prior nulls predict should flip **positive**. The M1.3 budget-sensing null happened
because every frame was clean (no signal to gate on); M1.6 proved the shell separates clean from corrupted.
So gating on the shell — which failed under a *budget* — should win under *shift*.

## Setup
Closed-loop estimation on `swm/PushT-v1`, open-loop random actions (the M1.3 rig). The agent maintains `ẑ`.
Each step `t≥1` the observation is corrupted with probability `p` (mask `m_t`). Per frame the agent either
- **trusts** — `ẑ_t ← encode(obs_t)` (adopt the observation), or
- **coasts** — `ẑ_t ← predict(ẑ_{<t}, a_{<t})` (free; error accumulates / drifts).

`t=0` is assumed clean (known good start). The gate decision uses only `encode(obs_t)`'s shell deviation —
a per-frame, label-free, no-retrain signal.

## Policies (all compared at matched trust-rate)
- **blind** — trust every frame (poisoned by corruption). The baseline.
- **random-gate** — coast a *random* subset sized to match shell-gate's trust count. Isolates whether the
  *signal* matters, or whether any coasting helps.
- **shell-gate** (ours) — trust iff `|‖encode(obs)‖ − √d| < τ`.
- **oracle-gate** — trust iff the frame is truly clean (`m_t = 0`). The ceiling (perfect detector).

`τ` is calibrated **label-free** from the clean in-distribution shell-dev distribution (a held-out clean
calibration set; `τ = 95th percentile`). Reported with a small `τ`-quantile sensitivity {0.90, 0.95, 0.99}.

## Metric
Latent-tracking error vs the **clean** latent (ground truth for eval, never shown to the agent):
`mean_t ‖ẑ_t − encode(clean_t)‖`, mean ± SEM over deploy rollouts. Sweep corruption rate
`p ∈ {0.2, 0.4, 0.6}` × corruption type {Gaussian noise, blackout} (noise pushes `‖emb‖` off-shell upward,
blackout collapses it downward — the two-sided shell-dev should catch both). Diagnostic: shell-gate's
precision/recall at *detecting* corruption vs the true mask.

## Verdict (pre-registered)
- **WIN** — shell-gate < blind **and** < random-gate (both beyond pooled SEM) **and** ≈ oracle (within ~1–2
  SEM): the free uncertainty recovers near-oracle robust estimation under shift, and the *signal* (not just
  coasting) is what does it.
- **PARTIAL** — shell-gate beats blind but not random-gate: the win is coasting, not the shell signal.
- **NULL** — shell-gate ≈ blind: the shell signal is not actionable for gating either (would extend the
  controller-null to deployment, a clean negative).

## Stage 2 — CONTROL under corruption (`src/gated_control.py`)
The headline. Closed-loop CEM control (the M1.2 planner, **vanilla β=0**) to do PushT, with observations
intermittently corrupted. We change *nothing* about the planner — we gate the **state estimate** it plans
from: a `blind` agent re-encodes every (possibly corrupted) frame → plans from a poisoned latent; a
`shell-gate` agent coasts (predict-forward with the executed action) through frames it distrusts → plans from
a clean estimate. **This is the foil to M1.2:** M1.2 gated the planner's *cost* (`+β·variance`) and found
nothing; gating *which observations the planner trusts* is a different lever.

- **Policies:** `clean` (no corruption, the ceiling) · `blind` · `random-gate` (coast a matched-rate random
  subset) · `shell-gate` · `oracle-gate`.
- **Metric:** best task reward / episode (PushT coverage), mean ± SEM + success-rate; reported as the
  fraction of the `clean→blind` corruption drop that shell-gate **recovers**. WIN = shell-gate > blind beyond
  SEM **and** ≈ oracle. First run: noise × p∈{0.3, 0.5}, 15 episodes (~30–60 min Colab).
- **Note (right facet matters):** the action-free *ensemble* would **fail** as this gate — M2.2 showed it is
  OOD-blind (heads agree, confidently wrong, on corrupted inputs). It is specifically the **shell/OOD** facet
  that does deployment work.

## If stage 2 is positive → strengthen for main-track
Add a **trained supervised corruption classifier** baseline (logistic probe on the latent, *with* labels) to
show the *free, label-free* shell matches it; add **blackout + subtler shift** (mild corruption where
detection isn't trivial); and a second substrate. The novelty: the JEPA's *own geometry* gives deploy-time
robustness for free, where M1.2's cost-shaping could not.

## Risks / honest scope
- **shell-gate > blind is near-certain** under heavy corruption (blind adopts garbage). The load-bearing
  comparisons are shell-gate **vs random-gate** (does the signal matter?) and **vs oracle** (how close to
  perfect?), plus the low-`p` regime where over-coasting could *hurt* (skipping clean frames it wrongly
  flags) — that tradeoff is where the result is non-trivial.
- **One substrate** (PushT) and **one estimator** (the shell). Stage 2 + a stochastic substrate (Direction
  B) would broaden it.
- Coast uses the deterministic predictor (`mc=0`); cheap (no MC sampling), much lighter than the Tier-2
  fine-tunes.

## Code
`src/gated_perception.py` (reuses the `src/active_sense.py` rig: `rollout`, `encode_all`, `act_encode`,
`_predict_one`, `HS`, `T_STEPS`, `cfg`). Colab GPU.
