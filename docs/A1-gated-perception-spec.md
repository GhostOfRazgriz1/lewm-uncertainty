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

## If positive → stage 2 (the paper's headline)
Plug the gate into **CEM planning** and measure **control cost / success under corruption** vs blind vs a
*trained*-uncertainty baseline (e.g., a small ensemble of predictors, or a learned corruption classifier).
The novelty is that the JEPA's *own geometry* gives the deploy-time robustness for free. The M1.2 cost-
shaping null is the foil: penalizing uncertainty in the cost didn't help, but gating *which observations to
trust* does — the distinction is the contribution.

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
