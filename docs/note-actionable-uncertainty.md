# A world model's uncertainty is a monitor, not a controller

*Technical note. Control, sensing, and selective prediction on a frozen LeWorldModel (LeWM,
arXiv:2603.19312) over `swm/PushT-v1`. No retraining anywhere; all signals are read off the frozen model.*

## Abstract

A JEPA world model yields several calibrated-looking uncertainty signals for free. We separate two uses:
as an *input that should make a decision better* (a **controller**), and as a *flag for when to distrust
the model's output* (a **monitor**). On the controller side we find a consistent negative with one
mechanism, across four experiments on real Push-T transitions. (1) Penalizing uncertain plans does not
improve CEM planning. For sensing — deciding when to spend a real observation under a budget — (2)
MC-dropout variance is flat and schedules no better than random, though an oracle that looks at the truly
surprising steps shows ~30% headroom; (3) a head that predicts one-step surprise from the *true* latent
(held-out Spearman +0.38) collapses on the *drifted* estimate it must use at deployment; (4) training it on
the exact deployment distribution still loses to the elapsed-time-since-observation clock. The actionable
signal lives in observations the agent is withholding. **On the monitor side the same signals succeed.**
(5) Used for selective prediction, MC-dropout variance abstains from in-distribution hard transitions
(AURC 1.82 vs 2.57 random; ~60% of the oracle gap) and the latent-shell signal abstains from corrupted/OOD
inputs (4.57 vs 8.36) — each *useless* on the other's failure mode, and their combination covers both
(4.22, approaching the 3.79 oracle). The dividing line is the contribution: **a world model's uncertainty
is for knowing you don't know, not for deciding better.**

## 1. Setup

LeWM is an action-conditioned JEPA: a ViT encoder maps a frame to a 192-d latent, an action-conditioned
predictor rolls it forward, and a SIGReg term keeps the latent on a Gaussian shell. It exposes two
uncertainty signals with no retraining: **shell deviation** `|‖emb‖ − shell|` (an OOD/epistemic signal)
and **MC-dropout variance** of the predictor (a predictive-error signal that correlates with rollout error
at Pearson +0.41 on real transitions). These signals are weakly *calibrated*. The question is whether they
are *actionable*, and we test two distinct uses: a **controller** (does uncertainty improve a decision —
planning, or scheduling observations?) and a **monitor** (does it tell you when to abstain?).

All experiments use random open-loop rollouts of 20 model-steps; the sensing experiments fix the action
sequence and only decide *when to observe*.

## 2. Controller, control: uncertainty-aware planning is null

Add an uncertainty penalty to CEM: rank plans by `z(dist-to-goal) + β·z(MC-dropout rollout variance)`. A
diagnostic first caught a confound — raw `[-1,1]` actions make the planner worse than random (LeWM was
trained on z-scored actions); at action-scale 2.0 CEM beats random by +40. On the working planner (20
episodes): vanilla `−243.8 ± 105.9` vs `β=1.0` `−250.3 ± 93.6`, delta `−6.4` inside the `±22` SEM. **No
improvement.** Penalizing uncertainty biases toward *predictable* rather than *goal-reaching* trajectories:
calibration ≠ actionability.

## 3. Controller, sensing: the question, the metric, an oracle

The natural home for an information-gain signal is perception scheduling. The agent maintains a latent `ẑ`
and at each step either **looks** (`ẑ ← encode(true frame)`, costs one observation, resets tracking error)
or **predicts** (`ẑ ← model.predict(...)`, free, error accumulates). Given a budget of `K` looks over
horizon `T`, *when* should it look? Frames stay full and real, so the encoder is in-distribution. We
compare policies at matched budget by **latent-tracking error** `mean_t ‖ẑ_t − encode(frame_t)‖`, against
**fixed-interval**, **random**, and a non-causal **oracle** that spends looks on the largest-true-error
steps (the ceiling on what any scheduler could gain from a perfect signal).

## 4. MC-dropout is flat (≈ random)

MC-dropout variance as the causal trigger fails. At budget 0.29: variance `3.12 ± 0.28` is worse than
fixed-interval `2.34 ± 0.17` (Δ −0.78, ≈2.4 SEM) and tied with random `3.24 ± 0.31`. The variance is flat
(~0.05–0.1) across the rollout while true surprise is sharply peaked at contacts — it carries essentially
zero scheduling information. But the **oracle** (2.12) beats fixed by ~0.2 here and up to ~0.4 (≈30%) at
mid-budgets: the headroom is real, MC-dropout just can't see it.

## 5. A learned surprise head: predictable, not deployable

Train a small MLP `(z, a) → log1p(one-step error)` on **true** latents (held-out split, no retrain). **Eval
1:** held-out Spearman `+0.38` vs true surprise (≫ MC-dropout's ≈0) — surprise *is* predictable — but the
trivial `latent-drift` baseline reaches `+0.36`, so most of it is motion autocorrelation. **Eval 2:** at
budget 0.29 the learned arm `2.83 ± 0.33` is within SEM of, and slightly worse than, fixed-interval `2.45 ±
0.21` (oracle 2.17). The disagreement: eval-1 is on *true* latents, deployment runs on the *drifted*
maintained latent. `latent-drift` deploys worst of all despite its true-latent correlation — once the agent
predicts forward the predictor yields smooth rollouts, so `‖ẑ−ẑ‖` reads the model's self-motion, not
reality's surprise. The constraint is the **train/deploy distribution gap**.

## 6. Closing the gap: drift-aware training still loses to the clock

Train `(ẑ_drifted, a, h) → realized error` on the *exact* deployment distribution (random-look free-runs),
`h` = steps-since-look given explicitly — so the head can trivially learn the `h`-trend (which is
fixed-interval) and beats it only by reading *state-dependent* drift. It does not. **Eval 1 is the smoking
gun:** the head's held-out correlation with realized error is `+0.20`, *worse* than `h`-alone (`+0.43`). On
a drifted estimate the only reliable predictor of "how wrong am I" is how long since you looked; the state
`(z, a)` dilutes `h`. **Eval 2:** the drift-aware arm `3.35 ± 0.61` sits at/above fixed-interval `2.88 ±
0.26` across the whole sweep (oracle 2.28). Even with the distribution gap closed and elapsed time provided,
**no learnable causal signal beats uniform spacing.**

## 7. Monitor: the same signals work for selective prediction

The controller results all share one shape: the actionable information lives in observations the agent does
not have at decision time. So we ask the *monitor* question instead — not "act better," but "abstain when
you'll be wrong." Per transition we compute the true error and rank by a signal, keep the most-confident
coverage `c`, and measure error on the kept set (**risk–coverage**, summarized by **AURC**, lower better).
Two failure modes, reported separately so the win is not a trivial OOD-sort:

| pool | signal | AURC | random | oracle |
|------|--------|------|--------|--------|
| in-dist (clean) | **MC-variance** | **1.82** | 2.57 | 1.31 |
| in-dist (clean) | shell | 2.68 | 2.57 | 1.31 |
| mixed (clean + corrupted) | shell | **4.57** | 8.36 | 3.79 |
| mixed (clean + corrupted) | MC-variance | 5.76 | 8.36 | 3.79 |
| mixed (clean + corrupted) | **combined** | **4.22** | 8.36 | 3.79 |

**MC-variance abstains from in-distribution hard transitions** (1.82 vs 2.57 random — ~60% of the way to
oracle), where the shell signal is useless (2.68 ≈ random; orthogonal, as the +0.05 calibration earlier
predicted). **The shell signal abstains from corrupted/OOD inputs** (4.57 vs 8.36), where MC-variance is the
weaker signal. **Their combination covers both** (4.22 on the mixed pool, below either single signal and
approaching the 3.79 oracle). The two facets are complementary: each is a working monitor for one failure
mode and blind to the other.

One caveat sharpens the practical rule rather than weakening it: combining *hurts* in-distribution (combined
2.13 vs MC-alone 1.82), because there the shell signal is pure noise. **Use the facet that matches the
failure mode you expect; combine only when both are in play.**

## 8. The dividing line

| use | experiment | result |
|-----|-----------|--------|
| controller — control | β·MC-variance in CEM cost | null (Δ −6.4, ±22 SEM) |
| controller — sensing | MC-dropout schedule | null (≈ random; oracle shows headroom) |
| controller — sensing | learned-on-truth head | partial (collapses on drift) |
| controller — sensing | drift-aware head | null (< `h`-alone; uniform unbeatable) |
| **monitor** | selective prediction | **positive (AURC ≪ random; facets complementary)** |

One mechanism explains the split. A world model's *actionable-for-control* uncertainty — the part that
would change which action you take or which step you look at — lives in information the agent does not have
at decision time (the true next state); an oracle with that information captures real headroom throughout,
but no signal computable from the agent's own state reaches it. The *monitor* use asks nothing of the
future: it only ranks the model's current outputs by how much to trust them, and for that the same signals
are immediately useful — MC-variance for in-distribution predictive error, the shell for distributional
shift, combined for both. **Uncertainty tells you when you don't know; it does not tell you what to do
about it.**

## 9. Scope and what would change the conclusion

The negatives are on *one* substrate (near-deterministic Push-T), in the *no-retrain* regime, with
*single-MLP / MC-dropout* estimators and *one-step* targets. Each is a place the *controller* conclusion
could move: stochastic tasks give the error-growth rate more exploitable state-dependence; a true deep
ensemble or a recurrent belief-state estimator of *cumulative* error sees information a feed-forward head
does not; a retrained stochastic predictor (M2) could carry a sharper intrinsic uncertainty. The *monitor*
result is more robust — it rests only on the two facets being calibrated, which they are — and would extend
naturally to selective *control* (abstain/replan under shift), the higher-variance follow-on.

---

*Reproduction: `src/plan_uncertainty.py` (§2), `src/active_sense.py` + `src/schedules.py` (§3–4),
`src/surprise_head.py` (§5), `src/surprise_head_drift.py` (§6), `src/monitor.py` (§7). All Colab GPU;
schedule logic unit-tested locally.*
