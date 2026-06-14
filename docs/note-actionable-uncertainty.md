# A world model's uncertainty is a monitor, not a controller

*Technical note. Control, sensing, selective prediction, and representation-shaping on LeWorldModel (LeWM,
arXiv:2603.19312) over `swm/PushT-v1`. §2–7 read every signal off the frozen model with no retraining; §8
adds lightweight trained readout heads on the still-frozen encoder, and one experiment fine-tunes the
encoder itself.*

## Abstract

A JEPA world model yields several calibrated-looking uncertainty signals for free. We separate three uses:
as an *input that should make a decision better* (a **controller**), as a *flag for when to distrust the
model's output* (a **monitor**), and as a *training signal that should make the model better* (a **shaper**).
On the controller side we find a consistent negative with one mechanism, across four experiments on real
Push-T transitions. (1) Penalizing uncertain plans does not improve CEM planning. For sensing — deciding
when to spend a real observation under a budget — (2) MC-dropout variance is flat and schedules no better
than random, though an oracle that looks at the truly surprising steps shows ~30% headroom; (3) a head that
predicts one-step surprise from the *true* latent (held-out Spearman +0.38) collapses on the *drifted*
estimate it must use at deployment; (4) training it on the exact deployment distribution still loses to the
elapsed-time-since-observation clock. The actionable signal lives in observations the agent is withholding.
**On the monitor side the same signals succeed.** (5) Used for selective prediction, MC-dropout variance
abstains from in-distribution hard transitions and the latent-shell signal abstains from corrupted/OOD
inputs — each *useless* on the other's failure mode, and their combination covers both. (6) An **action-free
ensemble** sharpens the monitor: its disagreement is both horizon-calibrated (growth r +0.98) and
per-instance sharp (within-horizon Spearman +0.58), where MC-dropout is flat, and as a selective-prediction
signal it recovers 77% of the random→oracle gap versus MC-dropout's 11%. **On the shaper side the result is
null.** Using that same action-free objective to fine-tune the encoder end-to-end leaves the latent's
physical structure unchanged (pose-probe R² 0.502 → 0.500±0.023 single / 0.513±0.008 ensemble over three
seeds, all tied), and the calibrated-uncertainty loss that would *amplify* disagreement (HAUWM-style) is
actively harmful. The dividing line is the contribution: **a world model's uncertainty is for knowing you
don't know — not for deciding better, and not for making the model better.**

## 1. Setup

LeWM is an action-conditioned JEPA: a ViT encoder maps a frame to a 192-d latent, an action-conditioned
predictor rolls it forward, and a SIGReg term keeps the latent on a Gaussian shell. It exposes two
uncertainty signals with no retraining: **shell deviation** `|‖emb‖ − shell|` (an OOD/epistemic signal)
and **MC-dropout variance** of the predictor (a predictive-error signal that correlates with rollout error
at Pearson +0.41 on real transitions). A third signal, introduced in §8, is the **disagreement of an
action-free ensemble** of small predictors trained on the frozen latent. These signals are weakly
*calibrated*. The question is whether they are *actionable*, across three uses: a **controller** (does
uncertainty improve a decision — planning, or scheduling observations?), a **monitor** (does it tell you
when to abstain?), and a **shaper** (does training with it improve the latent itself?).

All controller/monitor experiments use random open-loop rollouts of 20 model-steps; the sensing experiments
fix the action sequence and only decide *when to observe*.

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
failure mode you expect; combine only when both are in play.** §8 sharpens the in-distribution (predictive)
facet substantially.

## 8. Shaper: sharpen the monitor, but you cannot shape the latent

The monitor's predictive facet (MC-dropout) is *calibrated but blunt* — the same under-sharpness that sank
the sensing heads. Can a better uncertainty estimator both improve the monitor and, used as a loss, improve
the model? We train an **action-free ensemble**: `M = 8` small heads predict `emb_t → emb_{t+k}` (horizon
`k` embedded), on the frozen encoder, and read uncertainty off their **disagreement** `Var_i μ_i`. Action-
free prediction is genuinely multimodal (the future is underdetermined without the action), so the heads
have something real to disagree about.

**8.1 The ensemble is sharp and calibrated (where MC-dropout is not).** Its disagreement grows with horizon
almost perfectly (growth `r = +0.98`, tracking realized error 6.2 → 9.7) *and* ranks instances within a
fixed horizon (within-horizon Spearman `+0.58`). MC-dropout on the same model is flat on both axes
(`+0.02` / `+0.14`). The lever is ensemble disagreement in the action-free latent — not a special loss.

**8.2 It is a much sharper monitor.** Redoing §7's selective prediction with the ensemble (within-horizon,
confound-free): in-distribution it recovers **77%** of the random→oracle AURC gap, versus MC-dropout's
**11%** (shell −21%, worse than random). But the ensemble is **OOD-blind** — on the mixed clean+corrupted
pool it recovers only +23% (its heads agree, confidently wrong, on corrupted inputs), while the shell
recovers +42% and MC +55%. The complete monitor is `z(ensemble) + z(shell)`: 77% in-distribution, 78%
mixed. Same complementarity as §7, but the predictive facet is now sharp.

**8.3 Amplifying disagreement (a calibration loss) is harmful.** The natural next move — a HAUWM-style
horizon-calibrated-uncertainty loss `−k·Var_i μ_i` that *forces* disagreement to grow with horizon —
backfires. At `λ=1` it diverges (the reward is unbounded; predictions explode, fixed only with `log1p` +
gradient clipping). Stabilized, it gets growth `+1.00` *by construction* but kills sharpness (`+0.04`,
negative at long horizons): forcing disagreement ∝ `k` uniformly overwrites the real per-instance signal.
There is no "uncertainty collapse" to fix in a JEPA — that is an RSSM problem; here the plain ensemble is
already calibrated, and the loss only damages it.

**8.4 Shaping the encoder end-to-end is a no-op.** The representation question: does fine-tuning the encoder
with the action-free objective make the *latent itself* encode physical state better? We linear-probe the
PushT block pose (`info['block_pose']`, the clean target — a ridge probe shows the frozen latent already
encodes it at R² +0.53, while it barely localizes the small agent) from three encoders: **frozen-LeWM**,
**e2e-single** (encoder + one head, end-to-end, +VICReg anti-collapse), **e2e-ensemble** (encoder + the
8-head ensemble). Probe with closed-form ridge — an unregularized linear probe overfits 192→3 and reports
spurious R²<0 even on the frozen latent.

| encoder | pose-probe R² (mean ± SEM, 3 seeds) |
|---------|--------------------------------------|
| frozen-LeWM | +0.502 (deterministic) |
| e2e-single | +0.500 ± 0.023 |
| e2e-ensemble | +0.513 ± 0.008 |

All three are tied (single−ensemble `−0.013 ± 0.024`, −0.5 SEM). End-to-end shaping — with or without the
ensemble — leaves pose encoding exactly where pretraining left it. *A single seed had read* `e2e-single
0.561 / e2e-ensemble 0.450`, *a spurious 0.11 gap that would have supported a confident "the disagreement
objective trades current-state precision for future-spread" story; three seeds erased it.* (Same single-run
flip as the spatial-sensing false positive: always seed before claiming a mechanism.) The latent's physical
structure is already present from SIGReg pretraining and is not improvable by this objective.

## 9. The dividing line

| use | experiment | result |
|-----|-----------|--------|
| controller — control | β·MC-variance in CEM cost | null (Δ −6.4, ±22 SEM) |
| controller — sensing | MC-dropout schedule | null (≈ random; oracle shows headroom) |
| controller — sensing | learned-on-truth head | partial (collapses on drift) |
| controller — sensing | drift-aware head | null (< `h`-alone; uniform unbeatable) |
| **monitor** | selective prediction (shell + MC) | **positive (AURC ≪ random; facets complementary)** |
| **monitor** | selective prediction (ensemble) | **positive (77% of gap vs MC 11%; sharper)** |
| shaper | HAUWM-style calibration loss | negative (harmful; overwrites the signal) |
| shaper | end-to-end encoder fine-tune | null (pose-probe R² unchanged, 3 seeds) |

One mechanism explains the controller split. A world model's *actionable-for-control* uncertainty — the part
that would change which action you take or which step you look at — lives in information the agent does not
have at decision time (the true next state); an oracle with that information captures real headroom
throughout, but no signal computable from the agent's own state reaches it. The *monitor* use asks nothing
of the future: it only ranks the model's current outputs by how much to trust them, and for that the same
signals are immediately useful — sharpest from an action-free ensemble for in-distribution predictive error,
the shell for distributional shift, combined for both. The *shaper* use asks the most and gets the least:
the latent's structure is set by pretraining, and neither an uncertainty loss nor end-to-end action-free
fine-tuning improves it. **Uncertainty tells you when you don't know; it does not tell you what to do about
it, nor how to make the model know more.**

## 10. Scope and what would change the conclusion

The negatives are on *one* substrate (near-deterministic Push-T), in the *frozen-encoder* regime for the
controller/monitor work, with *single-MLP / MC-dropout / 8-head ensemble* estimators and *one-step* targets.
Each is a place the *controller* conclusion could move: stochastic tasks give the error-growth rate more
exploitable state-dependence; a recurrent belief-state estimator of *cumulative* error sees information a
feed-forward head does not. The *shaper* null is specific to fine-tuning a pretrained SIGReg encoder on a
deterministic task; training from scratch, or on a stochastic env where multimodality is intrinsic rather
than action-induced, could let the action-free objective shape structure that pretraining did not already
supply. The *monitor* result is the most robust — it rests only on the facets being calibrated, which they
are, the ensemble sharply so — and would extend naturally to selective *control* (abstain/replan under
shift), the higher-variance follow-on.

---

*Reproduction: `src/plan_uncertainty.py` (§2), `src/active_sense.py` + `src/schedules.py` (§3–4),
`src/surprise_head.py` (§5), `src/surprise_head_drift.py` (§6), `src/monitor.py` (§7), `src/hcu_jepa.py`
(§8.1, §8.3), `src/monitor_ensemble.py` (§8.2), `src/tier2_pose_probe.py` + `src/tier2_diag.py` (§8.4). All
Colab GPU; schedule logic unit-tested locally.*
