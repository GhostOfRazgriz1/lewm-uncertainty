# When is a world model's uncertainty actionable? A layered negative on LeWorldModel

*Technical note. Substrate: pretrained LeWorldModel (LeWM, arXiv:2603.19312) on `swm/PushT-v1`. No
retraining anywhere; all signals are read off the frozen model.*

## Abstract

A JEPA world model yields several calibrated-looking uncertainty signals for free. We ask whether any of
them can be turned into a *decision*: choosing actions (control) or choosing *when to observe* (sensing).
Across four experiments on real Push-T transitions we find a consistent negative with a single mechanism.
(1) Penalizing high-uncertainty plans does not improve CEM planning — *calibration ≠ actionability*. For
sensing — deciding when to spend a real observation versus predict forward under a fixed budget — (2)
MC-dropout predictive variance is flat across a rollout and schedules observations no better than random,
though an oracle that places looks at the truly-surprising steps beats uniform spacing by up to ~30%, so
the headroom is real. (3) A small head trained to predict one-step surprise from the *true* latent does
predict it (held-out Spearman +0.38) but collapses at deployment, where it must run on the model's own
*drifted* estimate. (4) Training that head on the exact deployment distribution, with elapsed-time-since-
observation given explicitly, still fails: on a drifted estimate the head predicts its error *worse* than
the elapsed-time clock alone, and uniform spacing is unbeatable among observation-free policies. The
mechanism is the same throughout: **the actionable part of a world model's uncertainty lives in the
observations one is trying to budget.** An oracle with ground-truth access captures the headroom; no
signal computable from the agent's own state reaches it.

## 1. Setup

LeWM is an action-conditioned JEPA: a ViT encoder maps a frame to a 192-d latent, an action-conditioned
predictor rolls the latent forward, and a SIGReg term keeps the latent on a Gaussian shell. It plans by
CEM in latent space. It exposes, with no retraining, at least two uncertainty signals: the deviation of
the latent norm from its Gaussian shell (an OOD/epistemic signal) and the MC-dropout variance of the
predictor (a predictive-error signal that correlates with rollout error at Pearson +0.41 on real
transitions). The question of this note is not whether these signals are *calibrated* — they are, in the
weak sense of correlating with error — but whether they are *actionable*.

We test two kinds of decision on the same frozen model: **control** (does uncertainty improve planning?)
and **sensing** (does uncertainty tell you when to look?). All experiments use random open-loop rollouts
of 20 model-steps on `swm/PushT-v1`; the sensing experiments decouple sensing from control by fixing the
action sequence and only deciding *when to observe*.

## 2. Control: uncertainty-aware planning is null

We add an uncertainty penalty to CEM: rank plans by `z-score(dist-to-goal) + β·z-score(MC-dropout rollout
variance)`. A diagnostic first caught a confound — raw `[-1,1]` actions make the planner *worse than
random* because LeWM was trained on z-scored actions; at action-scale 2.0 CEM beats random by +40. On the
*working* planner (20 episodes): vanilla `−243.8 ± 105.9` vs `β=1.0` `−250.3 ± 93.6`, a delta of `−6.4`
inside the `±22` SEM. **No improvement.** Penalizing uncertainty biases the planner toward *predictable*
rather than *goal-reaching* trajectories on a near-deterministic task: calibration ≠ actionability.

## 3. Sensing: the question, the metric, and an oracle

If uncertainty is not actionable for control, the natural home for an information-gain signal is
*perception scheduling*. The agent maintains a latent estimate `ẑ`. At each step it either **looks**
(`ẑ ← encode(true frame)`, costing one observation and resetting tracking error to zero) or **predicts**
(`ẑ ← model.predict(...)`, free, error accumulates). Given a budget of `K` looks over a horizon of `T`,
*when* should it look? Frames are always full and real, so the encoder stays in-distribution (unlike
spatial foveation, which would feed it masked, OOD frames).

We compare policies at matched budget by **latent-tracking error** = `mean_t ‖ẑ_t − encode(frame_t)‖`. The
references are **fixed-interval** (look every `T/K` steps), **random**, and a non-causal **oracle** that
spends its looks on the steps with the largest true one-step error. The oracle is the ceiling: it shows
how much *any* scheduler could gain from a perfect signal.

## 4. MC-dropout is flat (≈ random)

Using MC-dropout predictive variance as the causal look-trigger fails. At budget 0.29 (6/21 looks):
variance `3.12 ± 0.28` is *worse* than fixed-interval `2.34 ± 0.17` (Δ `−0.78`, ≈2.4 SEM) and
statistically tied with random `3.24 ± 0.31`. The cause is visible directly: MC-dropout variance is
**flat** (~0.05–0.1) across the whole rollout, while the true one-step surprise is sharply peaked at
contact events. The signal carries essentially zero scheduling information.

But the **oracle** (2.12) beats fixed-interval — by ~0.2 at this budget and up to ~0.4 (≈30%) at
mid-budgets. So error growth is *heterogeneous* and the headroom is real; MC-dropout simply can't see it.
This is the same under-sharpness that limits a trained foveal `u_hat` head, now confirmed across two
different models: a signal calibrated weakly *in expectation* but far too flat *per-instance* to act on.

## 5. A learned surprise head: predictable, but not deployable

Is one-step surprise *causally predictable*? We train a small MLP `(z, a) → log1p(one-step error)` on
**true** latents (free labels, held-out rollout split, no LeWM retrain). **Eval 1 (predictability): yes,
but shallow.** Held-out Spearman is `+0.38` versus true surprise — well above MC-dropout's ≈0 — so surprise
*is* predictable; but the trivial `latent-drift` baseline reaches `+0.36`, so most of the signal is motion
autocorrelation (`|action|` alone is `+0.17`). **Eval 2 (deployment): no.** At budget 0.29, the learned arm
`2.83 ± 0.33` is within SEM of, and slightly worse than, fixed-interval `2.45 ± 0.21` (oracle 2.17).

The two evals disagree for one reason. Eval-1's correlation is measured on *true* latents, but deployment
runs on the *drifted maintained* latent. `latent-drift` deploys *worst of all* despite its high true-latent
correlation: once the agent predicts forward, the predictor produces smooth rollouts, so `‖ẑ−ẑ‖` measures
the model's own self-motion, not reality's surprise. The head survives a little better only via the
(undrifted) action, which is weak. The binding constraint is the **train/deploy distribution gap**.

## 6. Closing the gap: drift-aware training still loses to the clock

We remove the gap. Training pairs `(ẑ_drifted, a, h) → realized one-step error` are collected by
free-running with *random looks*, so the training distribution equals the deployment distribution, and the
head is handed `h` = steps-since-look explicitly. It can therefore trivially learn "error grows with `h`"
(which is fixed-interval); it beats fixed only if it also reads *state-dependent* drift rate.

It does not. **Eval 1 is the smoking gun:** the head's held-out correlation with realized error is `+0.20`,
*worse* than `h`-alone (`+0.43`). On a drifted estimate the only reliable predictor of "how wrong am I" is
how long since you looked; the state `(z, a)` does not add signal, it *dilutes* `h`. **Eval 2:** the
drift-aware arm `3.35 ± 0.61` sits at or above fixed-interval `2.88 ± 0.26` across the entire budget sweep
(oracle 2.28). Even with the distribution gap closed and elapsed time provided, **no learnable causal
signal beats uniform spacing.**

## 7. The mechanism

| # | Decision | Best causal signal vs baseline | Oracle | Verdict |
|---|----------|--------------------------------|--------|---------|
| M1.2 | control (CEM) | β·MC-variance: Δ −6.4 (±22 SEM) | — | null |
| M1.3 | sensing | MC-dropout var ≈ random, < fixed (2.4 SEM) | beats fixed (~30%) | null, headroom real |
| M1.4 | sensing | learned-on-truth: ≈ fixed at deploy | beats fixed | partial (drift gap) |
| M1.5 | sensing | drift-aware: head < `h`-alone; ≥ fixed | beats fixed | null, structural |

One mechanism explains all four. A world model's *actionable* uncertainty — the part that would change a
decision — lives in information the agent does not have at decision time: the true next state. The oracle,
which has it, captures real headroom throughout. Every signal computable from the agent's *own* state
either fails to track error (MC-dropout, flat) or is sharp only on observations it no longer holds once it
starts predicting (the surprise head, on true vs drifted latents). In the limit this is almost definitional:
**you cannot read your own divergence off your diverged estimate.** The best observation-free predictor of
your error is elapsed time since you last looked — i.e., uniform sampling is near-optimal among policies
that do not get to observe.

## 8. Scope and what would change the conclusion

These are negatives on *one* substrate (near-deterministic Push-T), in the *no-retrain* regime, with
*single-MLP / MC-dropout* uncertainty estimators and *one-step* surprise targets. Each is a place the
conclusion could move:

- **Stochasticity.** Push-T is near-deterministic; aleatoric tasks (genuine branching) would give the
  error-growth rate more state-dependence for a causal signal to exploit. The oracle's modest low-budget
  margin here may itself be a determinism artifact.
- **Estimators.** MC-dropout is a weak ensemble; a true deep ensemble, or a recurrent belief-state estimator
  of *cumulative* (not one-step) error, sees information a single feed-forward head does not.
- **Retraining.** A stochastic predictor with an explicit posterior (M2 territory) could carry a sharper
  intrinsic uncertainty than any read-off signal.

What does *not* move with these is the central observation: as long as the scheduler must decide from the
agent's own state, elapsed-time-since-observation is a strong baseline, and the burden is on any uncertainty
signal to beat it — which, here, none did.

---

*Reproduction: `src/plan_uncertainty.py` (M1.2), `src/active_sense.py` + `src/schedules.py` (M1.3),
`src/surprise_head.py` (M1.4), `src/surprise_head_drift.py` (M1.5). All Colab GPU; schedule logic
unit-tested locally.*
