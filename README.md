# lewm-uncertainty

Augmenting **LeWorldModel** (LeWM) with calibrated uncertainty — for OOD detection and **uncertainty-aware planning**. Runs on a Colab (Linux + GPU) runtime, where the `stable_worldmodel` control sims work (they don't on Windows).

## Why (carried over from the perceptor exploration)

- **LeWM** ([le-wm.github.io](https://le-wm.github.io)) is an action-conditioned JEPA *control* world model: ViT-tiny → 192-d CLS latent, action-conditioned predictor (AdaLN), SIGReg Gaussian-latent regularizer, **CEM planning** (roll out action plans in latent space, pick the one whose final latent is closest to a goal latent). ~18M params.
- **Two uncertainty findings, complementary:**
  - LeWM's **SIGReg-Gaussian latent gives a *free OOD signal***: the latent norm sits on a Gaussian shell for in-distribution inputs and deviates for OOD (other envs/noise collapse below, corruption pushes above). Two-sided `|‖emb‖ − shell|` flags OOD. *(epistemic / input-familiarity facet)*
  - A foveal WM's trained `u_hat` head gives a **predictive-error** uncertainty, calibrated-in-expectation but per-instance under-sharp. *(predictive-error facet)*
  - **Neither model has both facets.** That's the analysis contribution. The constructive question — *how do you give a WM a sharp, actionable uncertainty, and does it help planning?* — needs the control loop, i.e. this repo.
- A heteroscedastic head did **not** sharpen the predictive uncertainty (structural: a single readout off the shared state). Remaining constructive levers: **ensemble disagreement** (epistemic) and a **stochastic RSSM/KL latent + uncertainty-aware planning** — both best tested *on LeWM*, where rollouts compound uncertainty and there's a real decision (the plan) to make with it.

## Plan (staged — gate each before the next)

- **M0 — infra gate (no retrain).** Reproduce LeWM's Push-T planning eval on Colab. Confirms `stable_worldmodel[env]` + the sim run, and that we can drive CEM with the pretrained checkpoint. *Needs the checkpoint + the env, NOT the 13 GB dataset.*
- **M1 — cheap real-substrate results (no retrain).**
  1. **Rigorous calibration probe** with *real* transitions from planning rollouts: does any derivable uncertainty (MC-dropout variance on the predictor, SIGReg latent-norm deviation, rollout latent-drift) **predict rollout error**? This upgrades the OOD probe to true predictive-error calibration.
  2. **Uncertainty-aware CEM:** score plans by `dist-to-goal + β·(rollout uncertainty)` and compare success rate vs vanilla CEM. The derived uncertainty needs no retrain.
  3. **Uncertainty-aware *sensing* (M1.3):** redirect the calibrated signal from control (M1.2 null) to **perception** — the white space nobody in the LeWM citation set occupies. *Temporal active sensing:* the agent maintains a latent, and MC-dropout variance decides *when* to spend a real observation (re-encode the true frame) vs predict forward. Compare latent-tracking error vs **fixed-interval / random / oracle** schedules **at matched budget**. Frames stay full+real so the ViT encoder is never OOD (unlike spatial foveation). `src/active_sense.py` (Colab); pure scheduling logic in `src/schedules.py` (unit-tested locally).
  4. **Learned surprise head (M1.4):** M1.3's deployable signal (MC-dropout) was flat, but the oracle showed real headroom. Train a tiny MLP `(z, a) → log1p(one-step error)` on true latents (free labels, no retrain) and ask: is one-step surprise *causally predictable* sharply enough to recover the oracle scheduling? `src/surprise_head.py`; spec `docs/M1.4-surprise-head-spec.md`.
  5. **Drift-aware head (M1.5):** M1.4 was PARTIAL (predictable on true latents, collapses on the drifted deploy estimate). Train the head on *exactly* the deploy distribution — pairs `(ẑ_drifted, a, h) → realized error` from random-look free-runs, `h` given explicitly. WIN (beats fixed) = first constructive positive; NULL (ties fixed) = obstacle is structural. `src/surprise_head_drift.py`; spec `docs/M1.5-drift-aware-spec.md`.
  6. **Runtime monitor (M1.6) — the positive complement.** The arc (M1.2–1.5) shows uncertainty can't *improve* a decision; M1.6 asks if it tells you *when to abstain*. Selective prediction: rank transitions by MC-variance / shell / combined, risk–coverage / AURC on prediction error, in-dist + corruption shift. POSITIVE if uncertainty beats random and the facets are complementary (MC in-dist, shell on shift). `src/monitor.py`; spec `docs/M1.6-monitor-spec.md`.
- **M2 — heavy (retrain, needs the 13 GB + sims).** Stochastic LeWM: predictor outputs `(μ, σ)`; loss `KL(posterior‖prior) + λ·SIGReg` (Dreamer ELBO with SIGReg replacing reconstruction). Proper predictive uncertainty → uncertainty-aware planning, benchmarked vs M1's derived signal and vs vanilla.

## Colab — Milestone 0  (GPU runtime; run cell by cell, absolute `/content` paths)

```python
# Cell 1 — deps. Do NOT pin transformers (stable-worldmodel resolves its own; pinning 4.49 conflicts).
# The checkpoint has old (4.x) ViT key names, so we just need whatever 4.x sw pulls.
!pip install -q "stable-worldmodel[train,env]" stable-pretraining
import transformers; print("transformers:", transformers.__version__)   # report this
# If Cell 3 later asserts on a key mismatch, transformers resolved to 5.x -> run:
#   !pip install -q "transformers<4.50"   then Runtime>Restart, then re-run from Cell 2.
```
```python
# Cell 2 — code (rm first so a re-run from a failed clone is clean)
!cd /content && rm -rf le-wm lewm-uncertainty \
 && git clone -q https://github.com/lucas-maes/le-wm.git \
 && git clone -q https://github.com/GhostOfRazgriz1/lewm-uncertainty.git && ls /content
```
```python
# Cell 3 — VALIDATED step: load the pretrained LeWM on GPU (no env needed).
import sys; sys.path.insert(0, "/content/lewm-uncertainty")
from src.load_lewm import load_lewm
model, cfg = load_lewm("/content/le-wm", device="cuda")
print("LeWM loaded:", sum(p.numel() for p in model.parameters()) / 1e6, "M params")
```
```python
# Cell 4 — convert to the _object.ckpt eval.py expects (uses the model from Cell 3)
import os, torch
os.environ["STABLEWM_HOME"] = "/content/stablewm"
out = "/content/stablewm/pusht/lewm_object.ckpt"
os.makedirs(os.path.dirname(out), exist_ok=True)
torch.save(model.cpu(), out); print("saved", out)
```
```python
# Cell 5 — Push-T planning eval (needs the env; NOT the 13GB dataset). Report the planner output + any error.
%cd /content/le-wm
!STABLEWM_HOME=/content/stablewm python eval.py --config-name=pusht.yaml policy=pusht/lewm
```

Cell 3 should print ~18M params (the validated loader). Cells 4–5 are the parts to debug from their output — paste back errors and the planner trace.

If M0's success rate reproduces the paper (~the bar charts in their gif), the infra is good and M1 is unblocked.

## Results

**M0 (infra gate):** LeWM loads on Colab (transformers 4.x), `swm/PushT-v1` builds/renders/steps, and the
rollout→encode→predict pipeline forward-models on real transitions (pred-err 2.30 < copy 2.73). ✅

**M1.1 — predictive-error calibration on 400 real Push-T transitions** (`src/probe_predictive.py`,
`lewm_predictive_calibration.png`): **MC-dropout variance on the predictor predicts LeWM's own rollout
error — Pearson +0.41, Spearman +0.40, monotone calibration curve.** No retraining. The OOD/latent-shell
signal is flat against prediction error (Pearson +0.05) → the two are **orthogonal facets**: MC-dropout =
*predictive-error* ("will I be wrong"), latent-shell = *epistemic/OOD* ("is this input familiar"). Neither
subsumes the other — the complementary-facets thesis, demonstrated within one model on real transitions.
→ unblocks **M1.2:** uncertainty-aware CEM with cost `get_cost + β·MC-variance`.

**M1.2 — uncertainty-aware CEM planning** (`src/plan_uncertainty.py`, `src/plan_diagnose.py`). First run
looked like a flat null — but the diagnostic (`plan_diagnose.py`) exposed a **confound**: at action-scale
1.0 (raw `[-1,1]`) the planner is *worse than random* (−16) because the model was trained on z-scored
actions; at **scale 2.0** CEM beats random by **+40** (planner actually steers). Re-run on the working
planner (scale 2.0, 20 eps): vanilla −243.8 ± 105.9 vs `β=1.0` uncertainty-aware −250.3 ± 93.6 →
**delta −6.4, inside the ~±22 SEM. CLEAN NULL.** So a calibrated predictive uncertainty does **not** improve
this planner via a cost penalty: **calibration ≠ actionability** (Push-T is near-deterministic; distrusting
uncertain plans biases toward *predictable*, not *goal-reaching*, plans).

**M1.3 — temporal active sensing (when to look)** (`src/active_sense.py`, `src/schedules.py`,
`lewm_active_sensing.png`). Redirect the signal from control (M1.2 null) to **perception**: the agent
maintains a latent and MC-dropout variance decides *when* to spend a real observation vs predict forward;
compare latent-tracking error vs fixed/random/oracle schedules at matched budget. **The deployable policy
fails.** At budget 0.29 (6/21 looks) the variance policy scores **3.12 ± 0.28 — worse than fixed-interval
2.34 ± 0.17** (Δ −0.78 ≈ 2.4 SEM) and **statistically tied with random (3.24 ± 0.31)**: MC-dropout variance
carries ~zero scheduling information. **But this is not "nothing to exploit":** the **oracle** (look at the
intrinsically-most-surprising steps) **beats fixed-interval** — 2.12 vs 2.34 here, up to ~0.4 (≈30%) at
mid-budgets — so error growth is **heterogeneous** and the headroom is real. The diagnostic figure shows
*why* the signal fails: **MC-dropout variance is flat (~0.05–0.1) across the whole rollout** while intrinsic
surprise is sharply peaked at contacts; the two don't track. This is the **same under-sharpness** that
limited the foveal `u_hat`, now confirmed **cross-model** — a signal weakly calibrated *in expectation*
(+0.41, M1.1) but far too flat *per-instance* to act on. Sensing refines the M1.2 control lesson: here
uncertainty **is** actionable in principle (the oracle wins, unlike control), but the available MC-dropout
signal can't realize it — **the bottleneck is signal sharpness, which is fixable** (→ M1.4: a learned
one-step-surprise head, no retrain, tested vs MC-dropout and the oracle ceiling).

**M1.4 — learned surprise head** (`src/surprise_head.py`, `lewm_surprise_head.png`). Can a sharp *causal*
signal capture the M1.3 oracle headroom MC-dropout missed? A tiny MLP `(z, a) → log1p(one-step error)`
trained on TRUE latents (held-out split, no retrain). **Eval 1 — predictability: yes, but shallow.**
Held-out Spearman **+0.38** vs true surprise (≫ MC-dropout's ~0) — surprise *is* causally predictable —
but the trivial `latent-drift` baseline gets **+0.36**, so most of it is motion-autocorrelation
(`|action|` +0.17). **Eval 2 — deployment: no.** At budget 0.29, `learned` **2.83 ± 0.33** is within-SEM
of, and slightly worse than, **fixed-interval 2.45 ± 0.21** (oracle 2.17). **PARTIAL.** The two evals
disagree for one reason: eval-1's correlation is on **true** latents, but deployment runs on the
**drifted maintained** latent. `latent-drift` deploys *worst of all* despite its high true-latent
correlation — once the agent predicts forward the predictor yields smooth rollouts, so `‖ẑ−ẑ‖` reads the
model's *self-motion*, not reality's surprise; the head survives a little better only via the (undrifted)
action, which is weak. **The binding constraint is the train/deploy distribution gap:** the useful
surprise signal lives in information you only have *after looking* (the true latent); from your own
drifted estimate you can't reliably tell you've diverged — almost definitionally. → M1.5 (drift-aware
training) is the pre-registered next lever, with a real chance of another principled negative.

**M1.5 — drift-aware head** (`src/surprise_head_drift.py`, `lewm_drift_aware.png`). Removes M1.4's
train/deploy gap: train `(ẑ_drifted, a, h) → realized one-step error` on the *exact* deploy distribution
(random-look free-runs), with `h` (steps-since-look) handed to the head explicitly. **Still NULL —
decisively.** Eval 1 is the smoking gun: the head's held-out correlation with realized error is **+0.20,
*worse* than `h`-alone (+0.43)** — on a drifted estimate the only reliable predictor of "how wrong am I"
is *how long since I looked*; the state `(z, a)` not only fails to add signal, it dilutes `h`. Eval 2
(budget 0.29): `drift-aware` 3.35 ± 0.61 is at/above fixed-interval 2.88 ± 0.26 across the whole sweep
(oracle 2.28). So even with the distribution gap closed and `h` provided, **no learnable causal signal
beats uniform spacing.** The obstacle is structural: you cannot read state-dependent divergence off your
own diverged estimate — the best observation-free predictor of your error is elapsed time, which *is*
fixed-interval.

**M1.6 — uncertainty as a runtime monitor** (`src/monitor.py`, `lewm_monitor.png`) — the **positive
complement**. The arc shows uncertainty can't *improve* a decision; this asks if it flags *when to abstain*.
Selective prediction (rank by a signal, keep coverage `c`, measure error on the kept set; AURC, lower
better). **In-dist:** MC-variance AURC **1.82** vs random 2.57 (oracle 1.31) — abstains from hard transitions
with no OOD tell; the shell signal is useless there (2.68 ≈ random, orthogonal). **Mixed (clean +
noise-corrupted):** the shell signal **4.57** vs random 8.36 catches the corruption, and **combined 4.22**
beats both singles, approaching oracle 3.79. So **each facet is a working monitor for its own failure mode
and blind to the other; combined covers both.** Caveat that sharpens the rule: combining *hurts* in-dist
(2.13 vs MC 1.82) — use the facet matching your failure mode, combine only when both are in play.
**POSITIVE** — the "complementary facets" analysis becomes an actionable monitor.

**Verdict.** The calibration story is: *world models carry different, incomplete, calibrated facets of "I
don't know" (LeWM OOD-geometry; MC-dropout predictive-error; ours `u_hat`) — but they are hard to sharpen
(heteroscedastic head failed, structural) and hard to act on (planning-penalty null on a working planner).*
The **analysis** (measuring the facets, across models, on real substrates, with the confounds caught) is the
contribution; the **constructive** side is a set of honest negatives. M2 (stochastic RSSM/KL retrain) would
give a "better" uncertainty, but the M1.2 null suggests the blocker is task-relevance, not uncertainty
quality — so M2 is a high-cost bet with uncertain payoff. **M1.3–M1.4 extend the negative from control to
sensing, and sharpen it:** the signal isn't only hard to *act on*, it's hard to *obtain causally* —
MC-dropout is flat (≈ random scheduling), and a learned surprise head is sharp on true latents but
collapses on the drifted estimate you actually hold at decision time. The throughline across M1.2–M1.4: a
world model's useful "I don't know" tends to require the very observations you are trying to budget.
**M1.5 closes the arc:** training the surprise predictor on the exact deploy distribution, with elapsed
time handed to it, *still* can't beat uniform spacing — on a drifted estimate the head is worse than the
steps-since-look clock alone. Four honest negatives, one mechanism: the oracle (truth) wins throughout, so
the headroom is real, but no signal computable from the agent's own state reaches it. The contribution is
the layered demonstration that a world model's actionable uncertainty lives in the observations you're
trying to avoid. **M1.6 flips the sign on the *other* use of uncertainty:** used as a *monitor* (selective
prediction) rather than a *controller*, the same signals work — MC-variance abstains from in-dist hard
transitions, the shell from OOD/corruption, combined covers both (AURC ≪ random, approaching oracle). The
dividing line is the result: **a world model's uncertainty is a monitor, not a controller — it tells you
when you don't know, not what to do about it.** Full writeup: `docs/note-actionable-uncertainty.md`.

## Repo layout

```
src/load_lewm.py         # reusable pretrained-LeWM loader (transformers-4.x note baked in)
src/probe_calibration.py # OOD probe (||emb|| in-dist vs OOD) — from the perceptor exploration
src/probe_predictive.py  # M1.1 — MC-dropout & latent-shell vs rollout error (predictive calibration)
src/plan_uncertainty.py  # M1.2 — uncertainty-aware CEM (+ plan_diagnose.py action-scale confound check)
src/active_sense.py      # M1.3 — temporal active sensing (when to look); Colab GPU. Also the importable rig for M1.4.
src/schedules.py         # M1.3 — pure look-scheduling policies (no torch/swm)
src/surprise_head.py     # M1.4 — learned (z,a)->surprise head + held-out corr + deploy; Colab GPU
src/surprise_head_drift.py  # M1.5 — drift-aware head (z,a,h) trained on the deploy distribution; Colab GPU
src/monitor.py           # M1.6 — uncertainty as a runtime monitor (selective prediction); Colab GPU
tests/test_schedules.py  # local unit tests for the scheduling logic — python tests/test_schedules.py
docs/M1.4-surprise-head-spec.md  # M1.4 design spec
docs/M1.5-drift-aware-spec.md    # M1.5 design spec
docs/M1.6-monitor-spec.md        # M1.6 design spec
docs/note-actionable-uncertainty.md  # technical note: the M1.2-1.5 negative arc
```

## Status

M0 scripts are **written-for-Colab and untested** (the sims don't run on the Windows box they were authored on) — expect to debug the env install + checkpoint conversion on the first Colab run. The OOD probe and the loader are validated locally against the pretrained checkpoint.

**M1.3 (active sensing)** is **written-for-Colab and untested on GPU** — same caveat (swm doesn't run on Windows). Its pure scheduling logic is validated locally: `python tests/test_schedules.py` (5 tests, budget-matching + threshold rule). Expect to debug the encode/predict loop on the first GPU run. *(M1.3 ran: MC-dropout variance ≈ random for scheduling; oracle proves headroom — see Results.)*

**M1.4 (learned surprise head)** is **written-for-Colab and untested on GPU**. `src/active_sense.py` was refactored so its rig helpers import cleanly (M1.3 still runs identically under `__main__`); `src/surprise_head.py` imports them. Schedule tests stay green. Run after M1.3: `python src/surprise_head.py`. *(M1.4 ran: surprise predictable on true latents (+0.38) but PARTIAL on deploy — train/deploy drift gap. See Results.)*

**M1.5 (drift-aware head)** is **written-for-Colab and untested on GPU**. `src/surprise_head_drift.py` reuses the rig; trains on the deploy distribution (random-look drift pairs). Run: `python src/surprise_head_drift.py`. *(M1.5 ran: decisive NULL — head < h-alone; uniform spacing unbeatable. Closes the negative arc.)*

**M1.6 (runtime monitor)** is **written-for-Colab and untested on GPU** — the *positive-complement* experiment. `src/monitor.py` reuses the rig; selective prediction (risk-coverage/AURC) on in-dist + corrupted transitions. Run: `python src/monitor.py`.
