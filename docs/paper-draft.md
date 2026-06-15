# Predictive Events Are Not Plannable Events

*Draft skeleton. Method/equation/algorithm detail lives in `actionable-events-technical-report.md`; this file
locks the framing, the narrative, and the section-by-section content.*

## Abstract

We show across **symbolic, structured-contact, and real-physics** substrates that learned predictive world
models and descriptive event codes fail to support planning through sparse contact modes. The missing
variable is **actionability**: event-conditioned controllers trained on *reachable* modes match oracle
execution even when model-based CEM fails completely. We further isolate *why* model-based control fails into
two independent causes — an inaccurate learned contact model and an intractable shooting search — each
demonstrated by a separate experiment, and we frame the problem as two layers, **mode acquisition** (reach the
precondition region) and **mode execution** (realize the contact event once in it). Our method solves
execution; we identify acquisition as the next hard subproblem.

**Thesis.** *Small learned world models can describe contact dynamics but fail to plan through them;
actionable event controllers can realize reachable contact modes even when model-based search fails.*

## 1. Introduction

Recent small world models (JEPA / value-equivalent) are *predictive but not plannable*: a latent that
predicts the future well does not plan well. We localize this for **events**. An event code can be made to
describe *what happened* (pickup, contact, delivered) far more easily than to tell a planner *how to make it
happen*. We call this gap **descriptive ≠ actionable**, and show it is the same obstacle as predictive ≠
plannable and latent-L2 ≠ reachability.

**Two layers.** Causing an event decomposes into (L1) **mode acquisition** — reaching the precondition region
from which the event is reachable (e.g., getting behind a block); and (L2) **mode execution** — realizing the
event once near that region (the contact push itself). Naive model-based planning must solve both, by shooting
through an imperfect model — and fails. We scope this paper to **L2: reachable event execution**, and name L1
(acquisition) as the next problem.

**Contributions.**
1. A controlled diagnosis that **descriptive event abstraction is insufficient for control** (the gap), with
   an oracle that quantifies the headroom.
2. **Actionable event controllers** (affordance + event-conditioned inverse model, robustified by DAgger) that
   **close the gap to oracle execution**, where descriptive codes and model-based planning do not.
3. A separation of *why model-based fails* into **two independent causes** (model inaccuracy; search
   hardness), each demonstrated by its own experiment.
4. A **three-level evidence ladder** showing the result is neither a toy-dynamics nor a hand-coded-dynamics
   artifact: it persists under real 2D contact physics.

## 2. Related work (pointers)

- *Predictive ≠ plannable* in JEPA / value-equivalent world models; latent-distance ≠ reachability.
- Options / affordances / skills and hierarchical control (the L1/L2 decomposition's lineage).
- Imitation under distribution shift (DAgger) — our robustification of L2 execution.
- Event-/object-centric and bottlenecked world models — the descriptive-code baseline we show is insufficient.

## 3. From descriptive to actionable events

- **Descriptive event code:** `e = what transition occurred`, discovered unsupervised by a sparse-additive
  event bottleneck (`z,a → e → z'`). Good at labeling *types*; for goal-relational events (delivered) it is
  weak (captured by *context*, not transition type).
- **Actionable event abstraction:** not just `e`, but **reachability** `p(reachable(e)|z)` and an
  event-conditioned controller `π(a|z,e)` — *which states an event is reachable from, and what actions cause
  it.* This is the missing variable.
- **The L1/L2 split** (Fig 1): acquisition vs execution; this paper solves L2.

## 4. Method (summary; full detail in the technical report)

Event-bottleneck (descriptive baseline), affordance head + event inverse model (actionable), and DAgger
(closed-loop imitation that fixes BC distribution-shift). Planners: dense-CEM (model-based), descriptive
event-CEM, the oracle/expert (true-sim MPC or competent scripted controller), and the learned controller.

## 5. Substrates: a three-level evidence ladder

| level | substrate | role |
|---|---|---|
| 1 | **EventEnv** (symbolic GridWorld) | clean causal diagnosis: events exist, are discoverable, but descriptive ≠ plannable; DAgger closes it |
| 2 | **PushEnv** (structured contact, hand-coded) | the gap reproduces under structured pushing; model-free reachability transfers; model-based refinement hurts |
| 3 | **PushPhysEnv** (real pymunk contact/friction/rotation) | same dense-model failure + model-free success on a real physics engine |

## 6. Results

**The gap (L1 EventEnv).** descriptive event-CEM ≈ dense ≈ 0.13 vs oracle-subgoal ≈ 0.85 (~7×). Event codes
know *what*, not *how*.

**Closing it (EventEnv).** learned affordance + inverse model reaches the oracle in 6/8 seeds but is brittle;
the failure is **BC distribution-shift** (three candidate mechanisms tested and refuted); **DAgger** robustifies
it (20 seeds: min 0.90, mean 0.99, 0 failures).

**Structured contact (PushEnv).** dense-CEM 0.00, oracle 0.90, learned-DAgger 0.60 → ~0.80–0.88 with more
coverage. Model-based *refinement* hurts (CEM-around-π 0.05–0.20): the residual gap is bottlenecked on
dynamics-model contact accuracy.

**Real physics (PushPhysEnv), 3 seeds.** dense-CEM **0.00**, scripted-oracle 0.90–0.95, learned-BC/DAgger
0.85–0.95. Model-free reachability matches the expert; model-based fails — on a real engine.

**Why model-based fails — two independent causes.**
1. *Model inaccuracy on contact (L2):* dense-CEM = 0.00 in the **behind-start** regime (acquisition given,
   pure execution) — the learned model can't predict the contact mode well enough to plan the push.
2. *Search hardness (L1+L2):* full-task **true-sim MPC = 0%** — a perfect model, but random shooting can't
   discover navigate-then-push.
Model-free actionability sidesteps both: no model to be wrong, imitation instead of search.

## 7. Scope and limitations (kept prominent — this is what makes the claim credible)

- We do **not** claim to solve general Push-T. We isolate the **contact execution** subproblem in the
  **behind-start** regime.
- **Mode acquisition (L1)** — go-around / pre-contact positioning — is the next hard subproblem: naive true-sim
  MPC 0%, scripted-with-go-around 25%. This is the *expert-acquisition wall*; causing the event is itself the
  hard control problem.
- **DAgger is regime-dependent:** load-bearing on EventEnv (drift) but not in real-physics behind-start (BC
  stays on-distribution). We report this honestly.
- **Discovery weakens on physical events** (goal-relational, not transition-type-distinct).
- **Untested:** pixels (a learned latent degraded event-recall to 0.39), the T-block **pose** (orientation),
  partial observability, stochastic contact.

## 8. Conclusion

Once the relevant event is reachable, **model-free actionability learns to execute it reliably — even where
model-based search fails completely** — across symbolic, structured-contact, and real-physics substrates. The
open frontier is **mode acquisition**: learning to reach the precondition region (the L1 affordance/precondition
policy), which is where the hard control problem now lives.

## Figures

1. Descriptive vs actionable event abstraction; the L1/L2 decomposition.
2. Event codes align with hidden events (discovery), and degrade on goal-relational events.
3. The gap: dense / descriptive-event fail, oracle succeeds (EventEnv).
4. Closing the gap + robustness: BC brittle → DAgger (EventEnv 20-seed min/mean).
5. **The ladder** (`docs/fig_ladder.png`, from `src/fig_ladder.py`): dense-CEM vs model-free vs oracle across
   EventEnv / PushEnv / PushPhysEnv — model-based collapses 0.13→0.00→0.00 as contact realism rises, model-free
   tracks the oracle. *This is the headline figure.*
6. Why model-based fails: model-inaccuracy (behind-start dense-CEM = 0) vs search (true-sim MPC = 0).
