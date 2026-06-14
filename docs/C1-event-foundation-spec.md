# C1 — Event-JEPA foundation test: intervention-discovery + event-planning efficiency

**Status:** RUN (3 seeds) → events emerge (robust) but **intervention and event-planning are both NULL**.
`src/c1_event_intervention_planning.py`. Extends C0's NO-GO by testing the two load-bearing claims C0
never touched.

## Why C1
C0 was a NO-GO but only tested the weakest legs. It also used **NMI**, which is dominated by the 90%
passive "none" class and *undersold event discovery*. C1 (a) re-measures discovery with the right metric
(per-event recall to a dedicated code + enrichment lift), (b) tests the **intervention/counterfactual**
loss (the proposal's "strongest part"), (c) tests **event-biased planning** sample-efficiency on a sparse
pickup→drop task. Self-contained (numpy env + torch, CPU). `env_step` is pure → counterfactuals are free.

## CLAIM A — intervention → event discovery
The robust finding (3 seeds): **events emerge as distinct dedicated codes WITHOUT intervention.**
| metric | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| event-recall on-policy | 0.94 | 0.90 | 0.90 |
| event-recall +counterfactual | 1.00 | 0.84 | 0.83 |

- **Events emerge (ROBUST):** on-policy recall ~0.90 all seeds; pickup/drop/switch each get a distinct
  code, enrichment-lift 2–18×. This **revises C0's "events don't emerge"** — that was an NMI (none-
  dominated) artifact, not a real failure.
- **Intervention loss is a NULL:** counterfactuals helped once (seed 0 → 1.00), hurt twice (→0.84, 0.83);
  net ≈ zero-to-negative. The proposal's "strongest part" does not deliver on discovery.
- Seed 0 alone looked like a PASS — **seeds caught the false positive** (the program's signature; cf.
  M2-Tier2, mover-only, C0-mean).

## CLAIM B — event-planning sample efficiency (sparse pickup→drop)
| planner | seed0 (64/128/256) | seed1 | seed2 |
|---|---|---|---|
| dense | .10/.15/.20 | .25/.20/.25 | .20/.20/.15 |
| event-BN | .35/.30/.15 | .15/.20/.15 | .05/.05/.10 |
| event-biased | .00/.05/.00 | .25/.20/.10 | .05/.10/.10 |

- **Robustly NULL.** No planner reliably solves the task (max 0.35, mostly ~0.20). No consistent winner;
  event-biasing is at best tied, often worst. seed-0's "event-BN best at low N" did **not** replicate.
- Caveat: the task is **too hard for these tiny models** (even dense fails), so this is inconclusive-to-
  negative rather than a perfectly clean refutation. My event-biasing (penalize event-free plans) is also
  cruder than the proposal's full event-level planner (inverse model + plan-over-event-codes).

## Synthesis C0 + C1 (the honest verdict)
**Discovery works; utility doesn't.** Events emerge as interpretable codes (robust), but event structure
does **not** help prediction (C0: median long-horizon tie/worse), **not** help planning (C1-B: null), and
the intervention loss does **not** help discovery (C1-A: null). This is exactly the failure mode the
user's own plan warned against: *"avoid only showing pretty event visualizations — you must connect event
discovery to control improvement."* The connection isn't there on a favorable toy.

Three constructive directions (LeWM-control, TD-MPC2-monitor, Event-JEPA) have now failed the cheap de-risk;
the consistent winner across the program is **rigorous analysis + the representation/monitoring
observations + the evaluation rigor that overturns single-metric/single-seed positives.**

## Recommendation
**Bank the analysis paper.** Fold Event-JEPA in as: "small WMs *can* discover sparse causal event codes
unsupervised, but the codes do not (yet) improve prediction or planning, and the intervention loss is a
null — discovery ≠ utility." If the absolute-cleanest planning refutation is wanted, a proper event-level
planner on a discriminable task is the one remaining fair test — but C0+C1+seeds make a flip unlikely.
