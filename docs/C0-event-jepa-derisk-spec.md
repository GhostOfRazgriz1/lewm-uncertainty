# C0 — De-risk gate: Event-JEPA ("causal event bottleneck for small world models")

**Status:** RUN → **NO-GO** (3 seeds). `src/c0_event_jepa_derisk.py`.

**Premise under test:** small world models fail because their latent *transitions* are too dense; a sparse
causal **event bottleneck** on the transition (`z,a → e → z'`) should (1) discover the events unsupervised
and (2) predict long-horizon better than a dense predictor. Test the falsifiable core on a self-contained
toy (numpy env + torch MLPs, CPU, minutes) before building the planner / benchmark / multi-env matrix.

- **GATE 1 (events emerge):** discrete event code `e=q(z,a,z')` aligns with the toy's TRUE event labels
  (pickup/drop/switch) it was never told. Metric = NMI / purity. Pass = NMI > 0.30.
- **GATE 2 (bottleneck helps):** capacity-matched event-bottleneck predictor beats a DENSE predictor at
  H=10/20/50 rollout (prior `p(e|z,a)` supplies `e` at rollout, future unavailable). Pass = robust
  (median) long-horizon win.

## What it took to make it a FAIR test (the iteration is the value)
1. First cut: events too rare (~2%, random actions) → bottleneck has nothing to encode. **Fix:** event-
   seeking data policy → events ~10–15%.
2. First architecture routed *every* transition through one categorical pick from K codes — a strawman
   that can't represent continuous motion (rollout diverged). **Fix:** faithful design = continuous base
   `B(z,a)` + **sparse additive** event correction `E(z,a,e)` (L1 so it fires only on events).
3. Tested in the **small-capacity regime** (the thesis's own claim: small models).
4. A mean-based GATE-2 showed a huge "win" (dense H50 mean 203 vs BN 0.74) — **an artifact of a few
   diverged dense rollouts.** Reporting the **median** reversed it. Added a codebook-usage anti-collapse
   term. Ran **3 seeds**.

## Result (3 seeds) — NO-GO
| seed | NMI (want >0.30) | H50 median BN/dense |
|---|---|---|
| 1 | 0.23 | 0.91 |
| 2 | 0.09 | 1.99 |
| 3 | 0.12 | 1.29 |

- **GATE 1 FAIL:** only the most distinctive events (drop, switch) get clean dedicated codes; pickup and
  passive transitions are not separated. NMI 0.09–0.23, never near 0.30.
- **GATE 2 FAIL:** on the typical (median) rollout the bottleneck is ≈ or *worse* than a capacity-matched
  dense predictor at long horizon — i.e. it does **not** help where the thesis says it should. Its only
  edge (bounding catastrophic divergence) is itself seed-dependent (seed 3 the BN diverged instead).

Same signature as the rest of the program: a single-metric/mean positive that **rigorous evaluation
(median + seeds) overturns** (cf. M2-Tier2 single-seed flip, mover-only-metric, M1.2 confound).

## Honest caveats (fairness to the proposal)
- This is a **low-dim, deterministic** toy. The proposal's strongest case is **high-dim pixels**, where
  small dense predictors genuinely fail — a regime this toy can't create cheaply. So a fair *full* test
  needs the LeWM-pixel substrate.
- BUT: the premise couldn't show a stable pulse even on a toy **designed to be favorable** (events present,
  sparse, learnable; small-model regime). The burden shifts: why would it work on pixels when it fails on
  the clean version? That is now a **big-build-on-faith** bet, not a cheap-de-risk-then-commit.

## Verdict / next
- **NO-GO on the cheap toy.** Do NOT build the planner+benchmark+multi-env on this premise yet.
- Options: (a) pixel de-risk on frozen LeWM (the only fair test of the real regime — but it is the
  big-build commitment the toy argues against); (b) drop Event-JEPA, bank the rigorous-analysis program
  (monitor/controller dividing line + geometry-dependence of free uncertainty + "structure doesn't help
  small WMs") which is the thread that has actually held up; (c) reconsider the program's framing — three
  constructive directions (LeWM-control, TD-MPC2-monitor, Event-JEPA) have now failed the cheap de-risk,
  while every rigorous-analysis result has held.
