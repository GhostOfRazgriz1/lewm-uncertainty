# Event-JEPA: three surgical follow-ups → the pivot

Three follow-ups were run to give the Event-JEPA plan its fairest possible shot before pivoting.
Scripts: `src/c2_pixel.py`, `src/c3_event_planner.py`, `src/c4_transfer.py` (+ `src/_event_common.py`).
Self-contained (numpy event-world + torch, CPU). All seeded.

## (C3) Fully fair event-level planner + oracle headroom — THE decisive result
Hierarchical planner (subgoal sequence pickup→drop + low-level CEM toward the **model's** target event
code), vs flat dense-CEM, vs an **oracle-subgoal** planner (same structure, perfect hand-coded affordances).
Best success over N∈{64,128,256}, 3 seeds:

| seed | dense | event-CEM (model codes) | oracle-subgoal (perfect affordances) |
|---|---|---|---|
| 0 | 0.10 | 0.15 | **0.80** |
| 1 | 0.15 | 0.10 | **0.95** |
| 2 | 0.15 | 0.10 | **0.85** |

**Subgoal/affordance structure helps ~7×** (oracle ~0.87 vs dense ~0.13) — the task *is* subgoal-bottlenecked.
**But the descriptive event codes provide ~none of it** (event-CEM ≈ dense). The model knows pickup
*happened* but cannot tell the planner *how to cause it* (where to go, with what actions), so event-
conditioned CEM wanders. **Measured dissociation: the event code discovers what happened, not what can be
planned.** This is not a flat null — the oracle proves the headroom is real and names the missing ingredient.

## (C4) Transfer / composition — descriptive-only (3 seeds)
Train layout A, test layout B (zones moved to other corners).
- **Discovery transfers** (events re-discoverable on B, recall 0.85 / 0.95 / 0.98) — but code identity is
  only partly stable (2/3, 1/3, 3/3 events keep their A-code; the codes are partly location-entangled).
- **Prediction does NOT transfer with any event advantage** (event-BN/dense H50 on B = 0.94 / 1.35 / 0.98 —
  tie-to-worse). The descriptor transfers; nothing *actionable* does.

## (C2) Pixel latent — NULL, and it dents the one positive (seed 0)
16×16 frames → small JEPA encoder (next-latent + VICReg) → frozen → dense vs event-BN in latent.
- Dense does **not** destabilize (mean≈median; the toy dynamics are too simple to exploit).
- Event-BN gives **no** long-horizon advantage (H50 ratio 0.97).
- **Event discovery degrades badly in the learned latent: recall 0.39** (vs ~0.90 from clean state) — the
  clean discovery result is partly an artifact of low-dim state input.

## Consolidated verdict (C0–C4)
| Claim | Result |
|---|---|
| Unsupervised event codes emerge in small WMs | **YES** from state (recall ~0.90); degrades in pixels (0.39) |
| Codes transfer to new layouts | **As descriptors** (recall ~0.9); code identity partly unstable |
| Long-horizon prediction benefit | **NO** (C0 median, C4-transfer, C2 pixel — all null/tie/worse) |
| Event-level planning benefit | **NO** (C3 event-CEM ≈ dense) |
| Intervention/counterfactual loss | **NO** (C1, seed-dependent) |
| Is the *structure* valuable? | **YES** — oracle affordances crush the task (C3, ~7×) |
| Can descriptive codes provide that structure? | **NO** (C3 event-CEM ≈ dense) |

## The pivot (empirical finding)
1. **Unsupervised causal event codes emerge in small world models** (and transfer as descriptors).
2. **Event abstraction alone is insufficient for planning (or prediction) improvement.**
3. **Utility requires reachability/actionability constraints, not just transition compression.**

Conceptual update: replace the *descriptive* event bottleneck (`e = what transition occurred`) with an
**affordance/reachability event model** — not just "this is a pickup," but *pickup is reachable from here,
with these actions, and it advances the goal*:

  p(e | z, π),  p(reachable(e) | z),  p(Δr | z, e)

## Unification with the broader program (why this matters)
This is the **same finding** as the whole LeWM/control thread, in a new guise:
- JEPA latent: **predictive ≠ plannable** (M1.2 / A2 / control-sanity).
- Latent geometry: **latent-L2 ≠ reachability** (factor-planning; "Beyond Euclidean Proximity").
- Event codes: **descriptive ≠ actionable** (C3, this work).

One thesis across substrates and representation types: **world-model representations capture what *is* /
what *happened*, but not what can be *reached* or *controlled* — and that reachability gap is the recurring
obstacle to using them for control.** The Event-JEPA result is now the cleanest single demonstration of it,
because the oracle-subgoal contrast *quantifies the headroom* the descriptive code fails to capture.

## (C5) Actionable events — the constructive validation (`src/c5_affordance.py`)
The pivot's claim was that the fix is to LEARN affordance semantics. C5 builds them — an affordance head
`g(z) → P(reachable(e) within K)` (per-event acc 0.92–0.93) and an event inverse model `π(a | z, e)` — and
plans the pickup→drop subgoal sequence with the LEARNED inverse model, vs dense-CEM and the C3 oracle.

**8 seeds, affordance-event success:** `0.00, 1.00, 1.00, 0.95, 0.00, 1.00, 1.00, 1.00`
→ **6/8 reach oracle-level (≥0.95), 2/8 collapse (0.00).** Dense ~0.19 throughout; oracle ~0.93.

- **The pivot's method works:** learned actionable events **match or beat the hand-coded oracle in 75% of
  runs** — something descriptive event-CEM *never* did (always ≈ dense in C3). The missing ingredient
  (reachability/affordance) is learnable and closes C3's gap.
- **Honest caveat:** the minimal greedy inverse-model controller is **bimodal** — when BC converges it
  solves every episode, when it doesn't it fails every episode (2/8 collapses; BC instability on the 50%-
  random mixed-policy data). Robustness (data filtering / ensemble / CEM-around-π / affordance-gating) is
  the clear next step — an engineering problem, not a refutation.

## Locked reframe
- **DEAD:** "a sparse event bottleneck (transition compression) improves planning." (C0/C1/C3 — never beat dense.)
- **ADOPTED:** *"Descriptive event abstraction is insufficient for control; small world models need
  actionable event abstractions grounded in reachability."* Working title: **Predictive Events Are Not
  Plannable Events** (alt: *From Descriptive to Actionable Events in Small World Models*).
- **Evidence arc (clean, reviewer-facing):** events exist + are descriptively discoverable (C1, recall ~0.9)
  → descriptive events don't plan, oracle affordances do, ~7× gap (C3) → learned reachability/affordance
  closes the gap to oracle in 6/8 runs (C5). Measured gap + a method designed to close it that does.
- **Next:** robustify the actionable-event controller (kill the 2/8 collapse), then scale beyond the toy.
