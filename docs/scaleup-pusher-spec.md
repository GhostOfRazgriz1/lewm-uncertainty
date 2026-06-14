# Scale-up de-risk: minimal PUSHER — stress the reachability pillar the toy trivializes

## Why this substrate
The toy (`EventEnv`) makes the clean C3→C5→C7 story work because **causing an event = navigate a point mass
to a coordinate that's in the state**. That triviality is exactly the part that is hard in Push-T / MuJoCo.
This substrate changes **one thing and only one thing**: reachability becomes structured *pushing* instead of
point-mass navigation. Everything else stays identical (from-state, deterministic, low-dim, CPU, seedable,
oracle available) so any change in the result is attributable to reachability hardness, not a confound.

It is deliberately the **essence of Push-T** (push a block to a goal) minus the parts that aren't about
reachability (pixels, the T's orientation, momentum, stochasticity) — those are *later* rungs.

## Env design (`PushEnv`, ~minimal, deterministic, from state)
- **State (4-dim):** `[agent_x, agent_y, block_x, block_y]` in [0,1]². Goal `g` is a fixed constant (e.g. (0.85,0.5)).
- **Action (2-dim):** `(dx, dy)`, clipped to ±STEP (≈0.05).
- **Push dynamics (no momentum):**
  ```
  agent ← clip(agent + action, box)
  d = block - agent;  if ||d|| < contact_r:  block ← agent + contact_r * d/||d||   # shoved ahead of agent
  ```
  i.e. walking into the block pushes it in the agent→block direction. **To move the block toward the goal you
  must approach from the side *opposite* the goal and push.** Approaching from the goal side pushes it *away*.
  That is the whole point: "go to the block" (the toy expert) is no longer sufficient.
- **Events (EMERGENT from dynamics, not flag-on-proximity; labels for eval only):**
  - `CONTACT` (1): agent crosses from not-touching to touching the block. *Navigation-reachable (toy-like).*
  - `MOVED` (2): block displaced > thresh this step. *Requires contact + a push.*
  - `DELIVERED` (3): block enters goal zone (||block − g|| < zone_r), first time. **Reachable only by structured
    pushing — the hard event.**
- **Task:** deliver the block to the goal zone. Natural subgoal sequence: `CONTACT → DELIVERED`.
  CONTACT is toy-easy (approach block); DELIVERED replaces the toy's easy "drop-navigation" with the hard
  push skill. This isolates reachability hardness on exactly one subgoal.

## What this tests — and what it deliberately does NOT
- **Tests:** does the discovery → affordance → inverse-model → DAgger pipeline survive when reachability is
  *structured control* (pushing from the correct side) rather than navigation?
- **Does NOT test (later rungs):** pixels (C2 already flags discovery degradation), the T's orientation
  (a second hard axis), momentum/stochasticity, partial observability. One hard thing at a time.

## Experiment ladder (reuse the C1/C3/C5/C7 pipeline verbatim)
1. **E1 — discovery (C1 analog):** do `CONTACT/MOVED/DELIVERED` emerge as distinct codes unsupervised
   (event-recall + lift)? *Harder than the toy: events are dynamics-emergent, not state flags.*
2. **E2 — the gap (C3 analog):** descriptive event-CEM vs dense-CEM vs **oracle-pusher**. Predict: descriptive
   ≈ dense (the code says "delivered" not how to push), oracle succeeds. If the C3 gap reproduces here, the
   phenomenon isn't a navigation artifact.
3. **E3 — actionable (C5 analog):** learn affordance head + event inverse model `π(a|z,e)`; does the learned
   controller close the gap to the oracle when the skill is *pushing*? **This is the real test** — the toy's
   π only had to learn 2D navigation; here it must learn the push skill (approach-behind + shove).
4. **E4 — robustness (C7 analog):** DAgger closed-loop. Does it robustify here too?

## The expert question (the honest crux)
DAgger and the oracle need an expert. In the toy it was **data-derivable** ("toward the object's coords,
which are in the state"). Here the fair expert is a **scripted pusher**:
```
behind = block + contact_r * unit(block − g)        # far side of the block from the goal
return toward(behind) if agent not at behind else toward(g)   # get into pushing position, then push
```
This is more hand-coded than the toy expert. That escalation is itself the finding: **as reachability gets
harder, the "expert" stops being free and starts being a skill.** So E3/E4 answer "given a competent expert,
does the learning pipeline work on hard reachability?" — and they surface the *next* open problem: **where does
the expert come from** in domains where you can't script it (the real Push-T / MuJoCo question). Two outcomes,
both useful:
- pipeline works with the scripted pusher → learning machinery generalizes; open problem = expert acquisition
  (RL? the WM itself? a few demos?).
- pipeline fails even with a good expert (π can't represent pushing, or discovery fails) → deeper limit, and
  we learn the toy was hiding it.

## Pre-registered failure criteria (so we can't rationalize)
- E1 fails if event-recall ≲ 0.6 / events don't get distinct codes → discovery doesn't survive emergent events.
- E2 fails to reproduce the gap if descriptive-CEM ≈ oracle (then the toy's gap *was* a navigation artifact).
- **E3 is the decisive one:** if the learned π can't approach the oracle on DELIVERED (the push subgoal), the
  toy's C5 win *was* mostly trivial reachability. Report the CONTACT subgoal (easy) and DELIVERED subgoal
  (hard) **separately** — expect a split.
- E4: report median + failure-rate separately (as in C7).

## Scope guards
From state, deterministic, disk block (no orientation), fixed goal, CPU, ≥3 seeds per claim, self-contained
(numpy env + the existing `_event_common` model/training code). Oracle + scripted expert included for the gap
and DAgger. ~the same compute envelope as C3/C5/C7.

## Next
- [ ] implement `PushEnv` + E1 (discovery) — cheapest rung, tells us if discovery survives emergent events.
- [ ] on E1 pass → E2 (does the gap reproduce), then E3 (the decisive reachability test), then E4 (DAgger).
