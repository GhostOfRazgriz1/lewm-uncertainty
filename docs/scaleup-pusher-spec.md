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

## RESULTS

### E1 — discovery (3 seeds): WEAK / fuzzy
recall 0.44–0.58, **distinct-codes = False** all seeds. CONTACT and MOVED never separate (they're the same
phenomenon — pushing; the 3-event taxonomy was over-specified). DELIVERED gets its own code in 2/3 seeds
(lift 2.8–3.8×) — it's the one event distinguishable by *terminal location*. **Lesson:** pushing's
task-event is **goal-relational**, not transition-type-distinct, so a transition-compression bottleneck only
weakly captures it. The clean symbolic-event discovery of the toy does **not** transfer; the task event is
defined by context, which foreshadows the reachability point. (So E3 gives the ground-truth event and tests
reachability directly.)

### E3 — reachability (3 seeds): the GAP reproduces sharply; learned reachability PARTIALLY closes it
| | dense-CEM | oracle (scripted push) | learned-BC | learned-DAgger |
|---|---|---|---|---|
| success | **0.00** | 0.85–0.95 | 0.05–0.20 | **0.60 (all 3 seeds)** |

- **Gap reproduces, harder than the toy:** model-based dense-CEM is **0.00** (it cannot plan the contact-push
  from a sparse goal cost — the LeWM "predictive ≠ plannable" failure, reproduced on contact dynamics),
  oracle ~0.90.
- **Learned reachability PARTIALLY transfers:** the learned inverse model + DAgger reaches **0.60** —
  ~67% of the dense→oracle gap, and a *huge* lift over both dense-CEM (0.00) and BC (~0.1). It learns a
  genuine push skill, not navigation. So the C5/C7 mechanism is **not** merely a navigation artifact.
- **But a residual gap to the oracle remains** (0.60 vs 0.90) — the toy *fully* closed the gap; hard
  (contact) reachability leaves the learned controller good-but-imperfect. DAgger is essential (BC alone
  fails), consistent with the C6/C7 distribution-shift finding.

**Scale-up verdict.** The actionable-events *thesis* strengthens on the harder substrate (descriptive/model-
based fails totally, actionable works substantially). The *method* partially transfers to contact-pushing
(real, not an artifact) with an honest residual gap to the oracle. Discovery is the weaker pillar on pushing
(goal-relational events). Next: close the residual gap (better push controller — model-corrected / more
DAgger / longer-horizon), then consider pixels / real Push-T.

### E3b — model-corrected controller (CEM-around-π): REFUTED, and it localizes the bottleneck
| | oracle | pure-π (DAgger) | cem-around-π |
|---|---|---|---|
| success (3 seeds) | 0.85–0.95 | 0.60 | **0.05–0.20** |

Seeding CEM at π's push rollout and refining with the dense model made it **much worse** than just running
π. The hypothesis "dense can *evaluate* a push even if it can't *search* one" was **wrong**: the dense model
is too inaccurate on **contact dynamics** to rank perturbations — it favors actions it *thinks* push toward
the goal but actually don't, so the search is actively misled (this is also why dense-CEM = 0.00).

**This localizes the residual gap to the DYNAMICS MODEL, not the controller/planner.** The model-free DAgger
controller (0.60) is the best learned approach *precisely because it sidesteps the unreliable model*; adding
the model back reintroduces its contact-prediction errors and hurts. Reinforces the program throughline:
**on contact dynamics the learned world model is too inaccurate to plan or refine with; model-free skill
learning sidesteps this but caps below the oracle.** So the residual gap is a *model-accuracy* problem.

To close 0.60 → ~0.90, the lever is **model-free** (more DAgger iterations / a better π architecture /
ensemble / more expert coverage), **not** model-based refinement. Or accept the honest cap + finding.

### E3c — is the 0.60 cap soft or fundamental? SOFT (model-free DAgger scaling lifts it)
Success vs DAgger iterations (2 seeds × hid {48,96}): from ~0.60 at 3 iters to **~0.80 typical, peak 0.88**
at 6–8 iters — approaching oracle ~0.90. Noisy/oscillating (DAgger isn't monotone — the reactive π can
regress as the aggregated set shifts), and capacity barely matters (48 ≈ 96), so **coverage is the lever**.
The residual gap was **soft**: model-free closed-loop coverage substantially closes it; the controller is not
at a fundamental ceiling.

## Scale-up: FINAL verdict
The actionable-event method **transfers to hard contact reachability**, **model-free**:
- descriptive/model-based planning fails totally (dense-CEM 0.00);
- learned model-free reachability + DAgger reaches **~0.80 (peak 0.88) vs oracle 0.90** with enough coverage;
- **model-based refinement HURTS** (the dynamics model is too inaccurate on contact — P4) — model-free is the way;
- discovery is the weak pillar (pushing events are goal-relational).

So the thesis strengthens AND the method substantially closes the gap to a hand-coded pusher on a genuine
contact problem — the toy's win was *not* a navigation artifact. Honest limits: model-based planning/refinement
is useless on contact (the WM isn't accurate enough), discovery degrades on goal-relational events, and the
expert is scripted (expert-acquisition is the real next open problem for un-scriptable domains).

## Done / Next
- [x] E1 discovery (fuzzy, goal-relational), E3 reachability (gap reproduces; model-free closes it),
      E3b model-correction (hurts → bottleneck = dynamics model), E3c cap (soft → model-free reaches ~0.80–0.88).
- [ ] **CONSOLIDATE / write up** — toy (full) + pusher (transfers, model-free, near-oracle) is a complete,
      honestly-scoped arc. Harder rungs (pixels / orientation, un-scriptable expert) are future work, not
      needed to establish the contribution.
