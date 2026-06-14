# B1 — E1: a free uncertainty MONITOR for a competent agent under deployment shift

**Status:** designed (gate B0 = GO). Script `src/b1_monitor.py` to follow.

**Claim:** the free Q-ensemble disagreement signal off a frozen, competent TD-MPC2 agent **detects
when the agent is operating on bad information under deployment shift, and flags the timesteps where
its predictions are wrong** — label-free, no retraining, and ≈ a supervised detector. This is the
M1.6/M2.2 monitor transplanted from LeWM to a *credible, competent* substrate (kills the MNIST /
broken-planner worry). It is the "method that wins" half of the paper; E2 (control) is the boundary.

This is **monitoring only** — detection + selective prediction. No control intervention here (that's
E2). Keeping them separate is deliberate: it isolates "the signal knows" from "acting on it helps."

---

## What B0 already established (don't re-litigate)

- Competence: cheetah-run checkpoint 853 vs published 850, random floor 7.4.
- Signal is alive in-dist: Q-ensemble disagreement CV 11.8, Spearman **+0.55** with realized one-step
  latent error — free, no head trained.

## The two caveats B1 must defuse

1. **Autocorrelation confound (M1.4 lesson):** part of +0.55 is "both signals track fast-changing
   states." B1 must show disagreement predicts error **beyond** a trivial latent-drift baseline
   (`||z_t − z_{t-1}||`). Report the partial/relative Spearman vs drift, not just the raw number.
2. **In-dist ≠ shift:** B0 measured clean rollouts. B1's claim is about **shift**. Test it directly.

---

## Setup

Frozen competent agent (`cheetah-run-1.pt`, model_size=5). Deploy it under observation corruption and
read the free monitor signal online at every step. **The signal needs only `(z_t, a_t)`** — Q-ensemble
disagreement = `var_k two_hot_inv(Q_k(z_t,a_t))` — so it is computable at deploy with **no clean
reference and no future obs**. That is the whole point: a deployable "I don't know."

### Shift models (deploy-time corruption of the state obs)
- **S1 — Gaussian noise**, σ ∈ {0.1, 0.3, 0.6}·(obs std). Detectable, severity-graded.
- **S2 — sustained blackout bursts**: obs frozen/zeroed for K∈{5,15,30} consecutive steps (carry A2's
  lesson — iid corruption self-corrects via replanning; *sustained* is what binds, and it pre-stages E2).
- **S3 — subtle shift** (defuses the A1 "corruption is obvious" reviewer point): small constant sensor
  bias / slow drift on a subset of obs dims, calibrated so detection is *not* trivial.

### Reference for "is the agent actually wrong here?"
Run a **paired clean rollout** (same seed, same action sequence the corrupted agent took) to get the
clean latent trajectory. Per-step error target = `||z_corrupt_t − z_clean_t||` (estimate divergence)
and the k-step predicted-vs-clean error. This is the honest target the monitor is trying to rank.

---

## Signals compared (all read off the frozen model)
| signal | cost | role |
|---|---|---|
| **Q-ensemble disagreement** | free (num_q=5) | the proposed monitor |
| latent-drift `||z_t − z_{t-1}||` | free | trivial baseline — must be beaten (caveat 1) |
| MC-dropout variance | free (cfg.dropout) | predictive baseline (was flat on LeWM — does it stay flat?) |
| trained detector (supervised) | small MLP | **the reviewer-proofing**: free disagreement should ≈ supervised |
| oracle (rank by true error) / random | — | ceiling / floor |

The supervised detector: a small MLP on `(z_t, a_t, disag, drift)` trained on **labeled** clean-vs-shift
(or on the realized error). If free disagreement ≈ supervised AUROC, the headline is "you get a
supervised-quality monitor for free." If supervised >> free, that's an honest limit to report.

---

## Metrics
1. **Shift detection AUROC** — does disagreement separate clean vs corrupted timesteps? Per shift type
   and severity. (Expect high for S1/S2; S3 is the honest stress test.)
2. **Selective prediction / within-horizon AURC** (M2.2 style, confound-free): rank timesteps by the
   signal; risk = error on the kept (high-confidence) fraction. Report % of random→oracle gap recovered.
3. **Beyond-drift check** (caveat 1): Spearman(disag, error) controlling for drift — partial correlation
   or Δ(AURC) of (disag) vs (drift alone) vs (disag+drift). Disag must add over drift.
4. **Free vs supervised** (caveat / reviewer-proofing): AUROC(disag) vs AUROC(trained detector).

## Verdict logic
- **WIN** = disag detection AUROC ≫ 0.5 on S1/S2 **and** disag AURC recovers a large share of the
  oracle gap **and** disag beats drift-alone **and** disag ≈ supervised (within a small margin).
- **PARTIAL** = detects obvious shift (S1/S2) but not subtle (S3), or doesn't beat drift on the easy
  regime → report honestly; still a usable monitor for gross corruption.
- **WEAK** = disag ≈ drift everywhere (signal is just motion) or ≪ supervised → the free signal is not
  the story; fall back to the trained head and reframe.

---

## Scope guards (Colab-tractable)
- One task (cheetah-run) for the main result; add walker-walk + one harder task (dog-run) only for the
  breadth table once cheetah-run is locked.
- Reuse B0's rollout+signal-dump loop and the M2.2 AURC helpers verbatim.
- No control here. No training of the agent. Only the tiny supervised detector trains.

## Next
- [ ] `src/b1_monitor.py` (mechanical from this spec).
- [ ] On WIN → `B2` spec = E2 (uncertainty-cost MPC / trust-gating under sustained outage — the swing).
