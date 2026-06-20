# Paper framing: Calibrated uncertainty in JEPA world models

*Adopted after a reviewer-style read of our verified results + LeWM / HAUWM / WIMLE. The framing matches our
data; do not claim "uncertainty improves JEPA control in general" (our results falsify that on full-data
in-distribution tasks).*

## Title
- Safe: **Calibrated Uncertainty in JEPA World Models: Separating Latent Support from Predictive Risk**
- Ambitious (only if off-support/DPP gives a real planning win): **When Should a JEPA World Model Trust Its
  Imagination? Calibrated Latent Uncertainty for Robust Long-Horizon Planning**

## Thesis
JEPA world models expose **two geometrically distinct uncertainty facets** — *latent support* uncertainty
from the SIGReg Gaussian shell, and *predictive* uncertainty from calibrated action-free ensembles. Together
they are a reliable **monitor** for long-horizon rollout risk and OOD robustness. They become useful for
**control only when planning is model-risk-limited** (off-support / limited-data) — not on full-data,
in-distribution, near-deterministic tasks.

## Three claims (each with our verified evidence)
1. **JEPA uncertainty is two-faceted.** Shell deviation `|‖z‖−√d|` (support/OOD) is ~orthogonal to predictive
   error (Pearson **+0.05**); shell OOD AUROC ~1.0; ensemble predictive within-horizon Spearman **+0.58**.
   OOD-uncertainty and transition-uncertainty are different axes, not one scalar.
2. **Calibration improves long-horizon latent world modeling.** Our realized-error NLL objective: scale-calib
   `se/var` **6.95 → 2.44**, long-horizon fidelity **+4.7% @k=12** (seeded). Emphasize *scale* calibration —
   ranking (Spearman) saturates while variance scale stays wrong. Refinement: `u = u_ens + u_ale + ε`
   (add a per-member aleatoric head) so total predictive uncertainty captures action-free multimodality too.
3. **Uncertainty is actionable as model-trust, not as generic control cost.** Negatives are the strength:
   5 control nulls (β·variance CEM, gating, sensing M1.2–1.5, A2, 3b) on full-data ID tasks. Positives:
   selective prediction (M1.6), combined OOD/predictive monitoring (M2.2), shell-gated perception (A1).

## Novelty (what to claim)
**Realized-error-calibrated uncertainty for SIGReg-JEPA latent spaces.** Distinction from HAUWM:
- HAUWM: `L_HCU ∼ −k·Var_m[μ_m]` — *forces* disagreement to grow with horizon.
- Ours: `L_cal = e_{t,k}/u_{t,k} + log u_{t,k}` with `e = (1/d)‖z_{t+k} − μ̄‖²` — variance *matches realized
  latent error*; horizon-growth emerges only when error actually grows.
We verified HAUWM's force-grow is **harmful in a JEPA** (kills sharpness; M2 Tier 1), motivating ours.

## Mapping to the references
- **LeWM** — the JEPA geometry: end-to-end pixel JEPA, next-embedding loss + SIGReg (isotropic Gaussian via
  random projections). This is *why* the shell `|‖z‖−√d|` is a geometry-induced support/OOD statistic, not a
  heuristic. LeWM uses latent CEM and notes long-horizon planning is a limitation.
- **HAUWM** — closest competitor; not JEPA-specific; force-grow-with-horizon (we improve on it, §Novelty).
- **WIMLE** — the controller-side lesson: use uncertainty to *down-weight unreliable model rollouts* (trust
  signal), not as an action penalty. Aligns with our monitor findings and the trust-boundary controller below.

## Controller method (avoid β·uncertainty cost — we have a clean null there)
Use uncertainty as a **trust boundary**, not a competing reward:
- adaptive-horizon MPC: `H* = max{h : Σ_{i≤h} u_i < τ}` (trust the rollout only as far as it stays reliable);
- or constrained planning: maximize reward **subject to** `Σ_h u_h < τ`.
DPP (`src/r_pessimism.py`) currently uses the **support** facet as a soft penalty (the disagreement facet is
~0 under action-conditioning — see below). If support-pessimism shows signal off-support, reimplement as the
adaptive-horizon trust boundary (cleaner; sidesteps the negative-reward sign issue).

## Boundary-condition experiment map (the decisive structure)
| regime | expected | interpretation |
|---|---|---|
| full-data ID planning | uncertainty-control ≈ vanilla | no model-risk bottleneck (our 5 nulls) |
| observation corruption / OOD perception | shell-gated trust **wins** | support uncertainty is actionable (A1) |
| long-horizon rollout eval | calibrated ensemble **wins** | predictive uncertainty actionable as monitor (M2.1/M2.2) |
| limited-data / off-support planning | **FALSIFIED** — support-pessimism = no effect (5-seed paired, gradient N∈{25,45,70}, all within 2 SEM) | uncertainty does NOT become controller-relevant even off-support |

**Verdict: the ambitious "controller off-support" claim is falsified; thesis lands firmly at MONITOR, NOT
CONTROLLER (safe title).** DPP support-pessimism (the canonical offline-RL method) had a 3-seed apparent
crossover (+4.97 @N=25) that **reversed under 5-seed paired testing** (−3.63 @N=25; curve [−3.6,−1.3,+0.1]) —
the 4th low-seed inflation caught. Six control failures total; this last one is the strongest (principled
method, favorable regime, proper stats). Claim 3 is now *strongly* evidenced: uncertainty is a monitor, and
its inability to help control holds even where theory predicts it should.

## Key mechanistic finding (strengthens Claim 3)
**Action-conditioned ensembles have ~0 epistemic disagreement.** M2.1's sharp ensemble uncertainty was
*action-free* multimodality; conditioning on the action (as planning must) collapses it (DPP-disagreement
runs gave disagreement ≈ 0). So the actionable controller signal is the **support** facet, not ensemble
disagreement — which is exactly why offline-RL-style support-pessimism (not a predictive-variance penalty) is
the right controller-side mechanism.
