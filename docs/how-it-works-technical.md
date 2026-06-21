# Calibrated Uncertainty in JEPA World Models — Complete Technical Writeup

*Full mechanism, with math. Substrate: LeWorldModel (LeWM, arXiv:2603.19312) on `swm/PushT-v1`, and a
from-scratch JEPA world model on gymnasium-MuJoCo (Reacher/Pusher). Notation is defined as it appears.*

---

## 0. Notation

- `o_t` observation (pixels at time t), `s_t` underlying env state, `a_t ∈ A` action.
- `E_θ : o ↦ z ∈ ℝ^d` the JEPA **encoder** (ViT), latent dim `d = 192` for LeWM.
- `z_t = E_θ(o_t)` the latent.
- `f_φ` the **action-conditioned predictor**: `ẑ_{t+1} = f_φ(z_t, a_t)`; k-step rollout `f_φ^{(k)}(z_t, a_{t:t+k})`.
- `r̂_ψ(z)` a learned reward head (for the control work).
- `π_β(a|s)` the **behavior policy** that generated the offline data `D = {(o_t, a_t, r_t)}`.

---

## 1. The world model and its geometry (why a "shell" exists)

LeWM is a **Joint-Embedding Predictive Architecture (JEPA)**: it is trained *only* in latent space, with a
next-embedding prediction loss plus a distributional regularizer **SIGReg** (Sketched Isotropic Gaussian
Regularizer). SIGReg pushes the marginal latent distribution toward an isotropic Gaussian `P_Z → N(0, I_d)`
by taking random 1-D projections `⟨z, v⟩, v ~ Unif(S^{d-1})` and applying an Epps–Pulley empirical-
characteristic-function normality test to each. By the Cramér–Wold theorem, all 1-D projections being N(0,1)
⟺ `P_Z = N(0, I_d)`.

**Consequence (the shell).** For `z ~ N(0, I_d)` in high dimension, the **Gaussian annulus theorem** gives
`‖z‖ ≈ √d` with fluctuations `O(1)`: the mass concentrates on a thin spherical shell of radius `√d ≈ 13.86`.
So in-distribution latents satisfy `‖z‖ ≈ √d`, and the **shell deviation**

```
s(z) = | ‖z‖ − √d |
```

is a *geometry-induced* out-of-distribution / support statistic — not a heuristic. (This shell framing is
ours; SIGReg/identifiability theory provides the Gaussian-latent guarantee it rests on.)

**Planning.** Control is done by CEM-MPC in latent space: sample action sequences, roll them out with `f_φ`,
score by terminal latent distance to a goal latent (or by `Σ r̂_ψ`), keep the elite, refit, execute the first
action, repeat.

---

## 2. The two uncertainty facets

### 2.1 Predictive uncertainty (an ensemble)
Train `M` predictors `{f_{φ_i}}_{i=1}^M`. Ensemble mean and disagreement:

```
μ̄(z,a) = (1/M) Σ_i f_{φ_i}(z,a),   u(z,a) = (1/M) Σ_i ‖ f_{φ_i}(z,a) − μ̄(z,a) ‖²  (total),  or  ū = u/d (per-dim).
```

MC-dropout variance is a weaker proxy: it correlates with rollout error (Pearson +0.41 on real PushT
transitions) but is **flat across horizon** and per-instance under-sharp.

### 2.2 Support/epistemic uncertainty (the shell)
`s(z) = |‖z‖ − √d|` from §1.

### 2.3 They are orthogonal facets
On real transitions, `corr(s(z), predictive error) ≈ +0.05`. The shell carries *no* predictive-error
information and vice-versa — they are **different axes** ("is this input familiar?" vs "will this transition
be hard to predict?"), not one scalar. This orthogonality is the structural fact the paper rests on.

---

## 3. The calibration objective (does the model know how wrong it is?)

### 3.1 Definitions
For a k-step rollout, ensemble mean `μ̄_{t,k}`, scalar predictive variance `u_{t,k}` (mean per-dim ensemble
variance), realized per-dim error

```
e_{t,k} = (1/d) ‖ z_{t+k} − μ̄_{t,k} ‖².
```

### 3.2 The objective (realized-error Gaussian NLL on the SIGReg latent)
Treat the ensemble as an isotropic Gaussian predictive distribution and minimize its negative log-likelihood:

```
L_cal  =  (1/2) [ e_{t,k} / u_{t,k}  +  log u_{t,k} ]      (constant dropped).
```

Stationarity in `u` gives the optimum `u* = e`: **the predictive variance is driven to match the realized
latent error.** This is principled precisely because SIGReg makes the latent Gaussian.

### 3.3 Contrast with HAUWM (the key novelty)
HAUWM (OpenReview pZuZWRuPyi) uses `L_HCU ∝ − k · Var_i[μ_i]`, which *forces* disagreement to grow with the
horizon `k`. Ours lets horizon-growth **emerge only if error actually grows**:

```
HAUWM:   u_{t,k} ∝ k            (imposed)
Ours:    u_{t,k} → E[ e_{t,k} ] (learned to match realized error)
```

We ran HAUWM's `L_HCU` in a JEPA: at `λ=1` it diverges (the reward `−k·Var` is unbounded; fixed with
`log1p` + grad-clip), and stabilized it gets growth `+1.00` *by construction* while **killing per-instance
sharpness** (within-horizon Spearman `+0.04`, negative at long `k`). So forcing growth is harmful here; the
plain ensemble is already horizon-calibrated and the NLL improves only what's actually miscalibrated (scale).

### 3.4 Results (3 seeds, k up to 12)
- **Scale calibration.** Define `ρ = E[e] / E[u]` (1.0 = calibrated). The plain ensemble is **overconfident**
  `ρ = 6.95` (disagreement ~7× too small for the error); the objective pulls it to `ρ = 2.44`.
- **Long-horizon fidelity.** k=12 rollout error improves **+4.7% (6.6 SEM)**, with a ~4% short-horizon
  trade-off (crossover ~k=7).
- **Ablations.** Confidence-weighting (WIMLE-style `w = 1/(1+βu)`) = null; **β-NLL** (Seitzer'22, weight NLL
  by `stopgrad(u^β)`) is *dominated* — it attenuates the calibration pressure that drives both gains.
- **Measurement subtlety.** *Rank* calibration (Spearman of `u` vs `e`) is saturated for the plain ensemble
  (~0.59) and cannot see the fix; the win is in *scale* (`ρ`). Report scale, not rank.

---

## 4. The monitor result (uncertainty as a warning light)

**Selective prediction / risk–coverage.** Rank transitions by a signal, keep the most-confident coverage
fraction `c`, measure mean error on the kept set `R(c)`; summarize by `AURC = mean_c R(c)` (lower = better).

- In-distribution: **MC-variance AURC 1.82** vs random 2.57 (oracle 1.31) — abstains from hard transitions;
  shell is useless here (2.68 ≈ random; orthogonal, as §2.3 predicts).
- On distribution shift (corrupted inputs): **shell AURC 4.57** vs random 8.36; **combined** (z-scored sum)
  **4.22** → oracle 3.79. Each facet is a working monitor for its *own* failure mode and blind to the other.
- Ensemble monitor (within-horizon, confound-free): recovers **77%** of the random→oracle gap vs MC-dropout's
  **11%** in-distribution; OOD-blind on shift (the shell carries that).
- **Shell-gated perception (A1).** Under intermittent observation corruption, gating on `s(z) < τ` (trust the
  encoding) vs coasting (predict forward) recovers near-oracle latent-tracking error: shell-gate beats blind
  6/6 settings (2–7×) and a matched-rate random gate 6/6 (4–25 SEM); detection recall 1.0.

**Takeaway:** as a *monitor* (knowing when to distrust the model), the uncertainty works.

---

## 5. The controller failures, and the mechanism behind them

Six attempts to use uncertainty to *improve a decision* failed (each with a metric and a SEM):
- uncertainty-penalized CEM cost (`cost = dist + β·Var`): Δ −6.4 within ±22 SEM (null).
- when-to-look scheduling (MC-variance): ≈ random; a non-causal oracle shows ~30% headroom the signal can't see.
- learned/drift-aware surprise heads: predictable on *true* latents (+0.38) but < elapsed-time-alone (+0.43) at deploy.
- gated control under iid corruption: estimate-quality decouples from control-quality (MPC self-corrects).
- factor-space planning (plan in a decoded-pose metric): ≈ random; probe R²=0.62 → localizes the gap to the
  *predictor dynamics*, not the representation or the metric.

### 5.1 The mechanism: action-conditioning collapses the predictive uncertainty
The sharp ensemble signal is **action-free**: `p(z_{t+k} | z_t)` is genuinely multimodal (the future is
underdetermined without the action), so `Var_i f_i(z_t)` is large and informative (within-horizon Spearman
+0.58). But planning evaluates **action-conditioned** rollouts `p(z_{t+k} | z_t, a_{t:t+k})`, which are
near-deterministic — conditioning on the action pins down the next latent, so

```
Var_i f_i(z_t, a_t)  ≈ 0      (measured 0.000 – 0.02 on Pusher).
```

The rich predictive uncertainty is *aleatoric multimodality* you lose the instant you commit to an action.
**That is why predictive-uncertainty-based control fails: the signal vanishes exactly when you plan.**

---

## 6. The pivot: support pessimism (offline-RL style)

If predictive uncertainty is empty under action-conditioning, use the *other* facet: **support**. Model-based
offline RL (MOPO/MOReL) penalizes leaving the data support, where the model extrapolates unreliably:

```
J(a_{0:H}) = (reward / goal term)  −  κ · Σ_t  U(ẑ_t, a_t),     U = support risk.
```

- **State-shell** support `U = s(z)`: tested (DPP). **Null** — penalizing state-shell does not help control.
- The realization: offline risk lives in unsupported **(z, a) pairs**, not states. Need a **state-action**
  support model.

### 6.1 The state-action support model (density ratio)
Train a classifier `g(z,a)` to separate real pairs from action-shuffled negatives. The Bayes-optimal logit is
the **pointwise mutual information**:

```
g*(z,a) = log [ p_D(z,a) / ( p_D(z) p_D(a) ) ]  =  PMI(z; a).
```

The support risk is `U_joint(z,a) = − g(z,a)` (large where the pair is off the joint support).

---

## 7. The identifiability theorem (why this is *untestable* under random data)

**Claim.** If the behavior policy is action-randomized, the state-action support model is unidentifiable: the
optimal classifier is constant and its AUROC is exactly 0.5.

**Proof.** A random policy takes `a_t` independent of `s_t`. The encoder is a (deterministic) function
`z_t = E_θ(o_t)`, a function of `s_t`, so `a_t ⊥ z_t`. Therefore

```
p_D(z,a) = p_D(z) · p_D(a)   ⟹   g*(z,a) = log 1 = 0   ⟹   AUROC = 0.5.
```

Equivalently: the shuffled-negative classifier is a mutual-information estimator (its AUROC is a monotone
function of `I(z; a)`), and `a ⊥ z ⟹ I(z;a) = 0`. The "unsupported action for this state" concept **does not
exist** in randomly-collected data. ∎

**The escape hatch.** A *structured* (state-dependent) behavior policy makes `a` depend on `s` hence on `z`,
so `I(z;a) > 0` and the support model becomes identifiable.

**Empirical confirmation (Gate 1).** Held-out density-ratio AUROC:
```
random policy     : 0.503 ± 0.006   (= 0.5, theorem confirmed to the decimal)
structured policy : 0.845 ± 0.005   (≫ 0.7, identifiable)
```
This reframes the earlier `(z,a)`-pessimism "failure": it was not falsified, it was **unidentifiable** in the
data regime we had.

---

## 8. The two gates (preconditions before any control claim)

To avoid mistaking noise for signal (we did, repeatedly — §10), formalize two checks; **both must pass**
before a control experiment counts.

**Gate 1 — identifiability.** `AUROC(g) > 0.7` against *meaningful* negatives (shuffled here; planner-proposed
in the strengthened version). → §7: structured 0.845, random 0.503. PASS for structured.

**Gate 2 — relevance.** The support score must predict *where the model errs*:

```
corr( U_joint(z,a) , e_model(z,a) ) > 0,    e_model(z,a) = ‖ f_φ(z,a) − E_θ(o') ‖,  o' = true next obs.
```

The off-support ground truth `o'` for actions the data never took requires the *real* next state, obtained by
resetting the simulator (record `(qpos, qvel)`, `set_state`, step `a`, render, encode). **Validity check:**
the no-step `set_state` round-trip `‖E_θ(reset render) − z_t‖` must be small (a v1 cross-episode reset gave
‖·‖≈13 ⟹ INVALID and was correctly rejected; the v2 inline same-episode reset gives ‖·‖≈0.26 ⟹ valid).

**Result (Gate 2, valid):** `corr(U, e_model) = +0.212 ± 0.018`, all 3 seeds positive; off-support actions
have *both* higher `U` and higher `e_model`. PASS.

Both gates pass for structured data → the preconditions for support-pessimism hold (first time across the
whole investigation).

---

## 9. The control experiment (the payoff)

**Data.** Structured-offline (state-dependent behavior policy `a = clip(W·standardize(obs) + 0.5·ξ)`, `W`
fixed-random per seed, `ξ ~ N(0,I)`) — makes `(z,a)` support identifiable (Gate 1) and relevant (Gate 2).
Train `E_θ, f_φ, r̂_ψ` (encoder + predictor + reward) and the support classifier `g` on it.

**Planner.** Vanilla vs support-pessimistic CEM, scored across the `S` candidates with z-scored terms so `κ`
is scale-free and the negative-cost reward ranks correctly:

```
score(a_{0:H}) = zscore_S( Σ_t r̂_ψ(ẑ_t) )  −  κ · zscore_S( Σ_t U_joint(ẑ_t, a_t) ),     κ = 3.
```

**Protocol.** Paired per-seed (vanilla and pessimistic share the per-seed world model), 10 seeds, plus a
random-action floor (competence guard) and inline Gate-1 AUROC.

**Result (10 seeds, Pusher).**
```
random floor      : −82.8 ± 1.5
vanilla CEM       : −69.6 ± 1.7        (competence: +13.1 over random, +5.7 SEM → CONTROLS)
support-pess CEM  : −65.3 ± 1.7
PAIRED Δ          : +4.33 ± 1.17  (+3.7 SEM),   9/10 seeds positive,   classifier AUROC 0.95
```

Support-pessimism recovers ≈ +33% of the control margin over vanilla. The pessimism keeps CEM on the reliable
data manifold; vanilla exploits off-manifold predictor errors (where `e_model` is high, §8) and is misled.

**Why this is a real effect, not the 6th false positive.** From 5→10 seeds the **mean held while the SEM
shrank**: `+4.64 (±2.83) → +4.33 (±1.17)` — the signature of a true effect (the estimator concentrating). Every
earlier "positive" did the opposite, the mean collapsing toward 0 under more seeds (§10). 9/10 positive is
itself significant (`P(≥9/10 | no effect) ≈ 0.011`).

---

## 10. Verification methodology (the part that did the most work)

- **Paired statistics.** Vanilla and treatment share the per-seed world model, so the correct test is the
  per-seed difference `Δ_s`, with `SEM = std(Δ_s)/√n`. An *unpaired* SEM (treating arms as independent) is
  inflated by across-seed spread in absolute returns and masks real effects — it flipped one verdict from
  "positive" to "null" until corrected.
- **Five inflations caught** (all rejected before becoming claims):
  1. Tier-2 pose probe: R² ≈ 0 from an unregularized Adam probe overfitting 192→3; ridge → +0.53.
  2. Calibration fidelity: single-run +6.2% → seeded +2.0%.
  3. Gated control (Reacher): seed-0 +14 → 3-seed null.
  4. DPP crossover: 3-seed +4.97 (+2.2 SEM) → 5-seed null (sign flipped).
  5. `(z,a)`-pessimism: 5-seed +9.2 (+2.2 SEM) → 8-seed null, *and* AUROC-gated INVALID (classifier 0.56).
- **The rule that emerged:** a finding is real only if it gets *sharper* under more seeds (mean stable, SEM
  shrinking), passes both gates, and is paired. The §9 win is the only control result that satisfies all three.

---

## 11. The unified thesis and the boundary condition

> **A JEPA world model's uncertainty is a monitor everywhere, and a controller exactly when there is a
> learnable, error-predictive support boundary in the latent state–action occupancy.**

The full causal chain, each link verified:

```
Theorem (random data ⟹ I(z;a)=0 ⟹ AUROC 0.5)
   └─ Gate 1: structured data ⟹ (z,a) support identifiable (AUROC 0.85)
        └─ Gate 2: support is relevant to model error (corr +0.21, valid reset)
             └─ Control: support-pessimistic planning improves return (+4.3, +3.7 SEM, 9/10 seeds)
```

The six in-distribution / random-data control nulls are not a blanket negative — they are the **boundary**
(regimes where support is unidentifiable, irrelevant, or where control isn't model-bottlenecked). The
structured-offline result is where the boundary is crossed and uncertainty becomes controller-relevant.

**Contribution stack.**
- **C1.** Two orthogonal JEPA uncertainty facets (shell support `s(z)`; ensemble predictive `u(z,a)`), `corr ≈ 0.05`.
- **C2.** A realized-error calibration objective `L_cal = ½(e/u + log u)` that improves *scale* calibration
  (`ρ: 6.95→2.44`) and long-horizon fidelity (+4.7%), distinct from — and empirically better than — HAUWM's
  force-grow `−k·Var`.
- **C3.** The *when* of control: the action-conditioning collapse `Var_i f_i(z,a)≈0`; the identifiability
  theorem; the two-gate protocol; the control nulls as boundary; and the structured-offline support-pessimism
  win as the positive.
- **Methodology.** Paired multi-seed testing + the two gates; five caught inflations.

---

## 12. Scope and limitations

- **Controlled demonstration.** From-scratch JEPA-WM on Pusher pixels with a synthetic (fixed-linear)
  structured behavior policy — not a benchmark-topping offline-RL system. The control effect is **real but
  modest** (≈ +33% of the margin, +3.7 SEM).
- **Relation to prior work.** The control mechanism is offline-RL pessimism (MOPO/MOReL); the novelty is
  doing it in a JEPA *latent* state-action occupancy, the calibration objective vs HAUWM, the two-facet
  separation, and the explicit identifiability/relevance gating that says *when* it can work at all.
- **Strengthening roadmap.** A natural (competent-but-suboptimal mixture) behavior policy with
  planner-proposed negatives; the SIGReg **shell** as the support signal on LeWM (the JEPA-specific, free
  version of `g`); a second substrate; and value-calibrated uncertainty (calibrate `u` to decision-relevant
  error `|d(z,z_g) − d(ẑ,z_g)|` rather than latent MSE).
