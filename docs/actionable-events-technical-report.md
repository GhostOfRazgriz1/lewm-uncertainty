# Actionable Event World Models — Technical Report

> **Rendering note.** Display equations use GitHub fenced ` ```math ` blocks; inline math uses unicode and
> code spans (no `$…$`). The ` ```math ` blocks render as math on GitHub and any MathJax/KaTeX viewer, and
> degrade to readable LaTeX (not broken text) in a plain editor preview without a math extension. **View this
> file on GitHub for the rendered equations** (VS Code's built-in preview needs a math extension).

**Working title of the contribution:** *Predictive Events Are Not Plannable Events: from descriptive to
actionable event abstractions in small world models.*

**One-sentence thesis.** A small world model can discover *what events occur* far more easily than it can
*plan to cause them*; closing that gap requires grounding events in **reachability/affordance** (which states
an event is reachable from, and with what actions), not in transition compression — and once grounded, a
**model-free** controller robustified by **DAgger** closes the gap to a hand-coded oracle even on contact
dynamics, whereas **model-based** planning/refinement fails because the learned dynamics are not accurate
enough on contact.

This report documents the full program: a symbolic GridWorld (`EventEnv`, experiments C0–C7) and a
contact-pushing world (`PushEnv`, experiments P1–P5). All code is self-contained (numpy envs + small PyTorch
MLPs, CPU), seeded, in `src/`.

---

## 1. Motivation: the reachability gap

Across a prior program on JEPA / value-equivalent world models the same phenomenon recurred under different
names: **predictive ≠ plannable** (a latent that predicts well plans poorly), **latent-L2 ≠ reachability**
(Euclidean latent distance ranks the wrong actions even when task factors are linearly decodable), and here
**descriptive ≠ actionable** (an event code that says *what happened* does not say *how to cause it*). These
are one law: **world-model representations encode what *is* / *happened*, not what can be *reached* /
*controlled*.** This report isolates that law for *events* and then closes it.

We use point-mass abstractions with **known ground-truth event labels** so that (a) discovery can be scored
against truth and (b) a scripted **oracle** controller exists to upper-bound planning and serve as the DAgger
expert.

---

## 2. Substrates

### 2.1 Symbolic GridWorld — `EventEnv`

State `s = (pᵃ, pᵒ, c, w) ∈ [0,1]² × [0,1]² × {0,1} × {0,1}` = (agent pos, object pos, carrying, switch),
`dim = 6`. Action `a ∈ ℝ²`, clipped to `‖a‖∞ ≤ η` with `η = 0.06`. Deterministic; fixed zones
`z_drop, z_sw`; radii `r_pick = 0.12`, `r_zone = 0.15`. Agent update:

```math
p^a_{t+1} = \mathrm{clip}(p^a_t + a_t,\ 0,\ 1)
```

Event `eₜ ∈ {∅, pickup, drop, switch}` and the flag updates it triggers:

```math
e_t = \begin{cases}
\text{pickup}  & c_t = 0 \ \wedge\ \lVert p^a_{t+1}-p^o_t\rVert < r_{\text{pick}} \ \Rightarrow\ c_{t+1}=1 \\
\text{drop}    & c_t = 1 \ \wedge\ \lVert p^a_{t+1}-z_{\text{drop}}\rVert < r_{\text{zone}} \ \Rightarrow\ c_{t+1}=0 \\
\text{switch}  & w_t = 0 \ \wedge\ \lVert p^a_{t+1}-z_{\text{sw}}\rVert < r_{\text{zone}} \ \Rightarrow\ w_{t+1}=1 \\
\varnothing    & \text{otherwise}
\end{cases}
```

If carrying, the object tracks the agent: `c_{t+1}=1 ⇒ pᵒ_{t+1} = pᵃ_{t+1}`.

**Key property (the toy's vulnerability):** causing any event = *navigate the agent to a coordinate that is in
the state*. Reachability is trivial point-mass navigation.

### 2.2 Contact pusher — `PushEnv`

State `s = (pᵃ, pᵇ) ∈ [0,1]⁴`; fixed goal `g`; contact radius `r_c = 0.10`, zone radius `r_z = 0.12`, move
threshold `δ = 0.008`. The agent shoves the block ahead of it (no momentum):

```math
p^a_{t+1} = \mathrm{clip}(p^a_t + a_t), \qquad
p^b_{t+1} = \begin{cases}
\mathrm{clip}\!\left(p^a_{t+1} + r_c\,\dfrac{p^b_t - p^a_{t+1}}{\lVert p^b_t - p^a_{t+1}\rVert}\right) & \lVert p^b_t - p^a_{t+1}\rVert < r_c \\
p^b_t & \text{otherwise}
\end{cases}
```

Emergent events: **contact** (touching, small displacement), **moved** (`‖pᵇ_{t+1}−pᵇₜ‖ > δ`), **delivered**
(`‖pᵇ_{t+1} − g‖ < r_z`, first time).

**Key property (the stress):** to push the block *toward* `g` the agent must reach the **far side**
`p_behind = pᵇ + r_c · (pᵇ − g)/‖pᵇ − g‖` and push toward `g`. "Go to the block" pushes it the *wrong* way.
Reachability is now **structured contact control**, and the task event *delivered* is **goal-relational** (a
push that *ends in the zone*), not a transition type.

---

## 3. Models and training objectives

Work from state, `z = s`, with one-step delta `Δzₜ = s_{t+1} − sₜ`. All nets are 2-hidden-layer GELU MLPs.

### 3.1 Dense dynamics predictor

```math
\widehat{\Delta z} = D_\theta(z,a), \qquad
\mathcal{L}_{\text{dyn}} = \mathbb{E}\,\big\lVert D_\theta(z,a) - \Delta z \big\rVert_2^2
```

### 3.2 Event-bottleneck predictor `EventBN` (the *descriptive* model)

A continuous **base** carries smooth dynamics; a discrete **event code** adds a *sparse additive* correction —
the faithful form of "`z,a → e → z'`". (A categorical-only `h(z,a,e)` was tried first and fails on continuous
motion.) Components: posterior `q_ψ(z,a,z') → ℓ_post ∈ ℝᴷ`, prior `p_φ(z,a) → ℓ_prior`, base `B_θ(z,a)`,
effect `E_χ(z,a,e)`. Code via **Gumbel-Softmax (straight-through, hard)** with `uᵢ ~ Unif(0,1)`,
`gᵢ = −log(−log uᵢ)`:

```math
\mathrm{GS}_\tau(\ell)_k = \frac{\exp((\ell_k + g_k)/\tau)}{\sum_j \exp((\ell_j + g_j)/\tau)},
\qquad e = \mathrm{onehot}\big(\arg\max_k \mathrm{GS}_\tau(\ell)_k\big)\ \text{(ST gradient)}
```

with temperature annealing `τ = max(0.5, 1 − epoch/epochs)`. The prediction is

```math
\widehat{\Delta z} = B_\theta(z,a) + E_\chi(z,a,e), \qquad \Delta z_{\text{ev}} \equiv E_\chi(z,a,e)
```

**Loss.** With batch code-usage `p̄ = (1/B) Σ_b softmax(ℓ_post,b)`:

```math
\mathcal{L}_{\text{BN}} =
\underbrace{\mathbb{E}\lVert \widehat{\Delta z} - \Delta z \rVert_2^2}_{\text{prediction}}
+ \underbrace{\mathrm{CE}\big(\ell_{\text{prior}},\ \overline{\arg\max\,\ell_{\text{post}}}\big)}_{\text{prior matches posterior code}}
+ \underbrace{\lambda_s\,\mathbb{E}\lVert \Delta z_{\text{ev}}\rVert_1}_{\text{sparse: fire only on events}}
+ \underbrace{\lambda_u \sum_k \bar p_k \log \bar p_k}_{=\,-\lambda_u H(\bar p)\ \text{(anti-collapse)}}
```

with `λ_s = 0.01`, `λ_u = 0.05` (overbar = stop-gradient). The CE term lets the **prior** supply `e` at
rollout, where the future `z'` is unavailable:

```math
e_{\text{roll}} = \mathrm{onehot}\big(\arg\max p_\phi(z,a)\big), \qquad
\widehat{\Delta z} = B(z,a) + E(z,a,e_{\text{roll}})
```

### 3.3 Affordance head (reachability)

Multi-label "which events are reachable within `K` steps", `reach_e(sₜ) = 1[∃ j ≤ K : e_{t+j} = e]`:

```math
\mathcal{L}_{\text{aff}} = \sum_e \mathrm{BCE}\big(g_\theta(z)_e,\ \mathrm{reach}_e(z)\big)
```

### 3.4 Event inverse model / controller (the *actionable* component)

`π_θ(z,e) → a` (toy) or `π_θ(z) → a` for a fixed target event (pusher), trained by behavior cloning on an
expert `a★ = E(z,e)`:

```math
\mathcal{L}_{\text{BC}} = \mathbb{E}_{(z,e)}\big\lVert \pi_\theta(z,e) - \mathcal{E}(z,e) \big\rVert_2^2
```

**Expert / oracle** — a scripted skill toward the event-trigger condition (also the planning upper bound and
the DAgger relabeler). Toy pickup: `E(s, pickup) = η · (pᵒ − pᵃ)/‖pᵒ − pᵃ‖` (object pos is in the state).
Pusher deliver (behind-then-push):

```math
\mathcal{E}(s,\text{deliver}) = \eta\,\frac{t - p^a}{\lVert t - p^a\rVert}, \qquad
t = \begin{cases} p_{\text{behind}} & \lVert p^a - p_{\text{behind}}\rVert > 0.6\,r_c \\ g & \text{otherwise} \end{cases}
```

**Action-relevant BC filter** (C6 / pusher): keep a pair `(zₜ, e, aₜ)` only if the action moved the agent
toward the location `L` where the event fired, `‖pᵃₜ + aₜ − L‖ < ‖pᵃₜ − L‖` — drops the 50%-random data noise.

---

## 4. Evaluation metrics

**Normalized mutual information** between codes `C` and true events `E` (arithmetic normalization):

```math
\mathrm{NMI} = \frac{I(C;E)}{\tfrac12\big(H(C)+H(E)\big)}, \quad
I = \sum_{c,e} \hat p(c,e)\log\frac{\hat p(c,e)}{\hat p(c)\hat p(e)}, \quad
H(X) = -\sum_x \hat p(x)\log \hat p(x)
```

NMI is **none-dominated** when the passive class is large; the discovery-appropriate metric is per-event
recall. With dominant code `k★(e) = argmax_k N(e,k)`:

```math
\mathrm{recall}_e = \frac{N(e,k^\star)}{N(e)}, \qquad
\mathrm{lift}_e = \frac{\hat P(E=e \mid C=k^\star(e))}{\hat P(E=e)}, \qquad
\overline{\mathrm{recall}} = \frac{1}{|\mathcal{E}|}\sum_e \mathrm{recall}_e
```

plus **distinct** = all `k★(e)` are distinct. Discovery "passes" iff `recall‾ > 0.6` and distinct.

**Selective-prediction AURC** (monitor strand). Rank items by uncertainty `u`; risk at coverage `κ` = mean
error over the lowest-`u` fraction:

```math
\mathrm{AURC} = \frac{1}{|\mathcal{K}|}\sum_{\kappa\in\mathcal{K}} \frac{1}{\lceil \kappa n\rceil}\sum_{i\le \lceil \kappa n\rceil}\mathrm{err}_{(i)}, \qquad
\%\text{gap} = \frac{\mathrm{random}-\mathrm{AURC}}{\mathrm{random}-\mathrm{oracle}}
```

**Planning success** = fraction of episodes reaching the goal within `T_plan` (toy: object in drop zone;
pusher: `‖pᵇ − g‖ < r_z`). We report **median and failure-rate separately** (a single seed can collapse).

---

## 5. Planning algorithms

### 5.1 Cross-Entropy Method (CEM / MPPI-style) over a learned model

```
CEM(cost, z0, N, H, iters):
  mu    <- 0 in R^{H x |A|};   sigma <- eta * 1
  repeat iters:
     a^(i)      ~ clip(mu + sigma * Normal(0, I)),  i = 1..N    # N action sequences
     z^(i)_{h+1} = z^(i)_h + Model(z^(i)_h, a^(i)_h)            # rollout the learned model
     c^(i)       = cost(z^(i)_{0..H})
     elite       = top-10% lowest-cost a^(i)
     mu, sigma  <- mean(elite), std(elite)
  return first action of the arg-min-cost sequence
```

Cost variants (block/object position written `z^[obj]`, agent `z^[ag]`):

```math
c_{\text{goal}} = \lVert z_H^{[\text{obj}]} - g\rVert, \quad
c_{\text{reach}(t)} = \min_h \lVert z_h^{[\text{ag}]} - t\rVert, \quad
c_{\text{event}(e)} = \min\{\,h : \arg\max p_\phi(z_h,a_h) = e\,\}\ \ (\text{else } H)
```

### 5.2 Hierarchical event-level planner (C3 / C5)

High level = fixed subgoal sequence (e.g. pickup → drop). Low level for the current target event `e`:

- **descriptive** event-CEM: `CEM(c_event(e))` over the **EventBN** rollout;
- **oracle-subgoal**: `CEM(c_reach(trigger(e)))` — perfect hand-coded affordance;
- **affordance-event (learned)**: greedy `a = π(z,e)` — the learned inverse model.

Advance the subgoal when the real environment fires `e`.

### 5.3 DAgger (closed-loop imitation; fixes BC distribution-shift)

```
DAgger(expert E, init states S0):
  D  <- {(s, E(s)) : s in S0};    pi <- BC(D)
  repeat n_dagger:
     for each rollout episode:                         # roll the CURRENT pi in the real env
        s = reset()
        while not done:  D <- D + {(s, E(s))};  s = step(s, pi(s))
     pi <- BC(D)                                        # retrain on aggregated states (incl. pi's own)
  return pi
```

Rationale: BC is correct on the *training* distribution but drifts off it in closed loop; DAgger trains on the
states `π` actually visits, relabeled by the expert — the canonical fix for closed-loop drift.

### 5.4 CEM-around-π (model-corrected controller)

```
seed = rollout pi greedily H steps under the dense model
CEM with mu = seed (small sigma), cost = c_goal (or affordance),
keep the pi-seed as a candidate; return the refined first action
```

Intended to combine a good learned seed with model-based local refinement; **fails on contact** (§7.2).

---

## 6. Experiments and results

### 6.1 Toy (`EventEnv`): C0–C7

| ID | Question | Result |
|---|---|---|
| **C0** | Event bottleneck → better **long-horizon prediction** vs capacity-matched dense? | **No** (3 seeds): median H=50 ratio BN/dense = 0.91 / 1.99 / 1.29. The "win" on the *mean* was a few diverged dense rollouts; the **median** reverses it. |
| **C1** | Do events **emerge unsupervised**? Does a counterfactual **intervention loss** help? | Events emerge: recall‾ ≈ 0.90 (3 seeds), distinct, lift 2–18×. NMI (0.09–0.23) under-reads it (none-dominated). The **intervention loss is null** (helped 1/3 seeds, hurt 2/3). |
| **C3** | Can you **plan** with descriptive event codes? | **The gap.** descriptive event-CEM ≈ dense ≈ 0.13; **oracle-subgoal** (perfect affordances) ≈ 0.85 (~7×). Codes know *what*, not *how*. |
| **C5** | Does a **learned affordance + inverse model** close the gap? | **Yes, but brittle.** reaches oracle in **6/8** seeds, collapses (0.00) in **2/8** (greedy BC instability). |
| **C6** | Do standard robustness fixes work? What is the failure? | Filtering / ensemble / CEM-around-π **don't** remove the ~20% collapse. Three mechanisms **tested & refuted** (policy uses the condition; cosines toward both targets high on collapse seeds) ⇒ **BC distribution-shift**. |
| **C7** | Does **DAgger** robustify it? | **Yes.** 20 seeds: ≥ BC-only on every seed; **min 0.90, mean 0.99**, 0 failures (vs BC min 0.60, mean 0.935). |

### 6.2 Pusher (`PushEnv`): P1–P5 — stressing reachability

| ID | Question | Result |
|---|---|---|
| **P1 (E1)** | Does **discovery** survive *emergent* events? | **Weak** (3 seeds): recall‾ 0.44–0.58, **not distinct**. contact+moved are one phenomenon; **delivered** is *goal-relational* (own code in 2/3 seeds, lift 2.8–3.8×). Transition compression captures *types*, not goal-relation. |
| **P3 (E3)** | When causing an event is **structured pushing**, does learned reachability close the gap? | **Gap reproduces, sharper:** dense-CEM = 0.00 (can't plan a contact-push from a sparse cost), oracle ≈ 0.90, learned-BC ≈ 0.1, **learned-DAgger = 0.60** (3 seeds) — a real push skill, not navigation. |
| **P4 (E3b)** | Does **CEM-around-π** close the residual 0.60 → 0.90? | **No — it hurts** (0.05–0.20). The dense model is **too inaccurate on contact** to rank perturbations (same reason dense-CEM = 0). **Localizes the residual gap to the dynamics model**, not the controller. |
| **P5 (E3c)** | Is the 0.60 cap soft (coverage) or fundamental? | **Soft.** Model-free **DAgger scaling** lifts success 0.60 → ~0.80 (peak 0.88) by 6–8 iters; capacity barely matters (48 ≈ 96) ⇒ **coverage** is the lever. |

---

## 7. Findings

### 7.1 The dividing line, on two substrates

> **Descriptive event abstraction is insufficient for control; small world models need *actionable* event
> abstractions grounded in reachability.**

C3 measures the gap (descriptive ≈ dense ≪ oracle); C5/C7 close it on the toy (DAgger → near-perfect); P3/P5
show it **transfers to genuine contact reachability, model-free** (0.00 model-based → ~0.80–0.88 learned vs
0.90 oracle). The toy win is **not** a navigation artifact.

### 7.2 Model-free beats model-based on contact — and we know why

dense-CEM = 0.00 and **CEM-around-π hurts** because the learned dynamics model is not accurate enough on
*contact* to plan, or even to *rank local perturbations*. The model-free DAgger controller wins precisely
**because it sidesteps the unreliable model.** This is the "predictive ≠ plannable" obstacle, isolated to
contact-prediction accuracy.

### 7.3 Methodological throughline: robust evaluation overturns single-shot positives

Every premature positive was killed by the right statistic: **mean → median** (C0), **NMI → event-recall**
(C1 / P1), **single-seed → many-seed** (C5 / C7), and three **refuted** collapse-mechanism hypotheses (C6).
The discipline (median + failure-rate + seeds + falsifiable mechanism probes) is itself a contribution.

---

## 8. Limitations and future work

- **Discovery is the weak pillar on physical events** — pushing's task event is *goal-relational*; a
  transition-compression bottleneck only weakly captures it. Goal/affordance-conditioned discovery is needed.
- **Model-based planning is useless on contact** with the learned model — accept model-free control, or invest
  in a contact-accurate dynamics model.
- **The expert is scripted.** As reachability hardens, the DAgger/oracle expert stops being data-derivable and
  becomes a skill. **Expert acquisition** (RL, a few demos, self-supervised affordance learning) is the real
  open problem for un-scriptable / pixel domains.
- **Untested rungs:** pixels (a parallel test degraded event discovery to recall 0.39 in a learned latent),
  the block's **orientation** (real Push-T), partial observability, stochastic contact.

---

## Appendix A — default hyperparameters

| | value |
|---|---|
| action clip η | 0.06 (toy), 0.05 (pusher) |
| MLP hidden | 16–48 (toy sweeps), 48 (pusher) |
| event codes K | 6 |
| Gumbel τ | anneal max(0.5, 1 − ep/eps) |
| λ_s, λ_u | 0.01, 0.05 |
| reachability horizon K | 12 |
| CEM N, H, iters | 64–256, 12–18, 2–3 |
| DAgger iters / eps | 3 (C7), up to 8 (P5) / 60–90 |
| seeds | 3 (most), 8 (C5), 20 (C7) |

## Appendix B — file map (`src/`, repo `GhostOfRazgriz1/lewm-uncertainty`)

| file | role |
|---|---|
| `_event_common.py` | toy env, `EventBN`, `train_bn`, NMI / `event_metrics`, mlp |
| `c0_event_jepa_derisk.py` | C0 long-horizon prediction (null) |
| `c1_event_intervention_planning.py` | C1 discovery + intervention + planning |
| `c5_affordance.py` | C3 gap + C5 affordance-event planner |
| `c6_robust.py` | C6 robustness fixes + mechanism probes |
| `c7_dagger.py` | C7 DAgger robustification |
| `_push_common.py` | `PushEnv`, scripted expert / oracle |
| `p1_push_discovery.py` | P1 discovery on emergent events |
| `p3_push_reachability.py` | P3 dense / oracle / BC / DAgger reachability |
| `p4_push_correct.py` | P4 CEM-around-π (model correction) |
| `p5_push_modelfree.py` | P5 model-free DAgger cap curve |

Specs: `docs/event-jepa-followups-and-pivot.md` (toy arc + pivot), `docs/scaleup-pusher-spec.md` (pusher arc).
