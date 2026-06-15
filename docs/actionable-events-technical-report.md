# Actionable Event World Models — Technical Report

**Working title of the contribution:** *Predictive Events Are Not Plannable Events: from descriptive to
actionable event abstractions in small world models.*

**One-sentence thesis.** A small world model can discover *what events occur* far more easily than it can
*plan to cause them*; closing that gap requires grounding events in **reachability/affordance** (which states
an event is reachable from, and with what actions), not in transition compression — and once grounded, a
**model-free** controller robustified by **DAgger** closes the gap to a hand-coded oracle, even on contact
dynamics, whereas **model-based** planning/refinement fails because the learned dynamics are not accurate
enough on contact.

This report documents the full experiment program: a symbolic GridWorld (`EventEnv`, experiments C0–C7) and a
contact-pushing world (`PushEnv`, experiments P1–P5). All code is self-contained (numpy envs + small PyTorch
MLPs, CPU), seeded, in `src/`; results are reproduced from the committed runs.

---

## 1. Motivation: the reachability gap

Across a prior program on JEPA/value-equivalent world models we repeatedly observed the same phenomenon under
different names:

- **predictive ≠ plannable** — a latent that predicts the future well does not plan well (LeWM/PushT control nulls);
- **latent-L2 ≠ reachability** — Euclidean latent distance ranks the wrong action sequences even when task
  factors are linearly decodable;
- and here, **descriptive ≠ actionable** — an event code that says *what happened* does not tell a planner *how
  to make it happen*.

These are one law: **world-model representations encode what *is* / what *happened*, not what can be *reached*
or *controlled*.** This report isolates that law for *events* and then closes it.

We use a **disk/point-mass abstraction with known ground-truth event labels** so that (a) discovery can be
scored against truth, and (b) a scripted **oracle** controller exists to upper-bound planning and to serve as
the DAgger expert.

---

## 2. Substrates

### 2.1 Symbolic GridWorld — `EventEnv` (src/_event_common.py)

State $s=(p^a, p^o, c, w)\in[0,1]^2\times[0,1]^2\times\{0,1\}\times\{0,1\}$ = (agent pos, object pos, carrying
flag, switch flag), $\dim=6$. Action $a\in\mathbb R^2$, clipped to $\|a\|_\infty\le\eta$ ($\eta=0.06$).

Transition (deterministic), with fixed zones $z_{\text{drop}}, z_{\text{sw}}$ and radii $r_{\text{pick}}=0.12$,
$r_{\text{zone}}=0.15$:

$$p^a_{t+1}=\mathrm{clip}(p^a_t+a_t,\,0,1).$$

Event $e_t\in\{\varnothing,\text{pickup},\text{drop},\text{switch}\}$ and flag updates:

$$
e_t=\begin{cases}
\text{pickup} & c_t=0 \wedge \|p^a_{t+1}-p^o_t\|<r_{\text{pick}}\ \Rightarrow\ c_{t+1}=1\\
\text{drop} & c_t=1 \wedge \|p^a_{t+1}-z_{\text{drop}}\|<r_{\text{zone}}\ \Rightarrow\ c_{t+1}=0\\
\text{switch} & w_t=0 \wedge \|p^a_{t+1}-z_{\text{sw}}\|<r_{\text{zone}}\ \Rightarrow\ w_{t+1}=1\\
\varnothing & \text{otherwise.}
\end{cases}
$$

If carrying, the object tracks the agent: $c_{t+1}=1\Rightarrow p^o_{t+1}=p^a_{t+1}$.

**Key property (the toy's vulnerability):** causing any event $=$ *navigate the agent to a coordinate that is in
the state* — reachability is trivial point-mass navigation.

### 2.2 Contact pusher — `PushEnv` (src/_push_common.py)

State $s=(p^a,p^b)\in[0,1]^4$ = (agent, block); goal $g$ fixed; contact radius $r_c=0.10$, zone radius
$r_z=0.12$, move threshold $\delta=0.008$. Push dynamics (no momentum): the agent shoves the block ahead of it,

$$
p^a_{t+1}=\mathrm{clip}(p^a_t+a_t),\qquad
p^b_{t+1}=\begin{cases}\mathrm{clip}\!\Big(p^a_{t+1}+r_c\,\tfrac{p^b_t-p^a_{t+1}}{\|p^b_t-p^a_{t+1}\|}\Big) & \|p^b_t-p^a_{t+1}\|<r_c\\[4pt] p^b_t & \text{otherwise.}\end{cases}
$$

Emergent events: $\text{contact}$ (touching, small displacement), $\text{moved}$ ($\|p^b_{t+1}-p^b_t\|>\delta$),
$\text{delivered}$ ($\|p^b_{t+1}-g\|<r_z$ first time).

**Key property (the stress):** to push the block *toward* $g$ the agent must position on the **far side**
$p^{\text{behind}}=p^b+r_c\,\widehat{(p^b-g)}$ and push toward $g$. "Go to the block" pushes it the wrong way.
Reachability is now **structured contact control**, and the task event "delivered" is **goal-relational** (a push
that *ends in the zone*), not a transition type.

---

## 3. Models and training objectives

Let $z=s$ (we work from state; $\Delta z_t=s_{t+1}-s_t$). All nets are 2-hidden-layer GELU MLPs.

### 3.1 Dense dynamics predictor

$$\widehat{\Delta z}=D_\theta(z,a),\qquad \mathcal L_{\text{dyn}}=\mathbb E\,\big\|D_\theta(z,a)-\Delta z\big\|_2^2.$$

### 3.2 Event-bottleneck predictor `EventBN` (the *descriptive* model)

A continuous **base** carries smooth dynamics; a discrete **event code** adds a *sparse additive* correction
(the faithful form of "$z,a\!\to\!e\!\to\!z'$"; a categorical-only $h(z,a,e)$ was tried first and fails on
continuous motion).

- posterior $q_\psi(z,a,z')\!\to\!\ell^{\text{post}}\in\mathbb R^K$, code $e=\mathrm{GS}_\tau(\ell^{\text{post}})$;
- prior $p_\phi(z,a)\!\to\!\ell^{\text{prior}}$;
- base $B_\theta(z,a)$, effect $E_\chi(z,a,e)$;
- prediction $\widehat{\Delta z}=B_\theta(z,a)+E_\chi(z,a,e)$, with $\Delta z_{\text{ev}}\equiv E_\chi(z,a,e)$.

**Gumbel-Softmax (straight-through, hard).** With $u_i\sim\mathrm{Unif}(0,1)$, $g_i=-\log(-\log u_i)$,

$$\mathrm{GS}_\tau(\ell)_k=\frac{\exp((\ell_k+g_k)/\tau)}{\sum_j\exp((\ell_j+g_j)/\tau)},\qquad e=\text{onehot}(\arg\max_k\,\mathrm{GS}_\tau(\ell)_k)\ \text{(ST gradient)},$$

with annealing $\tau=\max(0.5,\,1-\text{epoch}/\text{epochs})$.

**Loss.** With batch code-usage $\bar p=\tfrac1B\sum_b\mathrm{softmax}(\ell^{\text{post}}_b)$,

$$
\mathcal L_{\text{BN}}=\underbrace{\mathbb E\|\widehat{\Delta z}-\Delta z\|_2^2}_{\text{prediction}}
+\underbrace{\mathrm{CE}\!\big(\ell^{\text{prior}},\,\overline{\arg\max\,\ell^{\text{post}}}\big)}_{\text{prior matches posterior code}}
+\underbrace{\lambda_s\,\mathbb E\|\Delta z_{\text{ev}}\|_1}_{\text{sparse: fire only on events}}
+\underbrace{\lambda_u\!\sum_k \bar p_k\log\bar p_k}_{=-\lambda_u H(\bar p)\ \text{(anti-collapse)}}
$$

with $\lambda_s=0.01,\ \lambda_u=0.05$ (overbar = stop-gradient). The $\mathrm{CE}$ term lets the **prior** supply
$e$ at rollout (the future $z'$ is unavailable):
$\;e_{\text{roll}}=\text{onehot}(\arg\max p_\phi(z,a)),\;\widehat{\Delta z}=B(z,a)+E(z,a,e_{\text{roll}})$.

### 3.3 Affordance head (reachability)

Multi-label "which events are reachable within $K$ steps":

$$\text{reach}_e(s_t)=\mathbb 1\big[\exists\,j\le K:\ e_{t+j}=e\big],\qquad
\mathcal L_{\text{aff}}=\sum_e \mathrm{BCE}\big(g_\theta(z)_e,\ \text{reach}_e(z)\big).$$

### 3.4 Event inverse model / controller (the *actionable* component)

$\pi_\theta(z,e)\to a$ (toy) or $\pi_\theta(z)\to a$ for a fixed target event (pusher), trained by behavior
cloning on an expert $a^\star=\mathcal E(z,e)$:

$$\mathcal L_{\text{BC}}=\mathbb E_{(z,e)}\big\|\pi_\theta(z,e)-\mathcal E(z,e)\big\|_2^2.$$

**Expert / oracle.** A scripted skill toward the event-trigger condition (also used to upper-bound planning and
to relabel in DAgger):

- toy: $\mathcal E(s,\text{pickup})=\eta\,\widehat{(p^o-p^a)}$ (object pos is in the state);
- pusher (deliver): position behind the block then push,
$$\mathcal E(s,\text{deliver})=\eta\,\widehat{(t-p^a)},\quad t=\begin{cases}p^{\text{behind}} & \|p^a-p^{\text{behind}}\|>0.6\,r_c\\ g & \text{else.}\end{cases}$$

**Action-relevant BC filter** (C6/pusher): keep a pair $(z_t,e,a_t)$ only if the action moved the agent toward
the location $L$ where the event fired, $\|p^a_t+a_t-L\|<\|p^a_t-L\|$ — drops the 50%-random data noise.

---

## 4. Evaluation metrics

**Normalized mutual information** between codes $C$ and true events $E$ (arithmetic norm):

$$\mathrm{NMI}=\frac{I(C;E)}{\tfrac12\big(H(C)+H(E)\big)},\quad
I=\sum_{c,e}\hat p(c,e)\log\frac{\hat p(c,e)}{\hat p(c)\hat p(e)},\quad H(X)=-\sum_x \hat p(x)\log\hat p(x).$$

NMI is **none-dominated** when the passive class is large; the discovery-appropriate metric is per-event recall.

**Per-event recall and enrichment lift.** Dominant code $k^\star(e)=\arg\max_k N(e,k)$,

$$\text{recall}_e=\frac{N(e,k^\star)}{N(e)},\qquad
\text{lift}_e=\frac{\hat P(E=e\mid C=k^\star(e))}{\hat P(E=e)},\qquad
\overline{\text{recall}}=\tfrac1{|\mathcal E|}\sum_e\text{recall}_e,$$

plus the indicator **distinct** = all $k^\star(e)$ are distinct. (Discovery "passes" iff $\overline{\text{recall}}>0.6$ and distinct.)

**Selective-prediction AURC** (used in the monitor strand). Rank items by an uncertainty signal $u$; risk at
coverage $\kappa$ is the mean error over the lowest-$u$ fraction; $\mathrm{AURC}=\tfrac1{|\mathcal K|}\sum_{\kappa}\frac1{\lceil\kappa n\rceil}\sum_{i\le\lceil\kappa n\rceil} \text{err}_{(i)}$. Lower is better; "% gap recovered" $=\frac{\text{random}-\text{AURC}}{\text{random}-\text{oracle}}$.

**Planning success.** Fraction of episodes reaching the goal within $T_{\text{plan}}$: toy $=$ object in drop
zone; pusher $=\|p^b-g\|<r_z$. We report **median and failure-rate separately** (a single seed can collapse).

---

## 5. Planning algorithms

### 5.1 Cross-Entropy Method (CEM / MPPI-style) over a learned model

```
CEM(cost, z0, N, H, iters):
  mu <- 0 in R^{H x |A|};  sigma <- eta·1
  repeat iters:
     a^(i) ~ clip(mu + sigma·N(0,I)), i=1..N         # sample N action sequences
     z^(i)_{h+1} = z^(i)_h + Model(z^(i)_h, a^(i)_h)  # rollout the learned model
     c^(i) = cost(z^(i)_{0..H})
     elite = top-10% lowest-cost a^(i)
     mu, sigma <- mean(elite), std(elite)
  return first action of arg-min-cost sequence
```

Cost variants:

$$
c_{\text{goal}}=\|z_H^{[\text{obj/block}]}-g\|,\quad
c_{\text{reach}(t)}=\min_h\|z_h^{[\text{agent}]}-t\|,\quad
c_{\text{event}(e)}=\min\{h:\arg\max p_\phi(z_h,a_h)=e\}\ (\text{else }H).
$$

### 5.2 Hierarchical event-level planner (C3/C5)

High level = fixed subgoal sequence (e.g., pickup→drop). Low level for the current target event $e$:

- **descriptive** event-CEM: $\mathrm{CEM}(c_{\text{event}(e)})$ over the **EventBN** rollout;
- **oracle-subgoal**: $\mathrm{CEM}(c_{\text{reach}(\text{trigger}(e))})$ — perfect hand-coded affordance;
- **affordance-event (learned)**: greedy $a=\pi(z,e)$ — the learned inverse model.
Advance the subgoal when the real environment fires $e$.

### 5.3 DAgger (closed-loop imitation; fixes BC distribution-shift)

```
DAgger(expert E, init states S0):
  D <- {(s, E(s)) : s in S0};  pi <- BC(D)
  repeat n_dagger:
     for each rollout episode:                       # roll the CURRENT pi in the real env
        s = reset()
        while not done: D <- D ∪ {(s, E(s))};  s = step(s, pi(s))
     pi <- BC(D)                                      # retrain on aggregated states (incl. pi's own)
  return pi
```

Rationale: BC is correct on the *training* distribution but drifts off it in closed loop; DAgger trains on the
states $\pi$ actually visits, relabeled by the expert — the canonical fix for closed-loop drift.

### 5.4 CEM-around-π (model-corrected controller)

```
seed = rollout pi greedily H steps under the dense model
CEM with mu = seed (small sigma), cost = c_goal or affordance,
keep the pi-seed as a candidate, return refined first action
```

Intended to combine a good learned seed with model-based local refinement; **fails on contact** (§7.2).

---

## 6. Experiments and results

### 6.1 Toy (EventEnv): C0–C7

| ID | Question | Result |
|---|---|---|
| **C0** | Does an event bottleneck improve **long-horizon prediction** vs a capacity-matched dense model? | **No** (3 seeds): median $H{=}50$ rollout ratio BN/dense $=0.91/1.99/1.29$. The "win" seen on the *mean* was a few diverged dense rollouts; the **median** reverses it. |
| **C1** | Do events **emerge unsupervised**? Does a counterfactual **intervention loss** help? | Events emerge: $\overline{\text{recall}}\approx0.90$ (3 seeds), distinct codes, lift 2–18×. NMI ($0.09$–$0.23$) under-reads it (none-dominated). The **intervention loss is null** (helped 1/3 seeds, hurt 2/3). |
| **C3** | Can you **plan** with descriptive event codes? | **The gap.** descriptive event-CEM $\approx$ dense $\approx0.13$; **oracle-subgoal** (perfect affordances) $\approx0.85$ ($\sim$7×). Event codes know *what* happened, not *how* to cause it. |
| **C5** | Does a **learned affordance + inverse model** close the gap? | **Yes, but brittle.** reaches oracle in **6/8** seeds, collapses (0.00) in **2/8** (greedy BC instability). |
| **C6** | Do standard robustness fixes work? What is the failure? | Filtering/ensemble/CEM-around-π **don't** remove the ~20% collapse. Three mechanisms **tested and refuted** (the policy uses the condition; cosines toward both targets are high on collapse seeds) ⇒ **BC distribution-shift**. |
| **C7** | Does **DAgger** robustify it? | **Yes.** 20 seeds: $\ge$ BC-only on every seed; **min 0.90, mean 0.99**, 0 failures (vs BC min 0.60, mean 0.935). |

### 6.2 Pusher (PushEnv): P1–P5 — stressing reachability

| ID | Question | Result |
|---|---|---|
| **P1 (E1)** | Does **discovery** survive *emergent* events? | **Weak** (3 seeds): $\overline{\text{recall}}\,0.44$–$0.58$, **not distinct**. contact+moved are one phenomenon; **delivered** is *goal-relational* (own code in 2/3 seeds, lift 2.8–3.8×). Transition compression captures *types*, not goal-relation. |
| **P3 (E3)** | When causing an event is **structured pushing**, does learned reachability close the gap? | **Gap reproduces, sharper:** dense-CEM $=0.00$ (can't plan a contact-push from a sparse cost), oracle $\approx0.90$, learned-BC $\approx0.1$, **learned-DAgger $=0.60$** (3 seeds) — a genuine push skill, not navigation. |
| **P4 (E3b)** | Does **CEM-around-π** close the residual $0.60{\to}0.90$? | **No — it hurts** (0.05–0.20). The dense model is **too inaccurate on contact** to rank perturbations (same reason dense-CEM $=0$). **Localizes the residual gap to the dynamics model**, not the controller. |
| **P5 (E3c)** | Is the $0.60$ cap soft (coverage) or fundamental? | **Soft.** Model-free **DAgger scaling** lifts success $0.60\!\to\!\sim\!0.80$ (peak $0.88$) by 6–8 iters; capacity barely matters (48≈96) ⇒ **coverage** is the lever. |

---

## 7. Findings

### 7.1 The dividing line, on two substrates

> **Descriptive event abstraction is insufficient for control; small world models need *actionable* event
> abstractions grounded in reachability.**

C3 measures the gap (descriptive ≈ dense ≪ oracle); C5/C7 close it on the toy (DAgger → near-perfect); P3/P5
show it **transfers to genuine contact reachability, model-free** (0.00 model-based → ~0.80–0.88 learned vs
0.90 oracle). The toy win is **not** a navigation artifact.

### 7.2 Model-free beats model-based on contact — and we know why

dense-CEM $=0.00$ and **CEM-around-π hurts** because the learned dynamics model is not accurate enough on
*contact* to plan or even to *rank local perturbations*. The model-free DAgger controller wins precisely
**because it sidesteps the unreliable model.** This is the same "predictive ≠ plannable" obstacle, now isolated
to contact-prediction accuracy.

### 7.3 Methodological throughline: robust evaluation overturns single-shot positives

Every premature positive in the program was killed by the right statistic: **mean → median** (C0), **NMI →
event-recall** (C1/P1), **single-seed → many-seed** (C5/C7), and three **refuted** collapse-mechanism
hypotheses (C6). The discipline (median + failure-rate + seeds + falsifiable mechanism probes) is itself a
contribution.

---

## 8. Limitations and future work

- **Discovery is the weak pillar on physical events** — pushing's task event is *goal-relational*; a
  transition-compression bottleneck only weakly captures it. Goal/affordance-conditioned discovery is needed.
- **Model-based planning is useless on contact** with the learned model — either accept model-free control or
  invest in a contact-accurate dynamics model.
- **The expert is scripted.** As reachability hardens, the DAgger/oracle expert stops being data-derivable and
  becomes a skill. **Expert acquisition** (RL, a few demos, or self-supervised affordance learning) is the real
  open problem for un-scriptable / pixel domains.
- **Untested rungs:** pixels (a learned latent already degraded event discovery to recall 0.39 in a parallel
  test), the block's **orientation** (real Push-T), partial observability, and stochastic contact.

---

## Appendix A — default hyperparameters

| | value |
|---|---|
| action clip $\eta$ | 0.06 (toy), 0.05 (pusher) |
| MLP hidden | 16–48 (toy capacity sweeps), 48 (pusher) |
| event codes $K$ | 6 |
| Gumbel $\tau$ | anneal $\max(0.5,1-\text{ep}/\text{eps})$ |
| $\lambda_s,\lambda_u$ | 0.01, 0.05 |
| reachability horizon $K$ | 12 |
| CEM $N$, $H$, iters | 64–256, 12–18, 2–3 |
| DAgger iters / eps | 3 (C7), up to 8 (P5) / 60–90 |
| seeds | 3 (most), 8 (C5), 20 (C7) |

## Appendix B — file map (`src/`, repo `GhostOfRazgriz1/lewm-uncertainty`)

| file | role |
|---|---|
| `_event_common.py` | toy env, `EventBN`, `train_bn`, NMI / `event_metrics`, mlp |
| `c0_event_jepa_derisk.py` | C0 long-horizon prediction (null) |
| `c1_event_intervention_planning.py` | C1 discovery + intervention + planning |
| `c3` (in c1/dedicated) / `c5_affordance.py` | C3 gap + C5 affordance-event planner |
| `c6_robust.py` | C6 robustness fixes + mechanism probes |
| `c7_dagger.py` | C7 DAgger robustification |
| `_push_common.py` | `PushEnv`, scripted expert/oracle |
| `p1_push_discovery.py` | P1 discovery on emergent events |
| `p3_push_reachability.py` | P3 dense/oracle/BC/DAgger reachability |
| `p4_push_correct.py` | P4 CEM-around-π (model correction) |
| `p5_push_modelfree.py` | P5 model-free DAgger cap curve |

Specs: `docs/event-jepa-followups-and-pivot.md` (toy arc + pivot), `docs/scaleup-pusher-spec.md` (pusher arc).
