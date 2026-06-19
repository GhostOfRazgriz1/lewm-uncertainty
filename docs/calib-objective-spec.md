# A JEPA-latent uncertainty-calibration objective (spec)

## Thesis (the professor's framing)
Propose a **JEPA-latent-space-specific uncertainty-calibration objective** and show it improves three axes:
**(a) long-horizon latent rollout fidelity / planning, (b) OOD robustness, (c) uncertainty calibration.**
White space: WIMLE (arXiv:2602.14351, ICLR'26) and HAUWM (OpenReview pZuZWRuPyi) bring uncertainty-aware
objectives to **reconstruction/RSSM** world models; **neither touches the JEPA latent**.

## What our prior results contribute (we are not starting cold)
- **Estimator we already have:** the action-free / ensemble disagreement in the JEPA latent is sharp +
  horizon-calibrated (M2.1: growth r=+0.98, within-horizon Spearman +0.58) where MC-dropout is flat.
- **JEPA-specific hooks** (the novelty over WIMLE/HAUWM): the **SIGReg Gaussian latent** makes a Gaussian-NLL
  calibration objective *principled* (the latent is N(0,I) by construction), and the **shell** is a free OOD
  signal (A1/M1.6). RSSM/recon latents have neither.
- **Known trap:** HAUWM's `L_HCU = −k·Var` (force disagreement to grow with horizon) is **harmful in JEPA**
  (M2 Tier 1 — kills per-instance sharpness). Our objective must NOT be that. Clean contrast for the paper.
- **The target is localized:** factor-planning showed the plannability gap is the predictor's **long-horizon
  rollout fidelity** (compounding error / manifold drift), not the representation or the metric. WIMLE's
  confidence-weighted training attacks exactly that.

## The objective
Frozen LeWM encoder. An ensemble of `M` action-conditioned predictors `f_i(z_t, a_t) → z_{t+1}` (residual
MLPs) trained on frozen LeWM latents. Two variants, identical except the loss:
- **baseline** — `k`-step autoregressive rollout MSE only (this is M2.1's plain ensemble, already decent).
- **ours** — same rollout MSE, **confidence-weighted** (down-weight steps with high ensemble disagreement,
  WIMLE-style) **+ λ·Gaussian-NLL calibration** on the SIGReg latent: treat the ensemble as a Gaussian
  `(μ̄=mean_i f_i, σ̄²=Var_i f_i)` and minimize `0.5[‖z_true−μ̄‖²/σ̄² + d·log σ̄²]`. NLL ties variance to
  *realized error* (calibration) — the principled version, NOT HAUWM's grow-with-horizon. Variance floor +
  clip for stability (we learned this from the HCU divergence).

## Evaluation (the three axes), experiment (1) — LeWM/PushT, no working CEM needed
- **(a) long-horizon fidelity:** `k`-step rollout error of the ensemble mean vs horizon `k`, ours vs baseline
  (the localized bottleneck; ours should win at long `k`).
- **(b) calibration:** within-horizon Spearman(σ̄², realized error) per `k` + a calibration curve; ours ≥
  baseline (baseline already +0.58).
- **(c) OOD robustness:** on corrupted current frames, does uncertainty (ensemble disagreement + shell) rise?
  AUROC clean-vs-corrupted; ours preserves/improves it.
- **planning leg (latent-fidelity sense):** plan action sequences to a goal latent with the improved
  predictor; measure how close the *true* rollout lands (does better fidelity → better latent goal-reaching).
`src/calib_predictor.py`. Runnable on what we have; ~30 min Colab.

## Experiment (3) — a substrate where JEPA-WM control actually works
WIMLE used DMC / MyoSuite / HumanoidBench. **#1 risk is infra** (Colab mujoco pixel rendering — the OGBench
rabbit hole). So **de-risk first**: `src/dmc_smoke.py` only checks that a DMC pixel env renders on Colab. If
it passes → scope the full build (train a LeJEPA-style JEPA-WM on DMC pixels, apply the objective, measure
control/sample-efficiency à la WIMLE). If it fails → fall back to a clean-rendering pixel control env
(Atari/ALE or a custom matplotlib reacher) — lower credibility, no infra rabbit hole.

## Risks / honest scope
- **Shaper risk:** Tier 2 (encoder shaping) was null — but that shaped the *encoder* for *static* pose; this
  shapes the *predictor* for *long-horizon rollout fidelity* (the localized target), predictor in the loop.
- **Planning leg:** anchor on latent-rollout fidelity (1), not CEM-success on PushT (known-blocked). Real
  control success is (3)'s job, on a substrate where the WM-control loop works.
- **NLL pathology:** vanilla Gaussian-NLL can ignore high-variance points; use a variance floor (and β-NLL if
  needed). Carry over the HCU lesson: log/clip + modest λ.
