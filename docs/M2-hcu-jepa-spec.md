# M2 (HCU-for-JEPA), Tier 1 — horizon-calibrated ensemble uncertainty in a frozen LeWM latent (spec)

## Question
LeWM's read-off uncertainty is **flat across horizon** (M1.3) — the HAUWM paper (ICLR'26) calls this
"uncertainty collapse" and fixes it on RSSM with an **ensemble + Horizon-Calibrated Uncertainty (HCU)
loss** that forces ensemble disagreement to grow with the prediction horizon. **No JEPA version exists.**
Tier 1 ports HCU into a JEPA latent, cheaply, and asks: does it give the JEPA a **horizon-calibrated
predictive uncertainty** where the read-off signals couldn't?

## Method (cheap: frozen encoder, action-free, no sim beyond data-gen)
- **Data:** N random-action PushT rollouts (the existing `rollout()`), each frame encoded by the **frozen
  pretrained LeWM encoder** → latent sequences `z_{0:T}`. Random actions ⇒ genuine **action-free
  multimodality** (the paper's source of stochasticity), even though PushT is deterministic. Cache latents.
- **Model:** an **ensemble of M action-free predictors**. Head `i`: `(z_t ⊕ horizon-embed(k)) → ẑ_{t+k}`
  (no action). Variable-horizon (all `k ∈ 1..K_max`). Frozen encoder; only heads train → tiny/fast.
- **Loss:** `L = L_pred + λ·L_HCU`, with `L_pred = mean_i ‖ẑ_i − z_{t+k}‖²` (anchors the means) and
  `L_HCU = −k · Var_i(ẑ_i)` (disagreement grows with horizon).

## Baselines (so the win is attributable to HCU, not just ensembling)
- **MC-dropout:** a single action-free predictor with dropout, K passes → variance (our M1.3 signal, on
  the same action-free task).
- **Ensemble, λ=0:** ensemble without the HCU loss → does a plain ensemble also collapse to flat
  disagreement (as the paper finds for RSSM)?

## Eval (on held-out latents, no sim)
1. **Growth:** ensemble disagreement vs horizon `k` — does HCU's **grow** while MC-dropout / λ=0 stay flat?
   (Quantify: Pearson(`k`, mean disagreement).)
2. **Sharpness (confound-free):** **within-horizon** correlation between the uncertainty signal and the
   realized action-free error `‖ẑ̄ − z_{t+k}‖` — mean over `k` of Spearman(disagreement, error | fixed `k`).
   (Within-`k` so the trivial "both grow with `k`" confound is removed.) HCU should beat MC-dropout's flat ~0.
3. **Calibration curve** per horizon (predicted uncertainty vs realized error).

## Verdict
- **WIN** — HCU disagreement **grows with horizon** (growth ≫ baselines) **and** predicts error
  within-horizon (sharper than MC-dropout / λ=0) → *HCU gives a JEPA a horizon-calibrated uncertainty*
  (the novel gap; fixes the M1.3 flat-uncertainty problem). → unlocks Tier 2.
- **NULL** — HCU disagreement doesn't grow, or doesn't predict error within-horizon → the SIGReg-Gaussian
  JEPA latent resists horizon-calibration (itself a clean, novel finding: "JEPA latents collapse
  uncertainty differently than RSSM").

## Tier 2 (only if Tier 1 wins)
Unfreeze the encoder; train encoder+ensemble **end-to-end** with HCU → genuinely **shapes the latent
space**; then the downstream planning/probe eval.

## Code
- New `src/hcu_jepa.py`. One Colab script: generate rollouts → encode (cache) → train {ensemble-HCU,
  ensemble-λ0, MC-dropout} on the latents → eval (growth, within-horizon sharpness, calibration) → figures.
  Cheap (small MLP heads on precomputed latents). `λ`/`K_max` are the only knobs; start `λ=1.0`.

## Out of scope (Tier 1)
End-to-end encoder training (→ Tier 2), per-head Gaussian `(μ,σ)` / full GMM (start with μ-disagreement),
downstream planning eval (→ Tier 2), Cube/other substrates (PushT first).
