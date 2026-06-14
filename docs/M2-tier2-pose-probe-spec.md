# M2 Tier 2 — does end-to-end action-free shaping improve the JEPA latent's physical structure? (spec)

## Question
Tier 1 gave a sharp uncertainty from a **frozen** encoder. Tier 2 unfreezes it: does training the encoder
**end-to-end** with the action-free objective make the *latent itself* encode physical state (PushT pose)
better? Linear-probe protocol (the paper's "probing of physical quantities").

## Three encoders (probed identically)
- **frozen-LeWM** — pretrained encoder, as-is. The baseline to beat.
- **e2e-single** (control) — fine-tune encoder + ONE action-free predictor end-to-end. Isolates
  "end-to-end action-free training" from "ensemble."
- **e2e-ensemble** (ours) — fine-tune encoder + the action-free ensemble end-to-end.

## Method
- **Pose labels:** PushT's native **`info['block_pose']`** (the 3-d T-block pose `[x, y, angle]`), pulled in
  the data-gen phase and shape-asserted (**pose-gate runs first**, before any training). A diagnostic
  (`tier2_diag.py`) ridge-probed every candidate low-dim field on the frozen latent: `block_pose` is the
  clean target (**R² +0.53**); the full 7-d `state` is only +0.06 and `pos_agent` is **negative** — the
  latent encodes the big T-block well but barely localizes the small pusher, so we probe block_pose alone.
- **Fine-tune:** `L_pred` (action-free `emb_t → emb_{t+k}`, both encoded by the *training* encoder, no
  stop-grad, matching LeWM) + **VICReg variance/covariance** anti-collapse (stands in for SIGReg — simpler
  and robust; the e2e-single-vs-e2e-ensemble comparison is clean since both use it). Encode is treated as a
  differentiable black box (`model.encode`), so no LeWM internals are reverse-engineered.
- **Probe:** freeze encoder, encode frames → latents, fit a **closed-form ridge** probe latent→standardized
  block_pose (alpha-swept `{1,10,100,1000}`, same protocol for all three encoders), report held-out **MSE**
  (lower=better) and **R²** (higher=better). Ridge, not Adam — an unregularized Adam linear probe overfits
  192→3 and reported spurious R² < 0 even on the frozen latent that *does* encode pose.

## Verdict (R², frozen block_pose R² ≈ 0.53 is the bar)
- **WIN** — `e2e-ensemble` R² > frozen-LeWM (shaping helps) **and** > e2e-single (the ensemble/uncertainty-
  awareness, not just end-to-end, is what helps).
- **PARTIAL** — e2e helps but the ensemble adds little over single (it's the end-to-end, not the uncertainty).
- **NULL (legitimate, even expected)** — shaping doesn't beat the frozen LeWM latent → its structure is hard
  to improve; the Tier-1 *uncertainty* was the win, not latent-shaping.

## Risks / honest scope
- **Heaviest build:** first ViT-encoder fine-tune (from pretrained init; modest epochs, `MAX_PAIRS` cap).
  ~tens of minutes on Colab. Fail-fast ordering: pose-gate + frozen-baseline probe run first, so a break in
  the e2e step still yields the baseline.
- **Pose availability** is asserted in the gate (errors clearly if the env doesn't expose low-dim state).
- **Null-risk is real:** the frozen LeWM latent already encodes physical structure; action-free shaping may
  not improve (could blur) current-state encoding. A null is a clean result.
- **VICReg vs SIGReg:** e2e uses VICReg anti-collapse (not SIGReg) → the e2e-vs-frozen comparison has a
  regularizer difference; the *clean* claim is e2e-ensemble vs e2e-single (same regularizer).

## Code
`src/tier2_pose_probe.py`. Colab GPU. Caches data (`_tier2_data_v2.pt` — v2 carries block_pose labels; the
v1 cache held the wrong field). Diagnostic that found the right field: `src/tier2_diag.py`.
