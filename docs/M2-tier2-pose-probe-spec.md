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
- **Pose labels:** PushT's native low-dim state (agent + block pose), pulled from the env `obs`/`info` in
  the data-gen phase and asserted low-dim (**pose-gate runs first**, before any training).
- **Fine-tune:** `L_pred` (action-free `emb_t → emb_{t+k}`, both encoded by the *training* encoder, no
  stop-grad, matching LeWM) + **VICReg variance/covariance** anti-collapse (stands in for SIGReg — simpler
  and robust; the e2e-single-vs-e2e-ensemble comparison is clean since both use it). Encode is treated as a
  differentiable black box (`model.encode`), so no LeWM internals are reverse-engineered.
- **Probe:** freeze encoder, encode frames → latents, train a **linear** probe latent→standardized-state,
  report held-out **MSE** (lower=better) and **R²** (higher=better).

## Verdict
- **WIN** — `e2e-ensemble` MSE < frozen-LeWM (shaping helps) **and** < e2e-single (the ensemble/uncertainty-
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
`src/tier2_pose_probe.py`. Colab GPU. Caches data (`_tier2_data.pt`).
