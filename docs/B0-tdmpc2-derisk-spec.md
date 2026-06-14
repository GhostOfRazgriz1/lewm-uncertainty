# B0 — De-risk gate: a free uncertainty MONITOR on a COMPETENT world model

**Direction B in one line:** every control null in this repo (M1.2 cost-shaping, A2 gating, the CEM
sanity check, factor-planning) ran on **LeWM, which cannot plan PushT at all**. So the thesis
"uncertainty is a monitor, not a controller" is *confounded* with planner incompetence. Direction B
removes the confound: take a **pretrained, provably competent** value-equivalent world model
(TD-MPC2), freeze it, and re-run the monitoring + control questions on a planner that actually works.

`b0_tdmpc2_derisk.py` is the **go/no-go gate** before any of that. It commits zero research framing
and trains nothing. It answers two questions in one afternoon on a Colab GPU:

| Gate | Question | Pass condition |
|---|---|---|
| **GATE 1 — competence** | Does a pretrained checkpoint reproduce a competent return *in our setup*? | mean return > 600, and >> random-action floor (5× and >3σ separated) |
| **GATE 2 — live signal** | Is there a free uncertainty signal to monitor *with*? | Q-ensemble disagreement has CV > 0.1 **and** Spearman(disag, one-step latent error) \|·\| > 0.15 |

This is the OGBench-Cube risk handled correctly: we want it to fail **here**, cheaply, not three
weeks into a paper.

---

## Why TD-MPC2 (recap)

- **Competent + value-equivalent** (survey: 104 tasks, beats Dreamer/SAC; decoder-free latent MPC).
- **300+ pretrained checkpoints** on HF (`nicklashansen/tdmpc2`, `dmcontrol/` subfolder), single-task = **5M params**, eval fits a Colab T4. **No training.**
- **Decision-time MPPI planning** → E2 can inject an uncertainty cost straight into the planner (that *is* M1.2, on a competent planner).
- **Ships a 5-head Q-ensemble** → GATE-2 disagreement signal is *free*, no head to train.

**SimNorm caveat (important):** TD-MPC2's latent is simplicially normalized, **not Gaussian**, so the
LeJEPA **shell/OOD** signal does **not** port here. On this substrate the monitor is **ensemble
disagreement** (our sharpest signal anyway — M2.1/M2.2). The shell facet stays on the JEPA side; that
becomes the cross-substrate breadth story, not a loss.

---

## Colab setup (Runtime → GPU)

```python
# 1. clone TD-MPC2 (env wiring + agent code) and this repo (the gate script)
!git clone -q https://github.com/nicklashansen/tdmpc2 /content/tdmpc2
!git clone -q https://github.com/GhostOfRazgriz1/lewm-uncertainty /content/lewm-uncertainty

# 2. deps — DMControl path only (state obs); avoids the gym==0.21.0 / Meta-World hell entirely.
#    Pins mirror tdmpc2/docker/environment.yaml's DMC subset; tensordict/torchrl left unpinned to
#    match Colab's torch (the one fragile spot — see "If import fails" below).
!pip install -q dm-control==1.0.16 mujoco==3.1.2 gymnasium==0.29.1 \
    hydra-core==1.3.2 omegaconf==2.3.0 tensordict torchrl \
    kornia==0.7.2 termcolor tqdm imageio imageio-ffmpeg huggingface_hub
```

```python
# 3. run the gate (recommended first task: cheetah-run — robustly competent, simple physics)
!cd /content && python /content/lewm-uncertainty/src/b0_tdmpc2_derisk.py \
    --task cheetah-run --seed 1 --episodes 20
```

The script downloads the checkpoint itself via `hf_hub_download(...,"dmcontrol/cheetah-run-1.pt")`.
Other safe-competent tasks: `cup-catch`, `walker-walk`, `finger-spin`, `reacher-easy`. Seeds `{1,2,3}`.

Outputs: `/content/b0_<task>_s<seed>.png` (3-panel: competence bar, signal spread, disag-vs-error
scatter) and `_records.pt` (per-step `z`, `q_disag`, `onestep`, returns) for downstream E1/E2 reuse.

---

## Decision table

| GATE 1 | GATE 2 | Verdict | Next move |
|---|---|---|---|
| ✅ | ✅ | **GO** | Build **E1** (selective-prediction monitor under shift) + **E2** (uncertainty-cost MPC, the swing). Framing commits. |
| ✅ | ❌ | **NO-GO(2)** | Q-ensemble is flat (heads trained jointly → agree). Fall back to the **M2.1 recipe**: train K action-free forward heads on the frozen latent, re-gate. Cheap; doesn't kill the direction. |
| ❌ | — | **NO-GO(1)** | Not a research signal — a deps/runtime problem. Triage below. If unfixable on Colab-tier, pivot to the **safe fallback paper** (A1 + identifiability theory + free≈supervised + 2nd JEPA substrate). |

---

## If GATE 1 fails (triage, in order)

1. **Wrong/over-noisy eval** — confirm a GPU runtime (`torch.cuda.is_available()`); the script asserts it.
2. **`tensordict`/`torchrl` import error** — the one fragile pin. TD-MPC2's `docker/environment.yaml`
   uses `tensordict-nightly==2025.1.1` / `torchrl-nightly==2025.1.1`. If the stable wheels mismatch
   Colab's torch, pin: `pip install tensordict==0.6.* torchrl==0.6.*` (or the nightlies). Restart runtime after.
3. **`numpy` ABI clash** after install → `Runtime → Restart`, then re-run the gate cell (skip pip).
4. **`mujoco`/`dm_control` GL** → the gate sets `MUJOCO_GL=egl` and uses **state obs + no video**, so
   no rendering is needed for stepping. If physics init still fails, try `MUJOCO_GL=osmesa`.
5. Try another seed/task (`cup-catch-1` is among the easiest competent checkpoints).

A checkpoint that loads but returns near the random floor on an *easy* task = environment/version
mismatch, **not** an incompetent agent. Compare the printed return to `tdmpc2/results/` for the exact
published number.

---

## What GO unlocks (the paper, for reference — not built by this script)

- **E1 — monitor (likely positive):** selective prediction / within-horizon AURC on rollout error
  under deployment shift (obs corruption, the A1 setup), ranked by ensemble disagreement. This is
  M1.6/M2.2 transplanted to a **credible** substrate — kills the MNIST-credibility worry. *The method
  that wins.*
- **E2 — control (the swing):** inject ensemble disagreement into TD-MPC2's MPPI cost under
  **sustained** shift/outage (carry A2's lesson — iid corruption self-corrects via per-step replanning,
  so the shift must be sustained for estimation to bind). Positive → "uncertainty becomes actionable
  once the planner is competent" (thesis flip, main-track). Null → "the monitor/controller dividing
  line is fundamental, competence-controlled" (the thesis at full strength). *Can't lose the paper.*
- **E3 — de-confound framing:** re-run M1.2/A2 logic on TD-MPC2 side-by-side with the LeWM nulls. The
  controlled comparison is the spine.
- **Breadth:** keep the LeJEPA shell-monitor on V-JEPA latents → "complementary facets on two real
  substrates."

---

## Status

- [x] API verified against TD-MPC2 source (`encode`/`next`/`Q(return_type='all')`/`load`, `rand_act`, hydra `compose`+`parse_cfg`, HF checkpoint naming `dmcontrol/{task}-{seed}.pt`).
- [ ] **Run the gate** → record GO / NO-GO(1) / NO-GO(2).
- [ ] On GO: write `B1` (E1 monitor) spec + script.
