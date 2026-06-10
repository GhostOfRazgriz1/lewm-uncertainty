# lewm-uncertainty

Augmenting **LeWorldModel** (LeWM) with calibrated uncertainty — for OOD detection and **uncertainty-aware planning**. Runs on a Colab (Linux + GPU) runtime, where the `stable_worldmodel` control sims work (they don't on Windows).

## Why (carried over from the perceptor exploration)

- **LeWM** ([le-wm.github.io](https://le-wm.github.io)) is an action-conditioned JEPA *control* world model: ViT-tiny → 192-d CLS latent, action-conditioned predictor (AdaLN), SIGReg Gaussian-latent regularizer, **CEM planning** (roll out action plans in latent space, pick the one whose final latent is closest to a goal latent). ~18M params.
- **Two uncertainty findings, complementary:**
  - LeWM's **SIGReg-Gaussian latent gives a *free OOD signal***: the latent norm sits on a Gaussian shell for in-distribution inputs and deviates for OOD (other envs/noise collapse below, corruption pushes above). Two-sided `|‖emb‖ − shell|` flags OOD. *(epistemic / input-familiarity facet)*
  - A foveal WM's trained `u_hat` head gives a **predictive-error** uncertainty, calibrated-in-expectation but per-instance under-sharp. *(predictive-error facet)*
  - **Neither model has both facets.** That's the analysis contribution. The constructive question — *how do you give a WM a sharp, actionable uncertainty, and does it help planning?* — needs the control loop, i.e. this repo.
- A heteroscedastic head did **not** sharpen the predictive uncertainty (structural: a single readout off the shared state). Remaining constructive levers: **ensemble disagreement** (epistemic) and a **stochastic RSSM/KL latent + uncertainty-aware planning** — both best tested *on LeWM*, where rollouts compound uncertainty and there's a real decision (the plan) to make with it.

## Plan (staged — gate each before the next)

- **M0 — infra gate (no retrain).** Reproduce LeWM's Push-T planning eval on Colab. Confirms `stable_worldmodel[env]` + the sim run, and that we can drive CEM with the pretrained checkpoint. *Needs the checkpoint + the env, NOT the 13 GB dataset.*
- **M1 — cheap real-substrate results (no retrain).**
  1. **Rigorous calibration probe** with *real* transitions from planning rollouts: does any derivable uncertainty (MC-dropout variance on the predictor, SIGReg latent-norm deviation, rollout latent-drift) **predict rollout error**? This upgrades the OOD probe to true predictive-error calibration.
  2. **Uncertainty-aware CEM:** score plans by `dist-to-goal + β·(rollout uncertainty)` and compare success rate vs vanilla CEM. The derived uncertainty needs no retrain.
- **M2 — heavy (retrain, needs the 13 GB + sims).** Stochastic LeWM: predictor outputs `(μ, σ)`; loss `KL(posterior‖prior) + λ·SIGReg` (Dreamer ELBO with SIGReg replacing reconstruction). Proper predictive uncertainty → uncertainty-aware planning, benchmarked vs M1's derived signal and vs vanilla.

## Colab — Milestone 0  (GPU runtime; run cell by cell, absolute `/content` paths)

```python
# Cell 1 — deps. Do NOT pin transformers (stable-worldmodel resolves its own; pinning 4.49 conflicts).
# The checkpoint has old (4.x) ViT key names, so we just need whatever 4.x sw pulls.
!pip install -q "stable-worldmodel[train,env]" stable-pretraining
import transformers; print("transformers:", transformers.__version__)   # report this
# If Cell 3 later asserts on a key mismatch, transformers resolved to 5.x -> run:
#   !pip install -q "transformers<4.50"   then Runtime>Restart, then re-run from Cell 2.
```
```python
# Cell 2 — code (rm first so a re-run from a failed clone is clean)
!cd /content && rm -rf le-wm lewm-uncertainty \
 && git clone -q https://github.com/lucas-maes/le-wm.git \
 && git clone -q https://github.com/GhostOfRazgriz1/lewm-uncertainty.git && ls /content
```
```python
# Cell 3 — VALIDATED step: load the pretrained LeWM on GPU (no env needed).
import sys; sys.path.insert(0, "/content/lewm-uncertainty")
from src.load_lewm import load_lewm
model, cfg = load_lewm("/content/le-wm", device="cuda")
print("LeWM loaded:", sum(p.numel() for p in model.parameters()) / 1e6, "M params")
```
```python
# Cell 4 — convert to the _object.ckpt eval.py expects (uses the model from Cell 3)
import os, torch
os.environ["STABLEWM_HOME"] = "/content/stablewm"
out = "/content/stablewm/pusht/lewm_object.ckpt"
os.makedirs(os.path.dirname(out), exist_ok=True)
torch.save(model.cpu(), out); print("saved", out)
```
```python
# Cell 5 — Push-T planning eval (needs the env; NOT the 13GB dataset). Report the planner output + any error.
%cd /content/le-wm
!STABLEWM_HOME=/content/stablewm python eval.py --config-name=pusht.yaml policy=pusht/lewm
```

Cell 3 should print ~18M params (the validated loader). Cells 4–5 are the parts to debug from their output — paste back errors and the planner trace.

If M0's success rate reproduces the paper (~the bar charts in their gif), the infra is good and M1 is unblocked.

## Repo layout

```
src/load_lewm.py        # reusable pretrained-LeWM loader (transformers-4.x note baked in)
src/probe_calibration.py# OOD probe (from perceptor exploration) — upgrades to predictive in M1
notebooks/              # Colab notebooks per milestone (M0 bootstrap, M1 probe + uncertainty-CEM)
```

## Status

M0 scripts are **written-for-Colab and untested** (the sims don't run on the Windows box they were authored on) — expect to debug the env install + checkpoint conversion on the first Colab run. The OOD probe and the loader are validated locally against the pretrained checkpoint.
