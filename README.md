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

## Colab — Milestone 0

```bash
# 1. deps (transformers 4.x is REQUIRED — 5.x silently loads the ViT as random)
!pip install -q "stable-worldmodel[train,env]" stable-pretraining "transformers==4.49.0"

# 2. code
!git clone -q https://github.com/lucas-maes/le-wm.git
!git clone -q https://github.com/<you>/lewm-uncertainty.git

# 3. sanity: load the pretrained model + run a forward pass (no env needed)
%cd lewm-uncertainty
!python -c "from src.load_lewm import load_lewm; m,c=load_lewm('../le-wm'); print('loaded', sum(p.numel() for p in m.parameters())/1e6,'M params')"

# 4. convert HF checkpoint -> the _object.ckpt eval.py expects, then run the planning eval
#    (see le-wm/README 'Loading a checkpoint' + 'Planning'; eval uses the env, not the 13GB dataset)
%cd ../le-wm
# export STABLEWM_HOME=/content/stablewm ; hf download quentinll/lewm-pusht ... ; convert ; 
!python eval.py --config-name=pusht.yaml policy=pusht/lewm
```

If M0's success rate reproduces the paper (~the bar charts in their gif), the infra is good and M1 is unblocked.

## Repo layout

```
src/load_lewm.py        # reusable pretrained-LeWM loader (transformers-4.x note baked in)
src/probe_calibration.py# OOD probe (from perceptor exploration) — upgrades to predictive in M1
notebooks/              # Colab notebooks per milestone (M0 bootstrap, M1 probe + uncertainty-CEM)
```

## Status

M0 scripts are **written-for-Colab and untested** (the sims don't run on the Windows box they were authored on) — expect to debug the env install + checkpoint conversion on the first Colab run. The OOD probe and the loader are validated locally against the pretrained checkpoint.
