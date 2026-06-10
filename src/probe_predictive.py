"""M1.1 — predictive-error calibration on REAL Push-T transitions (no goals/CEM/dataset needed).
Does a derivable uncertainty predict LeWM's own rollout error? Two candidate signals:
  MC-dropout : enable the predictor's dropout at inference, K passes, variance = epistemic uncertainty.
  latent-norm: ||emb|| deviation from the Gaussian shell (the OOD signal) -- does it also track error?
Collect random rollouts (1 model step = FS env steps; action = FS env-actions concatenated), measure
prediction error ||pred - emb_next|| per transition, and correlate each signal with it.
Run on Colab GPU:  python src/probe_predictive.py
(Caveat: random un-normalized actions -- calibration is about the signal<->error relationship, robust to
action scale; the eventual planning use would use the real action normalizer.)"""
import sys
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import stable_worldmodel as swm                                   # registers swm/PushT-v1
from torchvision import transforms as TT
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/content/lewm-uncertainty")
from src.load_lewm import load_lewm

N_ROLLOUTS, T_STEPS, FS, HS, MC = 40, 12, 5, 3, 16                 # FS=frameskip; HS=history; MC=dropout samples
device = "cuda" if torch.cuda.is_available() else "cpu"
model, cfg = load_lewm("/content/le-wm", device=device)
prep = TT.Compose([TT.ToTensor(), TT.Resize((224, 224)), TT.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
shell = cfg["predictor"]["input_dim"] ** 0.5                       # Gaussian-shell norm (~13.9)


def rollout(env, T, gen):
    env.reset(seed=int(gen.integers(1_000_000_000)))
    frames = [env.render()]; acts = []
    for _ in range(T):
        blk = [env.action_space.sample().astype("float32") for _ in range(FS)]
        for a in blk:
            env.step(a)
        acts.append(np.concatenate(blk)); frames.append(env.render())
    return np.stack(frames), np.stack(acts)


def set_predictor_dropout(train):
    for m in model.predictor.modules():
        if isinstance(m, nn.Dropout):
            m.train(train)


@torch.no_grad()
def mc_predict(emb_hist, act_hist, K):
    set_predictor_dropout(True)
    preds = torch.stack([model.predict(emb_hist, act_hist)[:, -1] for _ in range(K)])   # [K,B,192]
    set_predictor_dropout(False)
    return preds.mean(0), preds.var(0).sum(-1)                     # mean pred, total predictive variance [B]


env = gym.make("swm/PushT-v1", render_mode="rgb_array")
gen = np.random.default_rng(0)
mc_unc, lat_norm, errs = [], [], []
for r in range(N_ROLLOUTS):
    frames, acts = rollout(env, T_STEPS, gen)
    pix = torch.stack([prep(f) for f in frames]).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode({"pixels": pix})["emb"]                # [1,T+1,192]
        act_emb = model.action_encoder(torch.tensor(acts).unsqueeze(0).to(device))   # [1,T,192]
    for t in range(HS - 1, T_STEPS):
        mu, var = mc_predict(emb[:, t - HS + 1:t + 1], act_emb[:, t - HS + 1:t + 1], MC)
        errs.append((mu - emb[:, t + 1]).norm(dim=-1).item())
        mc_unc.append(var.item())
        lat_norm.append(abs(emb[:, t].norm(dim=-1).item() - shell))   # two-sided shell deviation
    if r % 10 == 0:
        print(f"rollout {r}/{N_ROLLOUTS}", flush=True)

mc_unc, lat_norm, errs = np.array(mc_unc), np.array(lat_norm), np.array(errs)
corr = lambda a, b: (np.corrcoef(a, b)[0, 1], np.corrcoef(a.argsort().argsort(), b.argsort().argsort())[0, 1])
pe_mc, sp_mc = corr(mc_unc, errs); pe_ln, sp_ln = corr(lat_norm, errs)
print(f"\n==== LeWM predictive-error calibration on {len(errs)} real Push-T transitions ====")
print(f"prediction error: mean {errs.mean():.3f}  (copy-baseline-ish scale a few)")
print(f"MC-dropout variance vs error : Pearson {pe_mc:+.3f}  Spearman {sp_mc:+.3f}   (mc-var mean {mc_unc.mean():.4f})")
print(f"latent-shell deviation vs err: Pearson {pe_ln:+.3f}  Spearman {sp_ln:+.3f}")
print(">0 and growing => a usable predictive uncertainty for LeWM (the M1 hook for uncertainty-aware CEM).")

fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
for a, sig, name, c, p in [(ax[0], mc_unc, "MC-dropout variance", "#2980b9", pe_mc),
                           (ax[1], lat_norm, "latent-shell deviation", "#e67e22", pe_ln)]:
    a.scatter(sig, errs, s=8, alpha=.35, color=c)
    b = np.quantile(sig, np.linspace(0, 1, 9)); bi = np.clip(np.digitize(sig, b[1:-1]), 0, 7)
    a.plot([sig[bi == i].mean() for i in range(8)], [errs[bi == i].mean() for i in range(8)], "-o", color="black")
    a.set_xlabel(name); a.set_ylabel("prediction error"); a.set_title(f"{name}  (Pearson {p:+.2f})"); a.grid(alpha=.3)
fig.suptitle("Does an uncertainty signal predict LeWM's error on real Push-T transitions?", fontweight="bold")
fig.tight_layout(); fig.savefig("/content/lewm-uncertainty/lewm_predictive_calibration.png", dpi=110)
print("saved lewm_predictive_calibration.png")
