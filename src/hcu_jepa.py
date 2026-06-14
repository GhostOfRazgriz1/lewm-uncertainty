"""M2 (HCU-for-JEPA), Tier 1 -- horizon-calibrated ensemble uncertainty in a frozen LeWM latent (action-free).

LeWM's read-off uncertainty is FLAT across horizon (M1.3 -- the HAUWM paper's 'uncertainty collapse').
HAUWM (ICLR'26, RSSM) fixes it with an ensemble + a Horizon-Calibrated Uncertainty (HCU) loss that forces
ensemble disagreement to grow with the prediction horizon. No JEPA version exists. This ports HCU into a
JEPA latent, cheaply: freeze the pretrained LeWM encoder, encode random PushT rollouts -> latents, and
train an ensemble of ACTION-FREE predictors z_t -> z_{t+k} (no action -> the future is genuinely
multimodal, the paper's source of stochasticity) with the HCU loss.

Tests (held-out latents, no sim beyond data-gen):
  (1) GROWTH    : does ensemble disagreement grow with horizon k (vs flat MC-dropout / a lambda=0 ensemble)?
  (2) SHARPNESS : within-horizon, does disagreement predict the realized action-free error (vs MC's flat ~0)?
  (3) calibration curve per horizon.
Baselines: MC-dropout (single action-free predictor, K passes) and ensemble lambda=0 (no HCU) -- the latter
shows HCU, not just ensembling, makes uncertainty grow. WIN = HCU disagreement GROWS with horizon AND
predicts error within-horizon, where MC-dropout is flat and the lambda=0 ensemble collapses.
Tier 2 (if this wins): unfreeze the encoder, train end-to-end with HCU -> shapes the latent space.
Spec: docs/M2-hcu-jepa-spec.md.  Run on Colab GPU:  python src/hcu_jepa.py
"""
import os
import sys
import math
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import stable_worldmodel as swm                                   # noqa: F401  registers swm/PushT-v1
from torchvision import transforms as TT
import matplotlib; matplotlib.use("Agg")                          # noqa: E402
import matplotlib.pyplot as plt                                   # noqa: E402

sys.path.insert(0, "/content/lewm-uncertainty")
from src.load_lewm import load_lewm                               # noqa: E402

N_ROLLOUTS, T, FS = 150, 24, 5                                    # rollouts, model-steps/rollout, frameskip
K_MAX, M, MC = 12, 8, 16                                          # max horizon, ensemble size, MC-dropout passes
HE, HID, EPOCHS, LR, LAMBDA = 32, 256, 300, 1e-3, 1.0            # horizon-embed dim, hidden, epochs, lr, HCU weight
LAT_CACHE = "/content/lewm-uncertainty/_hcu_latents.pt"
torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
model, cfg = load_lewm("/content/le-wm", device=device)
prep = TT.Compose([TT.ToTensor(), TT.Resize((224, 224)), TT.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def rollout(env, gen):
    env.reset(seed=int(gen.integers(1_000_000_000)))
    frames = [env.render()]
    for _ in range(T):
        for _ in range(FS):
            env.step(env.action_space.sample().astype("float32"))
        frames.append(env.render())
    return np.stack(frames)


@torch.no_grad()
def encode(frames):
    pix = torch.stack([prep(f) for f in frames]).unsqueeze(0).to(device)     # [1,T+1,3,224,224]
    return model.encode({"pixels": pix})["emb"][0]                           # [T+1,192]


def hembed(k):                                                   # sinusoidal horizon embedding: k [B] -> [B,HE]
    kf = k.float()[:, None]
    div = torch.exp(torch.arange(0, HE, 2, device=device) * (-math.log(10000.0) / HE))
    e = torch.zeros(k.shape[0], HE, device=device)
    e[:, 0::2] = torch.sin(kf * div); e[:, 1::2] = torch.cos(kf * div)
    return e


class Head(nn.Module):
    """Action-free predictor: (z_t, horizon k) -> z_{t+k}. drop>0 enables MC-dropout."""
    def __init__(self, drop=0.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(192 + HE, HID), nn.GELU(), nn.Dropout(drop),
                                 nn.Linear(HID, HID), nn.GELU(), nn.Dropout(drop), nn.Linear(HID, 192))

    def forward(self, z, ke):
        return self.net(torch.cat([z, ke], -1))


def make_pairs(L):                                              # L [n,T+1,192] -> Z[P,192], K[P], Y[P,192]
    Z, K, Y = [], [], []
    n = L.shape[0]
    for k in range(1, K_MAX + 1):
        for t in range(0, T + 1 - k):
            Z.append(L[:, t]); Y.append(L[:, t + k]); K.append(torch.full((n,), k, device=device))
    return torch.cat(Z), torch.cat(K), torch.cat(Y)


BS = 4096                                                       # minibatch (memory-safe on small GPUs)


def train_ensemble(Z, K, Y, lam):
    heads = nn.ModuleList([Head() for _ in range(M)]).to(device)
    opt = torch.optim.Adam(heads.parameters(), lr=LR)
    P = len(K)
    for ep in range(EPOCHS):
        perm = torch.randperm(P, device=device); last = (0.0, 0.0)
        for i in range(0, P, BS):
            idx = perm[i:i + BS]; z, k, y = Z[idx], K[idx], Y[idx]
            preds = torch.stack([h(z, hembed(k)) for h in heads])            # [M,b,192]
            lpred = ((preds - y[None]) ** 2).mean()
            disag = preds.var(0).mean(-1)                                    # [b] per-dim-mean ensemble variance
            loss = lpred - lam * (k.float() * torch.log1p(disag)).mean()     # L_pred + lambda*L_HCU; log1p bounds the
            opt.zero_grad(); loss.backward()                                 # disagreement reward -> finite equilibrium (stable)
            torch.nn.utils.clip_grad_norm_(heads.parameters(), 5.0); opt.step()
            last = (lpred.item(), disag.mean().item())
        if ep % 100 == 0:
            print(f"  ens(lam={lam}) ep{ep}: lpred {last[0]:.4f}  disag {last[1]:.4f}", flush=True)
    return heads


def train_dropout(Z, K, Y):
    h = Head(drop=0.1).to(device)
    opt = torch.optim.Adam(h.parameters(), lr=LR)
    P = len(K)
    for ep in range(EPOCHS):
        perm = torch.randperm(P, device=device)
        for i in range(0, P, BS):
            idx = perm[i:i + BS]
            loss = ((h(Z[idx], hembed(K[idx])) - Y[idx]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return h


@torch.no_grad()
def ens_signal(heads, Z, K, Y):
    for h in heads:
        h.eval()
    preds = torch.stack([h(Z, hembed(K)) for h in heads])                    # [M,P,192]
    disag = preds.var(0).mean(-1)
    err = (preds.mean(0) - Y).norm(dim=-1)
    return disag.cpu().numpy(), err.cpu().numpy()


@torch.no_grad()
def drop_signal(h, Z, K, Y):
    h.train()                                                               # dropout ON for MC sampling
    preds = torch.stack([h(Z, hembed(K)) for _ in range(MC)])               # [MC,P,192]
    disag = preds.var(0).mean(-1)
    err = (preds.mean(0) - Y).norm(dim=-1)
    return disag.cpu().numpy(), err.cpu().numpy()


def spearman(a, b):
    if len(a) < 3 or np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return 0.0
    ra, rb = a.argsort().argsort().astype(float), b.argsort().argsort().astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


# ---- data: encode random rollouts (cache the latents) -------------------------------------------
if os.path.exists(LAT_CACHE):
    lat = torch.load(LAT_CACHE, map_location=device); print(f"loaded cached latents {tuple(lat.shape)}", flush=True)
else:
    gen = np.random.default_rng(0); seqs = []
    for r in range(N_ROLLOUTS):
        seqs.append(encode(rollout(gym.make("swm/PushT-v1", render_mode="rgb_array"), gen)))
        if r % 30 == 0:
            print(f"encoded rollout {r}/{N_ROLLOUTS}", flush=True)
    lat = torch.stack(seqs); torch.save(lat, LAT_CACHE)
ntr = int(0.8 * N_ROLLOUTS)
Ztr, Ktr, Ytr = make_pairs(lat[:ntr])
Zev, Kev, Yev = make_pairs(lat[ntr:])
Kev_np = Kev.cpu().numpy()
print(f"train pairs {len(Ktr)}, eval pairs {len(Kev)}", flush=True)

# ---- train the three models ---------------------------------------------------------------------
print("training ensemble + HCU ...", flush=True);    hcu = train_ensemble(Ztr, Ktr, Ytr, LAMBDA)
print("training ensemble (lambda=0) ...", flush=True); ens0 = train_ensemble(Ztr, Ktr, Ytr, 0.0)
print("training MC-dropout predictor ...", flush=True); drp = train_dropout(Ztr, Ktr, Ytr)

signals = {
    "HCU": ens_signal(hcu, Zev, Kev, Yev),
    "ensemble(lam=0)": ens_signal(ens0, Zev, Kev, Yev),
    "MC-dropout": drop_signal(drp, Zev, Kev, Yev),
}
ks = np.arange(1, K_MAX + 1)


def per_k_mean(d):
    return np.array([d[Kev_np == k].mean() for k in ks])


def within_k_sharpness(d, e):                                   # mean over k of Spearman(disag, err | fixed k)
    return float(np.mean([spearman(d[Kev_np == k], e[Kev_np == k]) for k in ks]))


# ---- verdict ------------------------------------------------------------------------------------
err_by_k = per_k_mean(signals["ensemble(lam=0)"][1])           # realized error vs horizon (from the STABLE lam=0 ensemble)
print("\n==== M2 HCU-for-JEPA Tier 1 ====")
print(f"  realized action-free error vs horizon: k1 {err_by_k[0]:.3f} -> k{K_MAX} {err_by_k[-1]:.3f}"
      f"  (grows: Pearson(k,err) {np.corrcoef(ks, err_by_k)[0,1]:+.2f})")
print(f"\n  {'signal':18}{'disag k1->kMax':>20}{'GROWTH r(k,disag)':>20}{'SHARPNESS within-k':>22}")
res = {}
for n, (d, e) in signals.items():
    dk = per_k_mean(d)
    growth = float(np.corrcoef(ks, dk)[0, 1])
    sharp = within_k_sharpness(d, e)
    res[n] = (growth, sharp)
    print(f"  {n:18}{dk[0]:8.3f} -> {dk[-1]:7.3f}{growth:>+20.2f}{sharp:>+22.2f}")
mcg, mcs = res["MC-dropout"]
print("\n  verdict (does an ACTION-FREE ENSEMBLE in the JEPA latent give horizon-calibrated, sharp uncertainty?):")
for n in ["ensemble(lam=0)", "HCU"]:
    g, s = res[n]
    print(f"    {n:18}: growth r={g:+.2f} (vs realized-error shape), within-horizon sharpness {s:+.2f}")
print(f"    {'MC-dropout':18}: growth r={mcg:+.2f}, sharpness {mcs:+.2f}  (the flat read-off baseline, M1.3)")
best = max(["ensemble(lam=0)", "HCU"], key=lambda n: res[n][1])
bg, bs = res[best]
if bg > 0.5 and bs > 0.3 and bs > mcs + 0.15:
    print(f"\n    WIN -- action-free ensembling ({best}) gives the JEPA a horizon-calibrated, per-instance-sharp")
    print(f"           uncertainty (growth {bg:+.2f}, sharpness {bs:+.2f}) where MC-dropout is flat ({mcs:+.2f}). Unlocks Tier 2.")
    print("    HCU vs plain ensemble: " + ("HCU ADDS sharpness." if res["HCU"][1] > res["ensemble(lam=0)"][1] + 0.05
          else "no gain -- action-free ensembling alone suffices; the HCU loss is unnecessary in this JEPA latent."))
else:
    print("\n    NULL -- no ensemble beats MC-dropout; the SIGReg-Gaussian JEPA latent resists calibrated uncertainty.")

# ---- figure: disagreement vs horizon + within-horizon sharpness ---------------------------------
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
col = {"HCU": "#8e44ad", "ensemble(lam=0)": "#2980b9", "MC-dropout": "#e67e22"}
for n, (d, e) in signals.items():
    dk = per_k_mean(d); dk = dk / dk.mean()                     # normalize for shape comparison
    ax[0].plot(ks, dk, "-o", ms=3, color=col[n], label=f"{n} (r={res[n][0]:+.2f})")
ax[0].plot(ks, err_by_k / err_by_k.mean(), "--", color="#27ae60", label="realized error (target shape)")
ax[0].set_xlabel("prediction horizon k"); ax[0].set_ylabel("disagreement (normalized)")
ax[0].set_title("Does uncertainty grow with horizon?"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
for n, (d, e) in signals.items():
    sk = [spearman(d[Kev_np == k], e[Kev_np == k]) for k in ks]
    ax[1].plot(ks, sk, "-o", ms=3, color=col[n], label=f"{n} (mean {res[n][1]:+.2f})")
ax[1].axhline(0, color="gray", lw=.8)
ax[1].set_xlabel("prediction horizon k"); ax[1].set_ylabel("Spearman(disagreement, error | k)")
ax[1].set_title("Does it predict error within-horizon?"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
fig.suptitle("M2 HCU-for-JEPA (Tier 1) -- horizon-calibrated ensemble uncertainty in a frozen LeWM latent", fontweight="bold")
fig.tight_layout(); fig.savefig("/content/lewm-uncertainty/lewm_hcu_jepa.png", dpi=110)
print("\nsaved lewm_hcu_jepa.png")
