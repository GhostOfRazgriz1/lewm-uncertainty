"""M2.2 -- M1.6 monitor REDUX with the action-free ENSEMBLE signal (the sharp Tier-1 uncertainty).

M1.6 showed a runtime monitor (selective prediction) works but its predictive signal (MC-dropout) was
flat/weak. M2 Tier-1 found the action-free ENSEMBLE disagreement is horizon-calibrated AND per-instance
sharp (within-horizon Spearman +0.58 vs MC-dropout +0.14). This wires that signal into the monitor: does
the sharper signal give a strictly better selective-prediction monitor?

Setup (cheap, frozen LeWM encoder, action-free): encode clean + noise-corrupted PushT rollouts -> latents.
Train an action-free ensemble (M heads, NO HCU loss -- Tier-1 showed HCU harmful) + a single MC-dropout
predictor on CLEAN latents. Monitor target = the ENSEMBLE-MEAN's prediction error (common to all signals,
so the comparison is apples-to-apples). Signals rank that error: ensemble-disag (predictive), MC-dropout
(predictive baseline), shell |‖z‖-shell| (OOD), combined = z(ens)+z(shell). WITHIN-horizon risk-coverage
/ AURC (lower=better; confound-free) on two pools: in-dist (clean) and mixed (clean + corrupted), vs oracle
(rank by true error) and random.

WIN = ensemble-disag AURC << MC-dropout in-dist (sharper predictive monitor); shell catches the shift;
combined(ens+shell) best on mixed -> a strictly sharper M1.6, complementary facets with the SHARP facet.
Spec mirrors docs/M1.6-monitor-spec.md + M2.  Run on Colab GPU:  python src/monitor_ensemble.py
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

N_ROLLOUTS, T, FS = 150, 24, 5
K_MAX, M, MC = 12, 8, 16
HE, HID, EPOCHS, LR, BS = 32, 256, 300, 1e-3, 4096
NOISE_SIGMA = 0.4                                                 # corruption (matches M1.6)
COVS = np.linspace(0.05, 1.0, 20)
LAT_CACHE = "/content/lewm-uncertainty/_monens_latents.pt"
torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
model, cfg = load_lewm("/content/le-wm", device=device)
prep = TT.Compose([TT.ToTensor(), TT.Resize((224, 224)), TT.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
SHELL = cfg["predictor"]["input_dim"] ** 0.5


def rollout(env, gen):
    env.reset(seed=int(gen.integers(1_000_000_000)))
    frames = [env.render()]
    for _ in range(T):
        for _ in range(FS):
            env.step(env.action_space.sample().astype("float32"))
        frames.append(env.render())
    return np.stack(frames)


def corrupt(frames, rng):
    f = frames.astype("float32") + rng.normal(0, NOISE_SIGMA * 255, frames.shape)
    return np.clip(f, 0, 255).astype("uint8")


@torch.no_grad()
def encode(frames):
    pix = torch.stack([prep(f) for f in frames]).unsqueeze(0).to(device)
    return model.encode({"pixels": pix})["emb"][0]                           # [T+1,192]


def hembed(k):
    kf = k.float()[:, None]
    div = torch.exp(torch.arange(0, HE, 2, device=device) * (-math.log(10000.0) / HE))
    e = torch.zeros(k.shape[0], HE, device=device)
    e[:, 0::2] = torch.sin(kf * div); e[:, 1::2] = torch.cos(kf * div)
    return e


class Head(nn.Module):
    def __init__(self, drop=0.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(192 + HE, HID), nn.GELU(), nn.Dropout(drop),
                                 nn.Linear(HID, HID), nn.GELU(), nn.Dropout(drop), nn.Linear(HID, 192))

    def forward(self, z, ke):
        return self.net(torch.cat([z, ke], -1))


def train(heads, Z, K, Y):
    opt = torch.optim.Adam(heads.parameters(), lr=LR); P = len(K)
    for ep in range(EPOCHS):
        perm = torch.randperm(P, device=device)
        for i in range(0, P, BS):
            idx = perm[i:i + BS]
            preds = torch.stack([h(Z[idx], hembed(K[idx])) for h in heads])
            loss = ((preds - Y[idx][None]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return heads


def train_pairs(L):                                            # clean latents -> training pairs
    Z, K, Y = [], [], []
    for k in range(1, K_MAX + 1):
        for t in range(0, T + 1 - k):
            Z.append(L[:, t]); Y.append(L[:, t + k]); K.append(torch.full((L.shape[0],), k, device=device))
    return torch.cat(Z), torch.cat(K), torch.cat(Y)


def eval_items(clean, corr):                                   # clean & corrupt latents [n,T+1,192]
    Zin, K, Y, ood = [], [], [], []
    for k in range(1, K_MAX + 1):
        for t in range(0, T + 1 - k):
            n = clean.shape[0]; kk = torch.full((n,), k, device=device)
            Zin.append(clean[:, t]); Y.append(clean[:, t + k]); K.append(kk); ood.append(torch.zeros(n, device=device))
            Zin.append(corr[:, t]);  Y.append(clean[:, t + k]); K.append(kk); ood.append(torch.ones(n, device=device))
    return torch.cat(Zin), torch.cat(K), torch.cat(Y), torch.cat(ood).bool()


def zscore(x):
    return (x - x.mean()) / (x.std() + 1e-9)


# ---- data: clean + corrupted latents ------------------------------------------------------------
if os.path.exists(LAT_CACHE):
    d = torch.load(LAT_CACHE, map_location=device); clean_lat, corr_lat = d["clean"], d["corr"]
    print(f"loaded cached latents {tuple(clean_lat.shape)}", flush=True)
else:
    gen = np.random.default_rng(0); crng = np.random.default_rng(1); cl, co = [], []
    for r in range(N_ROLLOUTS):
        fr = rollout(gym.make("swm/PushT-v1", render_mode="rgb_array"), gen)
        cl.append(encode(fr)); co.append(encode(corrupt(fr, crng)))
        if r % 30 == 0:
            print(f"encoded rollout {r}/{N_ROLLOUTS}", flush=True)
    clean_lat, corr_lat = torch.stack(cl), torch.stack(co)
    torch.save({"clean": clean_lat, "corr": corr_lat}, LAT_CACHE)
ntr = int(0.8 * N_ROLLOUTS)

# ---- train ensemble (no HCU) + MC-dropout predictor on CLEAN latents -----------------------------
Ztr, Ktr, Ytr = train_pairs(clean_lat[:ntr])
print("training action-free ensemble (no HCU) ...", flush=True)
ens = train(nn.ModuleList([Head() for _ in range(M)]).to(device), Ztr, Ktr, Ytr)
print("training MC-dropout predictor ...", flush=True)
drp = train(nn.ModuleList([Head(drop=0.1)]).to(device), Ztr, Ktr, Ytr)[0]

# ---- eval items (clean + corrupted), common target = ensemble-mean error ------------------------
Zin, K, Y, ood = eval_items(clean_lat[ntr:], corr_lat[ntr:])
for h in ens:
    h.eval()
with torch.no_grad():
    epred = torch.stack([h(Zin, hembed(K)) for h in ens])                    # [M,P,192]
    ens_mean = epred.mean(0)
    ens_disag = epred.var(0).mean(-1).cpu().numpy()
    err = (ens_mean - Y).norm(dim=-1).cpu().numpy()                          # COMMON target error (ensemble mean)
    drp.train()                                                              # dropout ON for MC sampling
    mc_disag = torch.stack([drp(Zin, hembed(K)) for _ in range(MC)]).var(0).mean(-1).cpu().numpy()
shell = (Zin.norm(dim=-1).cpu().numpy() - SHELL).__abs__()
Knp = K.cpu().numpy(); oodnp = ood.cpu().numpy()
sigs = {"ensemble": ens_disag, "MC-dropout": mc_disag, "shell": shell,
        "combined": zscore(ens_disag) + zscore(shell)}


def aurc_within(signal, mask):                                  # within-horizon risk-coverage AURC (confound-free)
    a = []
    for k in range(1, K_MAX + 1):
        m = mask & (Knp == k)
        if m.sum() < 10:
            continue
        e = err[m][np.argsort(signal[m])]
        a.append(np.mean([e[:max(1, int(c * len(e)))].mean() for c in COVS]))
    return float(np.mean(a))


def rand_within(mask):
    return float(np.mean([err[mask & (Knp == k)].mean() for k in range(1, K_MAX + 1) if (mask & (Knp == k)).sum() >= 10]))


pools = {"in-dist (clean)": ~oodnp, "mixed (clean+corrupted)": np.ones(len(err), bool)}
print("\n==== M2.2 monitor redux -- within-horizon AURC (lower=better) ====")
res = {}
for pname, mask in pools.items():
    print(f"\n  {pname}")
    rnd = rand_within(mask); orc = aurc_within(err, mask)
    row = {}
    for n, s in sigs.items():
        row[n] = aurc_within(s, mask)
        print(f"    {n:12s}: {row[n]:.3f}   (gap recovered {100*(rnd-row[n])/(rnd-orc+1e-9):+.0f}%)")
    print(f"    {'oracle':12s}: {orc:.3f}   (floor)")
    print(f"    {'random':12s}: {rnd:.3f}   (no-skill)")
    res[pname] = (row, rnd, orc)

# ---- verdict ------------------------------------------------------------------------------------
(ir, irnd, iorc) = res["in-dist (clean)"]; (mr, mrnd, morc) = res["mixed (clean+corrupted)"]
rec = lambda v, rnd, orc: 100 * (rnd - v) / (rnd - orc + 1e-9)
print("\n  verdict:")
print(f"    in-dist: ensemble recovers {rec(ir['ensemble'],irnd,iorc):.0f}% vs MC-dropout {rec(ir['MC-dropout'],irnd,iorc):.0f}%"
      f"  -> ensemble is the {'SHARPER' if ir['ensemble'] < ir['MC-dropout'] else 'NOT sharper'} predictive monitor")
print(f"    mixed:   shell recovers {rec(mr['shell'],mrnd,morc):.0f}%, combined {rec(mr['combined'],mrnd,morc):.0f}%"
      f"  -> combined {'covers both facets' if mr['combined'] <= min(mr['ensemble'], mr['shell'])+1e-6 else 'no extra gain'}")
WIN = ir["ensemble"] < ir["MC-dropout"] and mr["shell"] < mrnd
print("    " + ("WIN -- the Tier-1 ensemble is a strictly sharper M1.6 monitor; predictive+OOD facets, sharp facet."
              if WIN else "WEAK -- ensemble did not beat MC-dropout as a monitor (unexpected given Tier-1 sharpness)."))

# ---- figure: per-horizon AURC, in-dist + mixed --------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
col = {"ensemble": "#8e44ad", "MC-dropout": "#e67e22", "shell": "#2980b9", "combined": "#27ae60"}
for axi, (pname, mask) in zip(ax, pools.items()):
    for n, s in sigs.items():
        per_k = [aurc_within(s, mask & (Knp == k)) if (mask & (Knp == k)).sum() >= 10 else np.nan for k in range(1, K_MAX + 1)]
        axi.plot(range(1, K_MAX + 1), per_k, "-o", ms=3, color=col[n], label=n)
    axi.plot(range(1, K_MAX + 1), [aurc_within(err, mask & (Knp == k)) for k in range(1, K_MAX + 1)], "--", color="gray", label="oracle")
    axi.set_xlabel("horizon k"); axi.set_ylabel("AURC (risk on kept)"); axi.set_title(pname); axi.legend(fontsize=8); axi.grid(alpha=.3)
fig.suptitle("M2.2 -- M1.6 monitor with the action-free ENSEMBLE signal (sharper predictive facet)", fontweight="bold")
fig.tight_layout(); fig.savefig("/content/lewm-uncertainty/lewm_monitor_ensemble.png", dpi=110)
print("\nsaved lewm_monitor_ensemble.png")
