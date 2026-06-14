"""M2 Tier 2 -- does END-TO-END action-free-ensemble shaping improve the JEPA latent's physical structure?

Tier 1 gave a sharp uncertainty from a FROZEN encoder. Tier 2 unfreezes it: train encoder end-to-end with
the action-free objective and ask whether the *latent itself* encodes physical state (PushT pose) better.
Linear-probe protocol, three encoders:
  frozen-LeWM   : pretrained encoder, as-is (the baseline to beat)
  e2e-single    : fine-tune encoder + ONE action-free predictor end-to-end (control: end-to-end, no ensemble)
  e2e-ensemble  : fine-tune encoder + the action-free ENSEMBLE end-to-end (ours)
Each fine-tune = L_pred (action-free emb_t -> emb_{t+k}, both encoded by the training encoder, no stop-grad,
matching LeWM) + a VICReg variance/covariance anti-collapse term (stands in for SIGReg; simpler/robust). Then
freeze, encode frames, train a LINEAR probe latent->state, report held-out probe error / R^2.
WIN = e2e-ensemble probe error < frozen-LeWM (shaping helps) AND < e2e-single (the ensemble, not just
end-to-end, is what helps). HONEST: the frozen LeWM latent already encodes physical structure; a NULL/
regression is a legitimate outcome ("the latent is hard to improve; Tier-1 uncertainty was the win").

Fail-fast ordering: pose-gate + frozen-baseline probe run FIRST (cheap); the e2e fine-tunes (heaviest --
first ViT-encoder training in this project) run after. Run on Colab GPU:  python src/tier2_pose_probe.py
"""
import os
import sys
import math
import random
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
K_MAX, M, HE, HID = 12, 8, 32, 256
ENC_EPOCHS, ENC_LR, ENC_BS, MAX_PAIRS = 8, 1e-4, 16, 3000        # encoder fine-tune (modest, from pretrained init)
PROBE_EPOCHS, PROBE_LR = 300, 1e-2
VIC_VAR, VIC_COV = 1.0, 0.04                                      # VICReg anti-collapse weights
DATA_CACHE = "/content/lewm-uncertainty/_tier2_data.pt"
torch.manual_seed(0); random.seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
prep = TT.Compose([TT.ToTensor(), TT.Resize((224, 224)), TT.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def get_state(obs):                                              # robustly pull the low-dim pose from obs
    if isinstance(obs, dict):
        for v in obs.values():
            a = np.asarray(v).ravel()
            if a.size <= 32 and np.issubdtype(a.dtype, np.number):
                return a.astype("float32")
        raise ValueError(f"no low-dim state in obs dict (keys {list(obs.keys())}) -- locate the pose field")
    a = np.asarray(obs).ravel()
    if a.size <= 32:
        return a.astype("float32")
    raise ValueError(f"obs not low-dim (size {a.size}) -- pose labels unavailable; find the env state field")


def rollout(env, gen):
    obs, info = env.reset(seed=int(gen.integers(1_000_000_000)))
    frames, states = [env.render()], [get_state(obs)]
    for _ in range(T):
        for _ in range(FS):
            obs, _, term, trunc, info = env.step(env.action_space.sample().astype("float32"))
        frames.append(env.render()); states.append(get_state(obs))
    return np.stack(frames), np.stack(states)


def prep_batch(fr):                                             # fr: uint8 [B,H,W,3] -> [B,3,224,224]
    return torch.stack([prep(f) for f in fr]).to(device)


def encode_batch(model, fr, grad=False):                       # fr uint8 [B,H,W,3] -> emb [B,192]
    pix = prep_batch(fr).unsqueeze(1)                           # [B,1,3,224,224]
    if grad:
        return model.encode({"pixels": pix})["emb"][:, 0]
    with torch.no_grad():
        return model.encode({"pixels": pix})["emb"][:, 0]


def hembed(k):
    kf = k.float()[:, None]
    div = torch.exp(torch.arange(0, HE, 2, device=device) * (-math.log(10000.0) / HE))
    e = torch.zeros(k.shape[0], HE, device=device)
    e[:, 0::2] = torch.sin(kf * div); e[:, 1::2] = torch.cos(kf * div)
    return e


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(192 + HE, HID), nn.GELU(), nn.Linear(HID, HID), nn.GELU(), nn.Linear(HID, 192))

    def forward(self, z, ke):
        return self.net(torch.cat([z, ke], -1))


def vicreg(z):                                                 # variance + covariance anti-collapse
    std = (z.var(0) + 1e-4).sqrt()
    var = torch.relu(1.0 - std).mean()
    zc = z - z.mean(0)
    cov = (zc.T @ zc) / (z.shape[0] - 1)
    off = (cov - torch.diag(torch.diag(cov))).pow(2).sum() / z.shape[1]
    return VIC_VAR * var + VIC_COV * off


def fine_tune(model, n_heads, frames, idx):                    # end-to-end action-free shaping of the encoder
    for p in model.parameters():
        p.requires_grad_(True)                                  # only the encode path actually gets gradients (we call .encode)
    heads = nn.ModuleList([Head() for _ in range(n_heads)]).to(device)
    opt = torch.optim.Adam(list(model.parameters()) + list(heads.parameters()), lr=ENC_LR)
    pairs = [(r, t, k) for r in idx for k in range(1, K_MAX + 1) for t in range(0, T + 1 - k)]
    for ep in range(ENC_EPOCHS):
        random.shuffle(pairs); ep_pairs = pairs[:MAX_PAIRS]; last = 0.0
        for i in range(0, len(ep_pairs), ENC_BS):
            b = ep_pairs[i:i + ENC_BS]
            ft = np.stack([frames[r, t] for (r, t, k) in b]); ftk = np.stack([frames[r, t + k] for (r, t, k) in b])
            kk = torch.tensor([k for (_, _, k) in b], device=device)
            emb_t = encode_batch(model, ft, grad=True); emb_tk = encode_batch(model, ftk, grad=True)
            preds = torch.stack([h(emb_t, hembed(kk)) for h in heads])          # [H,B,192]
            loss = ((preds - emb_tk[None]) ** 2).mean() + vicreg(emb_t) + vicreg(emb_tk)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(heads.parameters()), 5.0); opt.step()
            last = loss.item()
        print(f"    ep{ep}: loss {last:.4f}", flush=True)
    model.eval()
    return model


@torch.no_grad()
def encode_rollouts(model, frames, idx):                       # -> Z [n*(T+1),192]
    Z = []
    for r in idx:
        for i in range(0, T + 1, 8):
            Z.append(encode_batch(model, frames[r, i:i + 8]))
    return torch.cat(Z)


def probe(model, frames, states, itr, iev, smean, sstd):
    Ztr = encode_rollouts(model, frames, itr); Zev = encode_rollouts(model, frames, iev)
    Ytr = ((states[itr].reshape(-1, states.shape[-1]) - smean) / sstd)
    Yev = ((states[iev].reshape(-1, states.shape[-1]) - smean) / sstd)
    Ztr, Zev = Ztr.to(device), Zev.to(device)
    Ytr, Yev = torch.tensor(Ytr, device=device), torch.tensor(Yev, device=device)
    lin = nn.Linear(192, Ytr.shape[1]).to(device)
    opt = torch.optim.Adam(lin.parameters(), lr=PROBE_LR)
    for _ in range(PROBE_EPOCHS):
        loss = ((lin(Ztr) - Ytr) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        pred = lin(Zev)
        err = (pred - Yev).pow(2).mean().item()                                 # standardized MSE (lower=better)
        r2 = (1 - (pred - Yev).pow(2).sum(0) / ((Yev - Yev.mean(0)).pow(2).sum(0) + 1e-9)).mean().item()
    return err, r2


# ---- data-gen + POSE GATE (first) ---------------------------------------------------------------
if os.path.exists(DATA_CACHE):
    d = torch.load(DATA_CACHE); frames, states = d["frames"], d["states"]
    print(f"loaded cached data frames {frames.shape} states {states.shape}", flush=True)
else:
    gen = np.random.default_rng(0); F, St = [], []
    for r in range(N_ROLLOUTS):
        fr, st = rollout(gym.make("swm/PushT-v1", render_mode="rgb_array"), gen)
        F.append(fr); St.append(st)
        if r == 0:
            print(f"POSE GATE: state dim = {st.shape[-1]} (example {st[0]})", flush=True)
        if r % 30 == 0:
            print(f"rolled {r}/{N_ROLLOUTS}", flush=True)
    frames, states = np.stack(F), np.stack(St)
    torch.save({"frames": frames, "states": states}, DATA_CACHE)
S = states.shape[-1]
smean, sstd = states.reshape(-1, S).mean(0), states.reshape(-1, S).std(0) + 1e-6
ntr = int(0.8 * N_ROLLOUTS); itr, iev = list(range(ntr)), list(range(ntr, N_ROLLOUTS))
print(f"pose dim {S}; train {len(itr)} / eval {len(iev)} rollouts", flush=True)

# ---- 1) frozen-LeWM probe (baseline, cheap, always runs) ----------------------------------------
print("\n[1/3] frozen-LeWM probe ...", flush=True)
frozen, cfg = load_lewm("/content/le-wm", device=device)
res = {}
res["frozen-LeWM"] = probe(frozen, frames, states, itr, iev, smean, sstd)
print(f"  frozen-LeWM: probe MSE {res['frozen-LeWM'][0]:.4f}  R2 {res['frozen-LeWM'][1]:.3f}", flush=True)

# ---- 2) e2e-single + 3) e2e-ensemble (heavy: ViT fine-tune) -------------------------------------
print("\n[2/3] e2e-single fine-tune ...", flush=True)
m1, _ = load_lewm("/content/le-wm", device=device)
res["e2e-single"] = probe(fine_tune(m1, 1, frames, itr), frames, states, itr, iev, smean, sstd)
print(f"  e2e-single: probe MSE {res['e2e-single'][0]:.4f}  R2 {res['e2e-single'][1]:.3f}", flush=True)
print("\n[3/3] e2e-ensemble fine-tune ...", flush=True)
m2, _ = load_lewm("/content/le-wm", device=device)
res["e2e-ensemble"] = probe(fine_tune(m2, M, frames, itr), frames, states, itr, iev, smean, sstd)
print(f"  e2e-ensemble: probe MSE {res['e2e-ensemble'][0]:.4f}  R2 {res['e2e-ensemble'][1]:.3f}", flush=True)

# ---- verdict ------------------------------------------------------------------------------------
print("\n==== M2 Tier 2 -- linear pose-probe (standardized MSE lower=better; R2 higher=better) ====")
for n in ["frozen-LeWM", "e2e-single", "e2e-ensemble"]:
    print(f"  {n:14}: MSE {res[n][0]:.4f}   R2 {res[n][1]:+.3f}")
fe, se, ee = res["frozen-LeWM"][0], res["e2e-single"][0], res["e2e-ensemble"][0]
print("\n  verdict:")
print(f"    shaping vs frozen : e2e-ensemble {'<' if ee < fe else '>='} frozen ({ee:.3f} vs {fe:.3f})")
print(f"    ensemble vs single: e2e-ensemble {'<' if ee < se else '>='} e2e-single ({ee:.3f} vs {se:.3f})")
if ee < fe * 0.98 and ee < se * 0.99:
    print("    WIN -- end-to-end ENSEMBLE shaping improves the latent's physical structure beyond frozen + single.")
elif min(se, ee) < fe * 0.98:
    print("    PARTIAL -- end-to-end shaping helps, but the ensemble adds little over single (it's the e2e, not the uncertainty).")
else:
    print("    NULL -- shaping does not beat the frozen LeWM latent: its structure is hard to improve (Tier-1 uncertainty was the win).")

# ---- figure -------------------------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
names = ["frozen-LeWM", "e2e-single", "e2e-ensemble"]; cols = ["#7f8c8d", "#e67e22", "#8e44ad"]
ax[0].bar(names, [res[n][0] for n in names], color=cols); ax[0].set_ylabel("probe MSE (lower=better)"); ax[0].set_title("Pose-probe error")
ax[1].bar(names, [res[n][1] for n in names], color=cols); ax[1].set_ylabel("probe R2 (higher=better)"); ax[1].set_title("Pose-probe R2")
for a in ax:
    a.grid(alpha=.3, axis="y"); a.tick_params(axis="x", labelrotation=15)
fig.suptitle("M2 Tier 2 -- does end-to-end action-free shaping improve the JEPA latent's pose encoding?", fontweight="bold")
fig.tight_layout(); fig.savefig("/content/lewm-uncertainty/lewm_tier2_pose.png", dpi=110)
print("\nsaved lewm_tier2_pose.png")
