"""(3b) THE HEADLINE: does the JEPA-latent calibration objective improve CONTROL? (Reacher pixels)

(3a) confirmed a from-scratch JEPA-WM + CEM controls Reacher (+22.6 SEM over random). (1) showed the plain
Gaussian-NLL objective improves LONG-horizon latent fidelity. So the causal prediction: a CEM controller
using the CALIBRATED world model should beat one using the BASELINE world model, and the advantage should
GROW with the planning horizon H (because that's exactly where the fidelity gain lives). That ties (1)->(3).

Protocol (isolates the predictor objective, matching (1)):
  1. Train encoder + reward head + a base predictor on random Reacher rollouts; FREEZE encoder + reward head.
  2. Encode all data -> latent sequences.
  3. Train two predictor ENSEMBLES on the frozen latents: baseline (k-step MSE) vs calibrated (MSE + NLL).
  4. CEM-MPC control with each (ensemble-mean rollout, frozen reward head as cost), swept over H in {8,12,16}.
WIN = calibrated return > baseline return beyond SEM, AND the margin grows with H (the (1)->(3) signature).
(VICReg anti-collapse for the pre-check; SIGReg Gaussian latent is the principled-framing refinement.)

Run on Colab GPU (pip install 'gymnasium[mujoco]' opencv-python-headless):  python src/r_control_calib.py
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import cv2

ENV_ID = "Reacher-v5"
IMG, D, M = 84, 128, 5
N_DATA_EP, ENC_EPOCHS, ENS_EPOCHS, BS, KSTEP = 120, 40, 60, 64, 12
H_SWEEP, S_CEM, CEM_ITERS, ELITE, EVAL_EP = [8, 12, 16], 256, 3, 26, 12
LAM, VFLOOR = 0.5, 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0); np.random.seed(0)


def render84(env):
    return cv2.resize(np.asarray(env.render()), (IMG, IMG), interpolation=cv2.INTER_AREA).astype("uint8")


def to_t(frames):
    return torch.tensor(np.asarray(frames), dtype=torch.float32, device=device).permute(0, 3, 1, 2) / 255.0


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.c = nn.Sequential(nn.Conv2d(3, 32, 4, 2, 1), nn.GELU(), nn.Conv2d(32, 64, 4, 2, 1), nn.GELU(),
                               nn.Conv2d(64, 128, 4, 2, 1), nn.GELU(), nn.Conv2d(128, 128, 4, 2, 1), nn.GELU(),
                               nn.Flatten())
        self.head = nn.Sequential(nn.Linear(128 * 5 * 5, D), nn.LayerNorm(D))

    def forward(self, x):
        return self.head(self.c(x))


class Predictor(nn.Module):
    def __init__(self, adim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(D + adim, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(),
                                 nn.Linear(256, D))

    def forward(self, z, a):
        return z + self.net(torch.cat([z, a], -1))


class RewardHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(D, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, z):
        return self.net(z).squeeze(-1)


def vicreg(z):
    std = (z.var(0) + 1e-4).sqrt(); var = torch.relu(1 - std).mean()
    zc = z - z.mean(0); cov = (zc.T @ zc) / (z.shape[0] - 1)
    return var + 0.04 * (cov - torch.diag(torch.diag(cov))).pow(2).sum() / D


def collect(n_ep, seed0):
    eps = []
    for ep in range(n_ep):
        env = gym.make(ENV_ID, render_mode="rgb_array"); env.reset(seed=seed0 + ep)
        fr = [render84(env)]; ac = []; rw = []; done = False
        while not done:
            a = env.action_space.sample().astype("float32")
            _, r, term, trunc, _ = env.step(a)
            fr.append(render84(env)); ac.append(a); rw.append(float(r)); done = term or trunc
        eps.append((np.stack(fr), np.stack(ac), np.array(rw, "float32"))); env.close()
    return eps


def sem(a):
    return float(np.std(a) / np.sqrt(len(a)))


# ---- env + data ----------------------------------------------------------------------------------
env = gym.make(ENV_ID, render_mode="rgb_array"); env.reset(seed=0); adim = env.action_space.shape[0]; env.close()
print(f"=== {ENV_ID} adim {adim} ===  collecting {N_DATA_EP} rollouts ...", flush=True)
data = collect(N_DATA_EP, 0)

# ---- 1) train encoder + reward head (+ base predictor), then FREEZE encoder + reward --------------
enc, rew, pred0 = Encoder().to(device), RewardHead().to(device), Predictor(adim).to(device)
opt = torch.optim.Adam(list(enc.parameters()) + list(rew.parameters()) + list(pred0.parameters()), lr=3e-4)
starts = [(e, t) for e in range(len(data)) for t in range(len(data[e][1]) - KSTEP)]
print("training encoder + reward head ...", flush=True)
for epoch in range(ENC_EPOCHS):
    np.random.shuffle(starts)
    for i in range(0, len(starts), BS):
        b = starts[i:i + BS]
        fr = to_t([data[e][0][t + k] for e, t in b for k in range(KSTEP + 1)]).view(len(b), KSTEP + 1, 3, IMG, IMG)
        ac = torch.tensor(np.stack([data[e][1][t:t + KSTEP] for e, t in b]), device=device)
        rw_t = torch.tensor(np.stack([data[e][2][t:t + KSTEP] for e, t in b]), device=device)
        zt = enc(fr.view(-1, 3, IMG, IMG)).view(len(b), KSTEP + 1, D)
        z = zt[:, 0]; lp = lr = 0.0
        for k in range(KSTEP):
            z = pred0(z, ac[:, k]); lp = lp + ((z - zt[:, k + 1]) ** 2).mean(); lr = lr + ((rew(z) - rw_t[:, k]) ** 2).mean()
        loss = lp / KSTEP + lr / KSTEP + 0.5 * vicreg(zt[:, 0])
        opt.zero_grad(); loss.backward(); opt.step()
enc.eval(); rew.eval()
for p in list(enc.parameters()) + list(rew.parameters()):
    p.requires_grad_(False)


# ---- 2) encode all data to frozen latent sequences -----------------------------------------------
@torch.no_grad()
def encode_eps():
    Z, A = [], []
    for fr, ac, _ in data:
        z = []
        for i in range(0, len(fr), 64):
            z.append(enc(to_t(fr[i:i + 64])))
        Z.append(torch.cat(z)); A.append(torch.tensor(ac, device=device))
    return Z, A


Zs, As = encode_eps()
print(f"  encoded {len(Zs)} latent sequences (D={D})", flush=True)


# ---- 3) train baseline vs calibrated predictor ENSEMBLES on frozen latents ------------------------
def ensemble_rollout(members, z0, acts):                          # z0[B,D], acts[B,k,adim] -> [M,B,k,D]
    outs = []
    for p in members:
        z = z0; seq = []
        for t in range(acts.shape[1]):
            z = p(z, acts[:, t]); seq.append(z)
        outs.append(torch.stack(seq, 1))
    return torch.stack(outs)


def train_ensemble(calibrated, seed):
    torch.manual_seed(seed)
    members = nn.ModuleList([Predictor(adim) for _ in range(M)]).to(device)
    opt = torch.optim.Adam(members.parameters(), lr=1e-3)
    for m in members:
        m.train()
    idx = [(e, t) for e in range(len(Zs)) for t in range(len(As[e]) - KSTEP)]
    for epoch in range(ENS_EPOCHS):
        np.random.shuffle(idx)
        for i in range(0, len(idx), BS):
            b = idx[i:i + BS]
            z0 = torch.stack([Zs[e][t] for e, t in b])
            acts = torch.stack([As[e][t:t + KSTEP] for e, t in b])
            tgt = torch.stack([Zs[e][t + 1:t + KSTEP + 1] for e, t in b])
            preds = ensemble_rollout(members, z0, acts)                       # [M,B,k,D]
            loss = ((preds - tgt[None]) ** 2).mean()
            if calibrated:
                mu = preds.mean(0); s = preds.var(0).mean(-1)
                se = ((mu - tgt) ** 2).mean(-1); sf = s.clamp(min=VFLOOR)
                loss = loss + LAM * (0.5 * (se / sf + torch.log(sf))).mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(members.parameters(), 5.0); opt.step()
    for m in members:
        m.eval()
    return members


print("training ensembles (baseline / calibrated) ...", flush=True)
WMS = {"baseline": train_ensemble(False, 0), "calibrated": train_ensemble(True, 0)}


# ---- 4) CEM control with each WM, swept over planning horizon ------------------------------------
@torch.no_grad()
def cem_action(members, z0, H, gen):
    mu = torch.zeros(H, adim, device=device); sig = torch.ones(H, adim, device=device)
    for _ in range(CEM_ITERS):
        plans = (mu + sig * torch.randn(S_CEM, H, adim, generator=gen, device=device)).clamp(-1, 1)
        z = z0[None].expand(S_CEM, D).clone(); ret = torch.zeros(S_CEM, device=device)
        for k in range(H):
            z = torch.stack([p(z, plans[:, k]) for p in members]).mean(0)      # ensemble-mean rollout
            ret = ret + rew(z)
        elite = plans[ret.argsort(descending=True)[:ELITE]]
        mu, sig = elite.mean(0), elite.std(0) + 1e-3
    return mu[0].clamp(-1, 1).cpu().numpy()


@torch.no_grad()
def eval_control(members, H):
    g = torch.Generator(device=device).manual_seed(0); rets = []
    for ep in range(EVAL_EP):
        env = gym.make(ENV_ID, render_mode="rgb_array"); env.reset(seed=20_000 + ep)
        R = 0.0; done = False
        while not done:
            a = cem_action(members, enc(to_t([render84(env)]))[0], H, g).astype("float32")
            _, r, term, trunc, _ = env.step(a); R += float(r); done = term or trunc
        rets.append(R); env.close()
    return np.array(rets)


print("\n==== (3b) Reacher control: baseline-WM vs calibrated-WM, vs planning horizon ====", flush=True)
res = {}
for H in H_SWEEP:
    for name in ("baseline", "calibrated"):
        res[(name, H)] = eval_control(WMS[name], H)
    b, c = res[("baseline", H)], res[("calibrated", H)]
    d = c.mean() - b.mean(); s = np.hypot(sem(c), sem(b))                     # >0: calibrated better (higher return)
    print(f"  H={H:2d}: baseline {b.mean():.2f}+/-{sem(b):.2f} | calibrated {c.mean():.2f}+/-{sem(c):.2f}"
          f" | margin {d:+.2f}+/-{s:.2f} ({d/(s+1e-9):+.1f} SEM)", flush=True)

# ---- verdict -------------------------------------------------------------------------------------
margins = [res[("calibrated", H)].mean() - res[("baseline", H)].mean() for H in H_SWEEP]
sems = [np.hypot(sem(res[("calibrated", H)]), sem(res[("baseline", H)])) for H in H_SWEEP]
grows = margins[-1] > margins[0]
best_sig = max(m / (s + 1e-9) for m, s in zip(margins, sems))
print("\n  verdict:")
print(f"    margins by H {H_SWEEP}: {[round(m,2) for m in margins]}  (grows with horizon: {grows})")
if best_sig > 2 and grows:
    print("    => POSITIVE: calibration objective improves CONTROL, and the margin GROWS with planning horizon")
    print("       -- the (1)->(3) signature (long-horizon fidelity gain shows up as control). The ICLR result.")
elif best_sig > 2:
    print("    => POSITIVE (flat): calibration improves control at some horizon but the H-trend isn't clean.")
else:
    print("    => NULL: calibrated WM does not beat baseline for control here (fidelity gain didn't translate).")
