"""(3a) CONTROL-COMPETENCE PRE-CHECK on a tractable MuJoCo task (Reacher-v5 pixels).

Before testing whether the calibration objective improves CONTROL (3b), confirm the substrate doesn't repeat
the PushT 'planner too weak' trap: can a minimal FROM-SCRATCH JEPA-style world model + CEM actually control
Reacher? If CEM >> random return -> the WM-control loop works here -> proceed to (3b). If CEM ~= random ->
too weak; pick an easier task or fix the WM before the objective matters.

WM (pixels): CNN encoder -> D-dim latent (+ VICReg anti-collapse, the simple robust regularizer for the
pre-check; SIGReg Gaussian latent comes in 3b for the principled NLL), action-conditioned residual-MLP
predictor f(z,a)->z', and a reward head r(z). Trained on random rollouts (k-step latent prediction + reward
+ VICReg). CEM-MPC plans H actions to maximize predicted cumulative reward, executes the first, replans.

Run on Colab GPU (after: pip install gymnasium 'gymnasium[mujoco]'):  python src/r_control_check.py
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import cv2

ENV_ID = "Reacher-v5"
IMG, D = 84, 128
N_DATA_EP, EPOCHS, BS, KSTEP = 120, 40, 64, 5
H_PLAN, S_CEM, CEM_ITERS, ELITE = 8, 256, 3, 26
EVAL_EP = 20
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0); np.random.seed(0)


def render84(env):
    f = env.render()
    return cv2.resize(np.asarray(f), (IMG, IMG), interpolation=cv2.INTER_AREA).astype("uint8")


def to_t(frames):                                                # uint8 [.,H,W,3] -> float [.,3,H,W] in [0,1]
    return torch.tensor(np.asarray(frames), dtype=torch.float32, device=device).permute(0, 3, 1, 2) / 255.0


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.c = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.GELU(), nn.Conv2d(32, 64, 4, 2, 1), nn.GELU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.GELU(), nn.Conv2d(128, 128, 4, 2, 1), nn.GELU(), nn.Flatten())
        self.head = nn.Sequential(nn.Linear(128 * 5 * 5, D), nn.LayerNorm(D))

    def forward(self, x):
        return self.head(self.c(x))


class Predictor(nn.Module):
    def __init__(self, adim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(D + adim, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, D))

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
    off = (cov - torch.diag(torch.diag(cov))).pow(2).sum() / D
    return var + 0.04 * off


# ---- fail-fast env inspection --------------------------------------------------------------------
env = gym.make(ENV_ID, render_mode="rgb_array"); obs, info = env.reset(seed=0)
adim = env.action_space.shape[0]
rs = []
for _ in range(20):
    _, r, term, trunc, _ = env.step(env.action_space.sample())
    rs.append(float(r))
    if term or trunc:
        env.reset()
print(f"=== {ENV_ID} ===  action dim {adim}  render {np.asarray(env.render()).shape} -> {IMG}x{IMG}"
      f"  reward[min {min(rs):.3f} max {max(rs):.3f}]", flush=True)
env.close()


# ---- collect random rollouts ---------------------------------------------------------------------
def collect(n_ep, seed0):
    eps = []
    for ep in range(n_ep):
        env = gym.make(ENV_ID, render_mode="rgb_array"); env.reset(seed=seed0 + ep)
        fr = [render84(env)]; ac = []; rw = []
        done = False
        while not done:
            a = env.action_space.sample().astype("float32")
            _, r, term, trunc, _ = env.step(a)
            fr.append(render84(env)); ac.append(a); rw.append(float(r)); done = term or trunc
        eps.append((np.stack(fr), np.stack(ac), np.array(rw, "float32"))); env.close()
    return eps


print("collecting random rollouts ...", flush=True)
data = collect(N_DATA_EP, 0)
EPLEN = min(len(a) for _, a, _ in data)
print(f"  {len(data)} eps, ~{EPLEN} steps each", flush=True)

enc, pred, rew = Encoder().to(device), Predictor(adim).to(device), RewardHead().to(device)
opt = torch.optim.Adam(list(enc.parameters()) + list(pred.parameters()) + list(rew.parameters()), lr=3e-4)

# build (ep, t) start index for k-step windows
starts = [(e, t) for e in range(len(data)) for t in range(len(data[e][1]) - KSTEP)]
print("training WM ...", flush=True)
for epoch in range(EPOCHS):
    np.random.shuffle(starts); tot = 0.0
    for i in range(0, len(starts), BS):
        b = starts[i:i + BS]
        frames = to_t([data[e][0][t + k] for e, t in b for k in range(KSTEP + 1)]).view(len(b), KSTEP + 1, 3, IMG, IMG)
        acts = torch.tensor(np.stack([data[e][1][t:t + KSTEP] for e, t in b]), device=device)          # [B,K,adim]
        rwd = torch.tensor(np.stack([data[e][2][t:t + KSTEP] for e, t in b]), device=device)            # [B,K]
        zt = enc(frames.view(-1, 3, IMG, IMG)).view(len(b), KSTEP + 1, D)                               # encode all
        z = zt[:, 0]; lp = 0.0; lr = 0.0
        for k in range(KSTEP):
            z = pred(z, acts[:, k])
            lp = lp + ((z - zt[:, k + 1]) ** 2).mean()
            lr = lr + ((rew(z) - rwd[:, k]) ** 2).mean()
        loss = lp / KSTEP + lr / KSTEP + 0.5 * vicreg(zt[:, 0])
        opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss)
    if epoch % 10 == 0:
        print(f"  epoch {epoch}: loss {tot/max(1,len(starts)//BS):.4f}", flush=True)
for m in (enc, pred, rew):
    m.eval()


@torch.no_grad()
def cem_action(z0, gen):
    mu = torch.zeros(H_PLAN, adim, device=device); sig = torch.ones(H_PLAN, adim) .to(device)
    for _ in range(CEM_ITERS):
        plans = (mu + sig * torch.randn(S_CEM, H_PLAN, adim, generator=gen, device=device)).clamp(-1, 1)
        z = z0[None].expand(S_CEM, D).clone(); ret = torch.zeros(S_CEM, device=device)
        for k in range(H_PLAN):
            z = pred(z, plans[:, k]); ret = ret + rew(z)
        elite = plans[ret.argsort(descending=True)[:ELITE]]
        mu, sig = elite.mean(0), elite.std(0) + 1e-3
    return mu[0].clamp(-1, 1).cpu().numpy()


@torch.no_grad()
def run_policy(kind):
    g = torch.Generator(device=device).manual_seed(0); arng = np.random.default_rng(1)
    rets = []
    for ep in range(EVAL_EP):
        env = gym.make(ENV_ID, render_mode="rgb_array"); env.reset(seed=10_000 + ep)
        R = 0.0; done = False
        while not done:
            a = arng.uniform(-1, 1, adim).astype("float32") if kind == "random" \
                else cem_action(enc(to_t([render84(env)]))[0], g).astype("float32")
            _, r, term, trunc, _ = env.step(a); R += float(r); done = term or trunc
        rets.append(R); env.close()
    return np.array(rets)


def sem(a):
    return float(np.std(a) / np.sqrt(len(a)))


print("\nevaluating ...", flush=True)
rnd = run_policy("random"); cem = run_policy("cem")
d = cem.mean() - rnd.mean(); s = np.hypot(sem(cem), sem(rnd))
print(f"\n==== (3a) Reacher control competence ({EVAL_EP} eps) ====")
print(f"  random : return {rnd.mean():.2f} +/- {sem(rnd):.2f}")
print(f"  CEM-WM : return {cem.mean():.2f} +/- {sem(cem):.2f}")
print(f"  margin : {d:+.2f} +/- {s:.2f}  ({d/(s+1e-9):+.1f} SEM)")
if d > 2 * s:
    print("  => COMPETENT: a from-scratch JEPA-WM + CEM controls Reacher -> substrate OK; proceed to (3b) objective test.")
else:
    print("  => WEAK: CEM ~= random -> WM/planner too weak here (PushT redux). Easier task or fix the WM before (3b).")
