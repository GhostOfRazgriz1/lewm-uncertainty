"""OUR OWN ALGORITHM -- Disagreement-Pessimistic Planning (DPP), proper test.

First run was VACUOUS: on Reacher the ensemble disagreement was ~0 (too-easy dynamics + jointly-trained
members converge), so kappa had zero effect -- not a test. This version fixes both: (a) BOOTSTRAP ensemble
(each member on its own resample -> genuine off-support disagreement), (b) PUSHER (controllable + real model
error), (c) a disagreement-magnitude DIAGNOSTIC so we see the signal is nonzero before trusting the verdict.

DPP: CEM ranking plans by  z(sum reward) - kappa * z(sum disagreement)  (z-scored across candidates, so kappa
is scale-free; kappa=0 = vanilla). Penalizes plans that bank imagined reward where the ensemble disagrees
(the WM is probably hallucinating off-support).

PREDICTION: DPP beats vanilla at LIMITED data (WM unreliable off-support, disagreement informative) and fades
at AMPLE data. WIN = margin > 2 SEM at low data AND shrinks with data, with NONZERO disagreement.
Run on Colab GPU (pip install 'gymnasium[mujoco]' opencv-python-headless):  python src/r_pessimism.py
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import cv2

ENV_ID = "Pusher-v5"
IMG, D, M = 84, 128, 5
N_POOL, ENC_EPOCHS, ENS_EPOCHS, BS, KSTEP = 70, 30, 60, 64, 8
N_SWEEP = [25, 70]                                               # limited vs ample training episodes
SEEDS, KAPPAS = [0, 1, 2], [0.0, 1.0, 3.0]                       # kappa scale-free (z-scored); 0 = vanilla
H_PLAN, S_CEM, CEM_ITERS, ELITE, EVAL_EP = 12, 256, 3, 26, 5
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


def train_base(pool, seed):                                     # encoder + reward head (+ a base predictor), then FREEZE
    torch.manual_seed(seed); np.random.seed(seed)
    enc, rew, p0 = Encoder().to(device), RewardHead().to(device), Predictor(adim).to(device)
    opt = torch.optim.Adam(list(enc.parameters()) + list(rew.parameters()) + list(p0.parameters()), lr=3e-4)
    starts = [(e, t) for e in range(len(pool)) for t in range(len(pool[e][1]) - KSTEP)]
    for epoch in range(ENC_EPOCHS):
        np.random.shuffle(starts)
        for i in range(0, len(starts), BS):
            b = starts[i:i + BS]
            fr = to_t([pool[e][0][t + k] for e, t in b for k in range(KSTEP + 1)]).view(len(b), KSTEP + 1, 3, IMG, IMG)
            ac = torch.tensor(np.stack([pool[e][1][t:t + KSTEP] for e, t in b]), device=device)
            rw_t = torch.tensor(np.stack([pool[e][2][t:t + KSTEP] for e, t in b]), device=device)
            zt = enc(fr.view(-1, 3, IMG, IMG)).view(len(b), KSTEP + 1, D)
            z = zt[:, 0]; lp = lr = 0.0
            for k in range(KSTEP):
                z = p0(z, ac[:, k]); lp = lp + ((z - zt[:, k + 1]) ** 2).mean(); lr = lr + ((rew(zt[:, k + 1]) - rw_t[:, k]) ** 2).mean()
            loss = lp / KSTEP + lr / KSTEP + 0.5 * vicreg(zt[:, 0])
            opt.zero_grad(); loss.backward(); opt.step()
    enc.eval(); rew.eval()
    for p in list(enc.parameters()) + list(rew.parameters()):
        p.requires_grad_(False)
    return enc, rew


@torch.no_grad()
def encode_pool(enc, pool):
    Z, A = [], []
    for fr, ac, _ in pool:
        z = [enc(to_t(fr[i:i + 64])) for i in range(0, len(fr), 64)]
        Z.append(torch.cat(z)); A.append(torch.tensor(ac, device=device))
    return Z, A


def train_bootstrap_ensemble(Z, A, seed):                      # M predictors, each on its OWN bootstrap resample
    torch.manual_seed(seed); np.random.seed(seed)
    members = nn.ModuleList([Predictor(adim) for _ in range(M)]).to(device)
    allstarts = [(e, t) for e in range(len(Z)) for t in range(len(A[e]) - KSTEP)]
    for mi, p in enumerate(members):
        opt = torch.optim.Adam(p.parameters(), lr=1e-3)
        rng = np.random.default_rng(1000 * seed + mi)
        boot = [allstarts[j] for j in rng.integers(0, len(allstarts), len(allstarts))]   # resample w/ replacement
        for epoch in range(ENS_EPOCHS):
            rng.shuffle(boot)
            for i in range(0, len(boot), BS):
                b = boot[i:i + BS]
                z0 = torch.stack([Z[e][t] for e, t in b]); acts = torch.stack([A[e][t:t + KSTEP] for e, t in b])
                tgt = torch.stack([Z[e][t + 1:t + KSTEP + 1] for e, t in b])
                z = z0; loss = 0.0
                for k in range(KSTEP):
                    z = p(z, acts[:, k]); loss = loss + ((z - tgt[:, k]) ** 2).mean()
                opt.zero_grad(); (loss / KSTEP).backward(); opt.step()
        p.eval()
    return members


def zsc(x):
    return (x - x.mean()) / (x.std() + 1e-9)


@torch.no_grad()
def cem_action(enc, members, rew, frame, kappa, gen):          # DPP: rank by z(reward) - kappa*z(disagreement)
    z0 = enc(to_t([frame]))[0]
    mu = torch.zeros(H_PLAN, adim, device=device); sig = torch.ones(H_PLAN, adim, device=device)
    last_u = 0.0
    for _ in range(CEM_ITERS):
        plans = (mu + sig * torch.randn(S_CEM, H_PLAN, adim, generator=gen, device=device)).clamp(-1, 1)
        z = z0[None].expand(S_CEM, D).clone(); R = torch.zeros(S_CEM, device=device); U = torch.zeros(S_CEM, device=device)
        for k in range(H_PLAN):
            preds = torch.stack([p(z, plans[:, k]) for p in members])            # [M,S,D]
            z = preds.mean(0); U = U + preds.var(0).mean(-1); R = R + rew(z)
        last_u = float(U.mean())
        score = zsc(R) - kappa * zsc(U)
        elite = plans[score.argsort(descending=True)[:ELITE]]
        mu, sig = elite.mean(0), elite.std(0) + 1e-3
    return mu[0].clamp(-1, 1).cpu().numpy(), last_u


@torch.no_grad()
def eval_return(enc, members, rew, kappa):
    g = torch.Generator(device=device).manual_seed(0); rets, us = [], []
    for ep in range(EVAL_EP):
        env = gym.make(ENV_ID, render_mode="rgb_array"); env.reset(seed=30_000 + ep)
        R = 0.0; done = False
        while not done:
            a, u = cem_action(enc, members, rew, render84(env), kappa, g); us.append(u)
            _, r, term, trunc, _ = env.step(a.astype("float32")); R += float(r); done = term or trunc
        rets.append(R); env.close()
    return np.array(rets), float(np.mean(us))


@torch.no_grad()
def eval_random():
    arng = np.random.default_rng(2); rets = []
    for ep in range(EVAL_EP):
        env = gym.make(ENV_ID, render_mode="rgb_array"); env.reset(seed=30_000 + ep)
        R = 0.0; done = False
        while not done:
            _, r, term, trunc, _ = env.step(arng.uniform(-1, 1, adim).astype("float32")); R += float(r); done = term or trunc
        rets.append(R); env.close()
    return np.array(rets)


# ---- run -----------------------------------------------------------------------------------------
env = gym.make(ENV_ID, render_mode="rgb_array"); env.reset(seed=0); adim = env.action_space.shape[0]; env.close()
print(f"=== {ENV_ID} adim {adim} ===  collecting pool of {N_POOL} eps ...", flush=True)
POOL = collect(N_POOL, 0)
rand = eval_random()
print(f"random-action reference: {rand.mean():.1f} +/- {sem(rand):.1f}\n", flush=True)

res, dis = {}, {}
for N in N_SWEEP:
    print(f"--- training data N={N} ---", flush=True)
    for kp in KAPPAS:
        res[(N, kp)] = []
    dis[N] = []
    for sd in SEEDS:
        enc, rew = train_base(POOL[:N], sd)
        Z, A = encode_pool(enc, POOL[:N])
        members = train_bootstrap_ensemble(Z, A, sd)
        for kp in KAPPAS:
            r, u = eval_return(enc, members, rew, kp)
            res[(N, kp)].append(r.mean())
            if kp == 0.0:
                dis[N].append(u)
        print(f"  seed {sd}: " + " | ".join(f"k={kp} {res[(N,kp)][-1]:.1f}" for kp in KAPPAS)
              + f"  [disagreement {dis[N][-1]:.3f}]", flush=True)
    for kp in KAPPAS:
        res[(N, kp)] = np.array(res[(N, kp)])

# ---- report + verdict ----------------------------------------------------------------------------
print("\n==== DPP on Pusher: return by (data N, kappa), mean +/- SEM over seeds ====")
margins = {}
for N in N_SWEEP:
    van = res[(N, 0.0)]
    print(f"  N={N:2d}: " + " | ".join(f"k={kp}: {res[(N,kp)].mean():.1f}+/-{sem(res[(N,kp)]):.1f}" for kp in KAPPAS)
          + f"  | mean-disagreement {np.mean(dis[N]):.3f}  | competent vs random({rand.mean():.0f}): {van.mean()>rand.mean()+2*np.hypot(sem(van),sem(rand))}")
    best = max([kp for kp in KAPPAS if kp > 0], key=lambda kp: res[(N, kp)].mean())
    d = res[(N, best)].mean() - van.mean(); s = np.hypot(sem(res[(N, best)]), sem(van))
    margins[N] = (d, s); print(f"        best DPP (k={best}) vs vanilla: {d:+.1f} +/- {s:.1f}  ({d/(s+1e-9):+.1f} SEM)")

lo, hi = N_SWEEP[0], N_SWEEP[-1]
print("\n  verdict:")
if np.mean(dis[lo]) < 1e-3:
    print("    => VACUOUS again: disagreement ~0 even with bootstrap+Pusher -> the ensemble can't produce usable")
    print("       epistemic uncertainty here; DPP has no signal to act on. (A finding, not a refutation of pessimism.)")
elif margins[lo][0] > 2 * margins[lo][1] and margins[lo][0] > margins[hi][0]:
    print("    => POSITIVE: pessimism helps at LIMITED data and fades with data -- the reliability signature. The result.")
elif margins[lo][0] > 2 * margins[lo][1]:
    print("    => POSITIVE (flat): DPP helps at low data; data-trend not clean.")
else:
    print("    => NULL: nonzero disagreement, but pessimism still doesn't improve control. Control leg truly done.")
