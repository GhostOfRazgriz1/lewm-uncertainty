"""OUR OWN ALGORITHM -- Disagreement-Pessimistic Planning (DPP) on a JEPA world model.

Our 5 control nulls all happened on FULL-DATA, IN-DIST, ~deterministic tasks -- exactly where uncertainty-
aware control SHOULD fail (no OOD to avoid; penalizing uncertainty just biases toward predictable plans,
M1.2's mechanism). Offline-RL theory says uncertainty-aware planning helps when the model is unreliable
OFF-SUPPORT (limited data / shift). DPP tests that with the JEPA's free calibrated ensemble disagreement.

DPP: CEM that maximizes a pessimistic objective  J = sum_k [ r_hat(z_k) - kappa * disagreement_k ]  -- it
won't bank imagined reward in regions the ensemble is unsure about (where the WM is probably hallucinating).
kappa=0 is vanilla CEM.

FALSIFIABLE PREDICTION (ties the 5 nulls into a mechanism): DPP beats vanilla when DATA IS LIMITED (WM
unreliable off-support) and CONVERGES to vanilla when data is AMPLE (the M1.2/3b regime). So the DPP margin
should SHRINK as training data grows -- that decay is the signature it's a real reliability effect.

Test: Reacher, data sweep N in {LOW, HIGH} x seeds, kappa in {0, 1, 3}; real env return.
Run on Colab GPU (pip install 'gymnasium[mujoco]' opencv-python-headless):  python src/r_pessimism.py
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
N_POOL, ENC_EPOCHS, BS, KSTEP = 70, 40, 64, 8
N_SWEEP = [20, 60]                                               # limited vs ample training episodes
SEEDS, KAPPAS = [0, 1, 2], [0.0, 1.0, 3.0]                       # kappa=0 is vanilla CEM
H_PLAN, S_CEM, CEM_ITERS, ELITE, EVAL_EP = 12, 256, 3, 26, 6
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


def train_wm(pool, seed):                                        # joint encoder + M-ensemble + reward head
    torch.manual_seed(seed); np.random.seed(seed)
    enc, rew = Encoder().to(device), RewardHead().to(device)
    members = nn.ModuleList([Predictor(adim) for _ in range(M)]).to(device)
    opt = torch.optim.Adam(list(enc.parameters()) + list(rew.parameters()) + list(members.parameters()), lr=3e-4)
    starts = [(e, t) for e in range(len(pool)) for t in range(len(pool[e][1]) - KSTEP)]
    for epoch in range(ENC_EPOCHS):
        np.random.shuffle(starts)
        for i in range(0, len(starts), BS):
            b = starts[i:i + BS]
            fr = to_t([pool[e][0][t + k] for e, t in b for k in range(KSTEP + 1)]).view(len(b), KSTEP + 1, 3, IMG, IMG)
            ac = torch.tensor(np.stack([pool[e][1][t:t + KSTEP] for e, t in b]), device=device)
            rw_t = torch.tensor(np.stack([pool[e][2][t:t + KSTEP] for e, t in b]), device=device)
            zt = enc(fr.view(-1, 3, IMG, IMG)).view(len(b), KSTEP + 1, D)
            lp = lr = 0.0
            for p in members:
                z = zt[:, 0]
                for k in range(KSTEP):
                    z = p(z, ac[:, k]); lp = lp + ((z - zt[:, k + 1]) ** 2).mean()
            for k in range(KSTEP):
                lr = lr + ((rew(zt[:, k + 1]) - rw_t[:, k]) ** 2).mean()
            loss = lp / (M * KSTEP) + lr / KSTEP + 0.5 * vicreg(zt[:, 0])
            opt.zero_grad(); loss.backward(); opt.step()
    enc.eval(); rew.eval()
    for p in members:
        p.eval()
    return enc, members, rew


@torch.no_grad()
def cem_action(enc, members, rew, frame, kappa, gen):           # DPP: maximize sum_k [reward - kappa*disagreement]
    z0 = enc(to_t([frame]))[0]
    mu = torch.zeros(H_PLAN, adim, device=device); sig = torch.ones(H_PLAN, adim, device=device)
    for _ in range(CEM_ITERS):
        plans = (mu + sig * torch.randn(S_CEM, H_PLAN, adim, generator=gen, device=device)).clamp(-1, 1)
        z = z0[None].expand(S_CEM, D).clone(); ret = torch.zeros(S_CEM, device=device)
        for k in range(H_PLAN):
            preds = torch.stack([p(z, plans[:, k]) for p in members])        # [M,S,D]
            z = preds.mean(0); u = preds.var(0).mean(-1)                      # [S,D],[S] disagreement
            ret = ret + rew(z) - kappa * u
        elite = plans[ret.argsort(descending=True)[:ELITE]]
        mu, sig = elite.mean(0), elite.std(0) + 1e-3
    return mu[0].clamp(-1, 1).cpu().numpy()


@torch.no_grad()
def eval_return(enc, members, rew, kappa):
    g = torch.Generator(device=device).manual_seed(0); rets = []
    for ep in range(EVAL_EP):
        env = gym.make(ENV_ID, render_mode="rgb_array"); env.reset(seed=30_000 + ep)
        R = 0.0; done = False
        while not done:
            a = cem_action(enc, members, rew, render84(env), kappa, g).astype("float32")
            _, r, term, trunc, _ = env.step(a); R += float(r); done = term or trunc
        rets.append(R); env.close()
    return np.array(rets)


# ---- data + sweep --------------------------------------------------------------------------------
env = gym.make(ENV_ID, render_mode="rgb_array"); env.reset(seed=0); adim = env.action_space.shape[0]; env.close()
print(f"=== {ENV_ID} adim {adim} ===  collecting pool of {N_POOL} eps ...", flush=True)
POOL = collect(N_POOL, 0)

res = {}                                                         # (N, kappa) -> per-seed mean returns
for N in N_SWEEP:
    print(f"\n--- training data N={N} episodes ---", flush=True)
    for kp in KAPPAS:
        res[(N, kp)] = []
    for sd in SEEDS:
        enc, members, rew = train_wm(POOL[:N], sd)
        for kp in KAPPAS:
            res[(N, kp)].append(eval_return(enc, members, rew, kp).mean())
        print(f"  seed {sd}: " + " | ".join(f"k={kp} {res[(N,kp)][-1]:.1f}" for kp in KAPPAS), flush=True)
    for kp in KAPPAS:
        res[(N, kp)] = np.array(res[(N, kp)])

# ---- report + verdict ----------------------------------------------------------------------------
print("\n==== DPP: return by (data N, kappa) -- mean +/- SEM over seeds (higher=better) ====")
margins = {}
for N in N_SWEEP:
    van = res[(N, 0.0)]
    line = " | ".join(f"k={kp}: {res[(N,kp)].mean():.1f}+/-{sem(res[(N,kp)]):.1f}" for kp in KAPPAS)
    best_kp = max([kp for kp in KAPPAS if kp > 0], key=lambda kp: res[(N, kp)].mean())
    d = res[(N, best_kp)].mean() - van.mean(); s = np.hypot(sem(res[(N, best_kp)]), sem(van))
    margins[N] = (d, s)
    print(f"  N={N:2d}: {line}")
    print(f"        best DPP (k={best_kp}) vs vanilla (k=0): {d:+.1f} +/- {s:.1f}  ({d/(s+1e-9):+.1f} SEM)")

lo, hi = N_SWEEP[0], N_SWEEP[-1]
shrinks = margins[lo][0] > margins[hi][0]
print("\n  verdict:")
print(f"    DPP margin: N={lo} {margins[lo][0]:+.1f}  ->  N={hi} {margins[hi][0]:+.1f}  (shrinks with data: {shrinks})")
if margins[lo][0] > 2 * margins[lo][1] and shrinks:
    print("    => POSITIVE: pessimism helps in the LIMITED-data regime and fades with data -- the reliability")
    print("       signature. Uncertainty-aware control works exactly where the model is unreliable. The result.")
elif margins[lo][0] > 2 * margins[lo][1]:
    print("    => POSITIVE (flat): DPP helps at low data but the data-trend isn't clean.")
else:
    print("    => NULL: pessimism does not help even at low data -> the disagreement signal isn't actionable here either.")
