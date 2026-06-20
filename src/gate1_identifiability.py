"""GATE 1 -- is (z,a) support IDENTIFIABLE? The fail-fast for the #5 escape hatch.

#5 showed state-action support pessimism is UNTESTABLE under random-action data: a _|_ z => p(z,a)=p(z)p(a)
=> density-ratio = 1 => classifier AUROC = 0.5 (theorem; observed 0.56). The escape hatch: STRUCTURED data
(a behavior policy where a depends on z) should make (z,a) support identifiable. This tests exactly that --
cheaply, with NO control eval -- before committing to the full structured-offline control experiment.

For each regime, collect data, train a JEPA encoder, encode (z_t,a_t), train a density-ratio classifier
(real (z,a) vs (z, shuffled-a)), and report held-out AUROC:
  random      a ~ U(-1,1)                         -> expect AUROC ~0.5 (theorem; reproduces #5)
  structured  a = clip(W @ standardized(obs)+noise)-> expect AUROC >0.7 if structure makes support learnable
W is a fixed random linear map per seed: it only needs to make a DEPEND on state (not be a good policy).

GATE: structured AUROC > 0.7 AND random AUROC ~0.5 => escape hatch open (structure -> identifiable support)
-> proceed to Gate 2 (relevance) + the structured-offline control test. Else => support unidentifiable in
this latent regardless of policy => the support-pessimism direction is closed in principle. Cheap either way.
Run on Colab GPU (pip install 'gymnasium[mujoco]' opencv-python-headless):  python src/gate1_identifiability.py
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import cv2

ENV_ID = "Pusher-v5"
IMG, D = 84, 128
N_DATA, ENC_EPOCHS, CLF_EPOCHS, BS, KSTEP = 40, 30, 40, 64, 8
SEEDS = [0, 1, 2]
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
        self.net = nn.Sequential(nn.Linear(D + adim, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, D))

    def forward(self, z, a):
        return z + self.net(torch.cat([z, a], -1))


class Classifier(nn.Module):
    def __init__(self, adim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(D + adim, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, z, a):
        return self.net(torch.cat([z, a], -1)).squeeze(-1)


def vicreg(z):
    std = (z.var(0) + 1e-4).sqrt(); var = torch.relu(1 - std).mean()
    zc = z - z.mean(0); cov = (zc.T @ zc) / (z.shape[0] - 1)
    return var + 0.04 * (cov - torch.diag(torch.diag(cov))).pow(2).sum() / D


def collect(n_ep, seed0, policy):
    eps = []
    for ep in range(n_ep):
        env = gym.make(ENV_ID, render_mode="rgb_array"); obs, info = env.reset(seed=seed0 + ep)
        fr = [render84(env)]; ac = []; done = False
        while not done:
            a = policy(obs).astype("float32")
            obs, r, term, trunc, info = env.step(a)
            fr.append(render84(env)); ac.append(a); done = term or trunc
        eps.append((np.stack(fr), np.stack(ac))); env.close()
    return eps


def train_encoder(pool, seed):                                  # encoder via next-latent prediction + VICReg
    torch.manual_seed(seed); np.random.seed(seed)
    enc, p0 = Encoder().to(device), Predictor(adim).to(device)
    opt = torch.optim.Adam(list(enc.parameters()) + list(p0.parameters()), lr=3e-4)
    st = [(e, t) for e in range(len(pool)) for t in range(len(pool[e][1]) - KSTEP)]
    for epoch in range(ENC_EPOCHS):
        np.random.shuffle(st)
        for i in range(0, len(st), BS):
            b = st[i:i + BS]
            fr = to_t([pool[e][0][t + k] for e, t in b for k in range(KSTEP + 1)]).view(len(b), KSTEP + 1, 3, IMG, IMG)
            ac = torch.tensor(np.stack([pool[e][1][t:t + KSTEP] for e, t in b]), device=device)
            zt = enc(fr.view(-1, 3, IMG, IMG)).view(len(b), KSTEP + 1, D)
            z = zt[:, 0]; lp = 0.0
            for k in range(KSTEP):
                z = p0(z, ac[:, k]); lp = lp + ((z - zt[:, k + 1]) ** 2).mean()
            (lp / KSTEP + 0.5 * vicreg(zt[:, 0])).backward(); opt.step(); opt.zero_grad()
    enc.eval()
    return enc


@torch.no_grad()
def encode(enc, pool):
    Z, A = [], []
    for fr, ac in pool:
        Z.append(torch.cat([enc(to_t(fr[i:i + 64])) for i in range(0, len(fr), 64)])[:-1]); A.append(torch.tensor(ac, device=device))
    return torch.cat(Z), torch.cat(A)                            # aligned (z_t, a_t)


def train_classifier(Z, A, seed):
    torch.manual_seed(seed + 99)
    clf = Classifier(adim).to(device); opt = torch.optim.Adam(clf.parameters(), lr=1e-3); bce = nn.BCEWithLogitsLoss()
    for epoch in range(CLF_EPOCHS):
        perm = torch.randperm(len(Z), device=device)
        for i in range(0, len(Z), BS):
            j = perm[i:i + BS]; z = Z[j]; a = A[j]; aneg = a[torch.randperm(len(a), device=device)]
            logit = torch.cat([clf(z, a), clf(z, aneg)])
            lab = torch.cat([torch.ones(len(z), device=device), torch.zeros(len(z), device=device)])
            bce(logit, lab).backward(); opt.step(); opt.zero_grad()
    clf.eval()
    return clf


@torch.no_grad()
def auroc(clf, Z, A):                                           # held-out: real (z,a) vs (z, shuffled-a)
    pos = clf(Z, A).cpu().numpy(); neg = clf(Z, A[torch.randperm(len(A), device=device)]).cpu().numpy()
    s = np.concatenate([pos, neg]); order = np.argsort(s); r = np.empty(len(s)); r[order] = np.arange(len(s))
    n = len(pos); return float((r[:n].sum() - n * (n - 1) / 2) / (n * len(neg)))


def sem(a):
    return float(np.std(a) / np.sqrt(len(a)))


# ---- env + obs standardization (for the structured policy) ---------------------------------------
env = gym.make(ENV_ID, render_mode="rgb_array"); obs0, _ = env.reset(seed=0); adim = env.action_space.shape[0]
obs_dim = int(np.asarray(obs0).ravel().shape[0])
obss = []
for _ in range(500):                                            # quick random pass for obs mean/std
    o, _, term, trunc, _ = env.step(env.action_space.sample().astype("float32")); obss.append(np.asarray(o).ravel())
    if term or trunc:
        env.reset()
env.close()
obss = np.stack(obss); OM, OS = obss.mean(0), obss.std(0) + 1e-6
print(f"=== {ENV_ID} adim {adim} obs_dim {obs_dim} ===", flush=True)


def make_policies(seed):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 1.0 / np.sqrt(obs_dim), (adim, obs_dim))   # fixed random linear map: a depends on state
    nz = np.random.default_rng(seed + 1)
    rand = lambda obs: nz.uniform(-1, 1, adim)
    struct = lambda obs: np.clip(W @ ((np.asarray(obs).ravel() - OM) / OS) + 0.5 * nz.standard_normal(adim), -1, 1)
    return {"random": rand, "structured": struct}


# ---- run: AUROC per regime -----------------------------------------------------------------------
res = {}
for regime in ("random", "structured"):
    res[regime] = []
    for sd in SEEDS:
        pol = make_policies(sd)[regime]
        pool = collect(N_DATA, sd, pol)
        ntr = int(0.8 * len(pool))
        enc = train_encoder(pool[:ntr], sd)
        Ztr, Atr = encode(enc, pool[:ntr]); Zev, Aev = encode(enc, pool[ntr:])
        clf = train_classifier(Ztr, Atr, sd)
        res[regime].append(auroc(clf, Zev, Aev))
        print(f"  {regime:11} seed {sd}: held-out (z,a) AUROC {res[regime][-1]:.3f}", flush=True)
    res[regime] = np.array(res[regime])

print("\n==== Gate 1: is (z,a) support identifiable? ====")
for regime in ("random", "structured"):
    print(f"  {regime:11}: AUROC {res[regime].mean():.3f} +/- {sem(res[regime]):.3f}")
rnd, struct = res["random"].mean(), res["structured"].mean()
print("\n  verdict:")
if struct > 0.7 and rnd < 0.6:
    print("    => OPEN: structured data makes (z,a) support IDENTIFIABLE (AUROC>0.7) while random ~0.5 (theorem).")
    print("       The escape hatch is real -> proceed to Gate 2 (relevance) + structured-offline control test.")
elif struct > rnd + 0.15:
    print("    => PARTIAL: structure helps identifiability but AUROC<0.7 -- support is weakly learnable; marginal.")
else:
    print("    => CLOSED: structured data does NOT make (z,a) support identifiable -> support-pessimism is dead in")
    print("       this latent regardless of behavior policy. Don't build the control test. Ship the monitor paper.")
