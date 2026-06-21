"""THE CONTROL TEST -- support-pessimistic planning on STRUCTURED offline data (both gates passed).

Gate 1: (z,a) support identifiable with structured data (AUROC 0.85 vs random 0.50). Gate 2: support score
relevant to model error (corr +0.21, off-support errs more, valid set_state). Both preconditions hold -- the
first time. Now the payoff: does support-pessimistic CEM beat vanilla CEM on structured-offline data?

Setup: collect STRUCTURED offline data (state-dependent behavior policy -> identifiable+relevant (z,a)
support). Train WM (encoder + predictor + reward head) + support classifier g(z,a). Then:
  random-action   (floor: is there any control to test?)
  vanilla CEM     (max predicted reward -- exploits WM errors off-support)
  support-pess CEM (max  z(reward) - kappa * z(sum_t -g(z_t,a_t))  -- stay where the WM is reliable)
PAIRED per-seed (vanilla & pess share the WM), 5 seeds. Inline AUROC confirms the gate holds for this data.

WIN = pess paired-delta > vanilla beyond 2 SEM (and clean-CEM >> random, i.e. there IS control to improve).
This is the principled realization of "JEPA uncertainty helps planning" -- on solid, gated footing.
Run on Colab GPU (pip install 'gymnasium[mujoco]' opencv-python-headless):  python src/control_structured.py
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
N_DATA, ENC_EPOCHS, CLF_EPOCHS, BS, KSTEP = 60, 30, 40, 64, 8
SEEDS, KAP = list(range(10)), 3.0                               # 10 seeds: 5-seed gave +4.6 (+1.6 SEM, 4/5 pos) on a VERIFIED signal -> resolve
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
        self.net = nn.Sequential(nn.Linear(D + adim, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, D))

    def forward(self, z, a):
        return z + self.net(torch.cat([z, a], -1))


class RewardHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(D, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, z):
        return self.net(z).squeeze(-1)


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
        fr = [render84(env)]; ac = []; rw = []; done = False
        while not done:
            a = policy(obs).astype("float32"); obs, r, term, trunc, info = env.step(a)
            fr.append(render84(env)); ac.append(a); rw.append(float(r)); done = term or trunc
        eps.append((np.stack(fr), np.stack(ac), np.array(rw, "float32"))); env.close()
    return eps


def train_wm(pool, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    enc, rew, p0 = Encoder().to(device), RewardHead().to(device), Predictor(adim).to(device)
    opt = torch.optim.Adam(list(enc.parameters()) + list(rew.parameters()) + list(p0.parameters()), lr=3e-4)
    stx = [(e, t) for e in range(len(pool)) for t in range(len(pool[e][1]) - KSTEP)]
    for epoch in range(ENC_EPOCHS):
        np.random.shuffle(stx)
        for i in range(0, len(stx), BS):
            b = stx[i:i + BS]
            fr = to_t([pool[e][0][t + k] for e, t in b for k in range(KSTEP + 1)]).view(len(b), KSTEP + 1, 3, IMG, IMG)
            ac = torch.tensor(np.stack([pool[e][1][t:t + KSTEP] for e, t in b]), device=device)
            rw_t = torch.tensor(np.stack([pool[e][2][t:t + KSTEP] for e, t in b]), device=device)
            zt = enc(fr.view(-1, 3, IMG, IMG)).view(len(b), KSTEP + 1, D)
            z = zt[:, 0]; lp = lr = 0.0
            for k in range(KSTEP):
                z = p0(z, ac[:, k]); lp = lp + ((z - zt[:, k + 1]) ** 2).mean(); lr = lr + ((rew(zt[:, k + 1]) - rw_t[:, k]) ** 2).mean()
            (lp / KSTEP + lr / KSTEP + 0.5 * vicreg(zt[:, 0])).backward(); opt.step(); opt.zero_grad()
    enc.eval(); rew.eval(); p0.eval()
    return enc, p0, rew


def train_classifier(pool, enc, seed):
    torch.manual_seed(seed + 99)
    with torch.no_grad():
        Z = torch.cat([torch.cat([enc(to_t(fr[i:i + 64])) for i in range(0, len(fr), 64)])[:-1] for fr, _, _ in pool])
        A = torch.cat([torch.tensor(ac, device=device) for _, ac, _ in pool])
    clf = Classifier(adim).to(device); opt = torch.optim.Adam(clf.parameters(), lr=1e-3); bce = nn.BCEWithLogitsLoss()
    for epoch in range(CLF_EPOCHS):
        perm = torch.randperm(len(Z), device=device)
        for i in range(0, len(Z), BS):
            j = perm[i:i + BS]; z = Z[j]; a = A[j]; aneg = a[torch.randperm(len(a), device=device)]
            bce(torch.cat([clf(z, a), clf(z, aneg)]),
                torch.cat([torch.ones(len(z), device=device), torch.zeros(len(z), device=device)])).backward()
            opt.step(); opt.zero_grad()
    clf.eval()
    with torch.no_grad():                                                       # inline AUROC (gate-1 confirm for this data)
        pos = clf(Z, A).cpu().numpy(); neg = clf(Z, A[torch.randperm(len(A), device=device)]).cpu().numpy()
    s = np.concatenate([pos, neg]); o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(len(s)); n = len(pos)
    return clf, float((r[:n].sum() - n * (n - 1) / 2) / (n * len(neg)))


def zsc(x):
    return (x - x.mean()) / (x.std() + 1e-9)


@torch.no_grad()
def cem_action(enc, p0, rew, clf, frame, kappa, gen):
    z0 = enc(to_t([frame]))[0]
    mu = torch.zeros(H_PLAN, adim, device=device); sig = torch.ones(H_PLAN, adim, device=device)
    for _ in range(CEM_ITERS):
        plans = (mu + sig * torch.randn(S_CEM, H_PLAN, adim, generator=gen, device=device)).clamp(-1, 1)
        z = z0[None].expand(S_CEM, D).clone(); R = torch.zeros(S_CEM, device=device); U = torch.zeros(S_CEM, device=device)
        for k in range(H_PLAN):
            a = plans[:, k]; U = U + (-clf(z, a))                               # support penalty on (z_t, a_t)
            z = p0(z, a); R = R + rew(z)
        score = zsc(R) if kappa == 0 else zsc(R) - kappa * zsc(U)
        elite = plans[score.argsort(descending=True)[:ELITE]]
        mu, sig = elite.mean(0), elite.std(0) + 1e-3
    return mu[0].clamp(-1, 1).cpu().numpy()


@torch.no_grad()
def eval_policy(kind, enc=None, p0=None, rew=None, clf=None, kappa=0.0):
    g = torch.Generator(device=device).manual_seed(0); arng = np.random.default_rng(1); rets = []
    for ep in range(EVAL_EP):
        env = gym.make(ENV_ID, render_mode="rgb_array"); env.reset(seed=50_000 + ep); R = 0.0; done = False
        while not done:
            a = arng.uniform(-1, 1, adim).astype("float32") if kind == "random" \
                else cem_action(enc, p0, rew, clf, render84(env), kappa, g).astype("float32")
            _, r, term, trunc, _ = env.step(a); R += float(r); done = term or trunc
        rets.append(R); env.close()
    return np.array(rets)


def sem(a):
    return float(np.std(a) / np.sqrt(len(a)))


# ---- structured behavior policy (same as the gates) ----------------------------------------------
env = gym.make(ENV_ID, render_mode="rgb_array"); obs0, _ = env.reset(seed=0); adim = env.action_space.shape[0]
obs_dim = int(np.asarray(obs0).ravel().shape[0]); obss = []
for _ in range(500):
    o, _, term, trunc, _ = env.step(env.action_space.sample().astype("float32")); obss.append(np.asarray(o).ravel())
    if term or trunc:
        env.reset()
env.close()
OM, OS = np.stack(obss).mean(0), np.stack(obss).std(0) + 1e-6
print(f"=== {ENV_ID} adim {adim} ===", flush=True)


def structured_policy(seed):
    rng = np.random.default_rng(seed); W = rng.normal(0, 1.0 / np.sqrt(obs_dim), (adim, obs_dim)); nz = np.random.default_rng(seed + 1)
    return lambda obs: np.clip(W @ ((np.asarray(obs).ravel() - OM) / OS) + 0.5 * nz.standard_normal(adim), -1, 1)


# ---- run -----------------------------------------------------------------------------------------
rand = eval_policy("random")
print(f"random-action floor: {rand.mean():.1f} +/- {sem(rand):.1f}\n", flush=True)
van, pess, aurocs = [], [], []
for sd in SEEDS:
    pool = collect(N_DATA, sd, structured_policy(sd))
    enc, p0, rew = train_wm(pool, sd); clf, au = train_classifier(pool, enc, sd); aurocs.append(au)
    v = eval_policy("cem", enc, p0, rew, clf, 0.0).mean()
    p = eval_policy("cem", enc, p0, rew, clf, KAP).mean()
    van.append(v); pess.append(p)
    print(f"  seed {sd}: vanilla {v:.1f} | support-pess {p:.1f} | delta {p-v:+.1f} | clf AUROC {au:.2f}", flush=True)
van, pess = np.array(van), np.array(pess)

print("\n==== control on structured-offline data (both gates passed) ====")
print(f"  random {rand.mean():.1f} | vanilla {van.mean():.1f}+/-{sem(van):.1f} | support-pess {pess.mean():.1f}+/-{sem(pess):.1f}")
print(f"  gate-1 AUROC (this data): {np.mean(aurocs):.2f}")
comp = van.mean() - rand.mean(); cs = np.hypot(sem(van), sem(rand))
delta = pess - van                                                              # PAIRED per-seed
fpos = float(np.mean(delta > 0))
print(f"  competence (vanilla vs random): {comp:+.1f} +/- {cs:.1f}  ({'CONTROLS' if comp > 2*cs else 'WEAK -- no control to improve'})")
print(f"  PAIRED support-pess delta: {delta.mean():+.2f} +/- {sem(delta):.2f}  ({delta.mean()/(sem(delta)+1e-9):+.1f} SEM)"
      f" | {int(fpos*len(delta))}/{len(delta)} seeds positive")
print("\n  verdict:")
if not comp > 2 * cs:
    print("    => INCONCLUSIVE: structured data gives no controllable WM (no reward coverage) -> add a goal-biased")
    print("       behavior-policy component so CEM has something to find, then re-test.")
elif delta.mean() > 2 * sem(delta) and fpos >= 0.7:
    print("    => CONTROL POSITIVE (verified-signal, multi-seed): support-pessimism improves planning where support")
    print("       is identifiable+relevant. JEPA uncertain-support IS controller-relevant given structured data.")
elif delta.mean() > 2 * sem(delta):
    print("    => POSITIVE but inconsistent across seeds (frac<0.7) -- report with the caveat.")
else:
    print(f"    => NOT SIGNIFICANT at {len(SEEDS)} seeds (delta {delta.mean():+.2f}, {delta.mean()/(sem(delta)+1e-9):+.1f} SEM,"
          f" {int(fpos*len(delta))}/{len(delta)} pos): directional but underpowered -> more seeds, or ship monitor paper.")
