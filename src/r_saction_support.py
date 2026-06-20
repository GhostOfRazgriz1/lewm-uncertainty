"""#5 -- STATE-ACTION support pessimism (the cheap probe before CUHL).

Our DPP null penalized STATE-shell support (|‖z‖-√d| / Mahalanobis on z). But offline-control risk lives in
unsupported (z,a) PAIRS -- actions the data never took in a given state. Hypothesis: state-action support is
the right risk variable; pessimism on it should help at LIMITED data where state-shell pessimism did not.

Test (paired, Pusher): vanilla CEM vs SHELL-pessimism (state Mahalanobis, the DPP that nulled) vs JOINT-
pessimism (learned (z,a) density-ratio penalty), at limited (N=25) and ample (N=70) data, 5 seeds.
WIN = joint paired-delta > 2 SEM at LOW data AND > shell's (i.e., (z,a) support helps where z-support didn't).

Density ratio: a classifier g(z,a) trained to tell real (z_t,a_t) from (z_t, shuffled-a); logit ~ log
p(a|z)/p(a). U_joint(z,a) = -g(z,a) is high off the (z,a) support. Penalize sum_k U over the imagined rollout.
Run on Colab GPU (pip install 'gymnasium[mujoco]' opencv-python-headless):  python src/r_saction_support.py
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
N_POOL, ENC_EPOCHS, ENS_EPOCHS, CLF_EPOCHS, BS, KSTEP = 70, 30, 50, 40, 64, 8
N_SWEEP, SEEDS, KAP = [25, 70], [0, 1, 2, 3, 4], 3.0
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
        self.net = nn.Sequential(nn.Linear(D + adim, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, D))

    def forward(self, z, a):
        return z + self.net(torch.cat([z, a], -1))


class RewardHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(D, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, z):
        return self.net(z).squeeze(-1)


class Classifier(nn.Module):                                     # density-ratio: real (z,a) vs (z, shuffled-a)
    def __init__(self, adim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(D + adim, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, z, a):
        return self.net(torch.cat([z, a], -1)).squeeze(-1)        # logit; high = in-support (z,a)


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


def zsc(x):
    return (x - x.mean()) / (x.std() + 1e-9)


def train_base(pool, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    enc, rew, p0 = Encoder().to(device), RewardHead().to(device), Predictor(adim).to(device)
    opt = torch.optim.Adam(list(enc.parameters()) + list(rew.parameters()) + list(p0.parameters()), lr=3e-4)
    st = [(e, t) for e in range(len(pool)) for t in range(len(pool[e][1]) - KSTEP)]
    for epoch in range(ENC_EPOCHS):
        np.random.shuffle(st)
        for i in range(0, len(st), BS):
            b = st[i:i + BS]
            fr = to_t([pool[e][0][t + k] for e, t in b for k in range(KSTEP + 1)]).view(len(b), KSTEP + 1, 3, IMG, IMG)
            ac = torch.tensor(np.stack([pool[e][1][t:t + KSTEP] for e, t in b]), device=device)
            rw_t = torch.tensor(np.stack([pool[e][2][t:t + KSTEP] for e, t in b]), device=device)
            zt = enc(fr.view(-1, 3, IMG, IMG)).view(len(b), KSTEP + 1, D)
            z = zt[:, 0]; lp = lr = 0.0
            for k in range(KSTEP):
                z = p0(z, ac[:, k]); lp = lp + ((z - zt[:, k + 1]) ** 2).mean(); lr = lr + ((rew(zt[:, k + 1]) - rw_t[:, k]) ** 2).mean()
            (lp / KSTEP + lr / KSTEP + 0.5 * vicreg(zt[:, 0])).backward(); opt.step(); opt.zero_grad()
    enc.eval(); rew.eval()
    for p in list(enc.parameters()) + list(rew.parameters()):
        p.requires_grad_(False)
    return enc, rew


@torch.no_grad()
def encode_pool(enc, pool):
    Z, A = [], []
    for fr, ac, _ in pool:
        Z.append(torch.cat([enc(to_t(fr[i:i + 64])) for i in range(0, len(fr), 64)])); A.append(torch.tensor(ac, device=device))
    return Z, A


def train_ensemble(Z, A, seed):                                 # for the rollout mean (predictor)
    torch.manual_seed(seed)
    members = nn.ModuleList([Predictor(adim) for _ in range(M)]).to(device)
    opt = torch.optim.Adam(members.parameters(), lr=1e-3)
    idx = [(e, t) for e in range(len(Z)) for t in range(len(A[e]) - KSTEP)]
    for epoch in range(ENS_EPOCHS):
        np.random.shuffle(idx)
        for i in range(0, len(idx), BS):
            b = idx[i:i + BS]
            z0 = torch.stack([Z[e][t] for e, t in b]); acts = torch.stack([A[e][t:t + KSTEP] for e, t in b])
            tgt = torch.stack([Z[e][t + 1:t + KSTEP + 1] for e, t in b])
            loss = 0.0
            for p in members:
                z = z0
                for k in range(KSTEP):
                    z = p(z, acts[:, k]); loss = loss + ((z - tgt[:, k]) ** 2).mean()
            (loss / (M * KSTEP)).backward(); opt.step(); opt.zero_grad()
    for p in members:
        p.eval()
    return members


def train_classifier(Z, A, seed):                               # density ratio g(z,a): real vs (z, shuffled-a)
    torch.manual_seed(seed + 99)
    clf = Classifier(adim).to(device); opt = torch.optim.Adam(clf.parameters(), lr=1e-3)
    zc = torch.cat([Z[e][:-1] for e in range(len(Z))]); ac = torch.cat([A[e] for e in range(len(A))])   # aligned (z_t,a_t)
    bce = nn.BCEWithLogitsLoss()
    for epoch in range(CLF_EPOCHS):
        perm = torch.randperm(len(zc), device=device)
        for i in range(0, len(zc), BS):
            j = perm[i:i + BS]; z = zc[j]; a = ac[j]
            aneg = a[torch.randperm(len(a), device=device)]                      # shuffle actions -> negatives
            logit = torch.cat([clf(z, a), clf(z, aneg)])
            lab = torch.cat([torch.ones(len(z), device=device), torch.zeros(len(z), device=device)])
            bce(logit, lab).backward(); opt.step(); opt.zero_grad()
    clf.eval()
    return clf


@torch.no_grad()
def cem_action(enc, members, rew, clf, mu_t, sig_t, frame, kappa, mode, gen):
    z0 = enc(to_t([frame]))[0]
    mu = torch.zeros(H_PLAN, adim, device=device); sig = torch.ones(H_PLAN, adim, device=device)
    for _ in range(CEM_ITERS):
        plans = (mu + sig * torch.randn(S_CEM, H_PLAN, adim, generator=gen, device=device)).clamp(-1, 1)
        z = z0[None].expand(S_CEM, D).clone(); R = torch.zeros(S_CEM, device=device); U = torch.zeros(S_CEM, device=device)
        for k in range(H_PLAN):
            a = plans[:, k]
            znext = torch.stack([p(z, a) for p in members]).mean(0)
            if mode == "shell":
                U = U + (((znext - mu_t) / sig_t) ** 2).mean(-1)                  # STATE support (DPP that nulled)
            elif mode == "joint":
                U = U + (-clf(z, a))                                             # STATE-ACTION support (this is (z_t,a_t))
            z = znext; R = R + rew(z)
        score = zsc(R) if kappa == 0 else zsc(R) - kappa * zsc(U)
        elite = plans[score.argsort(descending=True)[:ELITE]]
        mu, sig = elite.mean(0), elite.std(0) + 1e-3
    return mu[0].clamp(-1, 1).cpu().numpy()


@torch.no_grad()
def eval_return(enc, members, rew, clf, mu_t, sig_t, kappa, mode):
    g = torch.Generator(device=device).manual_seed(0); rets = []
    for ep in range(EVAL_EP):
        env = gym.make(ENV_ID, render_mode="rgb_array"); env.reset(seed=40_000 + ep)
        R = 0.0; done = False
        while not done:
            a = cem_action(enc, members, rew, clf, mu_t, sig_t, render84(env), kappa, mode, g)
            _, r, term, trunc, _ = env.step(a.astype("float32")); R += float(r); done = term or trunc
        rets.append(R); env.close()
    return np.array(rets)


# ---- run -----------------------------------------------------------------------------------------
env = gym.make(ENV_ID, render_mode="rgb_array"); env.reset(seed=0); adim = env.action_space.shape[0]; env.close()
print(f"=== {ENV_ID} adim {adim} ===  collecting {N_POOL} eps ...", flush=True)
POOL = collect(N_POOL, 0)
PLANNERS = [("vanilla", 0.0, "shell"), ("shell-pess", KAP, "shell"), ("joint-pess", KAP, "joint")]
res = {}
for N in N_SWEEP:
    print(f"\n--- N={N} ---", flush=True)
    for name, _, _ in PLANNERS:
        res[(N, name)] = []
    for sd in SEEDS:
        enc, rew = train_base(POOL[:N], sd)
        Z, A = encode_pool(enc, POOL[:N])
        allz = torch.cat(list(Z)); mu_t = allz.mean(0); sig_t = allz.std(0) + 1e-6
        members = train_ensemble(Z, A, sd); clf = train_classifier(Z, A, sd)
        for name, kp, mode in PLANNERS:
            res[(N, name)].append(eval_return(enc, members, rew, clf, mu_t, sig_t, kp, mode).mean())
        print(f"  seed {sd}: " + " | ".join(f"{nm} {res[(N,nm)][-1]:.1f}" for nm, _, _ in PLANNERS), flush=True)
    for name, _, _ in PLANNERS:
        res[(N, name)] = np.array(res[(N, name)])

# ---- report (PAIRED) -----------------------------------------------------------------------------
print("\n==== #5 state-action vs state-shell support pessimism on Pusher (PAIRED delta vs vanilla) ====")
for N in N_SWEEP:
    van = res[(N, "vanilla")]
    print(f"  N={N:2d}: vanilla {van.mean():.1f}+/-{sem(van):.1f}")
    for name in ("shell-pess", "joint-pess"):
        d = res[(N, name)] - van                                                 # paired per-seed delta
        print(f"        {name:11}: paired delta {d.mean():+.2f} +/- {sem(d):.2f}  ({d.mean()/(sem(d)+1e-9):+.1f} SEM)")

lo = N_SWEEP[0]
dj = res[(lo, "joint-pess")] - res[(lo, "vanilla")]; ds = res[(lo, "shell-pess")] - res[(lo, "vanilla")]
print("\n  verdict:")
if dj.mean() > 2 * sem(dj) and dj.mean() > ds.mean():
    print("    => POSITIVE: STATE-ACTION support pessimism helps at limited data (>2 SEM) and beats state-shell")
    print("       -- the support variable was the issue; control risk is in (z,a), not z. PROCEED to CUHL (#1).")
elif dj.mean() > 2 * sem(dj):
    print("    => POSITIVE (but ~= shell): joint helps but not clearly more than shell.")
else:
    print("    => NULL: state-action support pessimism also doesn't help -> support-pessimism is genuinely dead here;")
    print("       CUHL's value would have to come from the HIERARCHY + action-free subgoal scoring, not (z,a) pessimism.")
