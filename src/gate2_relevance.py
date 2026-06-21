"""GATE 2 -- is the (z,a) support score RELEVANT to model risk? (v2: INLINE branching; v1 set_state was unfaithful)

Gate 1 OPEN: structured data makes (z,a) support identifiable (AUROC 0.85 vs random 0.50). Gate 2: does the
support score predict WHERE THE MODEL IS WRONG?  corr( U(z,a)=-g(z,a) , true 1-step model error ) > 0 ?

v1 was INVALID: resetting a separate env (seed=0) to states from other episodes didn't restore Pusher's goal
(round-trip ||dz|| ~13). v2 fixes it by BRANCHING WITHIN THE SAME EPISODE'S ENV: roll the structured policy;
at probe steps, snapshot (qpos,qvel), try the data action + several off-support (random) actions via
set_state->step (same env/goal, so the reset is faithful), then restore and continue. Reports a round-trip
sanity (set_state with no step should re-render to ~the same latent).

PASS (corr>0 beyond SEM; off-support actions have higher U AND higher error) => support relevant to model
risk => both gates pass => proceed to the structured-offline control test. FAIL (corr~0) => identifiable but
irrelevant => ship the monitor paper.
Run on Colab GPU (pip install 'gymnasium[mujoco]' opencv-python-headless):  python src/gate2_relevance.py
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
SEEDS, N_EVAL_EP, PROBE_EVERY, N_OFF = [0, 1, 2], 4, 10, 5
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
            a = policy(obs).astype("float32"); obs, r, term, trunc, info = env.step(a)
            fr.append(render84(env)); ac.append(a); done = term or trunc
        eps.append((np.stack(fr), np.stack(ac))); env.close()
    return eps


@torch.no_grad()
def encode_one(enc, frame):
    return enc(to_t([frame]))[0]


def train_encoder(pool, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    enc, p0 = Encoder().to(device), Predictor(adim).to(device)
    opt = torch.optim.Adam(list(enc.parameters()) + list(p0.parameters()), lr=3e-4)
    stx = [(e, t) for e in range(len(pool)) for t in range(len(pool[e][1]) - KSTEP)]
    for epoch in range(ENC_EPOCHS):
        np.random.shuffle(stx)
        for i in range(0, len(stx), BS):
            b = stx[i:i + BS]
            fr = to_t([pool[e][0][t + k] for e, t in b for k in range(KSTEP + 1)]).view(len(b), KSTEP + 1, 3, IMG, IMG)
            ac = torch.tensor(np.stack([pool[e][1][t:t + KSTEP] for e, t in b]), device=device)
            zt = enc(fr.view(-1, 3, IMG, IMG)).view(len(b), KSTEP + 1, D)
            z = zt[:, 0]; lp = 0.0
            for k in range(KSTEP):
                z = p0(z, ac[:, k]); lp = lp + ((z - zt[:, k + 1]) ** 2).mean()
            (lp / KSTEP + 0.5 * vicreg(zt[:, 0])).backward(); opt.step(); opt.zero_grad()
    enc.eval(); p0.eval()
    return enc, p0


def train_classifier(pool, enc, seed):
    torch.manual_seed(seed + 99)
    with torch.no_grad():
        Z = torch.cat([torch.cat([enc(to_t(fr[i:i + 64])) for i in range(0, len(fr), 64)])[:-1] for fr, _ in pool])
        A = torch.cat([torch.tensor(ac, device=device) for _, ac in pool])
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


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float); ry = np.argsort(np.argsort(y)).astype(float)
    rx = (rx - rx.mean()) / (rx.std() + 1e-9); ry = (ry - ry.mean()) / (ry.std() + 1e-9)
    return float((rx * ry).mean())


@torch.no_grad()
def gate2_eval(enc, p0, clf, pol, seed):                        # INLINE branching: same env/goal -> faithful set_state
    rng = np.random.default_rng(seed + 7); env = gym.make(ENV_ID, render_mode="rgb_array")
    Us, Es, kinds, rts = [], [], [], []
    for ep in range(N_EVAL_EP):
        obs, info = env.reset(seed=9000 + ep); done = False; step = 0
        while not done:
            a_taken = pol(obs).astype("float32")
            if step % PROBE_EVERY == 0:
                z_t = encode_one(enc, render84(env))
                qpos, qvel = env.unwrapped.data.qpos.copy(), env.unwrapped.data.qvel.copy()
                env.unwrapped.set_state(qpos, qvel)                              # no-step round-trip sanity
                rts.append(float((encode_one(enc, render84(env)) - z_t).norm()))
                cands = [("data", a_taken)] + [("off", rng.uniform(-1, 1, adim).astype("float32")) for _ in range(N_OFF)]
                for kind, a in cands:
                    env.unwrapped.set_state(qpos, qvel); env.step(a)
                    z_next = encode_one(enc, render84(env)); at = torch.tensor(a, dtype=torch.float32, device=device)
                    Us.append(float(-clf(z_t[None], at[None])[0]))
                    Es.append(float((p0(z_t[None], at[None])[0] - z_next).norm())); kinds.append(kind)
                env.unwrapped.set_state(qpos, qvel)                              # restore to continue the trajectory
            obs, r, term, trunc, info = env.step(a_taken); done = term or trunc; step += 1
    env.close()
    Us, Es, kinds = np.array(Us), np.array(Es), np.array(kinds); d = kinds == "data"
    return spearman(Us, Es), float(np.mean(rts)), (Us[d].mean(), Es[d].mean()), (Us[~d].mean(), Es[~d].mean())


def sem(a):
    return float(np.std(a) / np.sqrt(len(a)))


# ---- env + obs standardization (structured policy, same as Gate 1) -------------------------------
env = gym.make(ENV_ID, render_mode="rgb_array"); obs0, _ = env.reset(seed=0); adim = env.action_space.shape[0]
obs_dim = int(np.asarray(obs0).ravel().shape[0]); obss = []
for _ in range(500):
    o, _, term, trunc, _ = env.step(env.action_space.sample().astype("float32")); obss.append(np.asarray(o).ravel())
    if term or trunc:
        env.reset()
env.close()
OM, OS = np.stack(obss).mean(0), np.stack(obss).std(0) + 1e-6
print(f"=== {ENV_ID} adim {adim} obs_dim {obs_dim} ===", flush=True)


def structured_policy(seed):
    rng = np.random.default_rng(seed); W = rng.normal(0, 1.0 / np.sqrt(obs_dim), (adim, obs_dim))
    nz = np.random.default_rng(seed + 1)
    return lambda obs: np.clip(W @ ((np.asarray(obs).ravel() - OM) / OS) + 0.5 * nz.standard_normal(adim), -1, 1)


# ---- run -----------------------------------------------------------------------------------------
corrs, rts = [], []
for sd in SEEDS:
    pol = structured_policy(sd)
    pool = collect(N_DATA, sd, pol)
    enc, p0 = train_encoder(pool, sd); clf = train_classifier(pool, enc, sd)
    corr, rt, (ud, ed), (uo, eo) = gate2_eval(enc, p0, clf, structured_policy(sd), sd)
    corrs.append(corr); rts.append(rt)
    print(f"  seed {sd}: corr(U,err) {corr:+.3f} | data(U {ud:+.2f},e {ed:.2f}) off(U {uo:+.2f},e {eo:.2f})"
          f" | round-trip ||dz|| {rt:.3f}", flush=True)
corrs = np.array(corrs)

print("\n==== Gate 2: is (z,a) support relevant to model error? ====")
print(f"  Spearman corr(support U, true 1-step model error): {corrs.mean():+.3f} +/- {sem(corrs):.3f}")
print(f"  reset round-trip ||dz|| (must be small for validity): {np.mean(rts):.3f}")
print("\n  verdict:")
if np.mean(rts) > 2.0:
    print("    => INVALID: set_state round-trip still large -> even inline reset unfaithful; need a different eval.")
elif corrs.mean() > 2 * sem(corrs) and corrs.mean() > 0.1:
    print("    => PASS: off-support (z,a) reliably has higher model error -> support is RELEVANT to model risk.")
    print("       Both gates pass -> proceed to the structured-offline pessimistic-control test.")
else:
    print("    => FAIL: support is identifiable (Gate 1) but NOT predictive of where the model errs -> pessimism")
    print("       has no real target. Ship the monitor paper.")
