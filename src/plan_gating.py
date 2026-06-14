"""M1.7 -- SELECTIVE CONTROL under shift: does uncertainty-GATING improve control RELIABILITY?

M1.2 found uncertainty as a planning COST PENALTY is null (penalizing uncertain plans != better plans).
M1.6 found uncertainty IS a good MONITOR (selective prediction). M1.7 applies the monitor to control as an
ABSTAIN GATE: run vanilla CEM on a pool of episodes, HALF with noise-corrupted observations (distribution
shift). Per episode record the outcome (best reward) + two confidence signals -- shell deviation (OOD) and
MC-dropout variance of the planned rollout (predictive). Then a risk-coverage curve OVER EPISODES: rank by
confidence, act on the most-confident fraction, measure risk (= -reward) on the kept set, vs oracle (rank by
true reward) and random. WIN = gating beats random and combined covers both axes (shell catches corrupted
episodes, MC-var the in-dist hard ones) -> the monitor improves control RELIABILITY under shift, even though
it never improves nominal planning (M1.2 stands). No retrain. Gating is PER-EPISODE ("decline tasks you'd
fail") -- Push-T has no safe mid-episode action. Spec: docs/M1.7-plan-gating-spec.md.

Run on Colab GPU:  python src/plan_gating.py
"""
import sys
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

FS, HS, HORIZON = 5, 3, 6                                         # frameskip, history, plan horizon (model steps)
S, CEM_ITERS, ELITE, MC = 96, 3, 12, 6
ACTION_SCALE = 2.0                                                # env [-1,1] -> model's z-scored range (M1.2 confound fix)
EPISODES, BUDGET, NOISE_SIGMA = 100, 15, 1.0                      # half corrupted; NOISE_SIGMA=1.0 -> destructive shift (strengthening shot)
COVS = np.linspace(0.1, 1.0, 19)
device = "cuda" if torch.cuda.is_available() else "cpu"
model, cfg = load_lewm("/content/le-wm", device=device)
prep = TT.Compose([TT.ToTensor(), TT.Resize((224, 224)), TT.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
SHELL = cfg["predictor"]["input_dim"] ** 0.5                      # Gaussian-shell norm (~13.9)


def set_drop(b):
    for m in model.predictor.modules():
        if isinstance(m, nn.Dropout):
            m.train(b)


def corrupt(frame, rng):
    """Gaussian pixel noise on the CURRENT observation -> distribution shift the planner fails on."""
    f = frame.astype("float32") + rng.normal(0, NOISE_SIGMA * 255, frame.shape)
    return np.clip(f, 0, 255).astype("uint8")


@torch.no_grad()
def encode_one(pix):                                             # [3,224,224] -> [192]
    return model.encode({"pixels": pix[None, None]})["emb"][0, 0]


@torch.no_grad()
def rollout_final(cur_emb, plans, dropout):
    """cur_emb [192], plans [S,P,10] -> final predicted emb [S,192] (autoregressive, last HS history)."""
    if dropout:
        set_drop(True)
    Sn, P, _ = plans.shape
    emb = cur_emb[None, None].expand(Sn, 1, 192).clone()
    act = model.action_encoder(plans * ACTION_SCALE)
    for t in range(P):
        emb = torch.cat([emb, model.predict(emb[:, -HS:], act[:, :t + 1][:, -HS:])[:, -1:]], dim=1)
    if dropout:
        set_drop(False)
    return emb[:, -1]


@torch.no_grad()
def cem(cur_emb, goal_emb, gen):
    """Vanilla CEM-MPC (no uncertainty in the cost -- that was the M1.2 null). Returns mu [HORIZON,10]."""
    mu = torch.zeros(HORIZON, 10, device=device)
    sigma = torch.full((HORIZON, 10), 0.5, device=device)
    for _ in range(CEM_ITERS):
        noise = torch.randn(S, HORIZON, 10, generator=gen, device=device)
        plans = (mu + sigma * noise).clamp(-1, 1)
        cost = (rollout_final(cur_emb, plans, False) - goal_emb).pow(2).sum(-1)
        elite = plans[cost.argsort()[:ELITE]]
        mu, sigma = elite.mean(0), elite.std(0) + 1e-3
    return mu.clamp(-1, 1)


@torch.no_grad()
def plan_mc_var(cur_emb, mu):
    """MC-dropout variance of the CHOSEN plan's rollout -- the planner's own uncertainty about its forecast."""
    samples = torch.stack([rollout_final(cur_emb, mu[None], True)[0] for _ in range(MC)])   # [MC,192]
    return samples.var(0).sum().item()


# ---- run episodes (half corrupted); record outcome + per-episode confidence signals --------------
recs = []                                                        # (best_reward, shell_mean, mcvar_mean, is_corrupt)
for ep in range(EPISODES):
    corr = (ep % 2 == 1)
    crng = np.random.default_rng(1000 + ep)
    g = torch.Generator(device=device).manual_seed(ep)
    env = gym.make("swm/PushT-v1", render_mode="rgb_array")
    _, info = env.reset(seed=ep)
    goal_emb = encode_one(prep(info["goal"]).to(device))         # goal stays clean
    best_r, shells, mcvars = -1e18, [], []
    for step in range(BUDGET):
        frame = env.render()
        if corr:
            frame = corrupt(frame, crng)                         # current observation corrupted
        cur_emb = encode_one(prep(frame).to(device))
        shells.append(abs(float(cur_emb.norm()) - SHELL))
        mu = cem(cur_emb, goal_emb, g)
        mcvars.append(plan_mc_var(cur_emb, mu))
        mstep = mu[0].cpu().numpy()
        done = False
        for j in range(FS):
            _, r, term, trunc, info = env.step(np.clip(mstep[2 * j:2 * j + 2], -1, 1).astype("float32"))
            best_r = max(best_r, float(r)); done = term or trunc
            if done:
                break
        if done:
            break
    recs.append((best_r, float(np.mean(shells)), float(np.mean(mcvars)), int(corr)))
    env.close()
    if ep % 10 == 0:
        print(f"episode {ep}/{EPISODES}", flush=True)

recs = np.array(recs)
reward, shell_sig, mcvar_sig, corr = recs[:, 0], recs[:, 1], recs[:, 2], recs[:, 3].astype(bool)
risk = -reward                                                   # lower risk = higher reward = better
print(f"\nsanity: reward clean {reward[~corr].mean():.1f} vs corrupted {reward[corr].mean():.1f}"
      f"   | shell clean {shell_sig[~corr].mean():.2f} vs corrupted {shell_sig[corr].mean():.2f}", flush=True)


def zscore(x):
    return (x - x.mean()) / (x.std() + 1e-9)


def risk_cov(sig):                                              # keep most-confident (low sig) coverage c, mean risk
    rk = risk[np.argsort(sig)]
    return np.array([rk[:max(1, int(c * len(rk)))].mean() for c in COVS])


def aurc(sig):
    return float(risk_cov(sig).mean())


sigs = {"shell": shell_sig, "MC-variance": mcvar_sig, "combined": zscore(shell_sig) + zscore(mcvar_sig)}
rand = float(risk.mean())
print("\n==== M1.7 selective control -- AURC (mean risk = -best_reward over coverage, lower=better) ====")
for n, s in sigs.items():
    print(f"  {n:12s}: {aurc(s):8.2f}")
print(f"  {'oracle':12s}: {aurc(risk):8.2f}   (floor)")
print(f"  {'random':12s}: {rand:8.2f}   (= mean risk, no-skill)")
def spearman(a, b):
    ra, rb = a.argsort().argsort().astype(float), b.argsort().argsort().astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


print("\n  signal<->risk coupling (Spearman; >0 = high signal predicts high risk = good gate):")
for n, s in sigs.items():
    print(f"    {n:12s}: {spearman(s, risk):+.3f}")

# bootstrap SEM on % of the random->oracle gap recovered (resample episodes) -> is the win beyond noise?
brng = np.random.default_rng(0); N = len(risk); B = 1000


def recovered(sig, idx):
    r = risk[idx]; sg = sig[idx]; n = len(r)
    rc = np.mean([r[np.argsort(sg)][:max(1, int(c * n))].mean() for c in COVS])
    rd = r.mean(); orc = np.mean([np.sort(r)[:max(1, int(c * n))].mean() for c in COVS])
    return 100 * (rd - rc) / (rd - orc) if rd - orc > 1e-9 else 0.0


print("\n  % of random->oracle gap recovered (bootstrap SEM over episodes):")
verdict = {}
for n, s in sigs.items():
    pt = recovered(s, np.arange(N))
    boot = np.array([recovered(s, brng.integers(0, N, N)) for _ in range(B)])
    verdict[n] = (pt, float(boot.std()))
    print(f"    {n:12s}: {pt:+.0f}% +/- {boot.std():.0f}%")
cp, cs = verdict["combined"]
print("\n  " + ("WIN: gating beats random beyond noise -- uncertainty improves control reliability under shift."
                if cp > 2 * cs and cp > 5 else
                "WEAK/NULL: gating within ~noise -- per-episode confidence is decoupled from episode reward."))

# ---- figure: episode risk-coverage --------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.6, 4.7))
for n, s, c in [("oracle", risk, "#27ae60"), ("shell", sigs["shell"], "#e67e22"),
                ("MC-variance", sigs["MC-variance"], "#2980b9"), ("combined", sigs["combined"], "#8e44ad")]:
    ax.plot(COVS, risk_cov(s), "-o", ms=3, color=c, label=f"{n} (AURC {aurc(s):.0f})")
ax.axhline(rand, ls="--", color="gray", label=f"random ({rand:.0f})")
ax.set_xlabel("coverage (fraction of episodes acted on)"); ax.set_ylabel("risk = -best reward on kept")
ax.set_title("M1.7 -- selective control under shift (lower-left = better gate)"); ax.legend(fontsize=8); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig("/content/lewm-uncertainty/lewm_plan_gating.png", dpi=110)
print("\nsaved lewm_plan_gating.png")
