"""Direction A, stage 2 -- CONTROL under intermittent observation corruption (the headline).

A1 (src/gated_perception.py) showed the free shell signal gates which observations to TRUST, keeping a
near-oracle latent ESTIMATE under corruption. Stage 2 asks the deployment question reviewers care about:
does that translate into better CONTROL?

THE DISTINCTION FROM M1.2 (which was a null). M1.2 gated the planner's COST -- cost = dist-to-goal +
beta*MC-variance -- and found no improvement (calibration != actionability). Here we change NOTHING about the
planner (vanilla beta=0 CEM); we gate the STATE ESTIMATE the planner plans from. Under corruption a BLIND
agent re-encodes every (possibly corrupted) frame -> plans from a poisoned latent; a SHELL-GATED agent coasts
through frames it distrusts (predict forward with the executed action) -> plans from a clean estimate.

Note: the action-free ENSEMBLE would FAIL as this gate -- M2.2 showed it is OOD-blind (heads agree,
confidently wrong, on corrupted inputs). It is specifically the shell/OOD facet that does deployment work.

POLICIES (closed-loop CEM control on swm/PushT-v1):
  clean        -- no corruption (the ceiling/reference).
  blind        -- re-encode every frame (poisoned under corruption). baseline.
  random-gate  -- coast a RANDOM matched subset (isolates: signal vs just-coasting).
  shell-gate   -- coast iff |‖encode(obs)‖-√d| >= tau  (ours; tau label-free from clean in-dist).
  oracle-gate  -- coast iff the frame is truly corrupted (knows the mask). ceiling under corruption.

METRIC: best task reward per episode (PushT coverage, higher=better), mean +/- SEM; success-rate @ thresh.
WIN = shell-gate > blind beyond SEM AND ~= oracle. Spec: docs/A1-gated-perception-spec.md (stage 2 section).

Run on Colab GPU:  python src/gated_control.py
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

FS, HS, HORIZON = 5, 3, 6                                         # frameskip, history window, plan horizon
S, CEM_ITERS, ELITE = 96, 3, 12
ACTION_SCALE = 2.0                                                # env [-1,1] -> model's z-scored range (M1.2)
EPISODES, BUDGET = 15, 12
P_GRID = [0.3, 0.5]                                               # corruption rate
CTYPES = ["noise"]                                               # add "blackout" once positive
NOISE_SIGMA = 0.4
SUCCESS_THRESH = 0.9                                              # PushT coverage counted as a "success"
N_CAL = 15                                                        # clean rollouts to calibrate tau
TAU_Q = 0.95
device = "cuda" if torch.cuda.is_available() else "cpu"
model, cfg = load_lewm("/content/le-wm", device=device)
prep = TT.Compose([TT.ToTensor(), TT.Resize((224, 224)), TT.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
SHELL = cfg["predictor"]["input_dim"] ** 0.5


def corrupt_noise(frame, rng):
    f = frame.astype("float32") + rng.normal(0, NOISE_SIGMA * 255, frame.shape)
    return np.clip(f, 0, 255).astype("uint8")


def corrupt_black(frame, rng):
    return np.zeros_like(frame)


CORRUPT = {"noise": corrupt_noise, "blackout": corrupt_black}


def set_drop(b):
    for m in model.predictor.modules():
        if isinstance(m, nn.Dropout):
            m.train(b)


@torch.no_grad()
def encode_one(frame):                                           # uint8 HWC -> [192]
    return model.encode({"pixels": prep(frame).to(device)[None, None]})["emb"][0, 0]


@torch.no_grad()
def act_emb_one(mstep):                                          # [10] env action -> [192] (scaled like the planner)
    a = torch.tensor(mstep, dtype=torch.float32, device=device) * ACTION_SCALE
    return model.action_encoder(a[None, None])[0, 0]            # [1,1,10] -> [1,1,192] -> [192]


@torch.no_grad()
def predict_step(Z, A):                                          # coast: next latent from last-HS (states, actions)
    e = torch.stack(Z[-HS:])[None]; a = torch.stack(A[-HS:])[None]
    return model.predict(e, a)[0, -1]


@torch.no_grad()
def rollout_final(cur_emb, plans):                              # cur_emb [192], plans [S,P,10] -> final emb [S,192]
    Sn, P, _ = plans.shape
    emb = cur_emb[None, None].expand(Sn, 1, 192).clone()
    act = model.action_encoder(plans * ACTION_SCALE)
    for t in range(P):
        emb = torch.cat([emb, model.predict(emb[:, -HS:], act[:, :t + 1][:, -HS:])[:, -1:]], dim=1)
    return emb[:, -1]


@torch.no_grad()
def cem(cur_emb, goal_emb, gen):                                # vanilla beta=0 CEM-MPC; returns [HORIZON,10]
    mu = torch.zeros(HORIZON, 10, device=device)
    sigma = torch.full((HORIZON, 10), 0.5, device=device)
    for _ in range(CEM_ITERS):
        plans = (mu + sigma * torch.randn(S, HORIZON, 10, generator=gen, device=device)).clamp(-1, 1)
        cost = (rollout_final(cur_emb, plans) - goal_emb).pow(2).sum(-1)
        elite = plans[cost.argsort()[:ELITE]]
        mu, sigma = elite.mean(0), elite.std(0) + 1e-3
    return mu.clamp(-1, 1)


def trust_decision(policy, dev, is_clean, coast_budget_rng, want_coast_frac):
    """Return True to TRUST (adopt obs), False to COAST. random-gate uses a Bernoulli matched in expectation."""
    if policy == "blind" or policy == "clean":
        return True
    if policy == "oracle-gate":
        return is_clean
    if policy == "shell-gate":
        return dev < TAU
    if policy == "random-gate":
        return coast_budget_rng.random() >= want_coast_frac     # coast with prob ~ corruption rate (matched)
    raise ValueError(policy)


def run_arm(policy, ctype, p, want_coast_frac):
    g = torch.Generator(device=device).manual_seed(0)
    crng = np.random.default_rng(2000 + int(p * 100) + (0 if ctype == "noise" else 7))
    rrng = np.random.default_rng(7)
    best_rewards = []
    for ep in range(EPISODES):
        env = gym.make("swm/PushT-v1", render_mode="rgb_array")
        _, info = env.reset(seed=ep)
        goal_emb = encode_one(info["goal"])
        z_hat = encode_one(env.render())                        # clean initial estimate
        Z, A = [z_hat], []
        best_r = -1e18
        for step in range(BUDGET):
            mstep = cem(z_hat, goal_emb, g)[0].cpu().numpy()    # first model-step (5 env actions)
            done = False
            for j in range(FS):
                _, r, term, trunc, info = env.step(np.clip(mstep[2 * j:2 * j + 2], -1, 1).astype("float32"))
                best_r = max(best_r, float(r)); done = term or trunc
                if done:
                    break
            A.append(act_emb_one(mstep))
            frame = env.render()
            is_clean = True if policy == "clean" else (crng.random() >= p)
            obs = frame if is_clean else CORRUPT[ctype](frame, crng)
            e_obs = encode_one(obs); dev = abs(float(e_obs.norm()) - SHELL)
            if trust_decision(policy, dev, is_clean, rrng, want_coast_frac):
                z_hat = e_obs
            else:
                z_hat = predict_step(Z, A)                       # coast through the distrusted frame
            Z.append(z_hat)
            if done:
                break
        best_rewards.append(best_r); env.close()
    return np.array(best_rewards)


def sem(a):
    return float(np.std(a) / np.sqrt(len(a)))


# ---- calibrate tau label-free from clean in-dist shell-devs --------------------------------------
gen = np.random.default_rng(0); cal = []
for r in range(N_CAL):
    env = gym.make("swm/PushT-v1", render_mode="rgb_array"); env.reset(seed=10_000 + r)
    cal.append(abs(float(encode_one(env.render()).norm()) - SHELL))
    for _ in range(BUDGET):
        env.step(env.action_space.sample().astype("float32"))
        cal.append(abs(float(encode_one(env.render()).norm()) - SHELL))
    env.close()
TAU = float(np.quantile(cal, TAU_Q))
print(f"clean shell-dev mean {np.mean(cal):.3f}; tau = q{int(TAU_Q*100)} = {TAU:.3f}\n", flush=True)

# ---- clean reference (no corruption) -------------------------------------------------------------
print("==== A2 gated CONTROL under corruption -- best reward/episode (higher=better) ====", flush=True)
clean = run_arm("clean", "noise", 0.0, 0.0)
print(f"  clean (no corruption): {clean.mean():.3f} +/- {sem(clean)}  succ {np.mean(clean>SUCCESS_THRESH):.2f}", flush=True)

# ---- corrupted: blind / random / shell / oracle --------------------------------------------------
res = {("clean", 0.0, "noise"): clean}
for ctype in CTYPES:
    for p in P_GRID:
        print(f"\n  --- {ctype}  p={p} ---", flush=True)
        for pol in ["blind", "random-gate", "shell-gate", "oracle-gate"]:
            rew = run_arm(pol, ctype, p, want_coast_frac=p)
            res[(pol, p, ctype)] = rew
            print(f"    {pol:12}: {rew.mean():.3f} +/- {sem(rew):.3f}  succ {np.mean(rew>SUCCESS_THRESH):.2f}", flush=True)

# ---- verdict -------------------------------------------------------------------------------------
print("\n==== verdict (does gating the STATE ESTIMATE recover control under shift?) ====")
cref = clean.mean()
allwin = True
for ctype in CTYPES:
    for p in P_GRID:
        b, rnd = res[("blind", p, ctype)], res[("random-gate", p, ctype)]
        sh, orc = res[("shell-gate", p, ctype)], res[("oracle-gate", p, ctype)]
        d_blind = sh.mean() - b.mean(); s_blind = np.hypot(sem(sh), sem(b))     # >0: shell beats blind
        d_orc = orc.mean() - sh.mean(); s_orc = np.hypot(sem(orc), sem(sh))     # ~0: shell ~ oracle
        drop = cref - b.mean(); recov = (sh.mean() - b.mean()) / (drop + 1e-9)  # frac of corruption drop recovered
        win = d_blind > s_blind and d_orc < 2 * s_orc
        allwin &= win
        print(f"  [{ctype} p={p}] shell-vs-blind {d_blind:+.3f}+/-{s_blind:.3f} | shell-vs-oracle {d_orc:+.3f}"
              f"+/-{s_orc:.3f} | recovers {recov*100:.0f}% of the {drop:+.3f} corruption drop  => {'WIN' if win else 'no'}")
print("\n  => " + ("POSITIVE: gating the state estimate on the free shell signal recovers control under shift"
                    " (blind cost-shaping was M1.2's null; gating WHICH OBSERVATIONS the planner trusts works)."
                    if allwin else
                    "MIXED/NULL: see rows -- shell-gate does not clearly beat blind and/or trails oracle."))

# ---- figure --------------------------------------------------------------------------------------
fig, ax = plt.subplots(1, len(CTYPES), figsize=(6 * len(CTYPES), 4.6), squeeze=False)
cols = {"blind": "#c0392b", "random-gate": "#bdc3c7", "shell-gate": "#2980b9", "oracle-gate": "#27ae60"}
for axi, ctype in zip(ax[0], CTYPES):
    axi.axhline(cref, ls="--", color="#34495e", label=f"clean ({cref:.2f})")
    for pol in ["blind", "random-gate", "shell-gate", "oracle-gate"]:
        m = [res[(pol, p, ctype)].mean() for p in P_GRID]; s = [sem(res[(pol, p, ctype)]) for p in P_GRID]
        axi.errorbar(P_GRID, m, yerr=s, fmt="-o", capsize=3, color=cols[pol], label=pol)
    axi.set_xlabel("corruption rate p"); axi.set_ylabel("best reward / episode"); axi.set_title(f"corruption: {ctype}")
    axi.grid(alpha=.3); axi.legend(fontsize=8)
fig.suptitle("A2 -- control under shift: gate the planner's STATE ESTIMATE on the free shell signal",
             fontweight="bold")
fig.tight_layout(); fig.savefig("/content/lewm-uncertainty/lewm_gated_control.png", dpi=110)
print("\nsaved lewm_gated_control.png")
