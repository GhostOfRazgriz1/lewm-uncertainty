"""Direction A, stage 2 (BURST) -- CONTROL through sustained sensor OUTAGES (the headline).

A2-iid (per-frame corruption) was a DECOUPLING null: corruption wrecks the latent estimate (A1) but MPC
re-plans from a fresh frame every step, so transient poisoning is self-corrected and control is unmoved.
The regime where estimate quality BINDS is a SUSTAINED outage: a burst of K consecutive corrupted/missing
frames during which there is no per-step correction. Then a BLIND agent plans from garbage for the whole
window, while a SHELL-GATED agent DETECTS the outage (shell signal off the Gaussian shell) and rides it out
by COASTING on the world model's dynamics (predict forward with the actions it executes) -- exactly what a
world model is for. The deployment story: survive sensor dropout by trusting the model, not the dead sensor.

THE FOIL TO M1.2: we change NOTHING about the planner (vanilla beta=0 CEM); we gate the STATE ESTIMATE it
plans from. M1.2 gated the planner's COST (+beta*variance) -> null.

POLICIES:
  clean         -- CEM, no outage (ceiling).
  random-action -- no planning (FLOOR): confirms the planner controls AND the metric/episodes resolve it.
  blind         -- CEM, re-encode every frame incl. the K-step blackout (poisoned through the outage).
  random-gate   -- coast a contiguous K-block at the WRONG location (controls: detecting the RIGHT window
                   vs just coasting for K steps somewhere).
  shell-gate    -- coast iff |‖encode(obs)‖-√d| >= tau  (ours; tau label-free from clean in-dist).
  oracle-gate   -- coast exactly during the true outage (ceiling detector).

METRIC: MEAN reward / episode (sensitive to sustained degradation; best-of-episode hid the iid effect),
also final + best. WIN = shell-gate > blind beyond SEM AND ~= oracle, AND clean >> random-action (planner
controls). Spec: docs/A1-gated-perception-spec.md (stage 2). Run on Colab GPU:  python src/gated_control.py
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

FS, HS, HORIZON = 5, 3, 6
S, CEM_ITERS, ELITE = 96, 3, 12
ACTION_SCALE = 2.0
EPISODES, BUDGET, WARMUP = 24, 16, 3
OUTAGE_LENS = [4, 8]                                              # consecutive blacked-out steps
N_CAL, TAU_Q = 15, 0.95
device = "cuda" if torch.cuda.is_available() else "cpu"
model, cfg = load_lewm("/content/le-wm", device=device)
prep = TT.Compose([TT.ToTensor(), TT.Resize((224, 224)), TT.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
SHELL = cfg["predictor"]["input_dim"] ** 0.5


def blackout(frame):                                             # sensor dropout: no signal -> ‖emb‖ collapses
    return np.zeros_like(frame)


def set_drop(b):
    for m in model.predictor.modules():
        if isinstance(m, nn.Dropout):
            m.train(b)


@torch.no_grad()
def encode_one(frame):
    return model.encode({"pixels": prep(frame).to(device)[None, None]})["emb"][0, 0]


@torch.no_grad()
def act_emb_one(mstep):
    a = torch.tensor(mstep, dtype=torch.float32, device=device) * ACTION_SCALE
    return model.action_encoder(a[None, None])[0, 0]


@torch.no_grad()
def predict_step(Z, A):                                          # coast: next latent from last-HS states+actions
    return model.predict(torch.stack(Z[-HS:])[None], torch.stack(A[-HS:])[None])[0, -1]


@torch.no_grad()
def rollout_final(cur_emb, plans):
    Sn, P, _ = plans.shape
    emb = cur_emb[None, None].expand(Sn, 1, 192).clone()
    act = model.action_encoder(plans * ACTION_SCALE)
    for t in range(P):
        emb = torch.cat([emb, model.predict(emb[:, -HS:], act[:, :t + 1][:, -HS:])[:, -1:]], dim=1)
    return emb[:, -1]


@torch.no_grad()
def cem(cur_emb, goal_emb, gen):
    mu = torch.zeros(HORIZON, 10, device=device)
    sigma = torch.full((HORIZON, 10), 0.5, device=device)
    for _ in range(CEM_ITERS):
        plans = (mu + sigma * torch.randn(S, HORIZON, 10, generator=gen, device=device)).clamp(-1, 1)
        cost = (rollout_final(cur_emb, plans) - goal_emb).pow(2).sum(-1)
        elite = plans[cost.argsort()[:ELITE]]
        mu, sigma = elite.mean(0), elite.std(0) + 1e-3
    return mu.clamp(-1, 1)


def outage_block(K, rng):                                        # contiguous True-block of length K -> (mask, start)
    m = np.zeros(BUDGET, bool)
    if K <= 0:
        return m, -1
    start = int(rng.integers(WARMUP, max(WARMUP + 1, BUDGET - K + 1)))
    m[start:start + K] = True
    return m, start


def wrong_block(K, true_start, rng):                            # contiguous K-block NOT at the true outage start
    m = np.zeros(BUDGET, bool)
    cands = [s for s in range(WARMUP, BUDGET - K + 1) if s != true_start]
    start = int(rng.choice(cands)) if cands else WARMUP
    m[start:start + K] = True
    return m


def run_arm(policy, K):
    g = torch.Generator(device=device).manual_seed(0)
    orng = np.random.default_rng(3000 + K)                       # outage placement (shared across policies via fixed seed)
    arng = np.random.default_rng(123)                            # random-action noise
    means, finals, bests = [], [], []
    for ep in range(EPISODES):
        outage, ostart = outage_block(0 if policy in ("clean", "random-action") else K, orng)
        rg = wrong_block(K, ostart, np.random.default_rng(9000 + ep)) if policy == "random-gate" else None
        env = gym.make("swm/PushT-v1", render_mode="rgb_array")
        _, info = env.reset(seed=ep)
        goal_emb = encode_one(info["goal"])
        z_hat = encode_one(env.render())
        Z, A = [z_hat], []
        ep_r = []
        for step in range(BUDGET):
            if policy == "random-action":
                mstep = arng.uniform(-1, 1, 10).astype("float32")
            else:
                mstep = cem(z_hat, goal_emb, g)[0].cpu().numpy()
            done = False
            for j in range(FS):
                _, r, term, trunc, info = env.step(np.clip(mstep[2 * j:2 * j + 2], -1, 1).astype("float32"))
                ep_r.append(float(r)); done = term or trunc
                if done:
                    break
            if policy != "random-action":                       # maintain the gated state estimate
                A.append(act_emb_one(mstep))
                frame = env.render()
                is_out = bool(outage[step])
                e_obs = encode_one(blackout(frame) if is_out else frame)
                dev = abs(float(e_obs.norm()) - SHELL)
                trust = {"clean": True, "blind": True,
                         "oracle-gate": not is_out,
                         "shell-gate": dev < TAU,
                         "random-gate": (rg is None or not bool(rg[step]))}[policy]
                z_hat = e_obs if trust else predict_step(Z, A)
                Z.append(z_hat)
            if done:
                break
        means.append(np.mean(ep_r)); finals.append(ep_r[-1]); bests.append(np.max(ep_r))
        env.close()
    return {"mean": np.array(means), "final": np.array(finals), "best": np.array(bests)}


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

# ---- references: clean ceiling + random-action floor ---------------------------------------------
print(f"==== A2-burst gated CONTROL through sensor outages ({EPISODES} eps, BUDGET {BUDGET}) ====", flush=True)
print("  metric = MEAN reward/episode (higher=better)\n", flush=True)
clean = run_arm("clean", 0); floor = run_arm("random-action", 0)
print(f"  clean (no outage, CEM)   : {clean['mean'].mean():.2f} +/- {sem(clean['mean']):.2f}")
print(f"  random-action (FLOOR)    : {floor['mean'].mean():.2f} +/- {sem(floor['mean']):.2f}")
controls = clean['mean'].mean() - floor['mean'].mean(); c_sem = np.hypot(sem(clean['mean']), sem(floor['mean']))
print(f"  -> planner control margin: {controls:+.2f} +/- {c_sem:.2f}  ({'CONTROLS' if controls > 2*c_sem else 'WEAK -- control conversion unlikely on PushT'})\n", flush=True)

# ---- outage sweep --------------------------------------------------------------------------------
res = {"clean": clean, "random-action": floor}
for K in OUTAGE_LENS:
    print(f"  --- outage length K={K} ({K}/{BUDGET} steps blacked out) ---", flush=True)
    for pol in ["blind", "random-gate", "shell-gate", "oracle-gate"]:
        res[(pol, K)] = run_arm(pol, K)
        m = res[(pol, K)]["mean"]
        print(f"    {pol:12}: {m.mean():.2f} +/- {sem(m):.2f}", flush=True)

# ---- verdict -------------------------------------------------------------------------------------
print("\n==== verdict (does outage-gating the STATE ESTIMATE recover control?) ====")
cmean = clean["mean"].mean()
allwin = True
for K in OUTAGE_LENS:
    b, rnd = res[("blind", K)]["mean"], res[("random-gate", K)]["mean"]
    sh, orc = res[("shell-gate", K)]["mean"], res[("oracle-gate", K)]["mean"]
    d_blind = sh.mean() - b.mean(); s_blind = np.hypot(sem(sh), sem(b))
    d_rand = sh.mean() - rnd.mean(); s_rand = np.hypot(sem(sh), sem(rnd))
    d_orc = orc.mean() - sh.mean(); s_orc = np.hypot(sem(orc), sem(sh))
    drop = cmean - b.mean(); recov = (sh.mean() - b.mean()) / (drop + 1e-9)
    win = d_blind > s_blind and d_orc < 2 * s_orc and controls > 2 * c_sem
    allwin &= win
    print(f"  [K={K}] shell-vs-blind {d_blind:+.2f}+/-{s_blind:.2f} | vs random-gate {d_rand:+.2f}+/-{s_rand:.2f}"
          f" | vs oracle {d_orc:+.2f}+/-{s_orc:.2f} | recovers {recov*100:.0f}% of the {drop:+.2f} outage drop"
          f"  => {'WIN' if win else 'no'}")
if not controls > 2 * c_sem:
    print("\n  => INCONCLUSIVE: planner ~= random-action -> control isn't estimate-bottlenecked on PushT;")
    print("     bank A1 (perception robustness) and move the CONTROL claim to a POMDP substrate.")
elif allwin:
    print("\n  => POSITIVE: detecting the outage (free shell) and coasting on the world model recovers control")
    print("     where a blind agent fails -- the deployment headline (M1.2 cost-shaping could not).")
else:
    print("\n  => MIXED/NULL: see rows -- coasting through the outage doesn't clearly beat blind / trails oracle.")

# ---- figure: mean reward vs outage length --------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.8))
ax.axhline(cmean, ls="--", color="#34495e", label=f"clean ({cmean:.0f})")
ax.axhline(floor["mean"].mean(), ls=":", color="#7f8c8d", label=f"random-action floor ({floor['mean'].mean():.0f})")
cols = {"blind": "#c0392b", "random-gate": "#e1b12c", "shell-gate": "#2980b9", "oracle-gate": "#27ae60"}
for pol in ["blind", "random-gate", "shell-gate", "oracle-gate"]:
    m = [res[(pol, K)]["mean"].mean() for K in OUTAGE_LENS]; s = [sem(res[(pol, K)]["mean"]) for K in OUTAGE_LENS]
    ax.errorbar(OUTAGE_LENS, m, yerr=s, fmt="-o", capsize=3, color=cols[pol], label=pol)
ax.set_xlabel("outage length (consecutive blacked-out steps)"); ax.set_ylabel("mean reward / episode")
ax.set_title("A2-burst -- survive sensor dropout by coasting on the world model", fontweight="bold")
ax.set_xticks(OUTAGE_LENS); ax.grid(alpha=.3); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig("/content/lewm-uncertainty/lewm_gated_control.png", dpi=110)
print("\nsaved lewm_gated_control.png")
