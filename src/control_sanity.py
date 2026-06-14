"""Path 1 de-risk: is PushT+LeWM control actually COMPETENT? (before outage-gating or a new substrate.)

A2's control null may be a PLANNER-CONFIG + WRONG-METRIC artifact, not a fundamental LeWM limit: we used a
weak hand-rolled CEM (S=96, horizon 6) and read a negative distance-like reward (~-262..-413), not PushT's
real success criterion. This checks the substrate before we invest:
  1) DUMP the env reward/info semantics -- find the true success/coverage signal.
  2) STRONG-CEM vs RANDOM-ACTION over N episodes on the PROPER metric. Does a competent planner clearly win?

READ:
  COMPETENT (strong-CEM success-rate >> random, beyond SEM) -> the substrate is fine; A2 was a config
    artifact. Next: make PushT estimate-bottlenecked (sensor outages) + gate on the free shell -> control
    positive, NO new infra.
  WEAK (strong-CEM ~= random) -> LeWM really is "predictive but not plannable" here; move control to a
    different WM/substrate (accept the infra risk).

Run on Colab GPU:  python src/control_sanity.py
"""
import sys
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import stable_worldmodel as swm                                   # noqa: F401  registers swm/PushT-v1
from torchvision import transforms as TT

sys.path.insert(0, "/content/lewm-uncertainty")
from src.load_lewm import load_lewm                               # noqa: E402

FS, HS = 5, 3
ACTION_SCALE = 2.0
EPISODES, BUDGET = 25, 20
CONFIGS = {"strong-CEM": dict(S=256, HORIZON=8, CEM_ITERS=5, ELITE=26)}   # one strong config; sweep later if WEAK
device = "cuda" if torch.cuda.is_available() else "cpu"
model, cfg = load_lewm("/content/le-wm", device=device)
prep = TT.Compose([TT.ToTensor(), TT.Resize((224, 224)), TT.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def set_drop(b):
    for m in model.predictor.modules():
        if isinstance(m, nn.Dropout):
            m.train(b)


@torch.no_grad()
def encode_one(frame):
    return model.encode({"pixels": prep(frame).to(device)[None, None]})["emb"][0, 0]


@torch.no_grad()
def rollout_final(cur_emb, plans, HORIZON):
    Sn, P, _ = plans.shape
    emb = cur_emb[None, None].expand(Sn, 1, 192).clone()
    act = model.action_encoder(plans * ACTION_SCALE)
    for t in range(P):
        emb = torch.cat([emb, model.predict(emb[:, -HS:], act[:, :t + 1][:, -HS:])[:, -1:]], dim=1)
    return emb[:, -1]


@torch.no_grad()
def cem(cur_emb, goal_emb, gen, S, HORIZON, CEM_ITERS, ELITE):
    mu = torch.zeros(HORIZON, 10, device=device)
    sigma = torch.full((HORIZON, 10), 0.5, device=device)
    for _ in range(CEM_ITERS):
        plans = (mu + sigma * torch.randn(S, HORIZON, 10, generator=gen, device=device)).clamp(-1, 1)
        cost = (rollout_final(cur_emb, plans, HORIZON) - goal_emb).pow(2).sum(-1)
        elite = plans[cost.argsort()[:ELITE]]
        mu, sigma = elite.mean(0), elite.std(0) + 1e-3
    return mu.clamp(-1, 1)


# ---- 1) inspect the env reward / success semantics -----------------------------------------------
env = gym.make("swm/PushT-v1", render_mode="rgb_array")
obs, info = env.reset(seed=0)
print("=== env semantics ===")
print("info keys:", {k: (np.asarray(v).shape if not np.isscalar(v) else type(v).__name__) for k, v in info.items()}
      if isinstance(info, dict) else type(info))
rs, succ_keys = [], [k for k in (info if isinstance(info, dict) else {}) if any(
    s in k.lower() for s in ("success", "coverage", "reward", "score", "done"))]
flags = {k: [] for k in succ_keys}
for t in range(60):
    obs, r, term, trunc, info = env.step(env.action_space.sample().astype("float32"))
    rs.append(float(r))
    for k in succ_keys:
        if k in info:
            flags[k].append(float(np.asarray(info[k]).ravel()[0]) if np.size(info[k]) else 0.0)
    if term or trunc:
        env.reset(seed=0)
print(f"reward over random rollout: min {min(rs):.3f}  max {max(rs):.3f}  mean {np.mean(rs):.3f}")
print("candidate success/coverage info fields:", {k: (min(v), max(v)) for k, v in flags.items() if v})
env.close()

# pick a success signal: prefer an explicit success/coverage info field, else fall back to best reward
SUCC_FIELD = next((k for k in succ_keys if "success" in k.lower()), None) \
    or next((k for k in succ_keys if "coverage" in k.lower()), None)
print(f"\nusing success signal: {SUCC_FIELD or 'NONE found -> will report best/final reward only'}\n", flush=True)


def episode(policy, ep, params, gen, arng):
    env = gym.make("swm/PushT-v1", render_mode="rgb_array")
    _, info = env.reset(seed=ep)
    goal_emb = encode_one(info["goal"])
    best_r, final_r, best_succ = -1e18, -1e18, 0.0
    for step in range(BUDGET):
        if policy == "random-action":
            mstep = arng.uniform(-1, 1, 10).astype("float32")
        else:
            cur = encode_one(env.render())
            mstep = cem(cur, goal_emb, gen, **params)[0].cpu().numpy()
        done = False
        for j in range(FS):
            _, r, term, trunc, info = env.step(np.clip(mstep[2 * j:2 * j + 2], -1, 1).astype("float32"))
            best_r = max(best_r, float(r)); final_r = float(r); done = term or trunc
            if SUCC_FIELD and SUCC_FIELD in info and np.size(info[SUCC_FIELD]):
                best_succ = max(best_succ, float(np.asarray(info[SUCC_FIELD]).ravel()[0]))
            if done:
                break
        if done:
            break
    env.close()
    return best_r, final_r, best_succ


def run(policy, params):
    gen = torch.Generator(device=device).manual_seed(0); arng = np.random.default_rng(123)
    out = np.array([episode(policy, ep, params, gen, arng) for ep in range(EPISODES)])
    return out[:, 0], out[:, 1], out[:, 2]                        # best_r, final_r, best_succ


def sem(a):
    return float(np.std(a) / np.sqrt(len(a)))


# ---- 2) strong-CEM vs random-action --------------------------------------------------------------
print(f"==== control competence on PushT ({EPISODES} eps, BUDGET {BUDGET}) ====", flush=True)
rb, rf, rs2 = run("random-action", {})
print(f"  random-action: best-r {rb.mean():.2f}+/-{sem(rb):.2f} | final-r {rf.mean():.2f} | "
      f"succ {rs2.mean():.3f}+/-{sem(rs2):.3f}", flush=True)
for name, params in CONFIGS.items():
    cb, cf, cs = run(name, params)
    print(f"  {name}: best-r {cb.mean():.2f}+/-{sem(cb):.2f} | final-r {cf.mean():.2f} | "
          f"succ {cs.mean():.3f}+/-{sem(cs):.3f}", flush=True)
    if SUCC_FIELD:
        d = cs.mean() - rs2.mean(); s = np.hypot(sem(cs), sem(rs2))
        metric = f"success {d:+.3f}+/-{s:.3f}"
    else:
        d = cb.mean() - rb.mean(); s = np.hypot(sem(cb), sem(rb))
        metric = f"best-reward {d:+.2f}+/-{s:.2f}"
    print(f"    -> {name} vs random: {metric}  ({'COMPETENT' if d > 2 * s else 'WEAK'})", flush=True)

print("\n  COMPETENT -> PushT control is estimate-worthy; make it outage-bottlenecked + gate (no new infra).")
print("  WEAK      -> LeWM not plannable here; move the control claim to a different WM/substrate.")
