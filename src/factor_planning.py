"""Path 1, fresh lever -- FACTOR-SPACE planning: is LeWM not-plannable because the raw-latent METRIC is wrong?

Every prior control attempt scored plans by ‖z_T - z_goal‖^2 in the raw latent -- a metric we have direct
evidence is broken (latent-L2 != reachability; the strong-CEM sanity got latent-closer but task-FARther, 0%
success). But Tier 2 showed the latent LINEARLY encodes the block pose (R2~0.53), and LeJEPA identifiability
(Klindt et al.) says that's structural: the task factors are linearly recoverable. So: decode the pose from
the predicted latent with a frozen linear probe and plan to minimize distance in POSE space (the actual task
quantity), not raw-latent space. Using identifiability to extract a reachable metric for free -- a different
lever than the crowded "add a reachability aux-loss" papers, attacking the failure mode (the metric) we have
evidence for.

This is NOT another attention/gaze trick. It tests a specific hypothesis and is informative either way:
  pose-CEM > latent-CEM (on the env's REAL success metric) -> the metric was the problem; JEPA WMs ARE
    plannable in their identified-factor space (constructive positive; turns the control-null arc positive).
  pose-CEM ~= latent-CEM ~= random -> the gap is the DYNAMICS/predictor, not the metric (informative; the
    plannability problem is localized downstream of representation).

Same CEM planner for both costs (only the cost differs) + random-action floor; scored on the env's real
success/coverage signal. Reuses the probe protocol from tier2_diag (ridge). Run on Colab GPU:
  python src/factor_planning.py
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
S, CEM_ITERS, ELITE, HORIZON = 256, 5, 26, 8                      # same planner for both costs (only cost differs)
N_CAL = 40                                                       # rollouts to fit the linear pose probe
device = "cuda" if torch.cuda.is_available() else "cpu"
model, cfg = load_lewm("/content/le-wm", device=device)
prep = TT.Compose([TT.ToTensor(), TT.Resize((224, 224)), TT.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


@torch.no_grad()
def encode_one(frame):
    return model.encode({"pixels": prep(frame).to(device)[None, None]})["emb"][0, 0]


@torch.no_grad()
def encode_batch(frames):                                        # list of uint8 HWC -> [N,192]
    out = []
    for i in range(0, len(frames), 32):
        pix = torch.stack([prep(f) for f in frames[i:i + 32]]).unsqueeze(1).to(device)
        out.append(model.encode({"pixels": pix})["emb"][:, 0])
    return torch.cat(out)


@torch.no_grad()
def rollout_final(cur_emb, plans):                              # [192],[S,P,10] -> [S,192]
    Sn, P, _ = plans.shape
    emb = cur_emb[None, None].expand(Sn, 1, 192).clone()
    act = model.action_encoder(plans * ACTION_SCALE)
    for t in range(P):
        emb = torch.cat([emb, model.predict(emb[:, -HS:], act[:, :t + 1][:, -HS:])[:, -1:]], dim=1)
    return emb[:, -1]


@torch.no_grad()
def cem(cur_emb, cost_fn, gen):                                 # cost_fn: [S,192] final emb -> [S] cost
    mu = torch.zeros(HORIZON, 10, device=device)
    sigma = torch.full((HORIZON, 10), 0.5, device=device)
    for _ in range(CEM_ITERS):
        plans = (mu + sigma * torch.randn(S, HORIZON, 10, generator=gen, device=device)).clamp(-1, 1)
        elite = plans[cost_fn(rollout_final(cur_emb, plans)).argsort()[:ELITE]]
        mu, sigma = elite.mean(0), elite.std(0) + 1e-3
    return mu.clamp(-1, 1)


# ---- 1) fit the linear pose probe (z -> standardized block_pose), report held-out R2 -------------
print("fitting linear pose probe (z -> block_pose) ...", flush=True)
gen = np.random.default_rng(0); frames, poses = [], []
for r in range(N_CAL):
    env = gym.make("swm/PushT-v1", render_mode="rgb_array"); _, info = env.reset(seed=5000 + r)
    frames.append(env.render()); poses.append(np.asarray(info["block_pose"], dtype="float64").ravel())
    for _ in range(BUDGET):
        for _ in range(FS):                                                            # one model-step of random actions
            _, _, _, _, info = env.step(env.action_space.sample().astype("float32"))
        frames.append(env.render()); poses.append(np.asarray(info["block_pose"], dtype="float64").ravel())
    env.close()
Z = encode_batch(frames).double().cpu().numpy(); Yraw = np.stack(poses)                # [N,192],[N,3]
pose_mean, pose_std = Yraw.mean(0), Yraw.std(0) + 1e-6
Y = (Yraw - pose_mean) / pose_std                                                      # standardized
n = len(Z); ntr = int(0.8 * n); zm = Z[:ntr].mean(0); Zc = Z[:ntr] - zm
best = None
for a in (1.0, 10.0, 100.0, 1000.0):
    W = np.linalg.solve(Zc.T @ Zc + a * np.eye(192), Zc.T @ Y[:ntr])
    pred = (Z[ntr:] - zm) @ W
    r2 = 1 - ((pred - Y[ntr:]) ** 2).sum(0) / (((Y[ntr:] - Y[ntr:].mean(0)) ** 2).sum(0) + 1e-9)
    if best is None or r2.mean() > best[0]:
        best = (float(r2.mean()), a, r2)
alpha = best[1]
zm_all = Z.mean(0); Zc_all = Z - zm_all                                                # refit on ALL data
W = np.linalg.solve(Zc_all.T @ Zc_all + alpha * np.eye(192), Zc_all.T @ Y)
print(f"  probe held-out R2 mean {best[0]:.3f}  per-dim {np.round(best[2], 3).tolist()} (alpha {alpha})", flush=True)
z_mean_t = torch.tensor(zm_all, dtype=torch.float32, device=device)
W_t = torch.tensor(W, dtype=torch.float32, device=device)
pmean_t = torch.tensor(pose_mean, dtype=torch.float32, device=device)
pstd_t = torch.tensor(pose_std, dtype=torch.float32, device=device)


def decode_pose(z):                                            # [.,192] -> [.,3] standardized pose
    return (z - z_mean_t) @ W_t


# ---- 2) detect the env's real success/coverage signal --------------------------------------------
env = gym.make("swm/PushT-v1", render_mode="rgb_array"); _, info0 = env.reset(seed=0)
succ_keys = [k for k in (info0 if isinstance(info0, dict) else {}) if any(
    s in k.lower() for s in ("success", "coverage", "score"))]
SUCC_FIELD = next((k for k in succ_keys if "success" in k.lower()), None) or (succ_keys[0] if succ_keys else None)
HAS_GOAL_POSE = isinstance(info0, dict) and "goal_pose" in info0
env.close()
print(f"success signal: {SUCC_FIELD or 'NONE (best/final reward only)'}; goal_pose in info: {HAS_GOAL_POSE}\n", flush=True)


def episode(policy, ep, gen, arng):
    env = gym.make("swm/PushT-v1", render_mode="rgb_array"); _, info = env.reset(seed=ep)
    goal_emb = encode_one(info["goal"])
    if HAS_GOAL_POSE:
        gp = (np.asarray(info["goal_pose"], dtype="float32").ravel() - pose_mean) / pose_std
        goal_pose = torch.tensor(gp, dtype=torch.float32, device=device)               # standardized target pose
    else:
        goal_pose = decode_pose(goal_emb[None])[0]                                      # fall back: decode goal image
    cost = {"latent-CEM": lambda zT: (zT - goal_emb).pow(2).sum(-1),
            "pose-CEM": lambda zT: (decode_pose(zT) - goal_pose).pow(2).sum(-1)}
    best_r, final_r, best_succ = -1e18, -1e18, 0.0
    for step in range(BUDGET):
        if policy == "random-action":
            mstep = arng.uniform(-1, 1, 10).astype("float32")
        else:
            mstep = cem(encode_one(env.render()), cost[policy], gen)[0].cpu().numpy()
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


def run(policy):
    g = torch.Generator(device=device).manual_seed(0); arng = np.random.default_rng(123)
    out = np.array([episode(policy, ep, g, arng) for ep in range(EPISODES)])
    return out[:, 0], out[:, 1], out[:, 2]


def sem(a):
    return float(np.std(a) / np.sqrt(len(a)))


# ---- 3) random / latent-CEM / pose-CEM -----------------------------------------------------------
print(f"==== factor-space planning on PushT ({EPISODES} eps, BUDGET {BUDGET}) ====", flush=True)
R = {}
for pol in ["random-action", "latent-CEM", "pose-CEM"]:
    R[pol] = run(pol)
    b, f, s = R[pol]
    print(f"  {pol:13}: best-r {b.mean():.2f}+/-{sem(b):.2f} | final-r {f.mean():.2f} | "
          f"succ {s.mean():.3f}+/-{sem(s):.3f}", flush=True)

# ---- 4) verdict ----------------------------------------------------------------------------------
print("\n==== verdict ====")
use_succ = SUCC_FIELD is not None and np.std(np.concatenate([R[p][2] for p in R])) > 1e-6
idx, name = (2, "success") if use_succ else (0, "best-reward")
rnd, lat, pos = R["random-action"][idx], R["latent-CEM"][idx], R["pose-CEM"][idx]
d_lat = pos.mean() - lat.mean(); s_lat = np.hypot(sem(pos), sem(lat))                  # pose vs latent
d_rnd = pos.mean() - rnd.mean(); s_rnd = np.hypot(sem(pos), sem(rnd))                  # pose vs random
print(f"  metric = {name}")
print(f"  pose-CEM vs latent-CEM : {d_lat:+.3f} +/- {s_lat:.3f}  ({d_lat / (s_lat + 1e-9):+.1f} SEM)")
print(f"  pose-CEM vs random     : {d_rnd:+.3f} +/- {s_rnd:.3f}  ({d_rnd / (s_rnd + 1e-9):+.1f} SEM)")
if d_lat > 2 * s_lat and d_rnd > 2 * s_rnd:
    print("  => POSITIVE: planning in the IDENTIFIED-FACTOR space beats raw-latent-L2 (and random). LeWM is")
    print("     plannable in its decoded-pose metric -- the raw-latent metric was the problem. Control positive.")
elif d_rnd > 2 * s_rnd and not d_lat > 2 * s_lat:
    print("  => BOTH plan: pose-CEM and latent-CEM both beat random but tie each other (metric wasn't the sole issue).")
else:
    print("  => NEGATIVE: pose-CEM does not beat latent-CEM/random -> the gap is the DYNAMICS/predictor, not the")
    print("     metric. Plannability problem localized downstream of representation (informative for the writeup).")
print(f"\n  (probe held-out R2 was {best[0]:.3f}; if low, factor recovery is the bottleneck, not the idea.)")
