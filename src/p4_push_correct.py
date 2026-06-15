"""Pusher E3b -- close the residual gap with a MODEL-CORRECTED push controller (CEM-around-pi).

p3: pure learned-DAgger pi reaches 0.60 vs oracle ~0.90; dense-CEM = 0.00. The asymmetry: dense-CEM can't
SEARCH a contact-push from scratch (sparse cost, no gradient until contact) but the model CAN EVALUATE one
given a good seed. So: seed CEM at pi's push rollout and let the dense model refine locally (cost = how
close the block gets to the goal over the predicted rollout). Tests whether model-correction closes
0.60 -> ~oracle. Compare oracle / pure-pi (DAgger) / CEM-around-pi. Run: python src/p4_push_correct.py --seed 0
"""
import argparse
import numpy as np
import torch
from _event_common import mlp
from _push_common import PushEnv, PSDIM, ADIM, PSTEP

ap = argparse.ArgumentParser()
ap.add_argument("--episodes", type=int, default=300); ap.add_argument("--T", type=int, default=80)
ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--hid", type=int, default=48)
ap.add_argument("--n_dagger", type=int, default=3); ap.add_argument("--dagger_eps", type=int, default=60)
ap.add_argument("--plan_episodes", type=int, default=20); ap.add_argument("--T_plan", type=int, default=70)
ap.add_argument("--H", type=int, default=15); ap.add_argument("--Ncem", type=int, default=128); ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
rng = np.random.default_rng(args.seed); torch.manual_seed(args.seed)
device = "cuda" if torch.cuda.is_available() else "cpu"
env = PushEnv(); GOAL = torch.tensor(env.goal, device=device)
to = lambda x: torch.tensor(x, device=device, dtype=torch.float32)

(Z, A, Zn, _), _ = env.collect(rng, args.episodes, args.T)
dense = mlp(PSDIM + ADIM, PSDIM, args.hid + 16).to(device); optd = torch.optim.Adam(dense.parameters(), 2e-3)
Zt, At, dZ = to(Z), to(A), to(Zn - Z); idx = np.arange(len(Z))
for ep in range(args.epochs):
    rng.shuffle(idx)
    for i in range(0, len(Z), 512):
        b = idx[i:i + 512]; loss = ((dense(torch.cat([Zt[b], At[b]], -1)) - dZ[b]) ** 2).mean()
        optd.zero_grad(); loss.backward(); optd.step()


def train_pi(states):
    pi = mlp(PSDIM, ADIM, args.hid).to(device); opt = torch.optim.Adam(pi.parameters(), 2e-3)
    X = to(np.array(states)); Y = to(np.array([env.expert(s, 3) for s in states])); ix = np.arange(len(X))
    for ep in range(args.epochs):
        rng.shuffle(ix)
        for i in range(0, len(X), 512):
            b = ix[i:i + 512]; loss = ((pi(X[b]) - Y[b]) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    return pi


@torch.no_grad()
def act_pi(pi, s):
    return np.clip(pi(to(s)).cpu().numpy(), -PSTEP, PSTEP)


# learned pi via DAgger (same recipe as p3)
agg = list(Z[:: max(1, len(Z) // 6000)]); pi = train_pi(agg)
for it in range(args.n_dagger):
    g = np.random.default_rng(500 + args.seed + it)
    for _ in range(args.dagger_eps):
        s = env.reset(g)
        for t in range(args.T_plan):
            agg.append(s.copy()); s, _ = env.step(s, act_pi(pi, s))
            if np.linalg.norm(s[2:4] - env.goal) < env.zone_r:
                break
    pi = train_pi(agg)


@torch.no_grad()
def cem_around_pi(s0, N, iters=2):
    seed = np.zeros((args.H, ADIM), "float32"); z = to(s0[None])              # seed = pi's push rollout (model)
    for h in range(args.H):
        a = np.clip(pi(z).cpu().numpy(), -PSTEP, PSTEP); seed[h] = a[0]; z = z + dense(torch.cat([z, to(a)], -1))
    mean, std = seed.copy(), np.ones((args.H, ADIM), "float32") * PSTEP * 0.5
    best = seed[0].copy()
    for _ in range(iters):
        acts = np.clip(mean + std * rng.standard_normal((N, args.H, ADIM)), -PSTEP, PSTEP).astype("float32")
        acts[0] = seed                                                         # keep the pi seed as a candidate
        z = to(np.repeat(s0[None], N, 0)); blocks = []
        for h in range(args.H):
            z = z + dense(torch.cat([z, to(acts[:, h])], -1)); blocks.append(z[:, 2:4])
        bt = torch.stack(blocks, 1)                                            # [N,H,2]
        cost = (bt - GOAL).norm(dim=2).min(1).values.cpu().numpy()            # closest the block gets to goal
        order = cost.argsort(); elite = acts[order[:max(1, N // 10)]]; mean, std, best = elite.mean(0), elite.std(0) + 1e-3, acts[order[0], 0]
    return best


def run(kind):
    g = np.random.default_rng(100 + args.seed); succ = 0
    for _ in range(args.plan_episodes):
        s = env.reset(g)
        for t in range(args.T_plan):
            a = env.expert(s, 3) if kind == "oracle" else (act_pi(pi, s) if kind == "pi" else cem_around_pi(s, args.Ncem))
            s, _ = env.step(s, a)
            if np.linalg.norm(s[2:4] - env.goal) < env.zone_r:
                succ += 1; break
    return succ / args.plan_episodes


print(f"SEED {args.seed}  oracle {run('oracle'):.2f}  pure-pi {run('pi'):.2f}  cem-around-pi {run('correct'):.2f}", flush=True)
