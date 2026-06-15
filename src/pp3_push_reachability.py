"""pp3 -- the reachability test on REAL physics (pymunk): does the actionable-events result transfer?

Task: push the box to the goal (behind-start, central goal -- the regime where a competent expert exists;
go-around navigation is a separable harder sub-problem, set aside for this first validation). Compare:
  - dense-CEM   : plan over the LEARNED dynamics model, cost = ||predicted block - goal|| (model-based);
  - scripted    : the competent feedback pusher (oracle upper bound, ~0.94);
  - learned-BC  : pi(z)->a, behavior-cloned on the scripted expert;
  - learned-DAgger : same, robustified by DAgger (relabel pi's own rollout states with the scripted expert).

Decisive: if learned-DAgger ~= scripted and dense-CEM fails, the toy/pusher result (model-free reachability
works, model-based fails on contact) TRANSFERS to real physics. Run: python src/pp3_push_reachability.py --seed 0
"""
import argparse
import numpy as np
import torch
from _event_common import mlp
from _pushphys_common import PushPhysEnv, scripted_push, W

ap = argparse.ArgumentParser()
ap.add_argument("--episodes", type=int, default=140); ap.add_argument("--T", type=int, default=120)
ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--hid", type=int, default=64)
ap.add_argument("--n_dagger", type=int, default=4); ap.add_argument("--dagger_eps", type=int, default=40)
ap.add_argument("--plan_episodes", type=int, default=20); ap.add_argument("--T_plan", type=int, default=130)
ap.add_argument("--H", type=int, default=15); ap.add_argument("--Ncem", type=int, default=256); ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
rng = np.random.default_rng(args.seed); torch.manual_seed(args.seed)
device = "cuda" if torch.cuda.is_available() else "cpu"
env = PushPhysEnv(block="box"); GOAL = torch.tensor(env.goal / W, device=device, dtype=torch.float32); SDIM, ADIM = 6, 2
to = lambda x: torch.tensor(x, device=device, dtype=torch.float32)


def collect(g, n_ep):
    Z, A, Zn, ST = [], [], [], []
    for _ in range(n_ep):
        env.reset(g)
        for t in range(args.T):
            s = env._state()
            a = scripted_push(env) if g.random() < 0.5 else g.uniform(-1, 1, ADIM).astype("float32")
            s2, _ = env.step(a)
            Z.append(s); A.append(a.astype("float32")); Zn.append(s2); ST.append(s)
            if np.linalg.norm(env.block_xy() - env.goal) < env.pos_tol:
                break
    return np.array(Z, "float32"), np.array(A, "float32"), np.array(Zn, "float32"), np.array(ST, "float32")


Z, A, Zn, ST = collect(rng, args.episodes)
dense = mlp(SDIM + ADIM, SDIM, args.hid).to(device); optd = torch.optim.Adam(dense.parameters(), 2e-3)
Zt, At, dZ = to(Z), to(A), to(Zn - Z); idx = np.arange(len(Z))
for ep in range(args.epochs):
    rng.shuffle(idx)
    for i in range(0, len(Z), 512):
        b = idx[i:i + 512]; loss = ((dense(torch.cat([Zt[b], At[b]], -1)) - dZ[b]) ** 2).mean()
        optd.zero_grad(); loss.backward(); optd.step()


def train_pi(states):
    pi = mlp(SDIM, ADIM, args.hid).to(device); opt = torch.optim.Adam(pi.parameters(), 2e-3)
    # relabel each state with the scripted expert (set env to that state, query script)
    Y = []
    for s in states:
        env.pusher.position = (float(s[0] * W), float(s[1] * W)); env.block.position = (float(s[2] * W), float(s[3] * W))
        env.block.angle = float(np.arctan2(s[5], s[4]))
        Y.append(scripted_push(env))
    X = to(np.array(states)); Y = to(np.array(Y)); ix = np.arange(len(X))
    for ep in range(args.epochs):
        rng.shuffle(ix)
        for i in range(0, len(X), 512):
            b = ix[i:i + 512]; loss = ((pi(X[b]) - Y[b]) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    return pi


@torch.no_grad()
def act_pi(pi, s):
    return np.clip(pi(to(s)).cpu().numpy(), -1, 1)


@torch.no_grad()
def cem(s0, N, iters=3):
    mean = np.zeros((args.H, ADIM), "float32"); std = np.ones((args.H, ADIM), "float32") * 0.7; best = np.zeros(ADIM, "float32")
    for _ in range(iters):
        acts = np.clip(mean + std * rng.standard_normal((N, args.H, ADIM)), -1, 1).astype("float32")
        z = to(np.repeat(s0[None], N, 0))
        for h in range(args.H):
            z = z + dense(torch.cat([z, to(acts[:, h])], -1))
        cost = (z[:, 2:4] - GOAL).norm(dim=1).cpu().numpy()
        order = cost.argsort(); elite = acts[order[:max(1, N // 10)]]; mean, std, best = elite.mean(0), elite.std(0) + 1e-3, acts[order[0], 0]
    return best


def run(kind, pi=None):
    g = np.random.default_rng(100 + args.seed); succ = 0
    for _ in range(args.plan_episodes):
        env.reset(g)
        for t in range(args.T_plan):
            s = env._state()
            a = scripted_push(env) if kind == "scripted" else (cem(s, args.Ncem) if kind == "dense" else act_pi(pi, s))
            env.step(a)
            if np.linalg.norm(env.block_xy() - env.goal) < env.pos_tol:
                succ += 1; break
    return succ / args.plan_episodes


pi_bc = train_pi(list(ST[:: max(1, len(ST) // 6000)]))
bc = run("pi", pi_bc)
agg = list(ST[:: max(1, len(ST) // 6000)]); pi = pi_bc
for it in range(args.n_dagger):
    g = np.random.default_rng(500 + args.seed + it)
    for _ in range(args.dagger_eps):
        env.reset(g)
        for t in range(args.T_plan):
            agg.append(env._state()); env.step(act_pi(pi, env._state()))
            if np.linalg.norm(env.block_xy() - env.goal) < env.pos_tol:
                break
    pi = train_pi(agg)
dag = run("pi", pi)
print(f"SEED {args.seed}  dense-CEM {run('dense'):.2f}  scripted(oracle) {run('scripted'):.2f}  learned-BC {bc:.2f}  learned-DAgger {dag:.2f}", flush=True)
