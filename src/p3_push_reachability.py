"""Pusher E3 -- the DECISIVE reachability test (C3+C5+C7 on hard, pushing reachability).

E1 showed discovery on the pusher is fuzzy (push-events aren't transition-type-distinct; the task event
'delivered' is goal-relational). So we SIDESTEP discovery and give the ground-truth task event, to isolate
the real scale-up question: when CAUSING the event is structured pushing (not point-mass navigation), does a
LEARNED controller close the gap to the oracle, or collapse to the model-based baseline?

Task = deliver the block to the goal zone. Planners:
  - dense-CEM        : model-based planner, cost = ||predicted block - goal|| (the 'descriptive'/predictive
                       baseline -- it knows the goal state, must plan the push through the learned model).
  - oracle-pusher    : the scripted push skill env.expert(.,DELIVERED) (upper bound, perfect affordance).
  - learned-BC       : inverse model pi(a|z) trained by BC on the expert over the data distribution.
  - learned-DAgger   : same, robustified by DAgger (relabel pi's OWN rollout states with the expert).

Decisive read: dense-CEM << oracle reproduces the gap on HARD reachability; learned-DAgger ~= oracle means
the learned controller LEARNS the push skill (reachability transfers); learned ~= dense means the toy's C5
win was mostly trivial (navigation) reachability. Run: python src/p3_push_reachability.py --seed 0
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
ap.add_argument("--plan_episodes", type=int, default=20); ap.add_argument("--T_plan", type=int, default=70); ap.add_argument("--H", type=int, default=15)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
rng = np.random.default_rng(args.seed); torch.manual_seed(args.seed)
device = "cuda" if torch.cuda.is_available() else "cpu"
env = PushEnv(); GOAL = torch.tensor(env.goal, device=device)
to = lambda x: torch.tensor(x, device=device, dtype=torch.float32)

(Z, A, Zn, _), _ = env.collect(rng, args.episodes, args.T)
# dynamics model (for dense-CEM); the push dynamics are in the data
dense = mlp(PSDIM + ADIM, PSDIM, args.hid + 16).to(device); optd = torch.optim.Adam(dense.parameters(), 2e-3)
Zt, At, dZ = to(Z), to(A), to(Zn - Z); idx = np.arange(len(Z))
for ep in range(args.epochs):
    rng.shuffle(idx)
    for i in range(0, len(Z), 512):
        b = idx[i:i + 512]; loss = ((dense(torch.cat([Zt[b], At[b]], -1)) - dZ[b]) ** 2).mean()
        optd.zero_grad(); loss.backward(); optd.step()


def train_pi(states):                                              # BC on the scripted expert over `states`
    pi = mlp(PSDIM, ADIM, args.hid).to(device); opt = torch.optim.Adam(pi.parameters(), 2e-3)
    X = to(np.array(states)); Y = to(np.array([env.expert(s, 3) for s in states]))
    ix = np.arange(len(X))
    for ep in range(args.epochs):
        rng.shuffle(ix)
        for i in range(0, len(X), 512):
            b = ix[i:i + 512]; loss = ((pi(X[b]) - Y[b]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return pi


@torch.no_grad()
def cem(s0, N, iters=3):                                           # dense-CEM toward the goal (model-based)
    mean = np.zeros((args.H, ADIM), "float32"); std = np.ones((args.H, ADIM), "float32") * PSTEP; best = np.zeros(ADIM, "float32")
    for _ in range(iters):
        acts = np.clip(mean + std * rng.standard_normal((N, args.H, ADIM)), -PSTEP, PSTEP).astype("float32")
        z = to(np.repeat(s0[None], N, 0))
        for h in range(args.H):
            z = z + dense(torch.cat([z, to(acts[:, h])], -1))
        cost = (z[:, 2:4] - GOAL).norm(dim=1).cpu().numpy()
        order = cost.argsort(); elite = acts[order[:max(1, N // 10)]]; mean, std, best = elite.mean(0), elite.std(0) + 1e-3, acts[order[0], 0]
    return best


@torch.no_grad()
def act_pi(pi, s):
    return np.clip(pi(to(s)).cpu().numpy(), -PSTEP, PSTEP)


def run(kind, N=0, pi=None):
    g = np.random.default_rng(100 + args.seed); succ = 0
    for _ in range(args.plan_episodes):
        s = env.reset(g)
        for t in range(args.T_plan):
            if kind == "dense":
                a = cem(s, N)
            elif kind == "oracle":
                a = env.expert(s, 3)
            else:
                a = act_pi(pi, s)
            s, _ = env.step(s, a)
            if np.linalg.norm(s[2:4] - env.goal) < env.zone_r:
                succ += 1; break
    return succ / args.plan_episodes


# learned controllers: BC over data states, then DAgger over pi's own rollout states
pi_bc = train_pi(list(Z[:: max(1, len(Z) // 6000)]))
bc_succ = run("pi", pi=pi_bc)
agg = list(Z[:: max(1, len(Z) // 6000)]); pi = pi_bc
for it in range(args.n_dagger):
    g = np.random.default_rng(500 + args.seed + it)
    for _ in range(args.dagger_eps):
        s = env.reset(g)
        for t in range(args.T_plan):
            agg.append(s.copy()); s, _ = env.step(s, act_pi(pi, s))
            if np.linalg.norm(s[2:4] - env.goal) < env.zone_r:
                break
    pi = train_pi(agg)
dagger_succ = run("pi", pi=pi)

NS = [128, 256]
dense_succ = [run("dense", N=n) for n in NS]
oracle_succ = run("oracle")
print(f"SEED {args.seed}  dense-CEM {dense_succ}  oracle {oracle_succ:.2f}  learned-BC {bc_succ:.2f}  learned-DAgger {dagger_succ:.2f}", flush=True)
