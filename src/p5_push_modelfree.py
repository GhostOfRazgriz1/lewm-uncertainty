"""Pusher E3c -- is the 0.60 cap SOFT (closable model-free) or the pi/BC ceiling?

p3/p4: pure model-free DAgger pi caps at 0.60 vs oracle ~0.90; model-based correction HURTS (the residual
gap is a dynamics-model problem). So push the MODEL-FREE lever: track pure-pi success as a function of DAgger
iterations (wider coverage of pi's own failure states), at default and larger capacity. If success climbs
toward ~0.90, the cap was soft (coverage) -> the method reaches the oracle. If it plateaus at ~0.60, it's the
reactive-pi/BC ceiling on this push skill. Pure model-free (no dynamics model). Run: python src/p5_push_modelfree.py --seed 0 --hid 48
"""
import argparse
import numpy as np
import torch
from _event_common import mlp
from _push_common import PushEnv, PSDIM, ADIM, PSTEP

ap = argparse.ArgumentParser()
ap.add_argument("--episodes", type=int, default=300); ap.add_argument("--T", type=int, default=80)
ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--hid", type=int, default=48)
ap.add_argument("--max_dagger", type=int, default=8); ap.add_argument("--dagger_eps", type=int, default=90)
ap.add_argument("--plan_episodes", type=int, default=25); ap.add_argument("--T_plan", type=int, default=70); ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
rng = np.random.default_rng(args.seed); torch.manual_seed(args.seed)
device = "cuda" if torch.cuda.is_available() else "cpu"
env = PushEnv()
to = lambda x: torch.tensor(x, device=device, dtype=torch.float32)

(Z, _, _, _), _ = env.collect(rng, args.episodes, args.T)


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


def success(pi):
    g = np.random.default_rng(100 + args.seed); sc = 0
    for _ in range(args.plan_episodes):
        s = env.reset(g)
        for t in range(args.T_plan):
            s, _ = env.step(s, act_pi(pi, s))
            if np.linalg.norm(s[2:4] - env.goal) < env.zone_r:
                sc += 1; break
    return sc / args.plan_episodes


agg = list(Z[:: max(1, len(Z) // 6000)]); pi = train_pi(agg)
curve = [success(pi)]
for it in range(args.max_dagger):
    g = np.random.default_rng(500 + args.seed + it)
    for _ in range(args.dagger_eps):                                  # roll pi, aggregate its own visited states
        s = env.reset(g)
        for t in range(args.T_plan):
            agg.append(s.copy()); s, _ = env.step(s, act_pi(pi, s))
            if np.linalg.norm(s[2:4] - env.goal) < env.zone_r:
                break
    pi = train_pi(agg); curve.append(success(pi))
print(f"SEED {args.seed} hid {args.hid}  success-by-DAgger-iter {['%.2f' % c for c in curve]}  (oracle ~0.90)", flush=True)
