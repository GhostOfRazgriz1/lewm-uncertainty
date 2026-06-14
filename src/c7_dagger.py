"""C7 -- DAgger closed-loop fix for the actionable-event inverse model (kill C6's ~20% collapse).

C6 diagnosed the collapse as BC distribution-shift: the greedy inverse model is correct on training-
distribution states (every on-distribution probe passed) but drifts off-distribution in closed loop and
collapses on ~20% of seeds. The principled fix is DAgger: retrain the policy on ITS OWN rollout states,
relabeled by an expert.

Expert (fair, derived from observation/data -- not hand-coded world knowledge):
  - pickup: move toward the object, whose position IS in the (fully observed) state s[2:4].
  - drop:   move toward the drop-zone centroid, ESTIMATED from data (mean agent pos at observed drops).

Loop: BC-init on action-relevant data -> {rollout pi greedily on the subgoal seq, relabel visited states
with the expert, aggregate, retrain} x n_dagger. Compare BC-only vs DAgger planning success across seeds;
report median + failure rate. If DAgger removes the collapse, the dist-shift diagnosis was right and the
actionable-event method is robust. Run one seed: python src/c7_dagger.py --seed 0 (loop seeds in bash).
"""
import argparse
import numpy as np
import torch
from _event_common import EventEnv, SDIM, ADIM, NEV, STEP, mlp

ap = argparse.ArgumentParser()
ap.add_argument("--episodes", type=int, default=300); ap.add_argument("--T", type=int, default=80)
ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--Klook", type=int, default=12); ap.add_argument("--hid", type=int, default=48)
ap.add_argument("--n_dagger", type=int, default=3); ap.add_argument("--dagger_eps", type=int, default=60)
ap.add_argument("--plan_episodes", type=int, default=20); ap.add_argument("--T_plan", type=int, default=50); ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
rng = np.random.default_rng(args.seed); torch.manual_seed(args.seed)
device = "cuda" if torch.cuda.is_available() else "cpu"
env = EventEnv()
to = lambda x: torch.tensor(x, device=device, dtype=torch.float32)
oh = lambda e: np.eye(NEV, dtype="float32")[e]


def collect_eps(g, n):
    eps = []
    for _ in range(n):
        s = env.reset(g); S, Av, Ev = [s], [], []
        for _ in range(args.T):
            a = env.policy(s, g); s2, ev = env.step(s, a); S.append(s2); Av.append(a); Ev.append(ev); s = s2
        eps.append((np.array(S, "float32"), np.array(Av, "float32"), np.array(Ev)))
    return eps


eps = collect_eps(rng, args.episodes)
# initial action-relevant BC pairs + estimate drop-zone centroid from data (mean agent pos at drops)
Z0, E0, A0, drops = [], [], [], []
for (S, Av, Ev) in eps:
    T = len(Av)
    for t in range(T):
        if Ev[t] == 2:
            drops.append(S[t + 1][:2])
        e_next, j = 0, 0
        for k in range(1, args.Klook + 1):
            if t + k - 1 < T and Ev[t + k - 1] > 0:
                e_next, j = Ev[t + k - 1], t + k - 1; break
        if e_next > 0:
            L = S[j][:2]
            if np.linalg.norm(S[t][:2] + Av[t] - L) < np.linalg.norm(S[t][:2] - L):
                Z0.append(S[t]); E0.append(oh(e_next)); A0.append(Av[t])
drop_c_hat = np.mean(drops, 0) if drops else env.drop_c.copy()
print(f"seed {args.seed}: {len(Z0)} BC pairs; drop-centroid est {drop_c_hat.round(2)} (true {env.drop_c.round(2)})", flush=True)


def expert(s, e):                                                  # toward the event-trigger location
    target = s[2:4] if e == 1 else drop_c_hat
    d = target - s[:2]
    return np.clip(d / (np.linalg.norm(d) + 1e-9) * STEP, -STEP, STEP).astype("float32")


def train_pi(Z, E, A, iters):
    pi = mlp(SDIM + NEV, ADIM, args.hid).to(device); opt = torch.optim.Adam(pi.parameters(), 2e-3)
    Zt, Et, At = to(np.array(Z)), to(np.array(E)), to(np.array(A)); idx = np.arange(len(Z))
    for ep in range(iters):
        rng.shuffle(idx)
        for i in range(0, len(Z), 512):
            b = idx[i:i + 512]; loss = ((pi(torch.cat([Zt[b], Et[b]], -1)) - At[b]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return pi


@torch.no_grad()
def act(pi, s, e):
    return np.clip(pi(torch.cat([to(s), to(oh(e))])).cpu().numpy(), -STEP, STEP)


def plan_success(pi):
    g = np.random.default_rng(100 + args.seed); succ = 0
    for _ in range(args.plan_episodes):
        s = env.reset(g); sg = 0
        for t in range(args.T_plan):
            s, ev = env.step(s, act(pi, s, 1 if sg == 0 else 2))
            if ev == (1 if sg == 0 else 2):
                sg = min(sg + 1, 1)
            if np.linalg.norm(s[2:4] - env.drop_c) < 0.12:
                succ += 1; break
    return succ / args.plan_episodes


# ---- BC-only baseline ----------------------------------------------------------------------------
pi_bc = train_pi(Z0, E0, A0, args.epochs)
bc_succ = plan_success(pi_bc)

# ---- DAgger: aggregate the policy's own rollout states, relabeled by the expert ------------------
Zd, Ed, Ad = list(Z0), list(E0), list(A0); pi = pi_bc
for it in range(args.n_dagger):
    g = np.random.default_rng(500 + args.seed + it)
    for _ in range(args.dagger_eps):                              # roll the CURRENT pi (closed loop)
        s = env.reset(g); sg = 0
        for t in range(args.T_plan):
            e = 1 if sg == 0 else 2
            Zd.append(s.copy()); Ed.append(oh(e)); Ad.append(expert(s, e))   # relabel visited state
            s, ev = env.step(s, act(pi, s, e))
            if ev == e:
                sg = min(sg + 1, 1)
            if np.linalg.norm(s[2:4] - env.drop_c) < 0.12:
                break
    pi = train_pi(Zd, Ed, Ad, args.epochs)
dagger_succ = plan_success(pi)

print(f"SEED {args.seed} BC-only {bc_succ:.2f}   DAgger {dagger_succ:.2f}   (aggregated {len(Zd)} pairs)", flush=True)
