"""C3 -- the FULLY FAIR event-level planner (C1's planner was a strawman null-penalty bias).

Hierarchical: high level = fixed subgoal sequence [pickup, drop] toward the task (object in drop zone);
low level = CEM that uses the learned model to find actions triggering the current target event. Compared
against:
  - dense-CEM   : flat planner, final goal cost only (the baseline).
  - event-CEM   : the fair method -- subgoal sequence + low-level CEM toward the MODEL's target event CODE.
  - oracle-subgoal: SAME subgoal structure but low-level cost = hand-coded affordance (reach the object,
                    then the drop zone). This is "if you had PERFECT actionable events."

The contrast is the whole point (user's conceptual update): if oracle-subgoal >> dense but event-CEM ~=
dense, then the descriptive event code discovers WHAT happened, not WHAT can be planned -- subgoal
structure helps, but the model's codes aren't actionable enough to provide it. If event-CEM ~= oracle,
the codes ARE actionable and the planning claim is rescued.

Run: python src/c3_event_planner.py
"""
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")                          # noqa: E402
import matplotlib.pyplot as plt                                   # noqa: E402
from _event_common import EventEnv, SDIM, ADIM, STEP, train_bn, event_metrics

ap = argparse.ArgumentParser()
ap.add_argument("--episodes", type=int, default=200); ap.add_argument("--T", type=int, default=80)
ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--K", type=int, default=6)
ap.add_argument("--hid", type=int, default=32)
ap.add_argument("--plan_episodes", type=int, default=20); ap.add_argument("--T_plan", type=int, default=50)
ap.add_argument("--H", type=int, default=15); ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
rng = np.random.default_rng(args.seed); torch.manual_seed(args.seed)
device = "cuda" if torch.cuda.is_available() else "cpu"
env = EventEnv()
DROP = torch.tensor(env.drop_c, device=device)

(Z, A, Zn, EV), _ = env.collect(rng, args.episodes, args.T)
bn = train_bn(Z, A, Zn, SDIM, ADIM, args.hid, args.K, args.epochs, device, rng)
print("trained event-BN", flush=True)
dense = torch.nn.Sequential(torch.nn.Linear(SDIM + ADIM, args.hid + 16), torch.nn.GELU(),
                            torch.nn.Linear(args.hid + 16, args.hid + 16), torch.nn.GELU(),
                            torch.nn.Linear(args.hid + 16, SDIM)).to(device)
optd = torch.optim.Adam(dense.parameters(), 2e-3)
Zt, At, dZ = (torch.tensor(x, device=device, dtype=torch.float32) for x in (Z, A, Zn - Z)); idx = np.arange(len(Z))
for ep in range(args.epochs):
    rng.shuffle(idx)
    for i in range(0, len(Z), 512):
        b = idx[i:i + 512]; loss = ((dense(torch.cat([Zt[b], At[b]], -1)) - dZ[b]) ** 2).mean()
        optd.zero_grad(); loss.backward(); optd.step()
print("trained dense", flush=True)

# identify which learned code = pickup / drop (privilege used ONLY to name subgoals, noted in writeup)
with torch.no_grad():
    pc = bn.post_code(Zt, At, torch.tensor(Zn, device=device, dtype=torch.float32)).cpu().numpy()
_, recall, distinct, tops = event_metrics(EV, pc, args.K)
PICK_C, DROP_C_code = tops.get(1, 0), tops.get(2, 1)
print(f"event-recall {recall:.2f} distinct {distinct}; pickup->code {PICK_C}, drop->code {DROP_C_code}", flush=True)


def to(x): return torch.tensor(x, device=device, dtype=torch.float32)


@torch.no_grad()
def rollout(kind, z0, acts):                                       # acts [N,H,ADIM]; returns zs[N,H,SDIM], codes[N,H] or None
    N, H = acts.shape[0], acts.shape[1]; z = to(np.repeat(z0[None], N, 0)); zs, codes = [], []
    for h in range(H):
        a = to(acts[:, h])
        if kind == "bn":
            dz, c = bn.step(z, a); codes.append(c)
        else:
            dz = dense(torch.cat([z, a], -1))
        z = z + dz; zs.append(z)
    return torch.stack(zs, 1), (torch.stack(codes, 1) if codes else None)


def cem(kind, cost_fn, z0, N, iters=3):
    mean = np.zeros((args.H, ADIM), "float32"); std = np.ones((args.H, ADIM), "float32") * STEP; best = np.zeros(ADIM, "float32")
    for _ in range(iters):
        acts = np.clip(mean + std * rng.standard_normal((N, args.H, ADIM)), -STEP, STEP).astype("float32")
        zs, codes = rollout(kind, z0, acts)
        cost = cost_fn(zs, codes).cpu().numpy()
        order = cost.argsort(); elite = acts[order[:max(1, N // 10)]]
        mean, std, best = elite.mean(0), elite.std(0) + 1e-3, acts[order[0], 0]
    return best


def cost_goal(zs, codes): return (zs[:, -1, 2:4] - DROP).norm(dim=1)                       # object -> drop zone


def cost_reach(target):                                                                     # min distance agent->target
    t = to(target)
    return lambda zs, codes: (zs[:, :, :2] - t).norm(dim=2).min(1).values


def cost_event(target_code):                                                                # earliest step triggering code
    def f(zs, codes):
        hit = (codes == target_code).float()                                                # [N,H]
        idx = torch.where(hit.any(1), hit.argmax(1).float(), torch.full((len(zs),), args.H, device=device))
        return idx
    return f


def run(planner, N, eps, g):
    succ = 0
    for _ in range(eps):
        s = env.reset(g); sg = 0
        for t in range(args.T_plan):
            if planner == "dense":
                a = cem("dense", cost_goal, s, N)
            elif planner == "event":
                a = cem("bn", cost_event(PICK_C if sg == 0 else DROP_C_code), s, N)
            else:                                                                           # oracle-subgoal
                tgt = s[2:4] if sg == 0 else env.drop_c
                a = cem("dense", cost_reach(tgt), s, N)
            s, ev = env.step(s, a)
            if planner != "dense" and ev == (1 if sg == 0 else 2):
                sg = min(sg + 1, 1)
            if np.linalg.norm(s[2:4] - env.drop_c) < 0.12:
                succ += 1; break
    return succ / eps


NS = [64, 128, 256]
print("\n  planning success vs CEM samples (pickup->drop)")
print(f"    {'planner':16s} " + " ".join(f"N={n:<4d}" for n in NS))
res = {}
for p in ["dense", "event", "oracle-subgoal"]:
    res[p] = [run(p, n, args.plan_episodes, np.random.default_rng(100 + args.seed)) for n in NS]
    print(f"    {p:16s} " + " ".join(f"{r:5.2f} " for r in res[p]))

best = {p: max(res[p]) for p in res}
print("\n  VERDICT:")
print(f"    oracle-subgoal best {best['oracle-subgoal']:.2f} vs dense {best['dense']:.2f}  -> subgoal STRUCTURE "
      f"{'helps' if best['oracle-subgoal'] > best['dense'] + 0.15 else 'does NOT help (task not subgoal-bottlenecked)'}")
print(f"    event-CEM best {best['event']:.2f} vs oracle {best['oracle-subgoal']:.2f}  -> the model's event codes are "
      f"{'AS ACTIONABLE as perfect affordances (planning claim rescued)' if best['event'] >= best['oracle-subgoal'] - 0.15 else 'NOT actionable enough (descriptive != plannable -- user thesis)'}")
GO = best["event"] >= best["dense"] + 0.15 and best["event"] >= best["oracle-subgoal"] - 0.15
print("    " + ("PASS -- fair event-level planning beats dense and ~matches oracle subgoals." if GO else
                "FAIL -- event-level planning does not beat dense; see whether oracle subgoals would have."))

fig, ax = plt.subplots(figsize=(6, 4.4))
for p, c in [("dense", "#95a5a6"), ("event", "#c0392b"), ("oracle-subgoal", "#16a085")]:
    ax.plot(NS, res[p], "-o", color=c, label=p)
ax.set_xlabel("CEM samples N"); ax.set_ylabel("success rate"); ax.set_title(f"C3 fair event-level planner ({'PASS' if GO else 'FAIL'})")
ax.legend(); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig("c3_event_planner.png", dpi=110)
print("\nsaved c3_event_planner.png")
