"""C5 -- the PIVOT's constructive test: do ACTIONABLE/REACHABLE event abstractions close C3's gap?

C3 measured the gap: descriptive event-CEM ~= dense (~0.13), oracle-subgoal (perfect affordances) ~0.85.
The pivot thesis ('predictive events are not plannable events; small WMs need actionable events grounded
in reachability') says the fix is to LEARN affordance semantics, not just transition compression. This
builds them and tests whether they reach the oracle bar:

  - affordance head      g(z) -> P(event e reachable within K steps)   (multi-label, from data lookahead)
  - event inverse model  pi(a | z, e_target)                            (BC: action that led to e within K)
  - affordance-event planner = subgoal sequence [pickup, drop] executed by the LEARNED inverse model pi
    (greedy MPC). The high-level structure matches the oracle; the low level is LEARNED, not hand-coded.

Compare success vs CEM samples N: dense-CEM (baseline), oracle-subgoal (upper bound, hand-coded reach),
affordance-event (learned pi, N-independent). (Descriptive event-CEM ~= dense, established in C3.)
Verdict: if affordance-event ~= oracle and >> dense, the actionable-event method is ALIVE; if it ~= dense,
even learned reachability/inverse doesn't close the gap (the thesis hardens). The inverse/affordance are
trained on TRUE event labels -> gives the method its FAIREST shot (isolates reachability, not identification).

Run: python src/c5_affordance.py
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")                          # noqa: E402
import matplotlib.pyplot as plt                                   # noqa: E402
from _event_common import EventEnv, SDIM, ADIM, NEV, STEP, mlp

ap = argparse.ArgumentParser()
ap.add_argument("--episodes", type=int, default=300); ap.add_argument("--T", type=int, default=80)
ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--Klook", type=int, default=12, help="reachability lookahead")
ap.add_argument("--hid", type=int, default=48)
ap.add_argument("--plan_episodes", type=int, default=20); ap.add_argument("--T_plan", type=int, default=50); ap.add_argument("--H", type=int, default=15)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
rng = np.random.default_rng(args.seed); torch.manual_seed(args.seed)
device = "cuda" if torch.cuda.is_available() else "cpu"
env = EventEnv(); DROP = torch.tensor(env.drop_c, device=device)
to = lambda x: torch.tensor(x, device=device, dtype=torch.float32)


def collect_eps(g, n):
    eps = []
    for _ in range(n):
        s = env.reset(g); S, Av, Ev = [s], [], []
        for _ in range(args.T):
            a = env.policy(s, g); s2, ev = env.step(s, a); S.append(s2); Av.append(a); Ev.append(ev); s = s2
        eps.append((np.array(S, "float32"), np.array(Av, "float32"), np.array(Ev)))
    return eps


eps = collect_eps(rng, args.episodes)
# build (a) inverse pairs (z_t, e_next_within_K, a_t) and (b) affordance multi-hot (z_t -> events within K)
invZ, invE, invA, affZ, affY = [], [], [], [], []
for (S, Av, Ev) in eps:
    T = len(Av)
    for t in range(T):
        reach = np.zeros(NEV, "float32")
        e_next = 0
        for j in range(1, args.Klook + 1):
            if t + j - 1 < T and Ev[t + j - 1] > 0:
                reach[Ev[t + j - 1]] = 1.0
                if e_next == 0:
                    e_next = Ev[t + j - 1]
        affZ.append(S[t]); affY.append(reach)
        if e_next > 0:
            invZ.append(S[t]); invE.append(np.eye(NEV, dtype="float32")[e_next]); invA.append(Av[t])
invZ, invE, invA = map(lambda x: to(np.array(x)), (invZ, invE, invA))
affZ, affY = to(np.array(affZ)), to(np.array(affY))
print(f"data: {len(invZ)} inverse pairs, {len(affZ)} affordance labels (event rate within K: {affY[:,1:].mean(0).cpu().numpy()})", flush=True)

# ---- learned components --------------------------------------------------------------------------
pi = mlp(SDIM + NEV, ADIM, args.hid).to(device)                    # event inverse model
aff = mlp(SDIM, NEV, args.hid).to(device)                          # affordance head (reachability)
dense = mlp(SDIM + ADIM, SDIM, args.hid + 16).to(device)           # dynamics (for dense/oracle planners)
opt = torch.optim.Adam(list(pi.parameters()) + list(aff.parameters()) + list(dense.parameters()), 2e-3)
Zall = to(np.concatenate([e[0][:-1] for e in eps])); Aall = to(np.concatenate([e[1] for e in eps]))
dZ = to(np.concatenate([e[0][1:] - e[0][:-1] for e in eps]))
nI, nA, nD = len(invZ), len(affZ), len(Zall)
for ep in range(args.epochs):
    for _ in range(max(nI, nA, nD) // 512 + 1):
        bi = rng.integers(0, nI, 512); ba = rng.integers(0, nA, 512); bd = rng.integers(0, nD, 512)
        l_inv = ((pi(torch.cat([invZ[bi], invE[bi]], -1)) - invA[bi]) ** 2).mean()
        l_aff = F.binary_cross_entropy_with_logits(aff(affZ[ba]), affY[ba])
        l_dyn = ((dense(torch.cat([Zall[bd], Aall[bd]], -1)) - dZ[bd]) ** 2).mean()
        opt.zero_grad(); (l_inv + l_aff + l_dyn).backward(); opt.step()
with torch.no_grad():                                              # affordance-head accuracy (held-outish)
    pred = (torch.sigmoid(aff(affZ)) > 0.5).float()
    aff_acc = (pred[:, 1:] == affY[:, 1:]).float().mean().item()
print(f"trained pi + affordance + dense; affordance-head per-event acc {aff_acc:.3f}", flush=True)


# ---- planners (dense-CEM, oracle-subgoal share C3's CEM; affordance-event = greedy pi) -----------
@torch.no_grad()
def cem(cost_fn, s0, N, iters=3):
    mean = np.zeros((args.H, ADIM), "float32"); std = np.ones((args.H, ADIM), "float32") * STEP; best = np.zeros(ADIM, "float32")
    for _ in range(iters):
        acts = np.clip(mean + std * rng.standard_normal((N, args.H, ADIM)), -STEP, STEP).astype("float32")
        z = to(np.repeat(s0[None], N, 0)); zs = []
        for h in range(args.H):
            z = z + dense(torch.cat([z, to(acts[:, h])], -1)); zs.append(z)
        zs = torch.stack(zs, 1); cost = cost_fn(zs).cpu().numpy()
        order = cost.argsort(); elite = acts[order[:max(1, N // 10)]]; mean, std, best = elite.mean(0), elite.std(0) + 1e-3, acts[order[0], 0]
    return best


def cost_goal(zs): return (zs[:, -1, 2:4] - DROP).norm(dim=1)
def cost_reach(t): tt = to(t); return lambda zs: (zs[:, :, :2] - tt).norm(dim=2).min(1).values


@torch.no_grad()
def greedy_pi(s, e):
    a = pi(torch.cat([to(s), to(np.eye(NEV, dtype="float32")[e])])).cpu().numpy()
    return np.clip(a, -STEP, STEP)


def run(planner, N, eps_n, g):
    succ = 0
    for _ in range(eps_n):
        s = env.reset(g); sg = 0
        for t in range(args.T_plan):
            if planner == "dense":
                a = cem(cost_goal, s, N)
            elif planner == "oracle-subgoal":
                a = cem(cost_reach(s[2:4] if sg == 0 else env.drop_c), s, N)
            else:                                                  # affordance-event: learned inverse model, greedy
                a = greedy_pi(s, 1 if sg == 0 else 2)
            s, ev = env.step(s, a)
            if planner != "dense" and ev == (1 if sg == 0 else 2):
                sg = min(sg + 1, 1)
            if np.linalg.norm(s[2:4] - env.drop_c) < 0.12:
                succ += 1; break
    return succ / eps_n


NS = [64, 128, 256]
res = {}
res["dense"] = [run("dense", n, args.plan_episodes, np.random.default_rng(100 + args.seed)) for n in NS]
res["oracle-subgoal"] = [run("oracle-subgoal", n, args.plan_episodes, np.random.default_rng(100 + args.seed)) for n in NS]
aff_succ = run("affordance-event", 0, args.plan_episodes, np.random.default_rng(100 + args.seed))   # N-independent
print("\n  planning success (pickup->drop)")
print(f"    {'dense (vs N)':22s} {res['dense']}")
print(f"    {'oracle-subgoal (vs N)':22s} {res['oracle-subgoal']}")
print(f"    {'affordance-event':22s} {aff_succ:.2f}   (learned inverse model, greedy, N-independent)")

dense_b, oracle_b = max(res["dense"]), max(res["oracle-subgoal"])
print("\n  VERDICT:")
closes = aff_succ >= oracle_b - 0.2 and aff_succ >= dense_b + 0.2
print(f"    affordance-event {aff_succ:.2f} vs dense {dense_b:.2f} vs oracle {oracle_b:.2f}")
print("    " + ("ALIVE -- learned actionable/reachable events CLOSE the gap to the oracle (the pivot's method works)."
                if closes else
                "NOT YET -- learned inverse/affordance does NOT close the gap; " +
                ("it beats dense but trails oracle (partial)" if aff_succ >= dense_b + 0.2 else "~= dense (reachability head insufficient as built)") +
                " -> thesis hardens or method needs longer-horizon reachability."))

fig, ax = plt.subplots(figsize=(6.2, 4.4))
ax.plot(NS, res["dense"], "-o", color="#95a5a6", label="dense-CEM")
ax.plot(NS, res["oracle-subgoal"], "-o", color="#16a085", label="oracle-subgoal (upper bound)")
ax.axhline(aff_succ, ls="--", color="#c0392b", label=f"affordance-event (learned) {aff_succ:.2f}")
ax.set_xlabel("CEM samples N"); ax.set_ylabel("success rate"); ax.set_ylim(-0.02, 1.0)
ax.set_title(f"C5 actionable events {'CLOSE' if closes else 'do NOT close'} C3's gap"); ax.legend(); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig("c5_affordance.png", dpi=110); print("\nsaved c5_affordance.png")
