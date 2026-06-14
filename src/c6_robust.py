"""C6 -- ROBUSTNESS for the actionable-event planner (kill C5's 2/8 collapse).

C5: learned affordance-event planner reached the oracle in 6/8 seeds but collapsed to 0.00 in 2/8
(greedy single inverse-model BC is unstable on 50%-random mixed-policy data). C6 applies the fixes:
  1. ACTION-RELEVANT filter: keep a BC pair only if the action moved the agent TOWARD where the event
     fired (drops the random-noise targets that destabilize BC).
  2. REACHABILITY-GATED: train the inverse model only on states where the target event is reachable (<=K).
  3. ENSEMBLE inverse model (M heads) + use the mean (and disagreement as a confidence signal).
  4. CEM-AROUND-pi: seed CEM at the ensemble-pi rollout, refine locally with the dynamics model under a
     LEARNED cost = the affordance head's predicted reachability of the target event (no oracle location).
Report MEDIAN success and FAILURE RATE separately across seeds.

Per-seed planners printed: greedy-single (C5 repro, filtered), greedy-ensemble, cem-around-pi.
Run one seed: python src/c6_robust.py --seed 0   (loop seeds in bash to get the distribution)
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from _event_common import EventEnv, SDIM, ADIM, NEV, STEP, mlp

ap = argparse.ArgumentParser()
ap.add_argument("--episodes", type=int, default=300); ap.add_argument("--T", type=int, default=80)
ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--Klook", type=int, default=12)
ap.add_argument("--hid", type=int, default=48); ap.add_argument("--M", type=int, default=4, help="inverse-model ensemble size")
ap.add_argument("--plan_episodes", type=int, default=20); ap.add_argument("--T_plan", type=int, default=50)
ap.add_argument("--H", type=int, default=12); ap.add_argument("--Ncem", type=int, default=128); ap.add_argument("--seed", type=int, default=0)
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
# inverse pairs with ACTION-RELEVANT filter + reachability gate; affordance multi-hot
invZ, invE, invA, affZ, affY = [], [], [], [], []
for (S, Av, Ev) in eps:
    T = len(Av)
    for t in range(T):
        reach = np.zeros(NEV, "float32"); e_next, j_next = 0, 0
        for j in range(1, args.Klook + 1):
            if t + j - 1 < T and Ev[t + j - 1] > 0:
                reach[Ev[t + j - 1]] = 1.0
                if e_next == 0:
                    e_next, j_next = Ev[t + j - 1], t + j - 1
        affZ.append(S[t]); affY.append(reach)
        if e_next > 0:
            L = S[j_next][:2]                                       # agent position when the event fired
            if np.linalg.norm(S[t][:2] + Av[t] - L) < np.linalg.norm(S[t][:2] - L):   # action-relevant
                invZ.append(S[t]); invE.append(oh(e_next)); invA.append(Av[t])
invZ, invE, invA = map(lambda x: to(np.array(x)), (invZ, invE, invA))
affZ, affY = to(np.array(affZ)), to(np.array(affY))
Zall = to(np.concatenate([e[0][:-1] for e in eps])); Aall = to(np.concatenate([e[1] for e in eps]))
dZ = to(np.concatenate([e[0][1:] - e[0][:-1] for e in eps]))

ens = nn.ModuleList([mlp(SDIM + NEV, ADIM, args.hid) for _ in range(args.M)]).to(device)
aff = mlp(SDIM, NEV, args.hid).to(device); dense = mlp(SDIM + ADIM, SDIM, args.hid + 16).to(device)
opt = torch.optim.Adam(list(ens.parameters()) + list(aff.parameters()) + list(dense.parameters()), 2e-3)
nI, nA, nD = len(invZ), len(affZ), len(Zall)
for ep in range(args.epochs):
    for _ in range(max(nI, nA, nD) // 512 + 1):
        ba = rng.integers(0, nA, 512); bd = rng.integers(0, nD, 512)
        l_inv = 0
        for pi in ens:                                             # each member its own bootstrap batch (diversity)
            bb = rng.integers(0, nI, 512)
            l_inv = l_inv + ((pi(torch.cat([invZ[bb], invE[bb]], -1)) - invA[bb]) ** 2).mean()
        l_aff = F.binary_cross_entropy_with_logits(aff(affZ[ba]), affY[ba])
        l_dyn = ((dense(torch.cat([Zall[bd], Aall[bd]], -1)) - dZ[bd]) ** 2).mean()
        opt.zero_grad(); (l_inv + l_aff + l_dyn).backward(); opt.step()
print(f"filtered inverse pairs {nI} (of {nA} states); trained ensemble(M={args.M})+aff+dense", flush=True)
with torch.no_grad():                                              # does the inverse model USE the event condition?
    zb = Zall[:512]; e1 = to(oh(1)).repeat(len(zb), 1); e2 = to(oh(2)).repeat(len(zb), 1)
    ap_ = torch.stack([pi(torch.cat([zb, e1], -1)) for pi in ens]).mean(0)
    ad_ = torch.stack([pi(torch.cat([zb, e2], -1)) for pi in ens]).mean(0)
    print(f"SEED {args.seed} cond-sensitivity ||pi(.,pickup)-pi(.,drop)|| {float((ap_ - ad_).norm(dim=1).mean()):.4f} (STEP={STEP})", flush=True)
    # does pi(.,pickup) actually point toward the object? (cos alignment); restrict to not-carrying states
    nc = zb[:, 4] < 0.5
    d_obj = zb[nc, 2:4] - zb[nc, :2]; d_obj = d_obj / (d_obj.norm(dim=1, keepdim=True) + 1e-9)
    pn = ap_[nc] / (ap_[nc].norm(dim=1, keepdim=True) + 1e-9)
    print(f"SEED {args.seed} pickup-direction-alignment(cos with toward-object) {float((pn * d_obj).sum(1).mean()):.3f}", flush=True)
    # drop phase: for CARRYING states, does pi(.,drop) point toward the drop zone? (drop data is rarer)
    car = Zall[Zall[:, 4] > 0.5][:512]
    if len(car) > 8:
        ad2 = torch.stack([pi(torch.cat([car, to(oh(2)).repeat(len(car), 1)], -1)) for pi in ens]).mean(0)
        d_dz = to(env.drop_c) - car[:, :2]; d_dz = d_dz / (d_dz.norm(dim=1, keepdim=True) + 1e-9)
        pn2 = ad2 / (ad2.norm(dim=1, keepdim=True) + 1e-9)
        print(f"SEED {args.seed} drop-direction-alignment(cos with toward-dropzone) {float((pn2 * d_dz).sum(1).mean()):.3f}  (n_carry={len(car)})", flush=True)


@torch.no_grad()
def ens_act(s, e):
    x = torch.cat([to(s), to(oh(e))]); acts = torch.stack([pi(x) for pi in ens])
    return acts.mean(0).cpu().numpy(), acts.std(0).mean().item()


@torch.no_grad()
def cem_around_pi(s, e, N):
    # seed: roll the ensemble-pi greedily H steps with the dynamics model
    seed = np.zeros((args.H, ADIM), "float32"); z = s.copy()
    for h in range(args.H):
        a, _ = ens_act(z, e); seed[h] = np.clip(a, -STEP, STEP)
        z = (z + dense(torch.cat([to(z), to(seed[h])], -1)).cpu().numpy())
    mean, std = seed.copy(), np.ones((args.H, ADIM), "float32") * STEP * 0.5
    et = to(oh(e))
    for _ in range(2):
        acts = np.clip(mean + std * rng.standard_normal((N, args.H, ADIM)), -STEP, STEP).astype("float32")
        z = to(np.repeat(s[None], N, 0)); aff_sum = torch.zeros(N, device=device)
        for h in range(args.H):
            z = z + dense(torch.cat([z, to(acts[:, h])], -1))
            aff_sum = aff_sum + torch.sigmoid(aff(z)) @ et                     # LEARNED cost: reach affordance of e
        cost = (-aff_sum).cpu().numpy(); order = cost.argsort()
        elite = acts[order[:max(1, N // 10)]]; mean, std = elite.mean(0), elite.std(0) + 1e-3
    return np.clip(mean[0], -STEP, STEP)


def run(planner):
    g = np.random.default_rng(100 + args.seed); succ = 0
    for _ in range(args.plan_episodes):
        s = env.reset(g); sg = 0
        for t in range(args.T_plan):
            e = 1 if sg == 0 else 2
            if planner == "greedy-single":
                a = ens[0](torch.cat([to(s), to(oh(e))])).detach().cpu().numpy()
            elif planner == "greedy-ensemble":
                a, _ = ens_act(s, e)
            else:
                a = cem_around_pi(s, e, args.Ncem)
            s, ev = env.step(s, np.clip(a, -STEP, STEP))
            if ev == e:
                sg = min(sg + 1, 1)
            if np.linalg.norm(s[2:4] - env.drop_c) < 0.12:
                succ += 1; break
    return succ / args.plan_episodes


for p in ["greedy-single", "greedy-ensemble", "cem-around-pi"]:
    print(f"SEED {args.seed} {p:16s} {run(p):.2f}", flush=True)
