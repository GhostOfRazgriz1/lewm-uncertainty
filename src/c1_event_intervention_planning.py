"""C1 -- Event-JEPA FOUNDATION test: the two load-bearing claims C0 never tested.

C0 was a seeded NO-GO, but it only tested the WEAKEST legs (event discovery from reconstruction; raw
long-horizon prediction on low-dim). The proposal's real headline claims were untested. C1 tests them,
still cheap (CPU, minutes):

  CLAIM A -- INTERVENTION fixes discovery: does adding COUNTERFACTUAL action data (same state, many
      actions -> observe which trigger events) make the event code align with true events? C0's reason
      events didn't emerge: on-policy data rarely shows the same state with/without an event. Interventions
      give that contrast densely. Test = NMI(no-CF) vs NMI(+CF), same architecture. (env_step is a pure
      function, so counterfactuals are free.)

  CLAIM B -- EVENT-BIASED PLANNING is sample-efficient: on a SPARSE pickup->drop task (goal = object in
      drop zone; goal-cost is FLAT until a pickup moves the object, so a planner gets no gradient toward
      the pickup event), does biasing CEM toward triggering events solve it with FEWER samples than dense
      CEM? Metric = success rate vs CEM sample budget N, three planners (dense / event-BN / event-biased).

Pre-registered read: if CF doesn't lift NMI AND event-biased planning doesn't beat dense on samples, the
proposal fails its OWN headline on a favorable toy -> don't build the pixel ladder. Run: python src/c1_event_intervention_planning.py
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")                          # noqa: E402
import matplotlib.pyplot as plt                                   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--episodes", type=int, default=200)
ap.add_argument("--T", type=int, default=80)
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--K", type=int, default=6)
ap.add_argument("--hid", type=int, default=32)
ap.add_argument("--n_cf", type=int, default=4, help="counterfactual actions per visited state")
ap.add_argument("--plan_episodes", type=int, default=20)
ap.add_argument("--T_plan", type=int, default=60)
ap.add_argument("--H", type=int, default=18, help="CEM horizon")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
rng = np.random.default_rng(args.seed)
torch.manual_seed(args.seed)
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---- env with known sparse causal events (pure function -> counterfactuals are free) -------------
E_NAMES = ["none", "pickup", "drop", "switch"]
SDIM, ADIM, NEV = 6, 2, 4
STEP, R_PICK = 0.06, 0.12
DROP_C, DROP_R = np.array([0.85, 0.15], "float32"), 0.15
SW_C, SW_R = np.array([0.15, 0.85], "float32"), 0.15


def env_reset(g):
    return np.array([*g.uniform(0.1, 0.9, 2), *g.uniform(0.1, 0.9, 2), 0.0, 0.0], "float32")


def env_step(s, a):
    s = s.copy(); s[:2] = np.clip(s[:2] + a, 0, 1); ev = 0
    if s[4] < 0.5 and np.linalg.norm(s[:2] - s[2:4]) < R_PICK:
        s[4] = 1.0; ev = 1
    elif s[4] > 0.5 and np.linalg.norm(s[:2] - DROP_C) < DROP_R:
        s[4] = 0.0; ev = 2
    if s[4] > 0.5:
        s[2:4] = s[:2]
    if ev == 0 and s[5] < 0.5 and np.linalg.norm(s[:2] - SW_C) < SW_R:
        s[5] = 1.0; ev = 3
    return s, ev


def policy(s, g):
    if g.random() < 0.5:
        target = s[2:4] if s[4] < 0.5 else (DROP_C if g.random() < 0.5 else SW_C)
        d = target - s[:2]; return (d / (np.linalg.norm(d) + 1e-9) * STEP).astype("float32")
    return (g.uniform(-1, 1, ADIM) * STEP).astype("float32")


def collect(g, n_ep, T, n_cf):
    on, cf, ep_rollouts = [], [], []
    for _ in range(n_ep):
        s = env_reset(g); ep_s = [s]
        for t in range(T):
            a = policy(s, g); s2, ev = env_step(s, a)
            on.append((s, a, s2, ev))
            for _ in range(n_cf):                                   # INTERVENTIONS: same state, other actions
                acf = (g.uniform(-1, 1, ADIM) * STEP).astype("float32")
                scf, evcf = env_step(s, acf); cf.append((s, acf, scf, evcf))
            ep_s.append(s2); s = s2
        ep_rollouts.append(np.array(ep_s))
    pack = lambda L: (np.array([x[0] for x in L], "float32"), np.array([x[1] for x in L], "float32"),
                      np.array([x[2] for x in L], "float32"), np.array([x[3] for x in L]))
    return pack(on), pack(cf), ep_rollouts


def nmi(a, b):
    a, b = a.astype(int), b.astype(int)
    j = np.zeros((a.max() + 1, b.max() + 1)); np.add.at(j, (a, b), 1.0); j /= j.sum()
    pa, pb = j.sum(1), j.sum(0)
    mi = np.sum(j[j > 0] * np.log(j[j > 0] / (pa[:, None] * pb[None, :])[j > 0] + 1e-12))
    ent = lambda p: -np.sum(p[p > 0] * np.log(p[p > 0]))
    Ha, Hb = ent(pa), ent(pb)
    return 0.0 if Ha == 0 or Hb == 0 else float(mi / ((Ha + Hb) / 2))


(onZ, onA, onZn, onEV), (cfZ, cfA, cfZn, cfEV), ep_rollouts = collect(rng, args.episodes, args.T, args.n_cf)
print(f"data: on-policy {len(onZ)} | counterfactual {len(cfZ)} | event freq(on) {np.bincount(onEV, minlength=NEV)/len(onEV)}", flush=True)
to = lambda x: torch.tensor(x, device=device, dtype=torch.float32)


def mlp(din, dout, hid, layers=2):
    seq = [nn.Linear(din, hid), nn.GELU()]
    for _ in range(layers - 1):
        seq += [nn.Linear(hid, hid), nn.GELU()]
    return nn.Sequential(*seq, nn.Linear(hid, dout))


class EventBN(nn.Module):
    def __init__(self, hid, K):
        super().__init__(); self.K = K
        self.base = mlp(SDIM + ADIM, SDIM, hid)
        self.post = mlp(SDIM + ADIM + SDIM, K, hid)
        self.prior = mlp(SDIM + ADIM, K, hid)
        self.eff = mlp(SDIM + ADIM + K, SDIM, hid)

    def forward(self, z, a, zn, tau=1.0):
        pl = self.post(torch.cat([z, a, zn], -1)); e = F.gumbel_softmax(pl, tau=tau, hard=True)
        dze = self.eff(torch.cat([z, a, e], -1))
        return self.base(torch.cat([z, a], -1)) + dze, pl, self.prior(torch.cat([z, a], -1)), dze

    def code(self, z, a):
        return self.prior(torch.cat([z, a], -1)).argmax(-1)

    def step(self, z, a):
        c = self.code(z, a)
        return self.base(torch.cat([z, a], -1)) + self.eff(torch.cat([z, a, F.one_hot(c, self.K).float()], -1)), c

    def post_code(self, z, a, zn):
        return self.post(torch.cat([z, a, zn], -1)).argmax(-1)


def train_bn(data, label):
    Z, A, Zn = to(data[0]), to(data[1]), to(data[2]); dZ = Zn - Z
    m = EventBN(args.hid, args.K).to(device); opt = torch.optim.Adam(m.parameters(), 2e-3)
    idx = np.arange(len(Z))
    for ep in range(args.epochs):
        rng.shuffle(idx); tau = max(0.5, 1.0 - ep / args.epochs)
        for i in range(0, len(Z), 512):
            b = idx[i:i + 512]
            dz, pl, prl, dze = m(Z[b], A[b], Zn[b], tau)
            pbar = F.softmax(pl, -1).mean(0)
            loss = ((dz - dZ[b]) ** 2).mean() + F.cross_entropy(prl, pl.detach().argmax(-1)) \
                + 0.01 * dze.abs().mean() + 0.05 * (pbar * torch.log(pbar + 1e-9)).sum()
            opt.zero_grad(); loss.backward(); opt.step()
    print(f"  trained event-BN ({label})", flush=True)
    return m


# ---- CLAIM A: does intervention (counterfactual) data lift event discovery? ----------------------
te = np.arange(int(0.8 * len(onZ)), len(onZ))                      # eval on held-out on-policy transitions
bn_nocf = train_bn((onZ, onA, onZn, onEV), "on-policy only")
cat_data = (np.concatenate([onZ, cfZ]), np.concatenate([onA, cfA]),
            np.concatenate([onZn, cfZn]), np.concatenate([onEV, cfEV]))
bn_cf = train_bn(cat_data, "on-policy + counterfactual")
with torch.no_grad():
    codes_nocf = bn_nocf.post_code(to(onZ[te]), to(onA[te]), to(onZn[te])).cpu().numpy()
    codes_cf = bn_cf.post_code(to(onZ[te]), to(onA[te]), to(onZn[te])).cpu().numpy()
true = onEV[te]


def event_metrics(codes):                                          # NMI is none-dominated; measure EVENT discovery
    base = np.bincount(true, minlength=NEV) / len(true)
    recalls, tops, lifts = {}, {}, {}
    for ev in range(1, NEV):                                       # skip 'none' (passive needn't be cleanly coded)
        m = true == ev
        if m.sum() < 3:
            continue
        h = np.bincount(codes[m], minlength=args.K); top = int(h.argmax())
        recalls[ev] = h[top] / m.sum(); tops[ev] = top
        in_top = codes == top
        lifts[ev] = ((true[in_top] == ev).mean() / (base[ev] + 1e-9)) if in_top.sum() else 0.0
    mean_recall = float(np.mean(list(recalls.values()))) if recalls else 0.0
    distinct = len(set(tops.values())) == len(tops) and len(tops) > 0
    return nmi(true, codes), mean_recall, distinct, recalls, tops, lifts


nmi_nocf, rec_nocf, _, _, _, _ = event_metrics(codes_nocf)
nmi_cf, rec_cf, distinct_cf, recalls, tops, lifts = event_metrics(codes_cf)
print("\n  CLAIM A -- intervention -> event discovery (NMI is none-dominated; event-recall is the real metric)")
print(f"    NMI:          on-policy {nmi_nocf:.3f} -> +counterfactual {nmi_cf:.3f}")
print(f"    event-recall: on-policy {rec_nocf:.3f} -> +counterfactual {rec_cf:.3f}   (events use distinct codes: {distinct_cf})")
for ev in range(1, NEV):
    if ev in recalls:
        print(f"      {E_NAMES[ev]:7s}: +CF dominant code {tops[ev]}  recall {recalls[ev]:.2f}  enrichment-lift {lifts[ev]:.1f}x")
gateA = rec_cf > 0.6 and distinct_cf and rec_cf > rec_nocf + 0.05

# ---- CLAIM B: event-biased planning sample efficiency --------------------------------------------
dense = mlp(SDIM + ADIM, SDIM, args.hid + 16).to(device)          # capacity-matched-ish dense baseline
optd = torch.optim.Adam(dense.parameters(), 2e-3)
Zc, Ac, dZc = to(cat_data[0]), to(cat_data[1]), to(cat_data[2] - cat_data[0]); idx = np.arange(len(Zc))
for ep in range(args.epochs):
    rng.shuffle(idx)
    for i in range(0, len(Zc), 512):
        b = idx[i:i + 512]; loss = ((dense(torch.cat([Zc[b], Ac[b]], -1)) - dZc[b]) ** 2).mean()
        optd.zero_grad(); loss.backward(); optd.step()
print("  trained dense baseline", flush=True)
with torch.no_grad():                                              # null code = modal prior code on data
    null_code = int(bn_cf.code(Zc, Ac).bincount(minlength=args.K).argmax())
DROP_t = to(DROP_C)


def make_step(kind):
    if kind == "dense":
        def f(z, a):
            return dense(torch.cat([z, a], -1)), torch.zeros(len(z), dtype=torch.bool, device=device)
    else:
        def f(z, a):
            dz, c = bn_cf.step(z, a); return dz, (c == null_code)
    return f


def cem(step_fn, s0, H, N, bias, iters=3):
    mean = np.zeros((H, ADIM), "float32"); std = np.ones((H, ADIM), "float32") * STEP
    best_a = np.zeros(ADIM, "float32")
    for it in range(iters):
        acts = np.clip(mean + std * rng.standard_normal((N, H, ADIM)), -STEP, STEP).astype("float32")
        z = to(np.repeat(s0[None], N, 0)); nullc = torch.zeros(N, device=device)
        with torch.no_grad():
            for h in range(H):
                dz, isnull = step_fn(z, to(acts[:, h])); z = z + dz; nullc += isnull.float()
            cost = (z[:, 2:4] - DROP_t).norm(dim=1)
            if bias > 0:
                cost = cost + bias * (nullc / H)                  # penalize event-free (null-heavy) plans
            cost = cost.cpu().numpy()
        order = cost.argsort(); elite = acts[order[:max(1, N // 10)]]
        mean, std = elite.mean(0), elite.std(0) + 1e-3; best_a = acts[order[0], 0]
    return best_a


def run_mpc(kind, N, bias, eps, g):
    step_fn = make_step(kind); succ = 0
    for _ in range(eps):
        s = env_reset(g)
        for t in range(args.T_plan):
            s, _ = env_step(s, cem(step_fn, s, args.H, N, bias))
            if np.linalg.norm(s[2:4] - DROP_C) < 0.12:            # object delivered to drop zone
                succ += 1; break
    return succ / eps


NS = [64, 128, 256]
planners = [("dense", "dense", 0.0), ("event-BN", "bn", 0.0), ("event-biased", "bn", 0.3)]
print("\n  CLAIM B -- planning success vs CEM samples (pickup->drop, sparse)")
print(f"    {'planner':14s} " + " ".join(f"N={n:<4d}" for n in NS))
res = {}
for name, kind, bias in planners:
    row = [run_mpc(kind, n, bias, args.plan_episodes, np.random.default_rng(100 + args.seed)) for n in NS]
    res[name] = row
    print(f"    {name:14s} " + " ".join(f"{r:5.2f} " for r in row))
# sample efficiency: smallest N reaching >=0.5 success
eff = {name: next((NS[i] for i, r in enumerate(res[name]) if r >= 0.5), None) for name in res}
gateB = (eff["event-biased"] is not None) and \
        (eff["dense"] is None or eff["event-biased"] <= eff["dense"]) and \
        (max(res["event-biased"]) > max(res["dense"]) + 0.1)

# ---- verdict ------------------------------------------------------------------------------------
print("\n  VERDICT (foundation test of the proposal's headline claims):")
print(f"    CLAIM A intervention->discovery : {'PASS' if gateA else 'FAIL'}  (event-recall {rec_nocf:.2f} -> {rec_cf:.2f}, distinct codes {distinct_cf})")
print(f"    CLAIM B event-planning efficiency: {'PASS' if gateB else 'FAIL'}  (success@best dense {max(res['dense']):.2f} vs event-biased {max(res['event-biased']):.2f}; N>=.5 dense {eff['dense']} biased {eff['event-biased']})")
go = gateA or gateB
print("    " + ("SOME LIFE -- at least one headline claim survives the toy; a pixel de-risk is now justified."
              if go else
              "NO-GO confirmed -- neither the intervention loss nor event-biased planning beats the baseline "
              "even on a favorable toy. The proposal's headline doesn't hold where it should be easiest."))

# ---- figure -------------------------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
ax[0].bar(["on-policy", "+counterfactual"], [nmi_nocf, nmi_cf], color=["#95a5a6", "#16a085"])
ax[0].axhline(0.30, ls="--", color="#c0392b", label="pass bar"); ax[0].set_ylabel("NMI(code, true event)")
ax[0].set_title("CLAIM A: intervention -> discovery"); ax[0].legend()
for name, col in [("dense", "#95a5a6"), ("event-BN", "#2980b9"), ("event-biased", "#c0392b")]:
    ax[1].plot(NS, res[name], "-o", color=col, label=name)
ax[1].set_xlabel("CEM samples N"); ax[1].set_ylabel("success rate"); ax[1].set_title("CLAIM B: planning efficiency"); ax[1].legend(); ax[1].grid(alpha=.3)
fig.suptitle(f"C1 Event-JEPA foundation test  |  {'SOME LIFE' if go else 'NO-GO'}", fontweight="bold")
fig.tight_layout(); fig.savefig("c1_event_foundation.png", dpi=110)
print("\nsaved c1_event_foundation.png")
