"""C0 -- DE-RISK GATE for Event-JEPA ('causal event bottleneck for small world models').

The project's whole thesis rides on one premise: a SMALL world model should not predict every latent
change equally; it should discover the SPARSE CAUSAL EVENTS that actually matter, and predict through a
compact 'transition vocabulary' (z,a -> e -> z') rather than a dense z,a -> z'. Before building the
planner / benchmark / multi-env matrix, this gate tests the falsifiable core in ~minutes on CPU:

  GATE 1 (events emerge UNSUPERVISED): the discrete event code e = q(z,a,z') aligns with the TRUE event
      labels of a synthetic env (collision/pickup/drop/switch) it was never told about. Metric = NMI and
      cluster purity of e-codes vs true events. If e is noise or just copies z', it FAILS.

  GATE 2 (the bottleneck HELPS long-horizon): a capacity-matched event-bottleneck predictor beats a DENSE
      predictor at H=10/20/50-step rollout (it commits to discrete events instead of MSE-smearing the rare
      high-impact transitions). At ROLLOUT the future z' is unavailable, so e comes from a learned PRIOR
      p(e|z,a); the prior is what must predict the event. If dense ties/beats bottleneck, the premise is
      not supported and we DON'T build the big thing.

This is a TOY designed to be the EASIEST case for events (small models, rare high-impact events, capacity
matched). So: GO here is necessary-not-sufficient (then scale to LeWM/pixels/real events); NO-GO here is a
strong kill signal (the premise failed even where it should be easiest). Self-contained -- numpy env +
torch MLPs, CPU-fine.  Run:  python src/c0_event_jepa_derisk.py
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")                          # noqa: E402
import matplotlib.pyplot as plt                                   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--episodes", type=int, default=300)
ap.add_argument("--T", type=int, default=80)
ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--K", type=int, default=6, help="event-code vocabulary size (>= #true events incl NONE)")
ap.add_argument("--hid", type=int, default=64)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
rng = np.random.default_rng(args.seed)
torch.manual_seed(args.seed)
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---- synthetic env with KNOWN sparse causal events (labels NOT used in training) -----------------
# state = [agent_x, agent_y, obj_x, obj_y, carrying, switch]   (6-d, all in [0,1] / {0,1})
# events: 0 NONE, 1 PICKUP (touch obj), 2 DROP (carry into drop-zone), 3 SWITCH (enter switch-zone once)
E_NAMES = ["none", "pickup", "drop", "switch"]
SDIM, ADIM, NEV = 6, 2, 4
STEP, R_PICK = 0.06, 0.12
DROP_C, DROP_R = np.array([0.85, 0.15]), 0.15
SW_C, SW_R = np.array([0.15, 0.85]), 0.15


def env_reset(g):
    return np.array([*g.uniform(0.1, 0.9, 2), *g.uniform(0.1, 0.9, 2), 0.0, 0.0], "float32")


def env_step(s, a):
    s = s.copy()
    s[:2] = np.clip(s[:2] + a, 0, 1)
    ev = 0
    if s[4] < 0.5 and np.linalg.norm(s[:2] - s[2:4]) < R_PICK:        # PICKUP
        s[4] = 1.0; ev = 1
    elif s[4] > 0.5 and np.linalg.norm(s[:2] - DROP_C) < DROP_R:      # DROP
        s[4] = 0.0; ev = 2
    if s[4] > 0.5:                                                    # carry: object follows agent
        s[2:4] = s[:2]
    if ev == 0 and s[5] < 0.5 and np.linalg.norm(s[:2] - SW_C) < SW_R:  # SWITCH (irreversible)
        s[5] = 1.0; ev = 3
    return s, ev


def policy(s, g):                                                    # mixed: half event-seeking, half random
    if g.random() < 0.5:                                            # -> events hit ~10-20% (like real envs)
        if s[4] < 0.5:
            target = s[2:4]                                         # not carrying: seek object (PICKUP)
        else:
            target = DROP_C if g.random() < 0.5 else SW_C           # carrying: drop-zone (DROP) or switch
        d = target - s[:2]
        return (d / (np.linalg.norm(d) + 1e-9) * STEP).astype("float32")
    return (g.uniform(-1, 1, ADIM) * STEP).astype("float32")


def collect(g, n_ep, T):
    Z, A, Zn, EV = [], [], [], []
    for _ in range(n_ep):
        s = env_reset(g)
        ep_s, ep_a, ep_e = [s], [], []
        for t in range(T):
            a = policy(s, g)
            s2, ev = env_step(s, a)
            Z.append(s); A.append(a); Zn.append(s2); EV.append(ev)
            ep_s.append(s2); ep_a.append(a); ep_e.append(ev); s = s2
        ep_rollouts.append((np.array(ep_s), np.array(ep_a, "float32"), np.array(ep_e)))
    return (np.array(Z, "float32"), np.array(A, "float32"), np.array(Zn, "float32"), np.array(EV))


ep_rollouts = []
Z, A, Zn, EV = collect(rng, args.episodes, args.T)
ntr = int(0.8 * len(Z))
print(f"data: {len(Z)} transitions, event freq {np.bincount(EV, minlength=NEV) / len(EV)}", flush=True)
to = lambda x: torch.tensor(x, device=device)
Zt, At, Znt, dZt = to(Z), to(A), to(Zn), to(Zn - Z)                   # predict delta


def nmi(a, b):                                                       # normalized MI (arithmetic), numpy
    a, b = a.astype(int), b.astype(int)
    j = np.zeros((a.max() + 1, b.max() + 1))
    np.add.at(j, (a, b), 1.0); j /= j.sum()
    pa, pb = j.sum(1), j.sum(0)
    mi = np.sum(j[j > 0] * np.log(j[j > 0] / (pa[:, None] * pb[None, :])[j > 0] + 1e-12))
    ent = lambda p: -np.sum(p[p > 0] * np.log(p[p > 0]))
    Ha, Hb = ent(pa), ent(pb)
    return 0.0 if Ha == 0 or Hb == 0 else float(mi / ((Ha + Hb) / 2))


def mlp(din, dout, hid, layers=2):
    seq = [nn.Linear(din, hid), nn.GELU()]
    for _ in range(layers - 1):
        seq += [nn.Linear(hid, hid), nn.GELU()]
    return nn.Sequential(*seq, nn.Linear(hid, dout))


# ---- dense predictor:  z,a -> dz  (capacity-matched below) ---------------------------------------
class Dense(nn.Module):
    def __init__(self, hid):
        super().__init__(); self.f = mlp(SDIM + ADIM, SDIM, hid)

    def forward(self, z, a):
        return self.f(torch.cat([z, a], -1))


# ---- event-bottleneck: continuous base B(z,a) + SPARSE additive event correction E(z,a,e) ---------
#   dz = base(z,a) + event_effect(z,a,e);  e is discrete (posterior q sees z', prior p does not).
#   L1 on the event correction => it is ~zero on passive steps, fires only on events (sparse overlay).
class EventBN(nn.Module):
    def __init__(self, hid, K):
        super().__init__(); self.K = K
        self.base = mlp(SDIM + ADIM, SDIM, hid)                      # continuous dynamics (carries motion)
        self.post = mlp(SDIM + ADIM + SDIM, K, hid)                  # train-only posterior
        self.prior = mlp(SDIM + ADIM, K, hid)
        self.eff = mlp(SDIM + ADIM + K, SDIM, hid)                   # event-driven correction

    def forward(self, z, a, zn, tau=1.0):
        post_logits = self.post(torch.cat([z, a, zn], -1))
        e = F.gumbel_softmax(post_logits, tau=tau, hard=True)
        dz_event = self.eff(torch.cat([z, a, e], -1))
        dz = self.base(torch.cat([z, a], -1)) + dz_event
        prior_logits = self.prior(torch.cat([z, a], -1))
        return dz, post_logits, prior_logits, dz_event

    @torch.no_grad()
    def rollout_dz(self, z, a):
        e = F.one_hot(self.prior(torch.cat([z, a], -1)).argmax(-1), self.K).float()
        return self.base(torch.cat([z, a], -1)) + self.eff(torch.cat([z, a, e], -1))

    @torch.no_grad()
    def post_code(self, z, a, zn):
        return self.post(torch.cat([z, a, zn], -1)).argmax(-1)


def nparams(m):
    return sum(p.numel() for p in m.parameters())


bn = EventBN(args.hid, args.K).to(device)
roll_params = nparams(bn.base) + nparams(bn.prior) + nparams(bn.eff)   # the rollout-path capacity
hid_d = args.hid
while nparams(Dense(hid_d)) < roll_params:                          # match dense to the rollout path
    hid_d += 8
dense = Dense(hid_d).to(device)
print(f"params: dense {nparams(dense)} (hid {hid_d}) vs bottleneck rollout-path {roll_params} "
      f"(prior+dec, hid {args.hid}); posterior {nparams(bn.post)} is train-only", flush=True)


def train(model, is_bn):
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    idx = np.arange(ntr)
    for ep in range(args.epochs):
        rng.shuffle(idx); tau = max(0.5, 1.0 - ep / args.epochs)
        for i in range(0, ntr, 512):
            b = idx[i:i + 512]
            if is_bn:
                dz, pl, prl, dze = model(Zt[b], At[b], Znt[b], tau)
                pbar = F.softmax(pl, -1).mean(0)                       # marginal code usage over batch
                usage_ent = -(pbar * torch.log(pbar + 1e-9)).sum()    # maximize -> use all codes (anti-collapse)
                loss = ((dz - dZt[b]) ** 2).mean() \
                    + F.cross_entropy(prl, pl.detach().argmax(-1)) \
                    + 0.01 * dze.abs().mean() \
                    - 0.05 * usage_ent
            else:
                loss = ((model(Zt[b], At[b]) - dZt[b]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return model


print("training dense ...", flush=True); train(dense, False)
print("training event-bottleneck ...", flush=True); train(bn, True)

# ---- GATE 1: do event codes emerge unsupervised? ------------------------------------------------
te = np.arange(ntr, len(Z))
codes = bn.post_code(Zt[te], At[te], Znt[te]).cpu().numpy()
true = EV[te]
nmi_val = nmi(true, codes)
# purity: each code -> its majority true event
purity = 0
for k in np.unique(codes):
    purity += np.bincount(true[codes == k], minlength=NEV).max()
purity /= len(true)
# per-event: is the rare event captured by a (mostly) dedicated code?
print("\n  GATE 1 -- unsupervised event discovery")
print(f"    NMI(code, true_event) = {nmi_val:.3f}   cluster purity = {purity:.3f}   (codes used: {len(np.unique(codes))}/{args.K})")
for ev in range(NEV):
    m = true == ev
    if m.sum() > 0:
        dom = np.bincount(codes[m], minlength=args.K)
        print(f"      {E_NAMES[ev]:7s} (n={m.sum():5d}): code hist {dom}  -> dominant code {dom.argmax()} gets {dom.max()/m.sum():.2f}")
gate1 = nmi_val > 0.30

# ---- GATE 2: does the bottleneck help long-horizon rollout? -------------------------------------
HS = [1, 10, 20, 50]
err_d, err_b = {h: [] for h in HS}, {h: [] for h in HS}
with torch.no_grad():
    for (es, ea, _) in ep_rollouts[int(0.8 * len(ep_rollouts)):]:    # held-out episodes
        es_t = to(es); ea_t = to(ea); Tlen = len(ea)
        for h in HS:
            if h > Tlen:
                continue
            for s0 in range(0, Tlen - h, 5):
                zd = es_t[s0:s0 + 1].clone(); zb = es_t[s0:s0 + 1].clone()
                for j in range(h):
                    a = ea_t[s0 + j:s0 + j + 1]
                    zd = zd + dense(zd, a)
                    zb = zb + bn.rollout_dz(zb, a)
                tgt = es_t[s0 + h:s0 + h + 1]
                err_d[h].append(float((zd - tgt).norm())); err_b[h].append(float((zb - tgt).norm()))
md = {h: np.mean(err_d[h]) for h in HS if err_d[h]}
mb = {h: np.mean(err_b[h]) for h in HS if err_b[h]}
medd = {h: np.median(err_d[h]) for h in HS if err_d[h]}              # robust to a few diverged rollouts
medb = {h: np.median(err_b[h]) for h in HS if err_b[h]}
print("\n  GATE 2 -- long-horizon rollout error (lower=better)")
print(f"    {'H':>4s} | {'dense(mean)':>11s} {'BN(mean)':>9s} | {'dense(med)':>10s} {'BN(med)':>9s} {'BN/dense med':>12s}")
for h in HS:
    if h in md:
        print(f"    {h:4d} | {md[h]:11.4f} {mb[h]:9.4f} | {medd[h]:10.4f} {medb[h]:9.4f} {medb[h]/(medd[h]+1e-9):12.2f}")
gate2 = (50 in medb) and (medb[50] < 0.9 * medd[50])                 # robust (median) long-horizon win

# ---- verdict ------------------------------------------------------------------------------------
go = gate1 and gate2
print("\n  VERDICT:")
print(f"    GATE1 events-emerge : {'PASS' if gate1 else 'FAIL'}  (NMI {nmi_val:.2f}, want >0.30)")
print(f"    GATE2 bottleneck-helps: {'PASS' if gate2 else 'FAIL'}  (H50 median BN {medb.get(50, float('nan')):.3f} vs dense {medd.get(50, float('nan')):.3f})")
print("    " + ("GO -- the core Event-JEPA premise survives the toy: events emerge unsupervised AND the "
                "bottleneck helps long-horizon. Scale to LeWM/pixels/real events (build the planner+benchmark)."
                if go else
                "NO-GO -- premise failed where it should be EASIEST. " +
                ("Events don't emerge unsupervised. " if not gate1 else "") +
                ("Bottleneck doesn't beat a capacity-matched dense predictor long-horizon. " if not gate2 else "") +
                "Reconsider before the big build (sweep K/capacity/event-rarity, or rethink the premise)."))

# ---- figure -------------------------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
hs = [h for h in HS if h in medd]
ax[0].plot(hs, [medd[h] for h in hs], "-o", color="#95a5a6", label="dense (median)")
ax[0].plot(hs, [medb[h] for h in hs], "-o", color="#c0392b", label="event-bottleneck (median)")
ax[0].set_xlabel("rollout horizon H"); ax[0].set_ylabel("state L2 error (median)"); ax[0].set_title("GATE 2: long-horizon prediction"); ax[0].legend(); ax[0].grid(alpha=.3)
cm = np.zeros((args.K, NEV))
for k, ev in zip(codes, true):
    cm[k, ev] += 1
cm = cm / (cm.sum(1, keepdims=True) + 1e-9)
im = ax[1].imshow(cm, aspect="auto", cmap="viridis", vmin=0, vmax=1)
ax[1].set_xticks(range(NEV)); ax[1].set_xticklabels(E_NAMES, rotation=30); ax[1].set_yticks(range(args.K))
ax[1].set_xlabel("true event"); ax[1].set_ylabel("event code"); ax[1].set_title(f"GATE 1: code->event (NMI {nmi_val:.2f})")
fig.colorbar(im, ax=ax[1], fraction=0.046)
fig.suptitle(f"C0 Event-JEPA de-risk  |  VERDICT: {'GO' if go else 'NO-GO'}", fontweight="bold")
fig.tight_layout(); fig.savefig("c0_event_jepa_derisk.png", dpi=110)
print("\nsaved c0_event_jepa_derisk.png")
