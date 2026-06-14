"""C4 -- transfer / composition test (the most plausible remaining win for Event-JEPA).

Train on layout A (drop/switch zones in two corners); test on layout B (zones moved to the OTHER corners).
Two questions, deliberately separated along the user's 'what happened vs what can be planned' axis:

  DISCOVERY transfer  (descriptive): does the event code still recognize pickup/drop/switch on B? The code
      is abstract (pickup = carrying 0->1) and the posterior sees z', so it SHOULD transfer regardless of
      where the zones are. This is the plausible win -- but it is still DESCRIPTIVE.

  PREDICTION transfer (actionable):  does long-horizon rollout transfer better with event structure? The
      PRIOR must predict WHERE drop happens -- which moved -- so this is the location-specific, actionable
      part. dense vs event-BN median rollout error on A (in-dist) vs B (transfer).

If discovery transfers but prediction does NOT (and event-BN is no better than dense on B), that confirms:
the event code transfers as a DESCRIPTOR but not as something a planner can use -- descriptive != actionable.

Run: python src/c4_transfer.py
"""
import argparse
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")                          # noqa: E402
import matplotlib.pyplot as plt                                   # noqa: E402
from _event_common import EventEnv, SDIM, ADIM, NEV, E_NAMES, train_bn, event_metrics

ap = argparse.ArgumentParser()
ap.add_argument("--episodes", type=int, default=200); ap.add_argument("--T", type=int, default=80)
ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--K", type=int, default=6)
ap.add_argument("--hid", type=int, default=32); ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
rng = np.random.default_rng(args.seed); torch.manual_seed(args.seed)
device = "cuda" if torch.cuda.is_available() else "cpu"

envA = EventEnv(drop_c=(0.85, 0.15), sw_c=(0.15, 0.85))           # train layout
envB = EventEnv(drop_c=(0.15, 0.15), sw_c=(0.85, 0.85))           # transfer layout (zones moved)
to = lambda x: torch.tensor(x, device=device, dtype=torch.float32)


def collect_eps(env, g, n_ep, T):
    eps = []
    for _ in range(n_ep):
        s = env.reset(g); S, Av, Ev = [s], [], []
        for _ in range(T):
            a = env.policy(s, g); s2, ev = env.step(s, a); S.append(s2); Av.append(a); Ev.append(ev); s = s2
        eps.append((np.array(S, "float32"), np.array(Av, "float32"), np.array(Ev)))
    return eps


def flat(eps):
    Z = np.concatenate([e[0][:-1] for e in eps]); A = np.concatenate([e[1] for e in eps])
    Zn = np.concatenate([e[0][1:] for e in eps]); EV = np.concatenate([e[2] for e in eps])
    return Z, A, Zn, EV


epsA = collect_eps(envA, rng, args.episodes, args.T)
epsA_te = collect_eps(envA, np.random.default_rng(args.seed + 7), 60, args.T)
epsB = collect_eps(envB, np.random.default_rng(args.seed + 99), 60, args.T)
Z, A, Zn, EV = flat(epsA)
bn = train_bn(Z, A, Zn, SDIM, ADIM, args.hid, args.K, args.epochs, device, rng)
dense = torch.nn.Sequential(torch.nn.Linear(SDIM + ADIM, args.hid + 16), torch.nn.GELU(),
                            torch.nn.Linear(args.hid + 16, args.hid + 16), torch.nn.GELU(),
                            torch.nn.Linear(args.hid + 16, SDIM)).to(device)
optd = torch.optim.Adam(dense.parameters(), 2e-3)
Zt, At, dZ = to(Z), to(A), to(Zn - Z); idx = np.arange(len(Z))
for ep in range(args.epochs):
    rng.shuffle(idx)
    for i in range(0, len(Z), 512):
        b = idx[i:i + 512]; loss = ((dense(torch.cat([Zt[b], At[b]], -1)) - dZ[b]) ** 2).mean()
        optd.zero_grad(); loss.backward(); optd.step()
print("trained event-BN + dense on layout A", flush=True)


# ---- DISCOVERY transfer -------------------------------------------------------------------------
def discovery(eps):
    Zf, Af, Znf, EVf = flat(eps)
    with torch.no_grad():
        codes = bn.post_code(to(Zf), to(Af), to(Znf)).cpu().numpy()
    return event_metrics(EVf, codes, args.K)


nmiA, recA, distA, topsA = discovery(epsA_te)
nmiB, recB, distB, topsB = discovery(epsB)
same_codes = sum(1 for ev in topsA if ev in topsB and topsA[ev] == topsB[ev])
print("\n  DISCOVERY transfer (event-recall; does the descriptive code still recognize events on B?)")
print(f"    layout A (in-dist): recall {recA:.3f}  codes {topsA}")
print(f"    layout B (transfer): recall {recB:.3f}  codes {topsB}   (same code as A for {same_codes}/{len(topsA)} events)")


# ---- PREDICTION transfer ------------------------------------------------------------------------
@torch.no_grad()
def rollout_err(kind, eps, HS=(10, 20, 50)):
    out = {h: [] for h in HS}
    for (S, Av, _) in eps:
        St, At2, Tlen = to(S), to(Av), len(Av)
        for h in HS:
            if h > Tlen:
                continue
            for s0 in range(0, Tlen - h, 5):
                z = St[s0:s0 + 1].clone()
                for j in range(h):
                    a = At2[s0 + j:s0 + j + 1]
                    z = z + (dense(torch.cat([z, a], -1)) if kind == "dense" else bn.step(z, a)[0])
                out[h].append(float((z - St[s0 + h:s0 + h + 1]).norm()))
    return {h: float(np.median(v)) for h, v in out.items() if v}


edA, ebA = rollout_err("dense", epsA_te), rollout_err("bn", epsA_te)
edB, ebB = rollout_err("dense", epsB), rollout_err("bn", epsB)
print("\n  PREDICTION transfer (median H50 rollout error; does event structure transfer to B?)")
print(f"    {'':10s} {'dense A':>8s} {'BN A':>8s} | {'dense B':>8s} {'BN B':>8s} {'BN/dense B':>11s}")
for h in (10, 20, 50):
    if h in edA:
        print(f"    H={h:<7d} {edA[h]:8.3f} {ebA[h]:8.3f} | {edB[h]:8.3f} {ebB[h]:8.3f} {ebB[h]/(edB[h]+1e-9):11.2f}")

# ---- verdict ------------------------------------------------------------------------------------
disc_transfers = recB > 0.6 and same_codes >= 2
pred_transfers = (50 in edB) and ebB[50] < 0.9 * edB[50]
print("\n  VERDICT:")
print(f"    DISCOVERY transfers : {'YES' if disc_transfers else 'NO'}  (B recall {recB:.2f}, {same_codes} events keep their code)")
print(f"    PREDICTION transfers (event-structure helps on B): {'YES' if pred_transfers else 'NO'}  (BN/dense H50 on B = {ebB.get(50, float('nan'))/(edB.get(50, 1)+1e-9):.2f})")
print("    " + ("WIN -- event codes transfer AND give a prediction advantage on a new layout." if (disc_transfers and pred_transfers)
                else "DESCRIPTIVE-ONLY -- " + ("codes transfer as descriptors" if disc_transfers else "codes don't even transfer descriptively") +
                ", but event structure gives no actionable (prediction) advantage on B = 'what happened != what can be planned'."))

fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
ax[0].bar(["A (in-dist)", "B (transfer)"], [recA, recB], color=["#16a085", "#e67e22"])
ax[0].axhline(0.6, ls="--", color="#c0392b"); ax[0].set_ylabel("event-recall"); ax[0].set_title("DISCOVERY transfer")
HSp = [h for h in (10, 20, 50) if h in edB]
ax[1].plot(HSp, [edB[h] for h in HSp], "-o", color="#95a5a6", label="dense (B)")
ax[1].plot(HSp, [ebB[h] for h in HSp], "-o", color="#c0392b", label="event-BN (B)")
ax[1].set_xlabel("horizon H"); ax[1].set_ylabel("median rollout err on B"); ax[1].set_title("PREDICTION transfer"); ax[1].legend(); ax[1].grid(alpha=.3)
fig.tight_layout(); fig.savefig("c4_transfer.png", dpi=110); print("\nsaved c4_transfer.png")
