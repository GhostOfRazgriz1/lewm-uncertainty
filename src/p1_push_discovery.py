"""Pusher E1 -- DISCOVERY (C1 analog on the harder substrate): do CONTACT/MOVED/DELIVERED emerge as
distinct event codes UNSUPERVISED, when the events are EMERGENT from push dynamics (not state flags)?

This is the cheapest rung of the scale-up. If discovery survives emergent events, proceed to E2 (the gap).
If recall is low / codes don't separate the events, the descriptive-discovery step itself doesn't transfer.
Reuses the EventBN + metrics from _event_common. Run: python src/p1_push_discovery.py --seed 0
"""
import argparse
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")                          # noqa: E402
import matplotlib.pyplot as plt                                   # noqa: E402
from _event_common import train_bn, event_metrics
from _push_common import PushEnv, PSDIM, ADIM, PNEV, P_ENAMES

ap = argparse.ArgumentParser()
ap.add_argument("--episodes", type=int, default=300); ap.add_argument("--T", type=int, default=80)
ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--K", type=int, default=6)
ap.add_argument("--hid", type=int, default=32); ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
rng = np.random.default_rng(args.seed); torch.manual_seed(args.seed)
device = "cuda" if torch.cuda.is_available() else "cpu"

env = PushEnv()
(Z, A, Zn, EV), _ = env.collect(rng, args.episodes, args.T)
freq = np.bincount(EV, minlength=PNEV) / len(EV)
print(f"data: {len(Z)} transitions; event freq {dict(zip(P_ENAMES, freq.round(3)))}", flush=True)

bn = train_bn(Z, A, Zn, PSDIM, ADIM, args.hid, args.K, args.epochs, device, rng)
te = np.arange(int(0.8 * len(Z)), len(Z))
to = lambda x: torch.tensor(x, device=device, dtype=torch.float32)
with torch.no_grad():
    codes = bn.post_code(to(Z[te]), to(A[te]), to(Zn[te])).cpu().numpy()
true = EV[te]
nmi_val, recall, distinct, tops = event_metrics(true, codes, args.K)

print("\n  E1 DISCOVERY on the pusher (emergent events)")
print(f"    NMI {nmi_val:.3f}   event-recall {recall:.3f}   distinct-codes {distinct}   codes {tops}")
base = np.bincount(true, minlength=PNEV) / len(true)
for ev in range(1, PNEV):
    m = true == ev
    if m.sum() >= 3:
        h = np.bincount(codes[m], minlength=args.K); top = int(h.argmax())
        lift = ((true[codes == top] == ev).mean() / (base[ev] + 1e-9)) if (codes == top).sum() else 0
        print(f"      {P_ENAMES[ev]:10s} (n={m.sum():5d}): code {top} recall {h[top]/m.sum():.2f} lift {lift:.1f}x")
gate = recall > 0.6 and distinct
print("\n  VERDICT: " + ("DISCOVERY SURVIVES emergent events -> proceed to E2 (the gap)." if gate else
                         "DISCOVERY WEAK on emergent events -> the descriptive-discovery step doesn't transfer cleanly."))

# figure: code->event confusion (row-normalized)
cm = np.zeros((args.K, PNEV))
for k, e in zip(codes, true):
    cm[k, e] += 1
cm = cm / (cm.sum(1, keepdims=True) + 1e-9)
fig, ax = plt.subplots(figsize=(5.2, 4.2))
im = ax.imshow(cm, aspect="auto", cmap="viridis", vmin=0, vmax=1)
ax.set_xticks(range(PNEV)); ax.set_xticklabels(P_ENAMES, rotation=30); ax.set_yticks(range(args.K))
ax.set_xlabel("true event"); ax.set_ylabel("event code")
ax.set_title(f"Pusher E1 discovery (recall {recall:.2f}, {'PASS' if gate else 'WEAK'})")
fig.colorbar(im, ax=ax, fraction=0.046); fig.tight_layout(); fig.savefig("p1_push_discovery.png", dpi=110)
print("\nsaved p1_push_discovery.png")
