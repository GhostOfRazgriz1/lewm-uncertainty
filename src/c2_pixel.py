"""C2 -- PIXEL version: does a small dense predictor genuinely destabilize in a learned pixel latent,
where the event-bottleneck stays stable? (The one regime that could rescue Exp-2 / long-horizon.)

Render the event-world to 16x16 frames -> train a small JEPA encoder (next-latent prediction + VICReg
anti-collapse) -> FREEZE it -> train dense vs event-BN predictors in the frozen latent -> compare
long-horizon (median) rollout error in latent, and check event discovery survives in the pixel latent.

Hypothesis (the proposal's): in a higher-dim learned latent the dense rollout compounds error / diverges
while the event-factorized one stays bounded. If dense median rollout blows up and event-BN doesn't -> Exp-2
has life in pixels. If both are fine (encoder just recovers state) -> the toy can't reproduce it and a fair
test needs real pixels (LeWM). Run (background; CNN on CPU is the slow part): python src/c2_pixel.py
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")                          # noqa: E402
import matplotlib.pyplot as plt                                   # noqa: E402
from _event_common import EventEnv, ADIM, NEV, mlp, EventBN, event_metrics

ap = argparse.ArgumentParser()
ap.add_argument("--episodes", type=int, default=140); ap.add_argument("--T", type=int, default=60)
ap.add_argument("--enc_epochs", type=int, default=30); ap.add_argument("--pred_epochs", type=int, default=40)
ap.add_argument("--lat", type=int, default=32); ap.add_argument("--K", type=int, default=6)
ap.add_argument("--hid", type=int, default=48); ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
rng = np.random.default_rng(args.seed); torch.manual_seed(args.seed)
device = "cuda" if torch.cuda.is_available() else "cpu"
env = EventEnv(); IMG = 16


def render(s):
    img = np.zeros((IMG, IMG), "float32")
    def put(x, y, v):
        i, j = int(np.clip(x, 0, 1) * (IMG - 1)), int(np.clip(y, 0, 1) * (IMG - 1)); img[i, j] = max(img[i, j], v)
    put(env.drop_c[0], env.drop_c[1], 0.25); put(env.sw_c[0], env.sw_c[1], 0.25)
    put(s[2], s[3], 0.6); put(s[0], s[1], 1.0)
    if s[5] > 0.5:
        img[0, 0] = 1.0
    return img


def collect_eps(g, n):
    eps = []
    for _ in range(n):
        s = env.reset(g); S, Av, Ev = [s], [], []
        for _ in range(args.T):
            a = env.policy(s, g); s2, ev = env.step(s, a); S.append(s2); Av.append(a); Ev.append(ev); s = s2
        eps.append((np.array(S, "float32"), np.array(Av, "float32"), np.array(Ev)))
    return eps


eps = collect_eps(rng, args.episodes)
to = lambda x: torch.tensor(x, device=device, dtype=torch.float32)


class Enc(nn.Module):
    def __init__(self, lat):
        super().__init__()
        self.c = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.GELU(), nn.Conv2d(16, 32, 3, stride=2, padding=1),
                               nn.GELU(), nn.Flatten(), nn.Linear(32 * 8 * 8, lat))

    def forward(self, x):
        return self.c(x[:, None])


enc = Enc(args.lat).to(device); jep = mlp(args.lat + ADIM, args.lat, 64).to(device)
optj = torch.optim.Adam(list(enc.parameters()) + list(jep.parameters()), 1e-3)
# flatten transitions (image_t, a_t, image_{t+1})
IMt = to(np.stack([render(s) for e in eps for s in e[0][:-1]]))
IMn = to(np.stack([render(s) for e in eps for s in e[0][1:]]))
Aflat = to(np.concatenate([e[1] for e in eps])); EVflat = np.concatenate([e[2] for e in eps])
idx = np.arange(len(Aflat))
print(f"pixel data: {len(idx)} transitions, {IMG}x{IMG}", flush=True)
for ep in range(args.enc_epochs):
    rng.shuffle(idx); tot = 0
    for i in range(0, len(idx), 512):
        b = idx[i:i + 512]
        z = enc(IMt[b]); zn = enc(IMn[b]).detach()                # JEPA: stop-grad target
        pred = z + jep(torch.cat([z, Aflat[b]], -1))
        inv = ((pred - zn) ** 2).mean()
        std = torch.sqrt(z.var(0) + 1e-4); var = F.relu(1 - std).mean()    # VICReg variance (anti-collapse)
        zc = z - z.mean(0); cov = (zc.T @ zc) / (len(z) - 1)
        covl = (cov.fill_diagonal_(0) ** 2).sum() / args.lat
        loss = inv + 1.0 * var + 0.04 * covl
        optj.zero_grad(); loss.backward(); optj.step(); tot += inv.item()
    if ep % 10 == 0:
        print(f"  enc epoch {ep}: jepa inv {tot/(len(idx)//512+1):.4f}", flush=True)
enc.eval()


@torch.no_grad()
def encode_eps(eps):
    out = []
    for (S, Av, Ev) in eps:
        zs = enc(to(np.stack([render(s) for s in S]))).cpu().numpy()
        out.append((zs, Av, Ev))
    return out


zeps = encode_eps(eps)
Z = np.concatenate([z[:-1] for z, _, _ in zeps]); Zn = np.concatenate([z[1:] for z, _, _ in zeps])
A = np.concatenate([a for _, a, _ in zeps]); EV = EVflat
LAT = args.lat
print(f"frozen-latent rollout amplitude (std of dz): {np.std(Zn-Z):.3f}", flush=True)

# train dense + event-BN in the frozen latent
dense = mlp(LAT + ADIM, LAT, args.hid + 16).to(device); optd = torch.optim.Adam(dense.parameters(), 2e-3)
Zt, At, dZ = to(Z), to(A), to(Zn - Z); ix = np.arange(len(Z))
for ep in range(args.pred_epochs):
    rng.shuffle(ix)
    for i in range(0, len(Z), 512):
        b = ix[i:i + 512]; loss = ((dense(torch.cat([Zt[b], At[b]], -1)) - dZ[b]) ** 2).mean()
        optd.zero_grad(); loss.backward(); optd.step()
bn = EventBN(LAT, ADIM, args.hid, args.K).to(device); optb = torch.optim.Adam(bn.parameters(), 2e-3)
Znt = to(Zn)
for ep in range(args.pred_epochs):
    rng.shuffle(ix); tau = max(0.5, 1.0 - ep / args.pred_epochs)
    for i in range(0, len(Z), 512):
        b = ix[i:i + 512]
        dz, pl, prl, dze = bn(Zt[b], At[b], Znt[b], tau); pbar = F.softmax(pl, -1).mean(0)
        loss = ((dz - dZ[b]) ** 2).mean() + F.cross_entropy(prl, pl.detach().argmax(-1)) \
            + 0.01 * dze.abs().mean() + 0.05 * (pbar * torch.log(pbar + 1e-9)).sum()
        optb.zero_grad(); loss.backward(); optb.step()
print("trained dense + event-BN in pixel latent", flush=True)

# event discovery in the pixel latent
with torch.no_grad():
    codes = bn.post_code(Zt, At, Znt).cpu().numpy()
_, recall, distinct, tops = event_metrics(EV, codes, args.K)


@torch.no_grad()
def rollout_err(kind, HS=(10, 20, 50)):
    out = {h: [] for h in HS}
    for (zs, Av, _) in zeps:
        Zt2, At2, Tlen = to(zs), to(Av), len(Av)
        for h in HS:
            if h > Tlen:
                continue
            for s0 in range(0, Tlen - h, 5):
                z = Zt2[s0:s0 + 1].clone()
                for j in range(h):
                    a = At2[s0 + j:s0 + j + 1]
                    z = z + (dense(torch.cat([z, a], -1)) if kind == "dense" else bn.step(z, a)[0])
                out[h].append(float((z - Zt2[s0 + h:s0 + h + 1]).norm()))
    return ({h: float(np.median(v)) for h, v in out.items() if v},
            {h: float(np.mean(v)) for h, v in out.items() if v})


med_d, mean_d = rollout_err("dense"); med_b, mean_b = rollout_err("bn")
print("\n  C2 pixel-latent results")
print(f"    event discovery in pixel latent: event-recall {recall:.3f}  distinct {distinct}  codes {tops}")
print("\n  long-horizon rollout error in pixel latent (does dense destabilize?)")
print(f"    {'H':>4s} | {'dense med':>9s} {'BN med':>8s} {'BN/dense':>9s} | {'dense mean':>10s} {'BN mean':>8s}")
for h in (10, 20, 50):
    if h in med_d:
        print(f"    {h:4d} | {med_d[h]:9.3f} {med_b[h]:8.3f} {med_b[h]/(med_d[h]+1e-9):9.2f} | {mean_d[h]:10.3f} {mean_b[h]:8.3f}")

dense_destabilizes = (50 in mean_d) and (mean_d[50] > 3 * med_d[50])      # heavy tail = diverging rollouts
bn_helps_median = (50 in med_d) and med_b[50] < 0.9 * med_d[50]
print("\n  VERDICT:")
print(f"    dense rollouts destabilize in pixel latent (mean>>median): {'YES' if dense_destabilizes else 'NO'}")
print(f"    event-BN helps median long-horizon: {'YES' if bn_helps_median else 'NO'}  (BN/dense H50 median {med_b.get(50, float('nan'))/(med_d.get(50, 1)+1e-9):.2f})")
print(f"    event discovery survives pixels: {'YES' if recall > 0.6 and distinct else 'NO'} (recall {recall:.2f})")
print("    " + ("PIXEL LIFE -- event structure gives a real long-horizon advantage in the learned latent."
                if bn_helps_median else
                "NULL in pixels too -- " + ("dense does destabilize but event-BN doesn't fix the median" if dense_destabilizes
                                            else "the encoder yields a predictable latent; dense is fine, no Exp-2 advantage")))

fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
ax[0].imshow(render(eps[0][0][3]), cmap="magma"); ax[0].set_title("example 16x16 frame"); ax[0].axis("off")
HSp = [h for h in (10, 20, 50) if h in med_d]
ax[1].plot(HSp, [med_d[h] for h in HSp], "-o", color="#95a5a6", label="dense (median)")
ax[1].plot(HSp, [med_b[h] for h in HSp], "-o", color="#c0392b", label="event-BN (median)")
ax[1].set_xlabel("horizon H"); ax[1].set_ylabel("latent rollout err"); ax[1].set_title(f"C2 pixel long-horizon (recall {recall:.2f})")
ax[1].legend(); ax[1].grid(alpha=.3); fig.tight_layout(); fig.savefig("c2_pixel.png", dpi=110)
print("\nsaved c2_pixel.png")
