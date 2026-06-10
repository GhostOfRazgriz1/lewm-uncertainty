"""M1.5 -- drift-aware surprise head: does training on the DEPLOY distribution close M1.4's PARTIAL gap?

M1.4: a head trained on TRUE latents predicts one-step surprise (held-out Spearman +0.38) but at DEPLOY,
on the DRIFTED maintained latent, no arm beats fixed-interval (PARTIAL) -- the train/deploy distribution
gap. M1.5 trains the head on exactly the deploy distribution: pairs (z_hat_drifted, a, h) -> realized
one-step error, collected by free-running with RANDOM looks (h = steps since last look). The head gets h
as an explicit input -- so it CAN learn the trivial 'error grows with h' trend (== fixed-interval); it
beats fixed only if it also learns STATE-DEPENDENT drift rate from (z, a). No LeWM retrain.

Two-sided, decisive:
  WIN  -- drift-aware learned arm beats fixed-interval beyond SEM and approaches oracle
          => the thread's FIRST constructive positive: calibrated WM uncertainty IS actionable for
             sensing once the surprise predictor sees the deploy distribution.
  NULL -- learned ~= fixed even trained on the deploy distribution
          => the obstacle is STRUCTURAL (h == uniform spacing; state-dependent divergence isn't readable
             off your own drifted estimate). Makes the M1.2-1.4 negative airtight.
Control: compare the head's predictability to h-alone (steps-since-look); the head only matters if it
clears the h trend a uniform schedule already exploits. Spec: docs/M1.5-drift-aware-spec.md.

Run on Colab GPU:  python src/surprise_head_drift.py    (run after M1.4; reuses the rig.)
"""
import sys
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import stable_worldmodel as swm                                   # noqa: F401  registers swm/PushT-v1
import matplotlib; matplotlib.use("Agg")                          # noqa: E402
import matplotlib.pyplot as plt                                   # noqa: E402

sys.path.insert(0, "/content/lewm-uncertainty")
from src.active_sense import (                                    # noqa: E402  the M1.3 rig (loads LeWM on import)
    device, rollout, encode_all, act_encode, _predict_one,
    tracking_errors, intrinsic_surprise, sem,
    HS, T_STEPS, T1, K_GRID, TARGET_K, N_TAU,
)
from src.schedules import fixed_interval_lookset, oracle_lookset, n_looks   # noqa: E402

N_TRAIN, N_EVAL = 40, 15
PASSES = 3                                                        # random-look passes per train rollout (samples varied h)
P_LOOK = 0.3                                                      # per-step look prob when collecting drift pairs
EPOCHS, LR, HIDDEN, WD = 400, 1e-3, 128, 1e-3
HNORM = float(T1)                                                # normalize steps-since-look by horizon
torch.manual_seed(0)


class DriftHead(nn.Module):
    """(z, a, h_norm) -> predicted log1p(one-step error from the drifted estimate)."""
    def __init__(self, d=192, h=HIDDEN):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * d + 1, h), nn.ReLU(), nn.Linear(h, h // 2), nn.ReLU(), nn.Linear(h // 2, 1))

    def forward(self, z, a, hn):
        return self.net(torch.cat([z, a, hn], dim=-1)).squeeze(-1)


def spearman(a, b):
    ra, rb = a.argsort().argsort().astype(float), b.argsort().argsort().astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


@torch.no_grad()
def collect_drift_pairs(emb_true, act_emb, rng):
    """Free-run with random looks; record (z_hat_{t-1}, a_{t-1}, h) -> realized one-step error e_t.
    h = steps since last look = the drift of the maintained latent z_hat_{t-1} (matches deploy exactly)."""
    zh = [emb_true[0]]; last = 0; Z, A, Hh, E = [], [], [], []
    for t in range(1, emb_true.shape[0]):
        a = max(0, t - HS); h = (t - 1) - last
        pred = _predict_one(torch.stack(zh[a:t]), act_emb[a:t])[0]
        Z.append(zh[t - 1]); A.append(act_emb[t - 1]); Hh.append(h)
        E.append(float((pred - emb_true[t]).norm()))
        if rng.random() < P_LOOK:
            zh.append(emb_true[t]); last = t                      # LOOK: reset drift
        else:
            zh.append(pred)
    return Z, A, Hh, E


@torch.no_grad()
def head_pred(z, a, h):
    hn = torch.tensor([[h / HNORM]], dtype=torch.float32, device=device)
    return float(torch.expm1(head(z[None], a[None], hn))[0])


@torch.no_grad()
def deploy_sim_h(emb_true, act_emb, tau):
    """Deploy the drift-aware head: predict forward; LOOK when predicted error >= tau; track h."""
    zh = [emb_true[0]]; looks = [0]; last = 0
    for t in range(1, emb_true.shape[0]):
        a = max(0, t - HS); h = (t - 1) - last
        s = head_pred(zh[t - 1], act_emb[t - 1], h)
        nxt = _predict_one(torch.stack(zh[a:t]), act_emb[a:t])[0]
        if s >= tau:
            zh.append(emb_true[t]); looks.append(t); last = t
        else:
            zh.append(nxt)
    err = (torch.stack(zh) - emb_true).norm(dim=-1).cpu().numpy()
    return err, looks


@torch.no_grad()
def signal_trace_h(emb_true, act_emb):
    """Free-run (never look) recording the head's signal -> distribution for the tau grid."""
    zh = [emb_true[0]]; sig = []; last = 0
    for t in range(1, emb_true.shape[0]):
        a = max(0, t - HS); h = (t - 1) - last
        sig.append(head_pred(zh[t - 1], act_emb[t - 1], h))
        zh.append(_predict_one(torch.stack(zh[a:t]), act_emb[a:t])[0])
    return sig


# ---- collect rollouts (train + eval split), encode once -----------------------------------------
gen = np.random.default_rng(2)
data = []
for r in range(N_TRAIN + N_EVAL):
    frames, acts = rollout(gym.make("swm/PushT-v1", render_mode="rgb_array"), T_STEPS, gen)
    et, ae = encode_all(frames), act_encode(acts)
    data.append((et, ae, intrinsic_surprise(et, ae)))
    if r % 10 == 0:
        print(f"encoded rollout {r}/{N_TRAIN + N_EVAL}", flush=True)
train, evalr = data[:N_TRAIN], data[N_TRAIN:]

# ---- build drift-aware training pairs (the deploy distribution) ----------------------------------
prng = np.random.default_rng(7)
Zl, Al, Hl, El = [], [], [], []
for et, ae, _ in train:
    for _ in range(PASSES):
        Z, A, Hh, E = collect_drift_pairs(et, ae, prng)
        Zl += Z; Al += A; Hl += Hh; El += E
Z = torch.stack(Zl); A = torch.stack(Al)
Hn = torch.tensor([[h / HNORM] for h in Hl], dtype=torch.float32, device=device)
Y = torch.tensor(np.log1p(El), dtype=torch.float32, device=device)
print(f"drift-aware training pairs: {len(El)}  (h range {min(Hl)}..{max(Hl)})", flush=True)

head = DriftHead().to(device)
opt = torch.optim.Adam(head.parameters(), lr=LR, weight_decay=WD)
for ep in range(EPOCHS):
    opt.zero_grad()
    loss = nn.functional.mse_loss(head(Z, A, Hn), Y)
    loss.backward(); opt.step()
    if ep % 100 == 0:
        print(f"epoch {ep}: train mse(log) {loss.item():.4f}", flush=True)
head.eval()

# ---- eval 1: predictability on the DEPLOY distribution (+ h-alone control) -----------------------
prng2 = np.random.default_rng(11)
ph, tt, hh = [], [], []
for et, ae, _ in evalr:
    Z, A, Hh, E = collect_drift_pairs(et, ae, prng2)
    with torch.no_grad():
        hn = torch.tensor([[h / HNORM] for h in Hh], dtype=torch.float32, device=device)
        pred = torch.expm1(head(torch.stack(Z), torch.stack(A), hn)).cpu().numpy()
    ph += list(pred); tt += E; hh += Hh
ph, tt, hh = np.array(ph), np.array(tt), np.array(hh, dtype=float)
corr, corr_h = spearman(ph, tt), spearman(hh, tt)
print(f"\n==== M1.5 eval 1 -- held-out Spearman with realized one-step error, DEPLOY distribution ({len(tt)}) ====")
print(f"  drift-aware head        : {corr:+.3f}")
print(f"  h alone (steps-since-look): {corr_h:+.3f}   (the trend a uniform schedule already exploits)")
print(f"  => head is interesting only if it clears h-alone by a margin.")

# ---- eval 2: deploy drift-aware vs fixed-interval vs oracle --------------------------------------
traces = []
for et, ae, _ in evalr:
    traces += signal_trace_h(et, ae)
TAU = np.quantile(np.array(traces), np.linspace(0.02, 0.98, N_TAU))
learn_b = np.zeros((N_TAU, N_EVAL)); learn_e = np.zeros((N_TAU, N_EVAL))
for ti, tau in enumerate(TAU):
    for ri, (et, ae, _) in enumerate(evalr):
        err, looks = deploy_sim_h(et, ae, tau)
        learn_e[ti, ri] = err.mean(); learn_b[ti, ri] = n_looks(looks) / T1
    print(f"deployed tau {ti + 1}/{N_TAU}: mean budget {learn_b[ti].mean():.2f}, err {learn_e[ti].mean():.3f}", flush=True)

fixed_e = np.zeros((len(K_GRID), N_EVAL)); orac_e = np.zeros((len(K_GRID), N_EVAL))
for ki, K in enumerate(K_GRID):
    for ri, (et, ae, s) in enumerate(evalr):
        fixed_e[ki, ri] = tracking_errors(fixed_interval_lookset(T1, K), et, ae).mean()
        orac_e[ki, ri] = tracking_errors(oracle_lookset(s, K), et, ae).mean()
fixed_b = np.array([k / T1 for k in K_GRID])

# ---- verdict ------------------------------------------------------------------------------------
ki = K_GRID.index(TARGET_K); tgt = TARGET_K / T1
li = int(np.argmin(np.abs(learn_b.mean(1) - tgt)))
le, ls = learn_e[li].mean(), sem(learn_e)[li]
fe, fs = fixed_e[ki].mean(), sem(fixed_e)[ki]
delta, pooled = fe - le, np.hypot(fs, ls)
print(f"\n==== M1.5 eval 2 -- deploy at budget ~{tgt:.2f} ({TARGET_K}/{T1} looks) ====")
print(f"  fixed-interval  : {fe:.3f} +/- {fs:.3f}")
print(f"  drift-aware     : {le:.3f} +/- {ls:.3f}   (achieved budget {learn_b[li].mean():.2f})")
print(f"  oracle (ceiling): {orac_e[ki].mean():.3f} +/- {sem(orac_e)[ki]:.3f}")
print(f"  drift-aware vs fixed: delta {delta:+.3f} (pooled SEM {pooled:.3f})")
print()
if delta > pooled:
    print("  WIN -- drift-aware training BEATS fixed-interval: state-dependent drift IS readable from the")
    print("        maintained estimate. The thread's first constructive positive.")
else:
    print("  NULL -- even trained on the deploy distribution, learned ~= fixed: the obstacle is STRUCTURAL")
    print("         (h == uniform spacing; state-dependent divergence isn't readable off your own drift).")

# ---- figure: predicted-vs-true (deploy dist.) + error-vs-budget ---------------------------------
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
ax[0].scatter(tt, ph, s=8, alpha=.3, color="#c0392b")
lim = float(max(tt.max(), ph.max()))
ax[0].plot([0, lim], [0, lim], "--", color="gray", alpha=.6)
ax[0].set_xlabel("true one-step error (deploy dist.)"); ax[0].set_ylabel("head predicted")
ax[0].set_title(f"Eval 1: drift-aware pred vs true (Spearman {corr:+.2f}; h-only {corr_h:+.2f})"); ax[0].grid(alpha=.3)

for b, e, name, c in [(fixed_b, fixed_e, "fixed-interval", "#7f8c8d"), (fixed_b, orac_e, "oracle", "#27ae60"),
                      (learn_b.mean(1), learn_e, "drift-aware (ours)", "#c0392b")]:
    o = np.argsort(b); b2, m, s = np.asarray(b)[o], e.mean(1)[o], sem(e)[o]
    ax[1].plot(b2, m, "-o", color=c, label=name); ax[1].fill_between(b2, m - s, m + s, color=c, alpha=.15)
ax[1].set_xlabel("observation budget (looks / horizon)"); ax[1].set_ylabel("latent-tracking error")
ax[1].set_title("Eval 2: drift-aware when-to-look (lower-left better)"); ax[1].legend(); ax[1].grid(alpha=.3)
fig.suptitle("M1.5 -- drift-aware surprise head (trained on the deploy distribution)", fontweight="bold")
fig.tight_layout(); fig.savefig("/content/lewm-uncertainty/lewm_drift_aware.png", dpi=110)
print("\nsaved lewm_drift_aware.png")
