"""M1.4 -- learned one-step-surprise head: is surprise causally predictable, sharply enough to schedule?

M1.3 found MC-dropout variance is flat -> ~random for look-scheduling, though the oracle shows real
headroom. M1.4 trains a tiny MLP (z, a) -> log1p(one-step error) on TRUE latents (free labels from the
M1.3 rig), with a held-out rollout split, then asks:
  (1) SCIENTIFIC: does the head's prediction correlate with true surprise on HELD-OUT rollouts?
      (MC-dropout's effective correlation was ~0.) Baselines: |action| and latent-drift.
  (2) DEPLOY: as a causal look-trigger, does the learned arm beat fixed-interval and approach the oracle?
No LeWM retrain. Spec: docs/M1.4-surprise-head-spec.md.

Verdict branches:
  WIN     -- head predicts surprise on held-out AND the learned arm beats fixed beyond SEM.
  PARTIAL -- head predicts surprise but deploy ~= fixed: train/deploy drift mismatch (-> M1.5).
  NULL    -- head can't predict surprise: it's intrinsically unpredictable one step ahead.

Run on Colab GPU:  python src/surprise_head.py
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

N_TRAIN, N_EVAL = 40, 15                                          # rollouts; ~T transitions each
EPOCHS, LR, HIDDEN, WD = 400, 1e-3, 128, 1e-3                     # head training (WD = weight decay, regularizes 384-d input)
torch.manual_seed(0)


class SurpriseHead(nn.Module):
    """(z, a) -> predicted log1p(one-step error). Tiny MLP; LeWM stays frozen."""
    def __init__(self, d=192, h=HIDDEN):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * d, h), nn.ReLU(), nn.Linear(h, h // 2), nn.ReLU(), nn.Linear(h // 2, 1))

    def forward(self, z, a):
        return self.net(torch.cat([z, a], dim=-1)).squeeze(-1)


def spearman(a, b):
    ra, rb = a.argsort().argsort().astype(float), b.argsort().argsort().astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


@torch.no_grad()
def head_pred(z, a):                                             # scalar predicted surprise (expm1 of log-output)
    return float(torch.expm1(head(z[None], a[None]))[0])


@torch.no_grad()
def deploy_sim(emb_true, act_emb, tau, score_fn):
    """Causal: predict forward deterministically; LOOK (reset to truth) when score_fn(zh, t) >= tau."""
    zh = [emb_true[0]]; looks = [0]
    for t in range(1, emb_true.shape[0]):
        a = max(0, t - HS)
        s = score_fn(zh, t)
        nxt = _predict_one(torch.stack(zh[a:t]), act_emb[a:t])[0]
        if s >= tau:
            zh.append(emb_true[t]); looks.append(t)
        else:
            zh.append(nxt)
    err = (torch.stack(zh) - emb_true).norm(dim=-1).cpu().numpy()
    return err, looks


@torch.no_grad()
def signal_trace(emb_true, act_emb, score_fn):
    """Free-run (never look) and record score_fn each step -> the signal distribution for a tau grid."""
    zh = [emb_true[0]]; sig = []
    for t in range(1, emb_true.shape[0]):
        a = max(0, t - HS)
        sig.append(score_fn(zh, t))
        zh.append(_predict_one(torch.stack(zh[a:t]), act_emb[a:t])[0])
    return sig


def make_score_fns(act_emb, raw_norm):
    """Causal per-step look-trigger signals (computed from the MAINTAINED latent zh, available at decision time)."""
    return {
        "learned": lambda zh, t: head_pred(zh[t - 1], act_emb[t - 1]),                     # head(z_hat_{t-1}, a_{t-1})
        "action-mag": lambda zh, t: float(raw_norm[t]),                                    # |action into step t|
        "latent-drift": lambda zh, t: float((zh[t - 1] - zh[t - 2]).norm()) if t >= 2 else 0.0,
    }


ARMS = ["learned", "action-mag", "latent-drift"]

# ---- collect rollouts (train + eval split), encode once -----------------------------------------
gen = np.random.default_rng(1)
data = []
for r in range(N_TRAIN + N_EVAL):
    frames, acts = rollout(gym.make("swm/PushT-v1", render_mode="rgb_array"), T_STEPS, gen)
    emb_true, act_emb = encode_all(frames), act_encode(acts)
    surp = intrinsic_surprise(emb_true, act_emb)
    raw_norm = np.concatenate([[0.0], np.linalg.norm(acts, axis=1)])       # |action| into each step ([T1]; idx t = into step t)
    data.append((emb_true, act_emb, surp, raw_norm))
    if r % 10 == 0:
        print(f"encoded rollout {r}/{N_TRAIN + N_EVAL}", flush=True)
train, evalr = data[:N_TRAIN], data[N_TRAIN:]

# ---- train the head on TRUE latents: (z_{t-1}, a_{t-1}) -> log1p(surprise[t]) --------------------
Z = torch.cat([et[:-1] for et, ae, s, rn in train])                        # [N*T,192]  z_{t-1}, t=1..T
A = torch.cat([ae for et, ae, s, rn in train])                             # [N*T,192]  a_{t-1} (action into t)
Y = torch.tensor(np.concatenate([np.log1p(s[1:]) for et, ae, s, rn in train]), dtype=torch.float32, device=device)
head = SurpriseHead().to(device)
opt = torch.optim.Adam(head.parameters(), lr=LR, weight_decay=WD)
for ep in range(EPOCHS):
    opt.zero_grad()
    loss = nn.functional.mse_loss(head(Z, A), Y)
    loss.backward(); opt.step()
    if ep % 100 == 0:
        print(f"head epoch {ep}: train mse(log) {loss.item():.4f}", flush=True)
head.eval()

# ---- eval 1: held-out correlation (the scientific question) -------------------------------------
ph, tt, am, dr = [], [], [], []
for et, ae, s, rn in evalr:
    with torch.no_grad():
        pred = torch.expm1(head(et[:-1], ae)).cpu().numpy()                # predicted surprise for steps 1..T
    ph.append(pred); tt.append(s[1:]); am.append(rn[1:])
    drift = [0.0] + [float((et[k] - et[k - 1]).norm()) for k in range(1, T_STEPS)]   # ||z_{t-1}-z_{t-2}||, causal
    dr.append(np.array(drift))
ph, tt, am, dr = map(np.concatenate, (ph, tt, am, dr))
corr_learned, corr_am, corr_dr = spearman(ph, tt), spearman(am, tt), spearman(dr, tt)
print(f"\n==== M1.4 eval 1 -- held-out Spearman with true one-step surprise ({len(tt)} transitions) ====")
print(f"  learned head : {corr_learned:+.3f}   (MC-dropout's effective corr was ~0 / flat)")
print(f"  |action|     : {corr_am:+.3f}")
print(f"  latent-drift : {corr_dr:+.3f}")

# ---- eval 2: deploy each causal arm + the fixed/oracle references on held-out rollouts -----------
sfns = [make_score_fns(ae, rn) for et, ae, s, rn in evalr]
traces = {arm: [] for arm in ARMS}
for (et, ae, s, rn), fns in zip(evalr, sfns):
    for arm in ARMS:
        traces[arm].extend(signal_trace(et, ae, fns[arm]))
tau_grids = {arm: np.quantile(np.array(traces[arm]), np.linspace(0.02, 0.98, N_TAU)) for arm in ARMS}

arm_b = {arm: np.zeros((N_TAU, N_EVAL)) for arm in ARMS}
arm_e = {arm: np.zeros((N_TAU, N_EVAL)) for arm in ARMS}
for arm in ARMS:
    for ti, tau in enumerate(tau_grids[arm]):
        for ri, ((et, ae, s, rn), fns) in enumerate(zip(evalr, sfns)):
            err, looks = deploy_sim(et, ae, tau, fns[arm])
            arm_e[arm][ti, ri] = err.mean(); arm_b[arm][ti, ri] = n_looks(looks) / T1
    print(f"deployed {arm}", flush=True)

fixed_e = np.zeros((len(K_GRID), N_EVAL)); orac_e = np.zeros((len(K_GRID), N_EVAL))
for ki, K in enumerate(K_GRID):
    for ri, (et, ae, s, rn) in enumerate(evalr):
        fixed_e[ki, ri] = tracking_errors(fixed_interval_lookset(T1, K), et, ae).mean()
        orac_e[ki, ri] = tracking_errors(oracle_lookset(s, K), et, ae).mean()
fixed_b = np.array([k / T1 for k in K_GRID])

# ---- verdict at TARGET_K ------------------------------------------------------------------------
ki = K_GRID.index(TARGET_K); tgt_b = TARGET_K / T1
li = int(np.argmin(np.abs(arm_b["learned"].mean(1) - tgt_b)))              # learned tau closest to target budget
learned_err, learned_sem = arm_e["learned"][li].mean(), sem(arm_e["learned"])[li]
fixed_err, fixed_s = fixed_e[ki].mean(), sem(fixed_e)[ki]
delta, pooled = fixed_err - learned_err, np.hypot(fixed_s, learned_sem)
print(f"\n==== M1.4 eval 2 -- deploy at budget ~{tgt_b:.2f} ({TARGET_K}/{T1} looks) ====")
print(f"  fixed-interval : {fixed_err:.3f} +/- {fixed_s:.3f}")
print(f"  learned        : {learned_err:.3f} +/- {learned_sem:.3f}   (achieved budget {arm_b['learned'][li].mean():.2f})")
print(f"  oracle (ceiling): {orac_e[ki].mean():.3f} +/- {sem(orac_e)[ki]:.3f}")
print(f"  learned vs fixed: delta {delta:+.3f} (pooled SEM {pooled:.3f})")
PREDICTABLE = corr_learned > 0.2 and corr_learned > max(corr_am, corr_dr)
print()
if PREDICTABLE and delta > pooled:
    print("  WIN -- surprise IS causally predictable AND the learned head schedules better than fixed-interval.")
elif PREDICTABLE:
    print("  PARTIAL -- head predicts surprise on held-out latents, but deploy ~= fixed: the train/deploy drift")
    print("            mismatch dominates -> M1.5 (drift-aware training).")
else:
    print("  NULL -- head cannot predict one-step surprise (corr ~ baselines): intrinsically unpredictable ahead.")

# ---- figure: predicted-vs-true surprise + error-vs-budget with the learned arm ------------------
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
ax[0].scatter(tt, ph, s=8, alpha=.3, color="#8e44ad")
lim = float(max(tt.max(), ph.max()))
ax[0].plot([0, lim], [0, lim], "--", color="gray", alpha=.6)
ax[0].set_xlabel("true one-step surprise"); ax[0].set_ylabel("head predicted")
ax[0].set_title(f"Eval 1: predicted vs true surprise (Spearman {corr_learned:+.2f})"); ax[0].grid(alpha=.3)

curves = [(fixed_b, fixed_e, "fixed-interval", "#7f8c8d"), (fixed_b, orac_e, "oracle", "#27ae60"),
          (arm_b["learned"].mean(1), arm_e["learned"], "learned (ours)", "#8e44ad"),
          (arm_b["action-mag"].mean(1), arm_e["action-mag"], "|action|", "#e67e22"),
          (arm_b["latent-drift"].mean(1), arm_e["latent-drift"], "latent-drift", "#16a085")]
for b, e, name, c in curves:
    o = np.argsort(b); b2, m, s = np.asarray(b)[o], e.mean(1)[o], sem(e)[o]
    ax[1].plot(b2, m, "-o", color=c, label=name); ax[1].fill_between(b2, m - s, m + s, color=c, alpha=.15)
ax[1].set_xlabel("observation budget (looks / horizon)"); ax[1].set_ylabel("latent-tracking error")
ax[1].set_title("Eval 2: when-to-look (lower-left better)"); ax[1].legend(); ax[1].grid(alpha=.3)
fig.suptitle("M1.4 -- learned surprise head for active sensing on LeWM", fontweight="bold")
fig.tight_layout(); fig.savefig("/content/lewm-uncertainty/lewm_surprise_head.png", dpi=110)
print("\nsaved lewm_surprise_head.png")
