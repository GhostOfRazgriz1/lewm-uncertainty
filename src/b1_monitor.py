"""B1 / E1 -- a FREE uncertainty MONITOR for a competent agent under deployment shift.

B0 (GO) established that frozen pretrained TD-MPC2 is competent on Colab AND that its Q-ensemble
disagreement is a live, free signal (Spearman +0.55 with one-step error in-dist). B1 tests the actual
monitor claim: deploy the competent agent under OBSERVATION corruption and ask whether the free signal
(computable online from (z,a), NO clean reference) flags the timesteps where the agent's latent estimate
is wrong.

Substrate caveat (B0): TD-MPC2's SimNorm latent is not Gaussian, so the LeJEPA shell does not port. The
monitor here is Q-ENSEMBLE disagreement = var_k two_hot_inv(Q_k(z,a)). Baselines: latent-drift
||z_t - z_{t-1}|| (the trivial motion signal disag must beat, M1.4 lesson), MC-dropout variance (flat on
LeWM -- does it stay flat?), and a TRAINED supervised detector (the reviewer-proofing: free should ~= supervised).

Corruption is applied ONLY to the observation handed to the agent; the env state is real, so we still have
the TRUE obs and define the honest target = estimate divergence ||encode(corrupt_obs) - encode(true_obs)||.
Shifts: S1 Gaussian noise (sigma in {.1,.3,.6}*obs_std, iid p=.5), S2 sustained blackout bursts (K in
{5,15,30}), S3 subtle constant bias on a dim subset (defuses the A1 'corruption is obvious' point).

Metrics per (shift,severity), on a held-out episode split:
  (1) detection AUROC -- does the signal separate clean vs corrupted timesteps?
  (2) selective-prediction AURC -- rank by signal, risk = divergence on the kept fraction (% gap recovered).
  (3) beyond-drift -- partial Spearman(disag, target | drift): disag must add over motion.
  (4) free vs supervised -- AUROC(disag) vs AUROC(trained detector on z,a).

MONITORING ONLY -- no control intervention (that is E2/B2). Run on Colab GPU after the B0 setup cells:
  python src/b1_monitor.py --task cheetah-run --seed 1 --episodes 8 --max_steps 200
"""
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr, rankdata
from sklearn.metrics import roc_auc_score
import matplotlib; matplotlib.use("Agg")                          # noqa: E402
import matplotlib.pyplot as plt                                   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--task", default="cheetah-run")
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--episodes", type=int, default=8, help="deploy episodes per (shift,severity)")
ap.add_argument("--max_steps", type=int, default=200, help="cap steps/episode (monitor stats need fewer than 500)")
ap.add_argument("--mc", type=int, default=4, help="MC-dropout samples (0 disables)")
ap.add_argument("--tdmpc2_dir", default="/content/tdmpc2/tdmpc2")
args = ap.parse_args()

assert torch.cuda.is_available(), "needs a GPU runtime."
os.environ.setdefault("MUJOCO_GL", "egl")
device = "cuda"
torch.manual_seed(0)

TDMPC2_DIR = args.tdmpc2_dir
assert os.path.isdir(TDMPC2_DIR), f"tdmpc2 dir not found: {TDMPC2_DIR}"
sys.path.insert(0, TDMPC2_DIR); os.chdir(TDMPC2_DIR)
from hydra import compose, initialize_config_dir                 # noqa: E402
from common.parser import parse_cfg                              # noqa: E402
from common.seed import set_seed                                 # noqa: E402
from common import math as tdmath                                # noqa: E402
from envs import make_env                                        # noqa: E402
from tdmpc2 import TDMPC2                                         # noqa: E402
from huggingface_hub import hf_hub_download                      # noqa: E402
import hydra.utils as _hu                                        # noqa: E402
_hu.get_original_cwd = lambda: os.getcwd()                       # compose() API has no Hydra runtime

ckpt = hf_hub_download("nicklashansen/tdmpc2", filename=f"dmcontrol/{args.task}-{args.seed}.pt")
with initialize_config_dir(config_dir=TDMPC2_DIR, version_base=None):
    cfg = compose(config_name="config", overrides=[
        f"task={args.task}", f"checkpoint={ckpt}", "model_size=5", "compile=false", "save_video=false", "seed=1"])
cfg = parse_cfg(cfg)
set_seed(cfg.seed)
env = make_env(cfg)
agent = TDMPC2(cfg); agent.load(cfg.checkpoint); agent.model.eval()
print(f"[cfg] task={cfg.task} num_q={cfg.num_q} latent_dim={cfg.latent_dim} action_dim={cfg.action_dim}", flush=True)


@torch.no_grad()
def enc(obs):
    return agent.model.encode(obs.to(device).unsqueeze(0), None)              # [1, D]


@torch.no_grad()
def q_disag(z, a):
    a = a.to(device).unsqueeze(0)
    qv = tdmath.two_hot_inv(agent.model.Q(z, a, None, return_type="all"), cfg).squeeze(-1).squeeze(-1)  # [num_q]
    return float(qv.var(0))


@torch.no_grad()
def mc_var(z, a, M):
    if M <= 0:
        return 0.0
    a = a.to(device).unsqueeze(0)
    agent.model.train()                                                       # dropout ON
    s = [float(tdmath.two_hot_inv(agent.model.Q(z, a, None, return_type="all"), cfg).mean()) for _ in range(M)]
    agent.model.eval()
    return float(np.var(s))


# ---- obs statistics from a short clean rollout (for noise/bias scale) ----------------------------
obs, done, t = env.reset(), False, 0
buf = []
while not done and t < args.max_steps:
    buf.append(obs.numpy()); obs, _, done, _ = env.step(agent.act(obs, t0=(t == 0), task=None)); t += 1
obs_std = torch.from_numpy(np.std(np.stack(buf), 0).astype("float32")).clamp_min(1e-3)   # [obs_dim]
D = obs_std.shape[0]
bias_dims = np.zeros(D, "float32"); bias_dims[: max(1, D // 2)] = 1.0                     # subtle: half the dims
bias_vec = torch.from_numpy(bias_dims) * obs_std

CONFIGS = [("noise", 0.1), ("noise", 0.3), ("noise", 0.6),
           ("blackout", 5), ("blackout", 15), ("blackout", 30), ("subtle", 0.5)]


def corrupt(obs, shift, sev, rng):
    if shift == "noise":
        return obs + torch.from_numpy(rng.normal(0, 1, size=obs.shape).astype("float32")) * (sev * obs_std)
    if shift == "blackout":
        return torch.zeros_like(obs)
    return obs + bias_vec * sev                                               # subtle constant bias


def mask_for(shift, sev, rng):
    if shift == "blackout":
        K = int(sev); m = np.zeros(args.max_steps, bool); i = K
        while i < args.max_steps:
            m[i:i + K] = True; i += 2 * K                                     # clean K, burst K, ...
        return m
    return rng.random(args.max_steps) < 0.5                                   # iid p=.5 for noise/subtle


# ---- deploy under each shift, collect per-step records ------------------------------------------
def rollout(shift, sev, ep):
    rng = np.random.default_rng(1000 * ep + int(sev * 10))
    m = mask_for(shift, sev, rng)
    obs, done, t, prev_z = env.reset(), False, 0, None
    rows = []
    while not done and t < args.max_steps:
        true_obs = obs
        cobs = corrupt(true_obs, shift, sev, rng) if m[t] else true_obs
        z = enc(cobs)                                                         # agent's (possibly corrupt) latent
        a = agent.act(cobs, t0=(t == 0), task=None)
        zc = enc(true_obs)                                                    # clean reference (env state is real)
        rows.append((int(m[t]), float((z - zc).norm()), q_disag(z, a),
                     0.0 if prev_z is None else float((z - prev_z).norm()),
                     mc_var(z, a, args.mc), z.squeeze(0).cpu().numpy(), a.cpu().numpy()))
        prev_z = z
        obs, _, done, _ = env.step(a); t += 1
    return rows


# keys: corrupt, target(divergence), disag, drift, mc, z[D], a[A]
data = {}
for shift, sev in CONFIGS:
    rec = {k: [] for k in ("corrupt", "target", "disag", "drift", "mc")}
    Z, A, ep_id = [], [], []
    for ep in range(args.episodes):
        for (c, tgt, dg, dr, mcv, z, a) in rollout(shift, sev, ep):
            rec["corrupt"].append(c); rec["target"].append(tgt); rec["disag"].append(dg)
            rec["drift"].append(dr); rec["mc"].append(mcv); Z.append(z); A.append(a); ep_id.append(ep)
    for k in rec:
        rec[k] = np.asarray(rec[k], "float64")
    rec["Z"] = np.stack(Z); rec["A"] = np.stack(A); rec["ep"] = np.asarray(ep_id)
    data[(shift, sev)] = rec
    print(f"  collected {shift}-{sev}: {len(rec['target'])} steps, corrupt frac {rec['corrupt'].mean():.2f}", flush=True)


# ---- metrics ------------------------------------------------------------------------------------
COVS = np.linspace(0.1, 1.0, 19)


def aurc(signal, err):
    e = err[np.argsort(signal)]
    return float(np.mean([e[:max(1, int(c * len(e)))].mean() for c in COVS]))


def auroc(label, signal):
    return roc_auc_score(label, signal) if (label.min() == 0 and label.max() == 1) else float("nan")


def partial_spear(x, y, z):                                       # Spearman(x,y) controlling for z
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    res = lambda u: u - np.polyval(np.polyfit(rz, u, 1), rz)
    return float(np.corrcoef(res(rx), res(ry))[0, 1])


def fit_detector(Xtr, ytr, Xte):
    det = nn.Sequential(nn.Linear(Xtr.shape[1], 128), nn.GELU(), nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 1)).to(device)
    opt = torch.optim.Adam(det.parameters(), lr=1e-3)
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device); ytr_t = torch.tensor(ytr, dtype=torch.float32, device=device)
    lossf = nn.BCEWithLogitsLoss()
    for _ in range(300):
        opt.zero_grad(); loss = lossf(det(Xtr_t).squeeze(-1), ytr_t); loss.backward(); opt.step()
    with torch.no_grad():
        return torch.sigmoid(det(torch.tensor(Xte, dtype=torch.float32, device=device)).squeeze(-1)).cpu().numpy()


print("\n==== B1 monitor under shift  (task", args.task, ") ====")
hdr = f"  {'shift':14s} {'detAUROC':>9s} {'drift':>7s} {'mc':>6s} {'superv':>7s} | {'AURC%gap':>8s} {'drift%':>7s} | {'partial':>8s}"
print(hdr)
summ = {}
for (shift, sev), r in data.items():
    tr = r["ep"] < int(0.6 * args.episodes); te = ~tr                          # episode-level split
    lab = r["corrupt"][te]
    det_disag = auroc(lab, r["disag"][te]); det_drift = auroc(lab, r["drift"][te]); det_mc = auroc(lab, r["mc"][te])
    Xtr = np.concatenate([r["Z"][tr], r["A"][tr]], 1); Xte = np.concatenate([r["Z"][te], r["A"][te]], 1)
    det_sup = auroc(lab, fit_detector(Xtr, r["corrupt"][tr], Xte))
    # selective prediction AURC on the held-out split (lower=better)
    err = r["target"][te]; rnd = err.mean(); orc = aurc(err, err)
    gap = lambda s: 100 * (rnd - aurc(s, err)) / (rnd - orc + 1e-9)
    g_disag, g_drift = gap(r["disag"][te]), gap(r["drift"][te])
    part = partial_spear(r["disag"][te], err, r["drift"][te])
    summ[(shift, sev)] = dict(det_disag=det_disag, det_drift=det_drift, det_mc=det_mc, det_sup=det_sup,
                              g_disag=g_disag, g_drift=g_drift, partial=part)
    print(f"  {f'{shift}-{sev}':14s} {det_disag:9.3f} {det_drift:7.3f} {det_mc:6.3f} {det_sup:7.3f} | "
          f"{g_disag:7.0f}% {g_drift:6.0f}% | {part:8.3f}")

# ---- verdict ------------------------------------------------------------------------------------
gross = [(s, v) for (s, _), v in summ.items() if s in ("noise", "blackout")]
det_ok = np.nanmean([v["det_disag"] for _, v in gross]) > 0.7
beats_drift = np.nanmean([v["det_disag"] - v["det_drift"] for _, v in gross]) > 0.03 and \
              np.nanmean([v["partial"] for _, v in gross]) > 0.1
near_sup = np.nanmean([v["det_disag"] - v["det_sup"] for _, v in gross]) > -0.07            # within .07 of supervised
subtle = summ[("subtle", 0.5)]["det_disag"]
print("\n  verdict:")
print(f"    gross-shift detection AUROC(disag) avg = {np.nanmean([v['det_disag'] for _,v in gross]):.3f}  ({'>.7 ok' if det_ok else 'WEAK'})")
print(f"    beats latent-drift (AUROC gap + partial>0.1): {'YES' if beats_drift else 'NO -- signal may be mostly motion'}")
print(f"    free ~= supervised (disag within .07 of trained): {'YES' if near_sup else 'NO -- supervised wins, honest limit'}")
print(f"    subtle-shift (S3) detection AUROC: {subtle:.3f}  ({'detects subtle too' if subtle>0.65 else 'misses subtle (gross-only monitor)'})")
WIN = det_ok and beats_drift and near_sup
print("    " + ("WIN -- free Q-ensemble disag is a label-free, supervised-grade monitor under shift; beats motion."
              if WIN else "PARTIAL/WEAK -- see which condition failed above; reframe per spec verdict logic."))

# ---- figure -------------------------------------------------------------------------------------
labels = [f"{s}-{v}" for (s, v) in CONFIGS]
x = np.arange(len(CONFIGS)); w = 0.2
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
for i, (k, col) in enumerate([("det_disag", "#8e44ad"), ("det_drift", "#95a5a6"), ("det_mc", "#e67e22"), ("det_sup", "#2c3e50")]):
    ax[0].bar(x + (i - 1.5) * w, [summ[c][k] for c in CONFIGS], w, label=k.replace("det_", ""), color=col)
ax[0].axhline(0.5, ls="--", color="k", lw=.8); ax[0].set_xticks(x); ax[0].set_xticklabels(labels, rotation=40, ha="right", fontsize=7)
ax[0].set_ylabel("detection AUROC"); ax[0].set_title("(1) shift detection"); ax[0].legend(fontsize=8)
ax[1].bar(x - w / 2, [summ[c]["g_disag"] for c in CONFIGS], w, label="disag", color="#8e44ad")
ax[1].bar(x + w / 2, [summ[c]["g_drift"] for c in CONFIGS], w, label="drift", color="#95a5a6")
ax[1].set_xticks(x); ax[1].set_xticklabels(labels, rotation=40, ha="right", fontsize=7)
ax[1].set_ylabel("% random->oracle gap"); ax[1].set_title("(2) selective-prediction AURC"); ax[1].legend(fontsize=8)
r = data[("noise", 0.3)]
ax[2].scatter(r["disag"], r["target"], s=6, alpha=.25, c=r["corrupt"], cmap="coolwarm")
ax[2].set_xlabel("Q-ensemble disagreement"); ax[2].set_ylabel("estimate divergence ||z_c - z_clean||")
ax[2].set_title(f"(3) noise-0.3  Spearman {spearmanr(r['disag'], r['target']).correlation:+.2f}")
for a in ax:
    a.grid(alpha=.3)
fig.suptitle(f"B1 -- free uncertainty monitor under shift (TD-MPC2 {args.task})  |  {'WIN' if WIN else 'PARTIAL/WEAK'}", fontweight="bold")
fig.tight_layout(); fig.savefig(f"/content/b1_{args.task}.png", dpi=110)
torch.save({"summary": {f"{s}-{v}": summ[(s, v)] for (s, v) in CONFIGS}, "win": bool(WIN)}, f"/content/b1_{args.task}.pt")
print(f"\nsaved /content/b1_{args.task}.png  and  .pt")
