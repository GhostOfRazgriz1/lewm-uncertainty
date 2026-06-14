"""B1 / E1 -- a FREE uncertainty MONITOR for a competent agent under deployment shift.

B0 (GO): frozen pretrained TD-MPC2 is competent on Colab AND its Q-ensemble disagreement is a live free
signal (Spearman +0.55 with one-step error in-dist). B1 deploys the competent agent under OBSERVATION
corruption and asks whether free signals (computable online from (z,a), NO clean reference) flag the
timesteps where the agent's latent estimate is wrong.

FIRST CUT FINDING (PARTIAL): Q-ensemble disagreement detects NOISE/SUBTLE corruption (severity-tracking,
beats motion-drift) but is BLIND to BLACKOUT (heads agree confidently on the degenerate zero-latent =
M2.2 ensemble-OOD-blindness, reproduced on TD-MPC2). The motion baseline (drift) is the mirror: ok on
noise, catastrophic on blackout (a stuck estimate has low drift but max error). => need a second FREE
facet. This refined version adds it:

  - OOD facet (free): diag-Mahalanobis distance of z to the CLEAN-latent cloud -- the SimNorm analog of
    the LeJEPA shell. Should catch blackout (encode-of-zeros is far from the clean cloud) without labels.
  - COMBINED = z(disag) + z(ood): predictive (disag) + epistemic/OOD (ood), as M1.6/M2.2.
  - FAIR supervised baseline: train a detector on ONE corruption family, test on ANOTHER (cross-family).
    The earlier same-family detector was unfairly easy (blackout z is a giveaway). Free signals need no
    corruption labels and should generalize where a cross-family supervised detector collapses.

Honest target = estimate divergence ||encode(corrupt_obs) - encode(true_obs)|| (env state is real).
Shifts: S1 noise (sigma in {.1,.3,.6}*obs_std, iid p=.5), S2 blackout bursts (K in {5,15,30}), S3 subtle
bias on a dim subset. MONITORING ONLY (control = E2/B2). Run on Colab GPU after B0 setup:
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
ap.add_argument("--max_steps", type=int, default=200)
ap.add_argument("--n_cal", type=int, default=3, help="clean calibration rollouts (obs std + OOD cloud)")
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
_hu.get_original_cwd = lambda: os.getcwd()

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
    return agent.model.encode(obs.to(device).unsqueeze(0), None)


@torch.no_grad()
def q_disag(z, a):
    a = a.to(device).unsqueeze(0)
    qv = tdmath.two_hot_inv(agent.model.Q(z, a, None, return_type="all"), cfg).squeeze(-1).squeeze(-1)
    return float(qv.var(0))


@torch.no_grad()
def mc_var(z, a, M):
    if M <= 0:
        return 0.0
    a = a.to(device).unsqueeze(0)
    agent.model.train()
    s = [float(tdmath.two_hot_inv(agent.model.Q(z, a, None, return_type="all"), cfg).mean()) for _ in range(M)]
    agent.model.eval()
    return float(np.var(s))


# ---- clean calibration: obs std (noise scale) + clean-latent cloud (free OOD facet) --------------
obs_buf, lat_buf = [], []
for c in range(args.n_cal):
    obs, done, t = env.reset(), False, 0
    while not done and t < args.max_steps:
        obs_buf.append(obs.numpy()); lat_buf.append(enc(obs).squeeze(0).cpu().numpy())
        obs, _, done, _ = env.step(agent.act(obs, t0=(t == 0), task=None)); t += 1
obs_std = torch.from_numpy(np.std(np.stack(obs_buf), 0).astype("float32")).clamp_min(1e-3)
D = obs_std.shape[0]
Zc = np.stack(lat_buf)
ood_mu, ood_var = Zc.mean(0), Zc.var(0) + 1e-6                                # diag-Gaussian on clean latents
maha = lambda Z: np.sqrt((((Z - ood_mu) ** 2) / ood_var).mean(1))             # OOD score (higher = OOD)
bias_dims = np.zeros(D, "float32"); bias_dims[: max(1, D // 2)] = 1.0
bias_vec = torch.from_numpy(bias_dims) * obs_std

CONFIGS = [("noise", 0.1), ("noise", 0.3), ("noise", 0.6),
           ("blackout", 5), ("blackout", 15), ("blackout", 30), ("subtle", 0.5)]
FAMILY = {"noise": ["noise"], "blackout": ["blackout"], "subtle": ["subtle"]}
CROSS = {"noise": "blackout", "blackout": "noise", "subtle": "blackout"}        # structurally-different family


def corrupt(obs, shift, sev, rng):
    if shift == "noise":
        return obs + torch.from_numpy(rng.normal(0, 1, size=obs.shape).astype("float32")) * (sev * obs_std)
    if shift == "blackout":
        return torch.zeros_like(obs)
    return obs + bias_vec * sev


def mask_for(shift, sev, rng):
    if shift == "blackout":
        K = int(sev); m = np.zeros(args.max_steps, bool); i = K
        while i < args.max_steps:
            m[i:i + K] = True; i += 2 * K
        return m
    return rng.random(args.max_steps) < 0.5


def rollout(shift, sev, ep):
    rng = np.random.default_rng(1000 * ep + int(sev * 10))
    m = mask_for(shift, sev, rng)
    obs, done, t, prev_z = env.reset(), False, 0, None
    rows = []
    while not done and t < args.max_steps:
        true_obs = obs
        cobs = corrupt(true_obs, shift, sev, rng) if m[t] else true_obs
        z = enc(cobs)
        a = agent.act(cobs, t0=(t == 0), task=None)
        zc = enc(true_obs)
        rows.append((int(m[t]), float((z - zc).norm()), q_disag(z, a),
                     0.0 if prev_z is None else float((z - prev_z).norm()),
                     mc_var(z, a, args.mc), z.squeeze(0).cpu().numpy(), a.cpu().numpy()))
        prev_z = z
        obs, _, done, _ = env.step(a); t += 1
    return rows


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
    rec["ood"] = maha(rec["Z"])                                               # free OOD facet (post-hoc on cached z)
    data[(shift, sev)] = rec
    print(f"  collected {shift}-{sev}: {len(rec['target'])} steps, corrupt frac {rec['corrupt'].mean():.2f}", flush=True)


# ---- supervised detectors per family (for the FAIR cross-family test) ----------------------------
def train_detector(X, y):
    det = nn.Sequential(nn.Linear(X.shape[1], 128), nn.GELU(), nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 1)).to(device)
    opt = torch.optim.Adam(det.parameters(), lr=1e-3)
    Xt = torch.tensor(X, dtype=torch.float32, device=device); yt = torch.tensor(y, dtype=torch.float32, device=device)
    lossf = nn.BCEWithLogitsLoss()
    for _ in range(300):
        opt.zero_grad(); lossf(det(Xt).squeeze(-1), yt).backward(); opt.step()

    @torch.no_grad()
    def predict(Xe):
        return torch.sigmoid(det(torch.tensor(Xe, dtype=torch.float32, device=device)).squeeze(-1)).cpu().numpy()
    return predict


def trainmask(r):
    return r["ep"] < int(0.6 * args.episodes)


dets = {}
for fam, shifts in FAMILY.items():
    Xtr, ytr = [], []
    for (s, v) in CONFIGS:
        if s in shifts:
            r = data[(s, v)]; tr = trainmask(r)
            Xtr.append(np.concatenate([r["Z"][tr], r["A"][tr]], 1)); ytr.append(r["corrupt"][tr])
    dets[fam] = train_detector(np.concatenate(Xtr), np.concatenate(ytr))


# ---- metrics ------------------------------------------------------------------------------------
COVS = np.linspace(0.1, 1.0, 19)
zc = lambda x: (x - x.mean()) / (x.std() + 1e-9)


def aurc(signal, err):
    e = err[np.argsort(signal)]
    return float(np.mean([e[:max(1, int(c * len(e)))].mean() for c in COVS]))


def auroc(label, signal):
    return roc_auc_score(label, signal) if (label.min() == 0 and label.max() == 1) else float("nan")


def partial_spear(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    res = lambda u: u - np.polyval(np.polyfit(rz, u, 1), rz)
    return float(np.corrcoef(res(rx), res(ry))[0, 1])


print("\n==== B1 monitor under shift -- REFINED (free OOD facet + combined + fair supervised) ====")
print(f"  {'shift':12s} | {'detection AUROC':^41s} | {'AURC %gap':^17s} | {'partial(d|drift)':>16s}")
print(f"  {'':12s} | {'disag':>6s} {'ood':>6s} {'comb':>6s} {'drift':>6s} {'sup=':>6s} {'sup~':>6s} | {'disag':>5s} {'ood':>5s} {'comb':>5s} |")
summ = {}
for (shift, sev), r in data.items():
    te = ~trainmask(r); lab = r["corrupt"][te]
    Xte = np.concatenate([r["Z"][te], r["A"][te]], 1)
    sig = {"disag": r["disag"][te], "ood": r["ood"][te], "drift": r["drift"][te], "mc": r["mc"][te]}
    sig["comb"] = zc(sig["disag"]) + zc(sig["ood"])
    det = {k: auroc(lab, s) for k, s in sig.items()}
    sup_same = auroc(lab, dets[shift](Xte))                                  # same family (unfair, upper bound)
    sup_cross = auroc(lab, dets[CROSS[shift]](Xte))                          # cross family (the FAIR test)
    err = r["target"][te]; rnd = err.mean(); orc = aurc(err, err)
    gap = lambda s: 100 * (rnd - aurc(s, err)) / (rnd - orc + 1e-9)
    summ[(shift, sev)] = dict(**{f"det_{k}": v for k, v in det.items()}, sup_same=sup_same, sup_cross=sup_cross,
                              g_disag=gap(sig["disag"]), g_ood=gap(sig["ood"]), g_comb=gap(sig["comb"]), g_drift=gap(sig["drift"]),
                              p_disag=partial_spear(sig["disag"], err, sig["drift"]),
                              p_ood=partial_spear(sig["ood"], err, sig["drift"]),
                              p_comb=partial_spear(sig["comb"], err, sig["drift"]))
    s = summ[(shift, sev)]
    print(f"  {f'{shift}-{sev}':12s} | {det['disag']:6.3f} {det['ood']:6.3f} {det['comb']:6.3f} {det['drift']:6.3f} "
          f"{sup_same:6.3f} {sup_cross:6.3f} | {s['g_disag']:4.0f}% {s['g_ood']:4.0f}% {s['g_comb']:4.0f}% | "
          f"{s['p_disag']:5.2f} {s['p_ood']:5.2f} {s['p_comb']:5.2f}")

# ---- verdict ------------------------------------------------------------------------------------
allc = list(summ.values())
comb_auroc = np.nanmean([v["det_comb"] for v in allc])
ood_blackout = np.nanmean([summ[c]["det_ood"] for c in CONFIGS if c[0] == "blackout"])
disag_blackout = np.nanmean([summ[c]["det_disag"] for c in CONFIGS if c[0] == "blackout"])
free_vs_cross = np.nanmean([v["det_comb"] - v["sup_cross"] for v in allc])
print("\n  verdict:")
print(f"    COMBINED detection AUROC (all shifts) avg = {comb_auroc:.3f}  ({'complete monitor' if comb_auroc>0.7 else 'still incomplete'})")
print(f"    OOD facet rescues blackout: ood AUROC {ood_blackout:.3f} vs disag {disag_blackout:.3f}  "
      f"({'YES -- the missing facet' if ood_blackout>disag_blackout+0.1 else 'NO'})")
print(f"    free COMBINED vs FAIR (cross-family) supervised: {free_vs_cross:+.3f}  "
      f"({'free generalizes better (no corruption labels)' if free_vs_cross>-0.03 else 'cross-supervised still wins'})")
WIN = (comb_auroc > 0.7) and (ood_blackout > disag_blackout + 0.1) and (free_vs_cross > -0.03)
print("    " + ("WIN -- a COMPLETE free monitor on a competent WM (disag predictive + ood OOD), beating a "
              "corruption-agnostic supervised baseline; complementary-facets law reproduced on TD-MPC2."
              if WIN else "PARTIAL -- combined helps but a condition above failed; see table."))

# ---- figure -------------------------------------------------------------------------------------
labels = [f"{s}-{v}" for (s, v) in CONFIGS]; x = np.arange(len(CONFIGS)); w = 0.15
fig, ax = plt.subplots(1, 3, figsize=(17, 4.7))
bars = [("det_disag", "disag", "#8e44ad"), ("det_ood", "ood", "#16a085"), ("det_comb", "combined", "#c0392b"),
        ("sup_same", "sup(same)", "#2c3e50"), ("sup_cross", "sup(cross)", "#95a5a6")]
for i, (k, lab, col) in enumerate(bars):
    ax[0].bar(x + (i - 2) * w, [summ[c][k] for c in CONFIGS], w, label=lab, color=col)
ax[0].axhline(0.5, ls="--", color="k", lw=.8); ax[0].set_xticks(x); ax[0].set_xticklabels(labels, rotation=40, ha="right", fontsize=7)
ax[0].set_ylabel("detection AUROC"); ax[0].set_title("(1) shift detection: free combined vs fair supervised"); ax[0].legend(fontsize=7, ncol=2)
for i, (k, lab, col) in enumerate([("g_disag", "disag", "#8e44ad"), ("g_ood", "ood", "#16a085"), ("g_comb", "combined", "#c0392b")]):
    ax[1].bar(x + (i - 1) * w, [summ[c][k] for c in CONFIGS], w, label=lab, color=col)
ax[1].set_xticks(x); ax[1].set_xticklabels(labels, rotation=40, ha="right", fontsize=7)
ax[1].set_ylabel("% random->oracle gap"); ax[1].set_title("(2) selective-prediction AURC"); ax[1].legend(fontsize=8)
r = data[("blackout", 15)]
ax[2].scatter(r["ood"], r["target"], s=6, alpha=.25, c=r["corrupt"], cmap="coolwarm")
ax[2].set_xlabel("OOD score (maha to clean cloud)"); ax[2].set_ylabel("estimate divergence")
ax[2].set_title(f"(3) blackout-15: OOD facet  Spearman {spearmanr(r['ood'], r['target']).correlation:+.2f}")
for a in ax:
    a.grid(alpha=.3)
fig.suptitle(f"B1 refined -- complete free monitor under shift (TD-MPC2 {args.task})  |  {'WIN' if WIN else 'PARTIAL'}", fontweight="bold")
fig.tight_layout(); fig.savefig(f"/content/b1_{args.task}.png", dpi=110)
torch.save({"summary": {f"{s}-{v}": summ[(s, v)] for (s, v) in CONFIGS}, "win": bool(WIN)}, f"/content/b1_{args.task}.pt")
print(f"\nsaved /content/b1_{args.task}.png  and  .pt")
