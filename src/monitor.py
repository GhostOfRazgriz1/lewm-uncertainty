"""M1.6 -- uncertainty as a RUNTIME MONITOR (selective prediction): the positive complement to M1.2-1.5.

The arc showed LeWM's uncertainty can't IMPROVE a decision (control null, sensing nulls). This asks the
other question: can it tell you WHEN TO ABSTAIN from the model's prediction? Per-transition, rank by an
uncertainty signal, keep the most-confident fraction (coverage c), measure prediction error on the kept
set (risk-coverage / AURC, lower=better). Two failure modes, reported separately so the win isn't a
trivial OOD-sort:
  IN-DIST -- clean Push-T transitions (error varies at contacts): can MC-dropout variance abstain from the
             hard ones? (the non-trivial monitor claim -- no OOD tell.)
  SHIFT   -- current frame corrupted with Gaussian noise: can the shell signal |‖emb‖-shell| abstain?
Signals: MC-dropout variance (predictive), shell deviation (OOD/epistemic), combined (z-score sum), vs
oracle (rank by true error) and random (flat). No retrain. Spec: docs/M1.6-monitor-spec.md.

POSITIVE if uncertainty AURC << random AND facets are COMPLEMENTARY (MC wins in-dist, shell wins on shift,
combined wins mixed) -> a free calibrated monitor; the 'complementary facets' analysis becomes constructive.

Run on Colab GPU:  python src/monitor.py
"""
import sys
import numpy as np
import torch
import gymnasium as gym
import stable_worldmodel as swm                                   # noqa: F401  registers swm/PushT-v1
import matplotlib; matplotlib.use("Agg")                          # noqa: E402
import matplotlib.pyplot as plt                                   # noqa: E402

sys.path.insert(0, "/content/lewm-uncertainty")
from src.active_sense import (                                    # noqa: E402  the rig (loads LeWM on import)
    cfg, encode_all, act_encode, _predict_one, rollout, HS, T_STEPS,
)

N_ROLLOUTS, MC, NOISE_SIGMA = 30, 16, 0.4                         # MC=dropout samples; NOISE_SIGMA = corruption std (frac of 255)
COVS = np.linspace(0.05, 1.0, 20)
SHELL = cfg["predictor"]["input_dim"] ** 0.5                      # Gaussian-shell norm (~13.9)


def corrupt(frame, sigma, rng):
    """Add Gaussian pixel noise -> an OOD 'current observation' (true next stays clean)."""
    f = frame.astype("float32") + rng.normal(0, sigma * 255, frame.shape)
    return np.clip(f, 0, 255).astype("uint8")


def zscore(x):
    return (x - x.mean()) / (x.std() + 1e-9)


def risk_coverage(signal, err):
    """Keep the lowest-signal (most-confident) coverage fraction; mean error on the kept set, per coverage."""
    e = err[np.argsort(signal)]
    return np.array([e[:max(1, int(c * len(e)))].mean() for c in COVS])


def aurc(signal, err):
    return float(risk_coverage(signal, err).mean())


# ---- build per-transition items: (mc_var, shell_dev, true_error, is_ood) ------------------------
gen = np.random.default_rng(0); crng = np.random.default_rng(1)
mc_v, sh_d, err, ood, nrm = [], [], [], [], []                   # nrm = raw ||emb|| (for the shell viz)
for r in range(N_ROLLOUTS):
    frames, acts = rollout(gym.make("swm/PushT-v1", render_mode="rgb_array"), T_STEPS, gen)
    emb = encode_all(frames)                                      # [T+1,192] clean
    emb_c = encode_all([corrupt(f, NOISE_SIGMA, crng) for f in frames])   # [T+1,192] corrupted
    ae = act_encode(acts)
    for t in range(HS - 1, T_STEPS):
        eh, ah = emb[t - HS + 1:t + 1], ae[t - HS + 1:t + 1]
        mu, var = _predict_one(eh, ah, mc=MC)
        mc_v.append(var); nrm.append(float(emb[t].norm())); sh_d.append(abs(nrm[-1] - SHELL))
        err.append(float((mu - emb[t + 1]).norm())); ood.append(0)
        eh_c = eh.clone(); eh_c[-1] = emb_c[t]                    # corrupt the CURRENT observation
        mu_c, var_c = _predict_one(eh_c, ah, mc=MC)
        mc_v.append(var_c); nrm.append(float(emb_c[t].norm())); sh_d.append(abs(nrm[-1] - SHELL))
        err.append(float((mu_c - emb[t + 1]).norm())); ood.append(1)
    if r % 10 == 0:
        print(f"rollout {r}/{N_ROLLOUTS}", flush=True)

mc_v, sh_d, err, ood, nrm = map(np.array, (mc_v, sh_d, err, ood, nrm))
ind = ood == 0
print(f"\nsanity: shell deviation  clean {sh_d[ind].mean():.2f}  vs  corrupted {sh_d[~ind].mean():.2f}"
      f"   | error clean {err[ind].mean():.2f} vs corrupted {err[~ind].mean():.2f}")

# ---- AURC table over two pools (lower = better) -------------------------------------------------
pools = {"in-dist (clean only)": ind, "mixed (clean + corrupted)": np.ones(len(err), bool)}
print("\n==== M1.6 selective prediction -- AURC (mean risk over coverage, lower=better) ====")
results = {}
for pname, mask in pools.items():
    e = err[mask]
    sigs = {"MC-variance": mc_v[mask], "shell": sh_d[mask],
            "combined": zscore(mc_v[mask]) + zscore(sh_d[mask]), "oracle": e,
            "random": None}
    rand = float(e.mean())                                        # random ordering -> flat at mean error
    print(f"\n  {pname}  (mean error {e.mean():.3f})")
    row = {}
    for sname, sig in sigs.items():
        a = rand if sname == "random" else aurc(sig, e)
        row[sname] = a
        print(f"    {sname:12s}: AURC {a:.3f}" + ("   (= mean error)" if sname == "random" else ""))
    results[pname] = (row, e, sigs, rand)

# ---- verdict ------------------------------------------------------------------------------------
ind_row = results["in-dist (clean only)"][0]
mix_row = results["mixed (clean + corrupted)"][0]
mc_beats_rand = ind_row["MC-variance"] < ind_row["random"]
shell_useless_indist = ind_row["shell"] >= 0.97 * ind_row["random"]
shell_helps_shift = mix_row["shell"] < mix_row["random"]
combined_best = mix_row["combined"] <= min(mix_row["MC-variance"], mix_row["shell"]) + 1e-6
print("\n  ---- verdict ----")
print(f"  MC-variance beats random IN-DIST : {mc_beats_rand}   (the non-trivial monitor claim)")
print(f"  shell ~ useless in-dist          : {shell_useless_indist}   (orthogonal facet, as M1.1)")
print(f"  shell beats random ON SHIFT      : {shell_helps_shift}")
print(f"  combined dominates on MIXED pool : {combined_best}")
if mc_beats_rand and shell_helps_shift and combined_best:
    print("  POSITIVE -- a free, calibrated runtime monitor; the two facets are COMPLEMENTARY")
    print("             (MC-variance catches in-dist hard transitions, shell catches OOD inputs).")
else:
    print("  WEAK/NULL -- at least one facet does not act as a usable monitor (see rows above).")

# ---- figure: risk-coverage curves, both pools ---------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
colors = {"MC-variance": "#2980b9", "shell": "#e67e22", "combined": "#8e44ad", "oracle": "#27ae60"}
for axi, (pname, (row, e, sigs, rand)) in zip(ax, results.items()):
    for sname in ["oracle", "MC-variance", "shell", "combined"]:
        axi.plot(COVS, risk_coverage(sigs[sname], e), "-o", ms=3, color=colors[sname],
                 label=f"{sname} (AURC {row[sname]:.2f})")
    axi.axhline(rand, ls="--", color="gray", label=f"random ({rand:.2f})")
    axi.set_xlabel("coverage (fraction acted on)"); axi.set_ylabel("risk (mean error on kept)")
    axi.set_title(pname); axi.legend(fontsize=8); axi.grid(alpha=.3)
fig.suptitle("M1.6 -- uncertainty as a runtime monitor (selective prediction on LeWM)", fontweight="bold")
fig.tight_layout(); fig.savefig("/content/lewm-uncertainty/lewm_monitor.png", dpi=110)
print("\nsaved lewm_monitor.png")

# ---- dump per-sample signal distributions for the HTML report (viz/monitor.html) ----------------
import json


def _ds(a, n=500):                                               # downsample for a compact JSON
    a = np.asarray(a, float)
    idx = np.linspace(0, len(a) - 1, min(n, len(a))).astype(int)
    return np.round(a[idx], 4).tolist()


viz = {"schematic": False, "shell_radius": round(float(SHELL), 3),
       "norm": {"in_dist": _ds(nrm[ind]), "ood": _ds(nrm[~ind])},               # raw ||emb|| (shell viz)
       "scatter": {"in_dist": {"mc": _ds(mc_v[ind]), "err": _ds(err[ind])},     # MC-variance vs true error
                   "ood": {"mc": _ds(mc_v[~ind]), "err": _ds(err[~ind])}},
       "aurc": {p: {k: round(v, 4) for k, v in results[p][0].items()} for p in results}}
json.dump(viz, open("/content/lewm-uncertainty/lewm_monitor_data.json", "w"))
print("saved lewm_monitor_data.json  (drop into viz/ to populate the HTML report's signal panels)")
