"""Direction A, stage 1 -- TRUST-GATED PERCEPTION under intermittent observation corruption.

Does LeWM's FREE shell signal |‖encode(obs)‖ - √d| decide *which observations to trust* well enough to keep
a near-oracle latent state estimate under distribution shift -- no corruption labels, no retraining?

This is the regime our prior nulls predict should flip POSITIVE: M1.3 (when-to-look under a budget) failed
because every frame was clean (nothing to gate on); M1.6 proved the shell separates clean from corrupted. So
gating on the shell -- which failed under a budget -- should WIN under shift.

Closed-loop estimation on swm/PushT-v1 (open-loop random actions; the M1.3 rig). The agent maintains z_hat;
each step the observation is corrupted with prob p. Per frame it either TRUSTS (z_hat <- encode(obs)) or
COASTS (z_hat <- predict(...), free, drifts). The non-trivial tradeoff: TRUST corrupted frames -> poisoned
state; COAST clean frames -> drift. The win needs to thread that needle from the noisy free signal.

POLICIES (matched trust-rate):
  blind        -- trust every frame (poisoned by corruption). baseline.
  random-gate  -- coast a RANDOM subset matched to shell-gate's trust count (isolates: does the SIGNAL matter
                  or is any coasting enough?).
  shell-gate   -- trust iff |‖encode(obs)‖-√d| < tau  (ours; tau calibrated label-free from clean in-dist).
  oracle-gate  -- trust iff the frame is truly clean (knows the corruption mask). ceiling.

METRIC: tracking error vs the CLEAN latent, mean_t ‖z_hat_t - encode(clean_t)‖ (clean = eval-only truth).
Sweep p in {0.2,0.4,0.6} x type {noise, blackout}. Verdict pre-registered in docs/A1-gated-perception-spec.md.

Run on Colab GPU:  python src/gated_perception.py
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

N_CAL, N_DEPLOY = 20, 40                                          # calibration (tau) / deploy rollouts
P_GRID = [0.2, 0.4, 0.6]                                          # corruption rate
NOISE_SIGMA = 0.4                                                 # Gaussian-noise std (frac of 255), matches M1.6
TAU_Q = 0.95                                                      # clean shell-dev quantile -> trust threshold
TAU_Q_SWEEP = [0.90, 0.95, 0.99]
SHELL = cfg["predictor"]["input_dim"] ** 0.5                     # Gaussian-shell norm (~13.9)
T1 = T_STEPS + 1


def corrupt_noise(frame, rng):                                   # additive Gaussian -> ‖emb‖ pushed UP off-shell
    f = frame.astype("float32") + rng.normal(0, NOISE_SIGMA * 255, frame.shape)
    return np.clip(f, 0, 255).astype("uint8")


def corrupt_black(frame, rng):                                   # blackout occlusion -> ‖emb‖ collapses DOWN
    return np.zeros_like(frame)


CORRUPT = {"noise": corrupt_noise, "blackout": corrupt_black}


def shell_dev(emb):                                              # [.,192] -> per-row |‖emb‖ - shell|
    return np.abs(emb.norm(dim=-1).cpu().numpy() - SHELL)


@torch.no_grad()
def deploy(emb_clean, emb_obs, act_emb, trust):
    """Follow a trust mask: TRUST -> adopt the observed encoding; COAST -> predict forward from z_hat history.
    Returns per-step tracking error vs the CLEAN latent."""
    zh = [emb_obs[0]]                                            # t=0 assumed clean
    for t in range(1, emb_clean.shape[0]):
        if trust[t]:
            zh.append(emb_obs[t])
        else:
            a = max(0, t - HS)
            zh.append(_predict_one(torch.stack(zh[a:t]), act_emb[a:t])[0])
    return (torch.stack(zh) - emb_clean).norm(dim=-1).cpu().numpy()


def policy_masks(dev, clean_mask, tau, rng):
    n = len(dev)
    shell = dev < tau; shell[0] = True                          # trust iff in-dist-looking
    blind = np.ones(n, bool)
    oracle = clean_mask.copy(); oracle[0] = True                # trust iff truly clean
    k = int(shell.sum())                                        # match random-gate to shell's trust count
    rnd = np.zeros(n, bool); rnd[0] = True
    pick = rng.permutation(np.arange(1, n))[:max(0, k - 1)]; rnd[pick] = True
    return {"blind": blind, "random-gate": rnd, "shell-gate": shell, "oracle-gate": oracle}


POLICIES = ["blind", "random-gate", "shell-gate", "oracle-gate"]


def sem(a):
    return float(np.std(a) / np.sqrt(len(a)))


# ---- 1) calibrate tau, label-free, from the clean in-dist shell-dev distribution ----------------
gen = np.random.default_rng(0)
cal = []
for r in range(N_CAL):
    frames, _ = rollout(gym.make("swm/PushT-v1", render_mode="rgb_array"), T_STEPS, gen)
    cal.append(shell_dev(encode_all(frames)))
cal = np.concatenate(cal)
TAUS = {q: float(np.quantile(cal, q)) for q in TAU_Q_SWEEP}
tau = TAUS[TAU_Q]
print(f"clean shell-dev: mean {cal.mean():.3f}  q90 {TAUS[0.90]:.3f}  q95 {TAUS[0.95]:.3f}  q99 {TAUS[0.99]:.3f}")
print(f"using tau = q{int(TAU_Q*100)} = {tau:.3f}\n", flush=True)

# ---- 2) deploy: per corruption type x rate, run every policy --------------------------------------
results = {}                                                     # (ctype, p) -> {policy: err[rollout]}
detect = {}                                                      # (ctype, p) -> (precision, recall) of shell-gate
for ctype, cfn in CORRUPT.items():
    for p in P_GRID:
        errs = {pol: [] for pol in POLICIES}
        tp = fp = fn = 0
        crng = np.random.default_rng(1000 + int(p * 100) + (0 if ctype == "noise" else 7))
        for r in range(N_DEPLOY):
            frames, acts = rollout(gym.make("swm/PushT-v1", render_mode="rgb_array"), T_STEPS, gen)
            clean_mask = crng.random(T1) >= p; clean_mask[0] = True              # True = clean
            obs = [f if clean_mask[t] else cfn(f, crng) for t, f in enumerate(frames)]
            emb_clean, emb_obs, ae = encode_all(frames), encode_all(obs), act_encode(acts)
            dev = shell_dev(emb_obs)
            M = policy_masks(dev, clean_mask, tau, crng)
            for pol in POLICIES:
                errs[pol].append(deploy(emb_clean, emb_obs, ae, M[pol]).mean())
            flagged = ~M["shell-gate"]; corrupted = ~clean_mask                 # detection bookkeeping
            tp += int((flagged & corrupted).sum()); fp += int((flagged & ~corrupted).sum())
            fn += int((~flagged & corrupted).sum())
        results[(ctype, p)] = {pol: np.array(errs[pol]) for pol in POLICIES}
        detect[(ctype, p)] = (tp / max(1, tp + fp), tp / max(1, tp + fn))
        print(f"[{ctype} p={p}] done ({N_DEPLOY} rollouts)", flush=True)

# ---- 3) report -----------------------------------------------------------------------------------
print("\n==== A1 trust-gated perception -- tracking error vs CLEAN latent (mean +/- SEM, lower=better) ====")
verdicts = {}
for ctype in CORRUPT:
    print(f"\n  corruption = {ctype}")
    print(f"    {'rate':>5} | " + " | ".join(f"{pol:>16}" for pol in POLICIES) + " |  detect P/R")
    for p in P_GRID:
        R = results[(ctype, p)]
        cells = " | ".join(f"{R[pol].mean():6.3f}+/-{sem(R[pol]):.3f}" for pol in POLICIES)
        pr, rc = detect[(ctype, p)]
        print(f"    {p:>5} | {cells} |  {pr:.2f}/{rc:.2f}")
        b, rnd, sh, orc = R["blind"], R["random-gate"], R["shell-gate"], R["oracle-gate"]
        d_blind = b.mean() - sh.mean(); s_blind = np.hypot(sem(b), sem(sh))                  # >0: shell beats blind
        d_rand = rnd.mean() - sh.mean(); s_rand = np.hypot(sem(rnd), sem(sh))                # >0: shell beats random
        d_orc = sh.mean() - orc.mean(); s_orc = np.hypot(sem(sh), sem(orc))                  # ~0: shell ~ oracle
        if d_blind > s_blind and d_rand > s_rand and d_orc < 2 * s_orc:
            v = "WIN"
        elif d_blind > s_blind:
            v = "PARTIAL"
        else:
            v = "NULL"
        verdicts[(ctype, p)] = v
        print(f"          -> shell vs blind {d_blind:+.3f}+/-{s_blind:.3f} | vs random {d_rand:+.3f}+/-{s_rand:.3f}"
              f" | vs oracle {d_orc:+.3f}+/-{s_orc:.3f}  => {v}")

wins = sum(v == "WIN" for v in verdicts.values()); parts = sum(v == "PARTIAL" for v in verdicts.values())
print(f"\n  OVERALL: {wins} WIN / {parts} PARTIAL / {len(verdicts)-wins-parts} NULL across {len(verdicts)} settings")
if wins >= len(verdicts) - 1:
    print("  => POSITIVE: the free shell signal gates trust to near-oracle robust estimation under shift, and")
    print("     the SIGNAL (not just coasting) is what does it. Stage 2 = plug the gate into CEM control.")
elif wins + parts >= len(verdicts) - 1:
    print("  => PARTIAL: gating beats blind but the shell signal doesn't clearly beat random coasting.")
else:
    print("  => NULL: shell-gating does not robustly beat blind -> uncertainty not actionable for deployment either.")

# ---- 4) figure: tracking error vs corruption rate, per type --------------------------------------
fig, ax = plt.subplots(1, len(CORRUPT), figsize=(6 * len(CORRUPT), 4.6), squeeze=False)
cols = {"blind": "#c0392b", "random-gate": "#bdc3c7", "shell-gate": "#2980b9", "oracle-gate": "#27ae60"}
for axi, ctype in zip(ax[0], CORRUPT):
    for pol in POLICIES:
        m = [results[(ctype, p)][pol].mean() for p in P_GRID]
        s = [sem(results[(ctype, p)][pol]) for p in P_GRID]
        axi.errorbar(P_GRID, m, yerr=s, fmt="-o", capsize=3, color=cols[pol], label=pol)
    axi.set_xlabel("corruption rate p"); axi.set_ylabel("tracking error vs clean latent")
    axi.set_title(f"corruption: {ctype}"); axi.grid(alpha=.3); axi.legend(fontsize=8)
fig.suptitle("A1 -- trust-gated perception under shift: does the free shell signal pick which obs to trust?",
             fontweight="bold")
fig.tight_layout(); fig.savefig("/content/lewm-uncertainty/lewm_gated_perception.png", dpi=110)
print("\nsaved lewm_gated_perception.png")
