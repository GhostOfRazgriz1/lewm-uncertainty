"""M1.3 -- TEMPORAL ACTIVE SENSING on LeWM: is the calibrated uncertainty actionable for *sensing*?

M1.2 showed LeWM's MC-dropout uncertainty does NOT help *control* (penalizing uncertain plans is a
clean null: calibration != actionability on a near-deterministic task). This asks the complementary,
under-served question -- the white space in the LeWM citation landscape: nobody uses a JEPA world
model's uncertainty to schedule PERCEPTION. Here "sensing" is temporal:

  The agent maintains a latent estimate z_hat of the world. At each model-step it either
    LOOK    -> z_hat = encode(true frame)   (costs one observation, resets tracking error to 0), or
    PREDICT -> z_hat = model.predict(...)    (free, error accumulates; calibrated by MC-variance, M1.1).
  Given a budget of K looks over a horizon of T steps, WHEN should it look?

Frames are always full and real (just fewer of them), so LeWM's ViT encoder stays in-distribution --
this is why temporal sensing is the safe first cut, unlike spatial foveation which would feed the
encoder OOD masked frames. The decision is causal: look when the MC-dropout predictive variance is
high. Variance naturally rises the longer the agent has predicted since its last look.

ARMS (compared AT MATCHED BUDGET -- same number of looks; see src/schedules.py):
  variance  -- causal: look when MC-dropout predictive variance >= tau   (the deployable policy)
  fixed     -- look every T/K steps                                       (the baseline to beat)
  random    -- K looks at random steps                                   (chance baseline)
  oracle    -- look at the K intrinsically-most-surprising steps          (non-causal ceiling)

METRIC: latent-tracking error = mean_t || z_hat_t - encode(frame_t) || over the horizon, vs budget
(looks / T). Lower-left on the curve = better. Secondary, qualitative: do variance-policy looks land
on the surprise peaks (contact events, where the model is least sure)?

PRE-REGISTERED VERDICT:
  WIN  if the variance curve sits below fixed-interval beyond SEM at a matched budget AND looks align
       with surprise peaks  -> calibrated WM uncertainty IS actionable for sensing (the M1.3 claim).
  NULL if variance ~= fixed -> error growth is ~uniform; even sensing-use needs heterogeneous
       predictability, which a near-deterministic control task may not provide (sharpens M1.2's lesson).

Run on Colab GPU:  python src/active_sense.py    (no retrain; pretrained LeWM + swm/PushT-v1 only.)
"""
import sys
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import stable_worldmodel as swm                                   # noqa: F401  registers swm/PushT-v1
from torchvision import transforms as TT
import matplotlib; matplotlib.use("Agg")                          # noqa: E402
import matplotlib.pyplot as plt                                   # noqa: E402

sys.path.insert(0, "/content/lewm-uncertainty")
from src.load_lewm import load_lewm                               # noqa: E402
from src.schedules import (                                       # noqa: E402
    fixed_interval_lookset, random_lookset, oracle_lookset, n_looks,
)

N_ROLLOUTS, T_STEPS, FS, HS, MC = 30, 20, 5, 3, 12                # FS=frameskip; HS=history; MC=dropout samples
K_GRID = [1, 2, 3, 4, 6, 9, 13, 18]                               # look budgets for the matched-budget arms
N_TAU = 9                                                         # variance-threshold sweep resolution
TARGET_K = 6                                                      # budget at which the summary compares arms
device = "cuda" if torch.cuda.is_available() else "cpu"
model, cfg = load_lewm("/content/le-wm", device=device)
prep = TT.Compose([TT.ToTensor(), TT.Resize((224, 224)), TT.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def rollout(env, T, gen):
    """One open-loop random rollout: T model-steps (each = FS env-steps). Returns frames, actions."""
    env.reset(seed=int(gen.integers(1_000_000_000)))
    frames = [env.render()]; acts = []
    for _ in range(T):
        blk = [env.action_space.sample().astype("float32") for _ in range(FS)]
        for a in blk:
            env.step(a)
        acts.append(np.concatenate(blk)); frames.append(env.render())
    return np.stack(frames), np.stack(acts)


def set_drop(b):
    for m in model.predictor.modules():
        if isinstance(m, nn.Dropout):
            m.train(b)


@torch.no_grad()
def encode_all(frames):
    pix = torch.stack([prep(f) for f in frames]).unsqueeze(0).to(device)   # [1,T+1,3,224,224]
    return model.encode({"pixels": pix})["emb"][0]                         # [T+1,192]


@torch.no_grad()
def act_encode(acts):
    return model.action_encoder(torch.tensor(acts).unsqueeze(0).to(device))[0]   # [T,192]


@torch.no_grad()
def _predict_one(emb_hist, act_hist, mc=0):
    """Predict the next latent from a history window. mc=0 -> deterministic; mc>0 -> (mean, variance).
    NOTE: for the first HS steps the window is shorter than HS (the ARPredictor is causal, so a length-1
    or length-2 prefix is valid). If Colab errors here first, that's the place to look -- pad the early
    window to HS by repeating emb_hist[:1] / act_hist[:1]."""
    eh, ah = emb_hist.unsqueeze(0), act_hist.unsqueeze(0)                   # [1,h,192]
    if mc == 0:
        return model.predict(eh, ah)[0, -1], None                          # [192], -
    set_drop(True)
    preds = torch.stack([model.predict(eh, ah)[0, -1] for _ in range(mc)]) # [mc,192]
    set_drop(False)
    return preds.mean(0), preds.var(0).sum().item()                        # [192], scalar total variance


@torch.no_grad()
def tracking_errors(lookset, emb_true, act_emb):
    """Deterministic: follow a fixed look-set, predicting forward between looks. Returns per-step error."""
    looks = set(lookset)
    zh = []
    for t in range(emb_true.shape[0]):
        if t == 0 or t in looks:
            zh.append(emb_true[t])
        else:
            a = max(0, t - HS)
            zh.append(_predict_one(torch.stack(zh[a:t]), act_emb[a:t])[0])
    zh = torch.stack(zh)
    return (zh - emb_true).norm(dim=-1).cpu().numpy()                       # [T+1]


@torch.no_grad()
def variance_sim(emb_true, act_emb, tau):
    """Online causal policy: predict forward, look (reset to truth) whenever MC-variance >= tau.
    Returns per-step tracking error, the look-set, and the per-step variance trace."""
    zh = [emb_true[0]]; looks = [0]; vtrace = [0.0]
    for t in range(1, emb_true.shape[0]):
        a = max(0, t - HS)
        mu, v = _predict_one(torch.stack(zh[a:t]), act_emb[a:t], mc=MC)
        vtrace.append(v)
        if v >= tau:
            zh.append(emb_true[t]); looks.append(t)                         # LOOK: snap to truth
        else:
            zh.append(mu)                                                   # PREDICT: keep drifting
    err = (torch.stack(zh) - emb_true).norm(dim=-1).cpu().numpy()
    return err, looks, vtrace


@torch.no_grad()
def intrinsic_surprise(emb_true, act_emb):
    """Per-step one-step prediction error from the TRUE history -- how surprising each transition
    intrinsically is (oracle score + the curve the variance signal is trying to estimate)."""
    s = [0.0]
    for t in range(1, emb_true.shape[0]):
        a = max(0, t - HS)
        pred = _predict_one(emb_true[a:t], act_emb[a:t])[0]
        s.append((pred - emb_true[t]).norm().item())
    return np.array(s)


# ---- collect rollouts (encode once each) ---------------------------------------------------------
gen = np.random.default_rng(0)
T1 = T_STEPS + 1
rolls = []
for r in range(N_ROLLOUTS):
    frames, acts = rollout(gym.make("swm/PushT-v1", render_mode="rgb_array"), T_STEPS, gen)
    emb_true, act_emb = encode_all(frames), act_encode(acts)
    rolls.append((emb_true, act_emb, intrinsic_surprise(emb_true, act_emb)))
    if r % 10 == 0:
        print(f"encoded rollout {r}/{N_ROLLOUTS}", flush=True)

# tau grid from the natural (never-look) variance scale, so the sweep spans empty->full budget
free_v = np.concatenate([variance_sim(et, ae, tau=np.inf)[2][1:] for et, ae, _ in rolls])
TAU_GRID = np.quantile(free_v, np.linspace(0.02, 0.98, N_TAU))

# ---- run every arm; collect (budget fraction, mean tracking error) per rollout --------------------
def run_lookset_arm(make_lookset):
    """Budget-exact arms (fixed/random/oracle): sweep K_GRID. Returns budgets[K], err[K,rollout]."""
    err = np.zeros((len(K_GRID), N_ROLLOUTS))
    for ki, K in enumerate(K_GRID):
        for ri, (et, ae, surp) in enumerate(rolls):
            err[ki, ri] = tracking_errors(make_lookset(K, ri, surp), et, ae).mean()
    return np.array([k / T1 for k in K_GRID]), err


fixed_b, fixed_e = run_lookset_arm(lambda K, ri, surp: fixed_interval_lookset(T1, K))
rand_b, rand_e = run_lookset_arm(lambda K, ri, surp: random_lookset(T1, K, seed=ri))
orac_b, orac_e = run_lookset_arm(lambda K, ri, surp: oracle_lookset(surp, K))

var_b = np.zeros((N_TAU, N_ROLLOUTS)); var_e = np.zeros((N_TAU, N_ROLLOUTS))
for ti, tau in enumerate(TAU_GRID):
    for ri, (et, ae, _) in enumerate(rolls):
        err, looks, _ = variance_sim(et, ae, tau)
        var_e[ti, ri] = err.mean(); var_b[ti, ri] = n_looks(looks) / T1
    print(f"variance tau {ti + 1}/{N_TAU}: mean budget {var_b[ti].mean():.2f}, err {var_e[ti].mean():.3f}", flush=True)


def sem(a, axis=1):
    return a.std(axis=axis) / np.sqrt(a.shape[axis])


# ---- summary at TARGET_K (pre-registered comparison) ---------------------------------------------
ki = K_GRID.index(TARGET_K); tgt_b = TARGET_K / T1
ti = int(np.argmin(np.abs(var_b.mean(1) - tgt_b)))                          # variance tau closest to target budget
print(f"\n==== M1.3 temporal active sensing: tracking error at budget ~{tgt_b:.2f} ({TARGET_K}/{T1} looks) ====")
print(f"  fixed-interval : {fixed_e[ki].mean():.3f} +/- {sem(fixed_e)[ki]:.3f}")
print(f"  random         : {rand_e[ki].mean():.3f} +/- {sem(rand_e)[ki]:.3f}")
print(f"  variance (tau)  : {var_e[ti].mean():.3f} +/- {sem(var_e)[ti]:.3f}   (achieved budget {var_b[ti].mean():.2f})")
print(f"  oracle (ceiling): {orac_e[ki].mean():.3f} +/- {sem(orac_e)[ki]:.3f}")
delta = fixed_e[ki].mean() - var_e[ti].mean()                              # >0 => variance beats fixed
pooled = np.hypot(sem(fixed_e)[ki], sem(var_e)[ti])
vs_rand = var_e[ti].mean() - rand_e[ki].mean()                             # ~0 => signal no better than chance
head = (fixed_e - orac_e).mean(1)                                          # oracle's gain over fixed, per matched-K budget
hk = int(np.argmax(head)); head_sem = np.hypot(sem(fixed_e)[hk], sem(orac_e)[hk])
print(f"\n  variance vs fixed : delta {delta:+.3f} (pooled SEM {pooled:.3f}) | vs random: {vs_rand:+.3f}")
print(f"  oracle headroom   : up to {head[hk]:+.3f} over fixed at budget {K_GRID[hk] / T1:.2f} (SEM {head_sem:.3f})")
# Three-way verdict -- distinguishes "no headroom" from "headroom exists but the signal can't see it".
if delta > pooled:
    print("  WIN -- MC-dropout variance schedules looks BETTER than fixed-interval.")
elif head[hk] > head_sem:
    print("  NULL (signal too FLAT) -- a perfect schedule (oracle) DOES beat fixed, but MC-dropout variance is")
    print("        ~indistinguishable from random: the headroom is real, the signal just can't capture it.")
else:
    print("  NULL (no headroom) -- fixed-interval is ~optimal; even the oracle barely beats it.")

# ---- figure: error-vs-budget curves + where the looks land ---------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
for b, e, name, c in [(fixed_b, fixed_e, "fixed-interval", "#7f8c8d"), (rand_b, rand_e, "random", "#bdc3c7"),
                      (var_b.mean(1), var_e, "variance (ours)", "#2980b9"), (orac_b, orac_e, "oracle", "#27ae60")]:
    o = np.argsort(b)                                                       # sort points by budget for a clean line
    b, m, s = np.asarray(b)[o], e.mean(1)[o], sem(e)[o]
    ax[0].plot(b, m, "-o", color=c, label=name)
    ax[0].fill_between(b, m - s, m + s, color=c, alpha=.18)
ax[0].set_xlabel("observation budget (looks / horizon)"); ax[0].set_ylabel("latent-tracking error")
ax[0].set_title("When-to-look: error vs budget (lower-left better)"); ax[0].legend(); ax[0].grid(alpha=.3)

et, ae, surp = rolls[0]
_, looks, vtr = variance_sim(et, ae, TAU_GRID[ti])
steps = np.arange(T1)
ax[1].plot(steps, surp, "-", color="#27ae60", label="intrinsic surprise")
ax[1].plot(steps, vtr, "-", color="#2980b9", alpha=.7, label="MC-dropout variance (never-look)")
for L in looks:
    ax[1].axvline(L, color="#2980b9", ls=":", alpha=.5)
ax[1].set_xlabel("model-step"); ax[1].set_title("Do variance looks land on surprise peaks? (rollout 0)")
ax[1].legend(); ax[1].grid(alpha=.3)
fig.suptitle("M1.3 -- temporal active sensing on LeWM (when to spend a real observation)", fontweight="bold")
fig.tight_layout(); fig.savefig("/content/lewm-uncertainty/lewm_active_sensing.png", dpi=110)
print("\nsaved lewm_active_sensing.png")
