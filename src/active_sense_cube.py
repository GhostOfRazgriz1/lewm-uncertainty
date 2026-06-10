"""M1.3-cube -- rerun temporal active sensing on the CONTACT-RICH Cube substrate (#3 flip-test).

The M1.3-1.5 sensing nulls were on near-deterministic Push-T; §8 of the note flags that determinism as the
likely culprit. This reruns the M1.3 comparison on `quentinll/lewm-cube` (a contact-rich manipulation
substrate, same LeWM family), to test whether the null was a Push-T artifact: on Cube, error growth should
be more state-dependent (contacts), so the ORACLE headroom over fixed-interval should be larger -- and the
key question is whether the causal MC-variance signal can finally read any of it.

Self-configuring: loads the Cube checkpoint, auto-discovers the swm Cube env id, and derives the frameskip
FS from the action dims (cfg action_encoder input_dim // env action dim). Set ENV_ID below if auto-discovery
picks the wrong env. Mirrors src/active_sense.py (variance / fixed / random / oracle at matched budget).

Run on Colab GPU:  python src/active_sense_cube.py
"""
import sys
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import stable_worldmodel as swm                                   # noqa: F401  registers swm envs
from torchvision import transforms as TT
import matplotlib; matplotlib.use("Agg")                          # noqa: E402
import matplotlib.pyplot as plt                                   # noqa: E402

sys.path.insert(0, "/content/lewm-uncertainty")
from src.load_lewm import load_lewm                               # noqa: E402
from src.schedules import fixed_interval_lookset, random_lookset, oracle_lookset, n_looks   # noqa: E402

REPO = "quentinll/lewm-cube"
ENV_ID = ""                                                       # "" => auto-discover; else set e.g. "swm/OGBench-Cube-v0"
N_ROLLOUTS, T_STEPS, HS, MC = 30, 20, 3, 12
K_GRID = [1, 2, 3, 4, 6, 9, 13, 18]
N_TAU, TARGET_K = 9, 6
T1 = T_STEPS + 1
device = "cuda" if torch.cuda.is_available() else "cpu"
model, cfg = load_lewm("/content/le-wm", repo=REPO, device=device)
prep = TT.Compose([TT.ToTensor(), TT.Resize((224, 224)), TT.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# ---- discover the Cube env + derive frameskip ---------------------------------------------------
if not ENV_ID:
    cands = [e for e in gym.envs.registry if ("cube" in e.lower() or "ogbench" in e.lower()) and "swm" in e.lower()]
    if not cands:
        cands = [e for e in gym.envs.registry if "cube" in e.lower()]
    print("Cube env candidates:", cands)
    assert cands, "no Cube env found in registry -- set ENV_ID manually (see printed swm envs)."
    ENV_ID = cands[0]
print("using ENV_ID =", ENV_ID, flush=True)
_probe = gym.make(ENV_ID, render_mode="rgb_array")
ACT_DIM = int(np.prod(_probe.action_space.shape))                 # env action dim
MODEL_ACT = cfg["action_encoder"]["input_dim"]                    # 25 for cube
assert MODEL_ACT % ACT_DIM == 0, f"action dims mismatch: model {MODEL_ACT} not divisible by env {ACT_DIM}"
FS = MODEL_ACT // ACT_DIM                                         # frameskip so FS*ACT_DIM == model action dim
_probe.close()
print(f"ACT_DIM {ACT_DIM}, model action {MODEL_ACT} => FS {FS}", flush=True)


def rollout(env, T, gen):
    env.reset(seed=int(gen.integers(1_000_000_000)))
    frames = [env.render()]; acts = []
    for _ in range(T):
        blk = [env.action_space.sample().astype("float32").reshape(-1) for _ in range(FS)]
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
    pix = torch.stack([prep(f) for f in frames]).unsqueeze(0).to(device)
    return model.encode({"pixels": pix})["emb"][0]


@torch.no_grad()
def act_encode(acts):
    return model.action_encoder(torch.tensor(acts).unsqueeze(0).to(device))[0]


@torch.no_grad()
def _predict_one(emb_hist, act_hist, mc=0):
    eh, ah = emb_hist.unsqueeze(0), act_hist.unsqueeze(0)
    if mc == 0:
        return model.predict(eh, ah)[0, -1], None
    set_drop(True)
    preds = torch.stack([model.predict(eh, ah)[0, -1] for _ in range(mc)])
    set_drop(False)
    return preds.mean(0), preds.var(0).sum().item()


@torch.no_grad()
def tracking_errors(lookset, emb_true, act_emb):
    looks = set(lookset); zh = []
    for t in range(emb_true.shape[0]):
        if t == 0 or t in looks:
            zh.append(emb_true[t])
        else:
            a = max(0, t - HS); zh.append(_predict_one(torch.stack(zh[a:t]), act_emb[a:t])[0])
    return (torch.stack(zh) - emb_true).norm(dim=-1).cpu().numpy()


@torch.no_grad()
def variance_sim(emb_true, act_emb, tau):
    zh = [emb_true[0]]; looks = [0]
    for t in range(1, emb_true.shape[0]):
        a = max(0, t - HS)
        mu, v = _predict_one(torch.stack(zh[a:t]), act_emb[a:t], mc=MC)
        if v >= tau:
            zh.append(emb_true[t]); looks.append(t)
        else:
            zh.append(mu)
    return (torch.stack(zh) - emb_true).norm(dim=-1).cpu().numpy(), looks


@torch.no_grad()
def intrinsic_surprise(emb_true, act_emb):
    s = [0.0]
    for t in range(1, emb_true.shape[0]):
        a = max(0, t - HS)
        s.append(float((_predict_one(emb_true[a:t], act_emb[a:t])[0] - emb_true[t]).norm()))
    return np.array(s)


def sem(a):
    return a.std(1) / np.sqrt(a.shape[1])


# ---- collect + run all arms ---------------------------------------------------------------------
gen = np.random.default_rng(0)
rolls = []
for r in range(N_ROLLOUTS):
    frames, acts = rollout(gym.make(ENV_ID, render_mode="rgb_array"), T_STEPS, gen)
    et, ae = encode_all(frames), act_encode(acts)
    rolls.append((et, ae, intrinsic_surprise(et, ae)))
    if r % 10 == 0:
        print(f"encoded rollout {r}/{N_ROLLOUTS}", flush=True)

# tau grid from the never-look variance trace (free-running), so the sweep spans empty->full budget
tr = []
for et, ae, _ in rolls:
    zh = [et[0]]
    for t in range(1, T1):
        a = max(0, t - HS); mu, v = _predict_one(torch.stack(zh[a:t]), ae[a:t], mc=MC); tr.append(v); zh.append(mu)
TAU_GRID = np.quantile(np.array(tr), np.linspace(0.02, 0.98, N_TAU))


def lookset_arm(make):
    e = np.zeros((len(K_GRID), N_ROLLOUTS))
    for ki, K in enumerate(K_GRID):
        for ri, (et, ae, surp) in enumerate(rolls):
            e[ki, ri] = tracking_errors(make(K, ri, surp), et, ae).mean()
    return np.array([k / T1 for k in K_GRID]), e


fixed_b, fixed_e = lookset_arm(lambda K, ri, s: fixed_interval_lookset(T1, K))
rand_b, rand_e = lookset_arm(lambda K, ri, s: random_lookset(T1, K, seed=ri))
orac_b, orac_e = lookset_arm(lambda K, ri, s: oracle_lookset(s, K))
var_b = np.zeros((N_TAU, N_ROLLOUTS)); var_e = np.zeros((N_TAU, N_ROLLOUTS))
for ti, tau in enumerate(TAU_GRID):
    for ri, (et, ae, _) in enumerate(rolls):
        err, looks = variance_sim(et, ae, tau); var_e[ti, ri] = err.mean(); var_b[ti, ri] = n_looks(looks) / T1
    print(f"variance tau {ti + 1}/{N_TAU}: budget {var_b[ti].mean():.2f} err {var_e[ti].mean():.3f}", flush=True)

# ---- verdict + cross-substrate comparison -------------------------------------------------------
ki = K_GRID.index(TARGET_K); tgt = TARGET_K / T1
ti = int(np.argmin(np.abs(var_b.mean(1) - tgt)))
head = (fixed_e - orac_e).mean(1); hk = int(np.argmax(head))
print(f"\n==== M1.3-CUBE active sensing at budget ~{tgt:.2f} ====")
print(f"  fixed-interval : {fixed_e[ki].mean():.3f} +/- {sem(fixed_e)[ki]:.3f}")
print(f"  variance (tau)  : {var_e[ti].mean():.3f} +/- {sem(var_e)[ti]:.3f}  (budget {var_b[ti].mean():.2f})")
print(f"  random         : {rand_e[ki].mean():.3f} +/- {sem(rand_e)[ki]:.3f}")
print(f"  oracle (ceiling): {orac_e[ki].mean():.3f} +/- {sem(orac_e)[ki]:.3f}")
print(f"\n  ORACLE HEADROOM over fixed: up to {head[hk]:+.3f} at budget {K_GRID[hk]/T1:.2f}"
      f"  ({100*head[hk]/fixed_e[hk].mean():.0f}% -- compare Push-T's ~30%)")
delta = fixed_e[ki].mean() - var_e[ti].mean(); pooled = np.hypot(sem(fixed_e)[ki], sem(var_e)[ti])
print(f"  variance vs fixed: delta {delta:+.3f} (pooled SEM {pooled:.3f}) -> "
      + ("MC-variance HELPS on Cube (flip!)" if delta > pooled else "still null for MC-variance"))
print("  (If headroom is much larger than Push-T but MC-variance still null -> rerun the learned head"
      " (surprise_head) on Cube; the determinism artifact is confirmed and a sharper signal may now pay off.)")

fig, ax = plt.subplots(figsize=(6.5, 4.8))
for b, e, name, c in [(fixed_b, fixed_e, "fixed-interval", "#7f8c8d"), (rand_b, rand_e, "random", "#bdc3c7"),
                      (var_b.mean(1), var_e, "variance", "#2980b9"), (orac_b, orac_e, "oracle", "#27ae60")]:
    o = np.argsort(b); b2, m, s = np.asarray(b)[o], e.mean(1)[o], sem(e)[o]
    ax.plot(b2, m, "-o", color=c, label=name); ax.fill_between(b2, m - s, m + s, color=c, alpha=.18)
ax.set_xlabel("observation budget (looks / horizon)"); ax.set_ylabel("latent-tracking error")
ax.set_title(f"M1.3 on Cube ({ENV_ID}) -- when to look"); ax.legend(); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig("/content/lewm-uncertainty/lewm_active_sensing_cube.png", dpi=110)
print("\nsaved lewm_active_sensing_cube.png")
